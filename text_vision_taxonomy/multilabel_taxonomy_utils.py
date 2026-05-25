from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from text_vision.config import TAXONOMY_LABELS
from text_vision.data_utils import normalize_transcript


RAW_DATASET_PATH = Path(r"D:\hvc\datasets\action\archieve\final_dataset.csv")
TEXT_FIELDS = ["Scene", "Action", "Subcategories", "ParentLabels", "action_clean"]
LABEL_COLUMNS = list(TAXONOMY_LABELS)
TRUE_PREFIX = "true_"
PROB_PREFIX = "prob_"
PRED_PREFIX = "pred_"
RARE_LABELS = {"threat", "illegal", "online_harm"}
RARE_SUPPORT_CUTOFF = 20
DEFAULT_THRESHOLD_FLOORS = {
    "threat": 0.90,
    "illegal": 0.95,
    "online_harm": 0.90,
    "sexual": 0.75,
    "fear": 0.75,
}
PRECISION_TARGETS = [0.60, 0.70, 0.80, 0.90]


def default_dataset_path() -> Path:
    if RAW_DATASET_PATH.exists():
        return RAW_DATASET_PATH
    return Path("data/final_dataset.csv")


def clean_label(value: Any) -> str:
    return str(value).strip().strip("\"'").lower()


def parse_label_list(value: Any) -> List[str]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = text.split("|") if "|" in text else text.split(",")
    if isinstance(parsed, str):
        labels = [parsed]
    elif isinstance(parsed, Iterable):
        labels = list(parsed)
    else:
        labels = []
    return [clean_label(label) for label in labels if clean_label(label)]


def text_block(row: Mapping[str, Any], fields: Sequence[str] = TEXT_FIELDS) -> str:
    parts: List[str] = []
    for field in fields:
        text = normalize_transcript(row.get(field, ""))
        if text:
            parts.append(f"[{field.upper()}]\n{text}")
    return "\n\n".join(parts)


def ensure_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    parsed_labels: List[List[str]] | None = None
    if "final_labels" in frame.columns:
        parsed_labels = [parse_label_list(value) for value in frame["final_labels"]]
    for label in LABEL_COLUMNS:
        if parsed_labels is not None:
            frame[label] = [int(label in labels) for labels in parsed_labels]
        elif label in frame.columns:
            frame[label] = pd.to_numeric(frame[label], errors="coerce").fillna(0).astype(int)
        elif f"tax_{label}" in frame.columns:
            frame[label] = pd.to_numeric(frame[f"tax_{label}"], errors="coerce").fillna(0).astype(int)
        else:
            frame[label] = 0
    if "neutral" not in frame.columns:
        label_sum = frame[LABEL_COLUMNS].sum(axis=1)
        frame["neutral"] = label_sum.eq(0).astype(int)
    return frame


def make_group_splits(
    frame: pd.DataFrame,
    group_column: str = "File Name",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, str]:
    groups = sorted(str(value) for value in frame[group_column].fillna("").unique() if str(value).strip() and str(value).lower() != "nan")
    rng = random.Random(int(seed))
    rng.shuffle(groups)
    n_train = int(round(len(groups) * float(train_ratio)))
    n_val = int(round(len(groups) * float(val_ratio)))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    return {
        group: "train" if group in train_groups else "val" if group in val_groups else "test"
        for group in groups
    }


def safe_file_stem(value: Any, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    try:
        return Path(text).stem or fallback
    except TypeError:
        return fallback


def load_multilabel_dataset(
    path: str | Path | None = None,
    seed: int = 42,
    split_path: str | Path | None = None,
    save_split_path: str | Path | None = None,
) -> pd.DataFrame:
    source = Path(path) if path else default_dataset_path()
    frame = ensure_label_columns(pd.read_csv(source))
    if "File Name" not in frame.columns:
        if "file_name" in frame.columns:
            frame["File Name"] = frame["file_name"]
        elif "video_file_name" in frame.columns:
            frame["File Name"] = frame["video_file_name"]
        else:
            raise ValueError("Dataset must contain File Name, file_name, or video_file_name")
    frame["row_id"] = np.arange(len(frame), dtype=int)
    frame["source_video_id"] = [
        safe_file_stem(value, f"row_{index}") for index, value in enumerate(frame["File Name"].tolist())
    ]
    frame["text_input"] = [text_block(row) for _, row in frame.iterrows()]
    synthetic_values = frame["synthetic"] if "synthetic" in frame.columns else pd.Series(0, index=frame.index)
    frame["synthetic"] = pd.to_numeric(synthetic_values, errors="coerce").fillna(0).astype(int)

    if split_path and Path(split_path).exists():
        split_frame = pd.read_csv(split_path)
        split_map = dict(zip(split_frame["File Name"].astype(str), split_frame["split"].astype(str)))
    else:
        split_map = make_group_splits(frame, seed=seed)
        if save_split_path:
            output = Path(save_split_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"File Name": group, "split": split} for group, split in sorted(split_map.items())]
            ).to_csv(output, index=False)
    frame["split"] = frame["File Name"].astype(str).map(split_map).fillna("train")
    frame.loc[frame["synthetic"].astype(int).eq(1), "split"] = "train"
    return frame


def target_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL_COLUMNS].astype(int).to_numpy()


def add_prediction_columns(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    thresholds = thresholds or {label: 0.5 for label in LABEL_COLUMNS}
    y_true = target_matrix(frame)
    for row_index, (_, row) in enumerate(frame.iterrows()):
        payload = {
            "row_id": int(row.get("row_id", row_index)),
            "file_name": row.get("File Name", row.get("file_name", "")),
            "source_video_id": row.get("source_video_id", ""),
            "split": row.get("split", ""),
            "synthetic": int(row.get("synthetic", 0)),
            "text_input": row.get("text_input", ""),
        }
        for label_index, label in enumerate(LABEL_COLUMNS):
            prob = float(probabilities[row_index, label_index])
            payload[f"{TRUE_PREFIX}{label}"] = int(y_true[row_index, label_index])
            payload[f"{PROB_PREFIX}{label}"] = prob
            payload[f"{PRED_PREFIX}{label}"] = int(prob >= float(thresholds.get(label, 0.5)))
        rows.append(payload)
    return pd.DataFrame(rows)


def probability_columns(prefix: str = PROB_PREFIX, labels: Sequence[str] = LABEL_COLUMNS) -> List[str]:
    return [f"{prefix}{label}" for label in labels]


def true_columns(labels: Sequence[str] = LABEL_COLUMNS) -> List[str]:
    return [f"{TRUE_PREFIX}{label}" for label in labels]


def load_thresholds(path: str | Path | None, labels: Sequence[str] = LABEL_COLUMNS, default: float = 0.5) -> Dict[str, float]:
    if not path or not Path(path).exists():
        return {label: float(default) for label in labels}
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(blob, Mapping):
        if "production_thresholds" in blob:
            values = blob["production_thresholds"]
        elif "thresholds" in blob:
            values = blob["thresholds"]
        else:
            values = blob
    else:
        values = {}
    return {label: float(values.get(label, default)) for label in labels}


def metric_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = y_prob >= float(threshold)
    true = y_true.astype(bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    predicted = int(pred.sum())
    support = int(true.sum())
    precision = float(tp / predicted) if predicted else 0.0
    recall = float(tp / support) if support else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if precision + recall else 0.0
    return {
        "support": support,
        "predicted_positive_count": predicted,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_metric(fn, *args, default: float = 0.0, **kwargs) -> float:
    try:
        value = float(fn(*args, **kwargs))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def evaluate_probabilities(
    predictions: pd.DataFrame,
    thresholds: Mapping[str, float],
    system: str = "model",
    labels: Sequence[str] = LABEL_COLUMNS,
    use_final_labels: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    y_true = predictions[[f"{TRUE_PREFIX}{label}" for label in labels]].astype(int).to_numpy()
    y_prob = predictions[[f"{PROB_PREFIX}{label}" for label in labels]].astype(float).to_numpy()
    if use_final_labels and "final_labels" in predictions.columns:
        y_pred = np.zeros_like(y_true)
        for row_index, value in enumerate(predictions["final_labels"].fillna("")):
            selected = set(str(value).split("|")) if value else set()
            for label_index, label in enumerate(labels):
                y_pred[row_index, label_index] = int(label in selected)
    else:
        threshold_values = np.array([float(thresholds.get(label, 0.5)) for label in labels]).reshape(1, -1)
        y_pred = (y_prob >= threshold_values).astype(int)
    neutral = y_true.sum(axis=1) == 0
    rare_indices = [labels.index(label) for label in labels if label in RARE_LABELS]
    summary = {
        "system": system,
        "rows": int(len(predictions)),
        "micro_precision": _safe_metric(precision_score, y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": _safe_metric(recall_score, y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": _safe_metric(f1_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": _safe_metric(precision_score, y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": _safe_metric(recall_score, y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": _safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "neutral_false_positive_rate": float(((y_pred.sum(axis=1) > 0) & neutral).sum() / max(int(neutral.sum()), 1)),
    }
    if rare_indices:
        rare_true = y_true[:, rare_indices]
        rare_pred = y_pred[:, rare_indices]
        summary["rare_false_positive_count"] = int(((rare_pred == 1) & (rare_true == 0)).sum())
        summary["rare_predicted_positive_count"] = int(rare_pred.sum())
        summary["rare_true_positive_count"] = int(rare_true.sum())
    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        counts = metric_counts(y_true[:, index], y_pred[:, index].astype(float), 0.5)
        rows.append(
            {
                "system": system,
                "label": label,
                "threshold": float(thresholds.get(label, 0.5)),
                "true_count": counts["support"],
                "predicted_count": counts["predicted_positive_count"],
                **counts,
            }
        )
    return summary, pd.DataFrame(rows)


def candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.01, 0.99, 99)
    extras = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95])
    values = np.unique(np.concatenate([grid, extras, y_prob.astype(float)]))
    return np.clip(np.sort(values), 0.0, 1.0)


def threshold_for_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds(y_prob):
        score = metric_counts(y_true, y_prob, float(threshold))["f1"]
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def threshold_for_precision(y_true: np.ndarray, y_prob: np.ndarray, target: float) -> Tuple[float, bool]:
    best: Tuple[float, float, float] | None = None
    for threshold in candidate_thresholds(y_prob):
        metrics = metric_counts(y_true, y_prob, float(threshold))
        if metrics["predicted_positive_count"] == 0:
            continue
        if metrics["precision"] >= float(target):
            candidate = (metrics["recall"], metrics["precision"], float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return 0.99, False
    return best[2], True


def threshold_for_prevalence(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    support = int(y_true.sum())
    if support <= 0:
        return 0.99
    sorted_probs = np.sort(y_prob.astype(float))[::-1]
    index = min(support - 1, len(sorted_probs) - 1)
    return float(np.clip(sorted_probs[index], 0.0, 1.0))


def calibrate_thresholds(
    predictions: pd.DataFrame,
    labels: Sequence[str] = LABEL_COLUMNS,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    thresholds: Dict[str, Dict[str, float]] = {
        "max_f1": {},
        "top_k_by_prevalence": {},
        "production": {},
    }
    for target in PRECISION_TARGETS:
        thresholds[f"precision_{int(target * 100)}"] = {}
    support_by_label: Dict[str, int] = {}
    for label in labels:
        y_true = predictions[f"{TRUE_PREFIX}{label}"].astype(int).to_numpy()
        y_prob = predictions[f"{PROB_PREFIX}{label}"].astype(float).to_numpy()
        support = int(y_true.sum())
        support_by_label[label] = support
        thresholds["max_f1"][label] = threshold_for_max_f1(y_true, y_prob)
        thresholds["top_k_by_prevalence"][label] = threshold_for_prevalence(y_true, y_prob)
        precision_ok: Dict[str, bool] = {}
        for target in PRECISION_TARGETS:
            key = f"precision_{int(target * 100)}"
            threshold, ok = threshold_for_precision(y_true, y_prob, target)
            thresholds[key][label] = threshold
            precision_ok[key] = ok
        is_rare = label in RARE_LABELS
        if is_rare:
            base = thresholds["precision_90"][label] if precision_ok["precision_90"] else 0.99
        else:
            base = thresholds["precision_80"][label] if precision_ok["precision_80"] else thresholds["top_k_by_prevalence"][label]
            base = max(base, thresholds["top_k_by_prevalence"][label])
        floor = DEFAULT_THRESHOLD_FLOORS.get(label, 0.0)
        thresholds["production"][label] = float(max(base, floor))

    report_rows: List[Dict[str, Any]] = []
    for mode, values in thresholds.items():
        for label in labels:
            y_true = predictions[f"{TRUE_PREFIX}{label}"].astype(int).to_numpy()
            y_prob = predictions[f"{PROB_PREFIX}{label}"].astype(float).to_numpy()
            threshold = float(values[label])
            report_rows.append(
                {
                    "mode": mode,
                    "label": label,
                    "threshold": threshold,
                    "rare_label": int(label in RARE_LABELS),
                    **metric_counts(y_true, y_prob, threshold),
                }
            )
    return thresholds, pd.DataFrame(report_rows)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_json_map(value: str | None) -> Dict[str, float]:
    if not value:
        return {}
    text = str(value)
    path = Path(text)
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
    else:
        blob = json.loads(text)
    if not isinstance(blob, Mapping):
        return {}
    return {str(key): float(val) for key, val in blob.items()}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "model"

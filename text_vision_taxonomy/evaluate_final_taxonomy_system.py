from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS
from text_vision.ensemble_taxonomy import load_thresholds


def safe_metric(default: float, fn, *args, **kwargs) -> float:
    try:
        value = fn(*args, **kwargs)
        if value is None or not np.isfinite(value):
            return default
        return float(value)
    except Exception:
        return default


def evaluate_predictions(name: str, predictions: pd.DataFrame, thresholds: Mapping[str, float]) -> tuple[Dict[str, Any], pd.DataFrame]:
    y_true = predictions[[f"true_{label}" for label in TAXONOMY_LABELS]].astype(int).to_numpy()
    y_prob = predictions[[f"prob_{label}" for label in TAXONOMY_LABELS]].astype(float).to_numpy()
    threshold_values = np.array([float(thresholds.get(label, 0.5)) for label in TAXONOMY_LABELS]).reshape(1, -1)
    y_pred = (y_prob >= threshold_values).astype(int)
    summary = {
        "system": name,
        "rows": int(len(predictions)),
        "micro_precision": safe_metric(0.0, precision_score, y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": safe_metric(0.0, recall_score, y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": safe_metric(0.0, f1_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": safe_metric(0.0, precision_score, y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": safe_metric(0.0, recall_score, y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": safe_metric(0.0, f1_score, y_true, y_pred, average="macro", zero_division=0),
    }
    neutral = y_true.sum(axis=1) == 0
    summary["neutral_false_positive_rate"] = float(((y_pred.sum(axis=1) > 0) & neutral).sum() / max(int(neutral.sum()), 1))
    rare_indices = [TAXONOMY_LABELS.index(label) for label in ["threat", "illegal", "online_harm"]]
    rare_true = y_true[:, rare_indices]
    rare_pred = y_pred[:, rare_indices]
    summary["rare_label_precision"] = safe_metric(0.0, precision_score, rare_true, rare_pred, average="micro", zero_division=0)
    summary["rare_label_recall"] = safe_metric(0.0, recall_score, rare_true, rare_pred, average="micro", zero_division=0)

    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(TAXONOMY_LABELS):
        true_count = int(y_true[:, index].sum())
        pred_count = int(y_pred[:, index].sum())
        tp = int(((y_true[:, index] == 1) & (y_pred[:, index] == 1)).sum())
        fp = int(((y_true[:, index] == 0) & (y_pred[:, index] == 1)).sum())
        fn = int(((y_true[:, index] == 1) & (y_pred[:, index] == 0)).sum())
        rows.append(
            {
                "system": name,
                "label": label,
                "threshold": float(thresholds.get(label, 0.5)),
                "true_positive_label_count": true_count,
                "predicted_positive_count": pred_count,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": float(tp / pred_count) if pred_count else 0.0,
                "recall": float(tp / true_count) if true_count else 0.0,
                "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "false_positive_rate": float(fp / max((y_true[:, index] == 0).sum(), 1)),
            }
        )
    return summary, pd.DataFrame(rows)


def maybe_eval(name: str, path: str, thresholds: Mapping[str, float]) -> tuple[Dict[str, Any], pd.DataFrame] | None:
    if not path or not Path(path).exists():
        return None
    return evaluate_predictions(name, pd.read_csv(path), thresholds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare taxonomy systems with precision-focused metrics.")
    parser.add_argument("--thresholds", default="reports/taxonomy_head/calibrated_thresholds.json")
    parser.add_argument("--taxonomy-head-original", default="reports/taxonomy_head/predictions.csv")
    parser.add_argument("--taxonomy-head-original-thresholds", default="reports/taxonomy_head/calibrated_thresholds.json")
    parser.add_argument("--taxonomy-head-no-pos-weight", default="reports/taxonomy_head_no_pos_weight/predictions.csv")
    parser.add_argument("--taxonomy-head-no-pos-weight-thresholds", default="reports/taxonomy_head_no_pos_weight/calibrated_thresholds.json")
    parser.add_argument("--taxonomy-head-focal", default="reports/taxonomy_head_focal/predictions.csv")
    parser.add_argument("--taxonomy-head-focal-thresholds", default="reports/taxonomy_head_focal/calibrated_thresholds.json")
    parser.add_argument("--taxonomy-head-capped-pos-weight", default="reports/taxonomy_head_capped_pos_weight/predictions.csv")
    parser.add_argument("--taxonomy-head-capped-pos-weight-thresholds", default="reports/taxonomy_head_capped_pos_weight/calibrated_thresholds.json")
    parser.add_argument("--text-only", default="reports/text_taxonomy_classifier/predictions.csv")
    parser.add_argument("--text-only-thresholds", default="reports/text_taxonomy_classifier/calibrated_thresholds.json")
    parser.add_argument("--ensemble", default="reports/taxonomy_ensemble/predictions.csv")
    parser.add_argument("--ensemble-thresholds", default="reports/taxonomy_ensemble/calibrated_thresholds.json")
    parser.add_argument("--output-dir", default="reports/final_taxonomy_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = [
        ("taxonomy_head_original", args.taxonomy_head_original, args.taxonomy_head_original_thresholds),
        ("taxonomy_head_no_pos_weight", args.taxonomy_head_no_pos_weight, args.taxonomy_head_no_pos_weight_thresholds),
        ("taxonomy_head_capped_pos_weight", args.taxonomy_head_capped_pos_weight, args.taxonomy_head_capped_pos_weight_thresholds),
        ("taxonomy_head_focal", args.taxonomy_head_focal, args.taxonomy_head_focal_thresholds),
        ("text_only", args.text_only, args.text_only_thresholds),
        ("ensemble", args.ensemble, args.ensemble_thresholds),
    ]
    summaries = []
    per_label_frames = []
    fallback_thresholds = load_thresholds(args.thresholds)
    for name, path, threshold_path in systems:
        thresholds = load_thresholds(threshold_path) if threshold_path and Path(threshold_path).exists() else fallback_thresholds
        result = maybe_eval(name, path, thresholds)
        if result is None:
            continue
        summary, per_label = result
        summaries.append(summary)
        per_label_frames.append(per_label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    per_label_frame = pd.concat(per_label_frames, ignore_index=True) if per_label_frames else pd.DataFrame()
    summary_frame.to_csv(output_dir / "metrics.csv", index=False)
    per_label_frame.to_csv(output_dir / "per_label_metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()

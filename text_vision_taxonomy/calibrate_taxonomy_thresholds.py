from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS


RARE_SUPPORT_CUTOFF = 10
RARE_PRODUCTION_THRESHOLDS = {
    "threat": 0.90,
    "illegal": 0.95,
    "online_harm": 0.90,
    "sexual": 0.85,
}
PRECISION_TARGETS = {
    "precision_target_70": 0.70,
    "precision_target_80": 0.80,
    "precision_target_90": 0.90,
}


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
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predicted_positive_count": predicted,
    }


def _candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.99, 95)
    values = np.unique(np.concatenate([grid, y_prob.astype(float), np.array([0.5, 0.7, 0.8, 0.9, 0.95])]))
    return np.clip(np.sort(values), 0.0, 1.0)


def f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in _candidate_thresholds(y_prob):
        metrics = metric_counts(y_true, y_prob, float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def precision_target_threshold(y_true: np.ndarray, y_prob: np.ndarray, target: float) -> float:
    best: Tuple[float, float, float] | None = None
    for threshold in _candidate_thresholds(y_prob):
        metrics = metric_counts(y_true, y_prob, float(threshold))
        if metrics["predicted_positive_count"] == 0:
            continue
        if metrics["precision"] >= float(target):
            candidate = (metrics["recall"], metrics["precision"], float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is not None:
        return best[2]
    return 0.99


def top_k_by_prevalence_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    support = int(y_true.astype(int).sum())
    if support <= 0:
        return 0.99
    sorted_probs = np.sort(y_prob.astype(float))[::-1]
    index = min(support - 1, len(sorted_probs) - 1)
    return float(np.clip(sorted_probs[index], 0.0, 1.0))


def thresholds_for_predictions(predictions: pd.DataFrame, labels: Iterable[str] = TAXONOMY_LABELS) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {
        "f1_optimal": {},
        "precision_target_70": {},
        "precision_target_80": {},
        "precision_target_90": {},
        "top_k_by_prevalence": {},
        "production": {},
    }
    for label in labels:
        y_true = predictions[f"true_{label}"].astype(int).to_numpy()
        y_prob = predictions[f"prob_{label}"].astype(float).to_numpy()
        support = int(y_true.sum())
        thresholds["f1_optimal"][label] = f1_optimal_threshold(y_true, y_prob)
        for mode, target in PRECISION_TARGETS.items():
            thresholds[mode][label] = precision_target_threshold(y_true, y_prob, target)
        thresholds["top_k_by_prevalence"][label] = top_k_by_prevalence_threshold(y_true, y_prob)
        if support < RARE_SUPPORT_CUTOFF:
            thresholds["production"][label] = float(RARE_PRODUCTION_THRESHOLDS.get(label, max(0.85, thresholds["precision_target_90"][label])))
        else:
            thresholds["production"][label] = float(max(thresholds["precision_target_80"][label], thresholds["top_k_by_prevalence"][label]))
    return thresholds


def threshold_report_rows(
    predictions: pd.DataFrame,
    thresholds_by_mode: Mapping[str, Mapping[str, float]],
    labels: Iterable[str] = TAXONOMY_LABELS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode, thresholds in thresholds_by_mode.items():
        for label in labels:
            y_true = predictions[f"true_{label}"].astype(int).to_numpy()
            y_prob = predictions[f"prob_{label}"].astype(float).to_numpy()
            threshold = float(thresholds[label])
            rows.append(
                {
                    "mode": mode,
                    "label": label,
                    "threshold": threshold,
                    **metric_counts(y_true, y_prob, threshold),
                    "rare_label": int(y_true.sum()) < RARE_SUPPORT_CUTOFF,
                }
            )
    return rows


def calibrate(predictions_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    predictions = pd.read_csv(predictions_path)
    missing = [column for label in TAXONOMY_LABELS for column in (f"true_{label}", f"prob_{label}") if column not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions file is missing columns: {missing}")

    thresholds = thresholds_for_predictions(predictions)
    report = pd.DataFrame(threshold_report_rows(predictions, thresholds))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.parent / "threshold_report.csv"
    report.to_csv(report_path, index=False)
    payload = {
        "labels": list(TAXONOMY_LABELS),
        "production_mode": "precision_target_80_or_top_k_with_rare_overrides",
        "rare_support_cutoff": RARE_SUPPORT_CUTOFF,
        "rare_production_thresholds": dict(RARE_PRODUCTION_THRESHOLDS),
        "production_thresholds": thresholds["production"],
        "thresholds_by_mode": thresholds,
        "threshold_report": str(report_path),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate taxonomy thresholds for precision-first production use.")
    parser.add_argument("--predictions", default="reports/taxonomy_head/predictions.csv")
    parser.add_argument("--output", default="reports/taxonomy_head/calibrated_thresholds.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = calibrate(args.predictions, args.output)
    print(json.dumps({"output": args.output, "production_thresholds": payload["production_thresholds"]}, indent=2))


if __name__ == "__main__":
    main()

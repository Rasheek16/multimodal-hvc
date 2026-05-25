from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS
from text_vision.ensemble_taxonomy import load_thresholds


RARE_REVIEW_LABELS = {"threat", "illegal", "online_harm"}


def selected_labels(row: Mapping[str, object], thresholds: Mapping[str, float], prefix: str = "prob_") -> List[str]:
    labels = []
    for label in TAXONOMY_LABELS:
        if float(row.get(f"{prefix}{label}", 0.0) or 0.0) >= float(thresholds.get(label, 0.5)):
            labels.append(label)
    return labels


def true_labels(row: Mapping[str, object]) -> List[str]:
    return [label for label in TAXONOMY_LABELS if int(row.get(f"true_{label}", 0) or 0) == 1]


def review_reasons(row: Mapping[str, object], thresholds: Mapping[str, float]) -> List[str]:
    pred = set(selected_labels(row, thresholds))
    true = set(true_labels(row))
    reasons: List[str] = []
    if pred & RARE_REVIEW_LABELS:
        reasons.append("rare label predicted")
    if not true and pred:
        reasons.append("neutral true label but high taxonomy probability")
    if len(pred) >= 4:
        reasons.append("many labels predicted for one segment")
    if float(row.get("prob_hate", 1.0) or 1.0) >= 0.65 and not pred:
        reasons.append("prob_hate high but no taxonomy label")
    if float(row.get("prob_hate", 1.0) or 1.0) < 0.35 and (pred & {"hate_speech", "discrimination", "contextual_hate", "threat", "illegal", "online_harm"}):
        reasons.append("taxonomy labels high but binary hate low")
    if "suppressed_labels" in row and str(row.get("suppressed_labels", "")).strip():
        reasons.append("labels suppressed by guardrails")
    return reasons


def disagreement_reasons(row: Mapping[str, object], thresholds: Mapping[str, float]) -> List[str]:
    reasons = []
    for label in TAXONOMY_LABELS:
        video_col = f"video_prob_{label}"
        text_col = f"text_prob_{label}"
        if video_col in row and text_col in row:
            video_pred = float(row.get(video_col, 0.0) or 0.0) >= thresholds.get(label, 0.5)
            text_pred = float(row.get(text_col, 0.0) or 0.0) >= thresholds.get(label, 0.5)
            if video_pred != text_pred:
                reasons.append(f"video/text disagree on {label}")
    return reasons


def build_review_queue(predictions: pd.DataFrame, thresholds: Mapping[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    review_rows = []
    accept_rows = []
    reason_counts: Dict[str, int] = {}
    for _, row in predictions.iterrows():
        reasons = review_reasons(row, thresholds) + disagreement_reasons(row, thresholds)
        payload = row.to_dict()
        payload["review_reasons"] = "|".join(reasons)
        if reasons:
            review_rows.append(payload)
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            accept_rows.append(payload)
    return pd.DataFrame(review_rows), pd.DataFrame(accept_rows), reason_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create precision-focused taxonomy review and auto-accept queues.")
    parser.add_argument("--predictions", default="reports/taxonomy_ensemble/predictions.csv")
    parser.add_argument("--thresholds", default="reports/taxonomy_head/calibrated_thresholds.json")
    parser.add_argument("--output-dir", default="reports/taxonomy_review_queue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    thresholds = load_thresholds(args.thresholds)
    review, auto_accept, reason_counts = build_review_queue(predictions, thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_dir / "review_queue.csv", index=False)
    auto_accept.to_csv(output_dir / "auto_accept.csv", index=False)
    summary = {
        "rows": int(len(predictions)),
        "review_rows": int(len(review)),
        "auto_accept_rows": int(len(auto_accept)),
        "reason_counts": reason_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

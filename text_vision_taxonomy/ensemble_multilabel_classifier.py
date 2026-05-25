from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, evaluate_probabilities, load_thresholds, write_json
from text_vision.rare_label_rules import apply_rules_to_predictions


SEMANTIC_LABELS = {"hate_speech", "discrimination", "contextual_hate", "illegal", "online_harm"}
VISUAL_LABELS = {"violence", "sexual", "fear"}


def read_predictions(path: str | Path | None) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    return pd.read_csv(path)


def rename_prob_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = frame.copy()
    for label in LABEL_COLUMNS:
        if f"prob_{label}" in renamed.columns:
            renamed = renamed.rename(columns={f"prob_{label}": f"{prefix}_prob_{label}"})
    return renamed


def merge_sources(tfidf: pd.DataFrame | None, transformer: pd.DataFrame | None, video: pd.DataFrame | None) -> pd.DataFrame:
    base = transformer if transformer is not None else tfidf if tfidf is not None else video
    if base is None:
        raise ValueError("At least one prediction source is required")
    keys = ["row_id"] if "row_id" in base.columns else [key for key in ["file_name", "source_video_id", "split"] if key in base.columns]
    merged = rename_prob_columns(base, "transformer" if transformer is not None else "tfidf" if tfidf is not None else "video")
    for source_name, frame in [("tfidf", tfidf), ("transformer", transformer), ("video", video)]:
        if frame is None or frame is base:
            continue
        source = rename_prob_columns(frame, source_name)
        keep = keys + [f"{source_name}_prob_{label}" for label in LABEL_COLUMNS if f"{source_name}_prob_{label}" in source.columns]
        merged = merged.merge(source[keep], on=keys, how="left")
    return merged


def source_prob(row: Mapping[str, Any], source: str, label: str) -> float | None:
    key = f"{source}_prob_{label}"
    value = row.get(key, None)
    if value is None or pd.isna(value):
        return None
    return float(value)


def weighted_prob(label: str, tfidf_prob: float | None, transformer_prob: float | None, video_prob: float | None) -> float:
    text_prob = transformer_prob if transformer_prob is not None else tfidf_prob
    if text_prob is None and tfidf_prob is not None:
        text_prob = tfidf_prob
    if text_prob is None and video_prob is None:
        return 0.0
    if text_prob is None:
        return float(video_prob or 0.0)
    if video_prob is None:
        if transformer_prob is not None and tfidf_prob is not None:
            return float(0.85 * transformer_prob + 0.15 * tfidf_prob)
        return float(text_prob)
    tfidf_value = float(tfidf_prob if tfidf_prob is not None else text_prob)
    transformer_value = float(transformer_prob if transformer_prob is not None else text_prob)
    video_value = float(video_prob)
    if label in SEMANTIC_LABELS:
        return 0.75 * transformer_value + 0.15 * tfidf_value + 0.10 * video_value
    if label in VISUAL_LABELS:
        return 0.55 * transformer_value + 0.25 * tfidf_value + 0.20 * video_value
    if label == "threat":
        return 0.50 * transformer_value + 0.20 * tfidf_value + 0.30 * video_value
    return 0.70 * transformer_value + 0.20 * tfidf_value + 0.10 * video_value


def ensemble_predictions(
    tfidf: pd.DataFrame | None,
    transformer: pd.DataFrame | None,
    video: pd.DataFrame | None,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    merged = merge_sources(tfidf, transformer, video)
    rows = []
    for _, row in merged.iterrows():
        payload = row.to_dict()
        for label in LABEL_COLUMNS:
            payload[f"prob_{label}"] = weighted_prob(
                label,
                source_prob(row, "tfidf", label),
                source_prob(row, "transformer", label),
                source_prob(row, "video", label),
            )
            if f"true_{label}" not in payload:
                for source in ["transformer", "tfidf", "video"]:
                    source_true = row.get(f"{source}_true_{label}", None)
                    if source_true is not None and not pd.isna(source_true):
                        payload[f"true_{label}"] = int(source_true)
                        break
            payload[f"pred_{label}"] = int(float(payload[f"prob_{label}"]) >= float(thresholds.get(label, 0.5)))
        rows.append(payload)
    predictions = pd.DataFrame(rows)
    return apply_rules_to_predictions(predictions, thresholds)


def run(args: argparse.Namespace) -> None:
    tfidf = read_predictions(args.tfidf_predictions)
    transformer = read_predictions(args.transformer_predictions)
    video = read_predictions(args.video_predictions)
    thresholds = load_thresholds(args.thresholds)
    predictions = ensemble_predictions(tfidf, transformer, video, thresholds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    eval_frame = predictions[predictions.get("synthetic", 0).astype(int).eq(0)] if "synthetic" in predictions.columns else predictions
    if all(f"true_{label}" in eval_frame.columns for label in LABEL_COLUMNS):
        summary, per_label = evaluate_probabilities(eval_frame, thresholds, system="ensemble")
        per_label.to_csv(output.parent / "per_label_metrics.csv", index=False)
        write_json(output.parent / "metrics.json", summary)
    else:
        summary = {"rows": int(len(predictions))}
    print(json.dumps({"output": str(output), "rows": len(predictions), "metrics": summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble text, transformer, and optional video taxonomy probabilities.")
    parser.add_argument("--tfidf-predictions", default="reports/text_multilabel_classifier/predictions.csv")
    parser.add_argument("--transformer-predictions", default="reports/transformer_multilabel_classifier/predictions.csv")
    parser.add_argument("--video-predictions", default="")
    parser.add_argument("--thresholds", default="reports/multilabel_calibration/thresholds.json")
    parser.add_argument("--output", default="reports/multilabel_ensemble/predictions.csv")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

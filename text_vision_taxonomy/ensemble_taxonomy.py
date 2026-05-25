from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS
from text_vision.taxonomy_postprocessing import postprocess_taxonomy


SEMANTIC_LABELS = {"hate_speech", "discrimination", "contextual_hate", "illegal", "online_harm"}
VISUAL_LABELS = {"violence", "sexual", "fear"}


def load_thresholds(path: str | Path | None) -> Dict[str, float]:
    if not path or not Path(path).exists():
        return {label: 0.5 for label in TAXONOMY_LABELS}
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if "production_thresholds" in blob:
        values = blob["production_thresholds"]
    elif "thresholds" in blob:
        values = blob["thresholds"]
    else:
        values = blob
    return {label: float(values.get(label, 0.5)) for label in TAXONOMY_LABELS}


def combine_probs(label: str, video_prob: float, text_prob: float, has_text: bool = True) -> float:
    if not has_text:
        return float(video_prob)
    if label in SEMANTIC_LABELS:
        return float(0.70 * text_prob + 0.30 * video_prob)
    if label in VISUAL_LABELS:
        return float(0.65 * video_prob + 0.35 * text_prob)
    if label == "threat":
        return float(0.50 * text_prob + 0.50 * video_prob)
    return float(0.50 * text_prob + 0.50 * video_prob)


def merge_predictions(video: pd.DataFrame, text: pd.DataFrame | None = None) -> pd.DataFrame:
    keys = [key for key in ["source_video_id", "segment_id", "split"] if key in video.columns]
    if text is None or text.empty:
        merged = video.copy()
        for label in TAXONOMY_LABELS:
            merged[f"video_prob_{label}"] = merged[f"prob_{label}"]
            merged[f"text_prob_{label}"] = 0.0
        return merged
    text_cols = keys + [f"prob_{label}" for label in TAXONOMY_LABELS]
    text_renamed = text[text_cols].rename(columns={f"prob_{label}": f"text_prob_{label}" for label in TAXONOMY_LABELS})
    video_renamed = video.rename(columns={f"prob_{label}": f"video_prob_{label}" for label in TAXONOMY_LABELS})
    return video_renamed.merge(text_renamed, on=keys, how="left")


def ensemble_predictions(
    video_predictions: pd.DataFrame,
    text_predictions: pd.DataFrame | None,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    merged = merge_predictions(video_predictions, text_predictions)
    rows = []
    for _, row in merged.iterrows():
        payload = row.to_dict()
        probs = {}
        has_text = not any(pd.isna(row.get(f"text_prob_{label}", None)) for label in TAXONOMY_LABELS)
        for label in TAXONOMY_LABELS:
            video_prob = float(row.get(f"video_prob_{label}", row.get(f"prob_{label}", 0.0)) or 0.0)
            text_prob = float(row.get(f"text_prob_{label}", 0.0) or 0.0)
            probs[label] = combine_probs(label, video_prob, text_prob, has_text=has_text)
            payload[f"prob_{label}"] = probs[label]
            if f"true_{label}" not in payload and f"video_true_{label}" in payload:
                payload[f"true_{label}"] = payload[f"video_true_{label}"]
        processed = postprocess_taxonomy(
            probs,
            thresholds,
            prob_hate=float(row.get("prob_hate", 1.0) or 1.0),
            evidence_text=str(row.get("text_input", "")),
        )
        payload["predicted_labels"] = "|".join(processed["labels"])
        payload["suppressed_labels"] = "|".join(item["label"] for item in processed["suppressed"])
        rows.append(payload)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble video and text taxonomy probabilities.")
    parser.add_argument("--video-predictions", default="reports/taxonomy_head/predictions.csv")
    parser.add_argument("--text-predictions", default="reports/text_taxonomy_classifier/predictions.csv")
    parser.add_argument("--thresholds", default="reports/taxonomy_head/calibrated_thresholds.json")
    parser.add_argument("--output", default="reports/taxonomy_ensemble/predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = pd.read_csv(args.video_predictions)
    text = pd.read_csv(args.text_predictions) if args.text_predictions and Path(args.text_predictions).exists() else None
    thresholds = load_thresholds(args.thresholds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = ensemble_predictions(video, text, thresholds)
    predictions.to_csv(output, index=False)
    print(json.dumps({"output": str(output), "rows": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()

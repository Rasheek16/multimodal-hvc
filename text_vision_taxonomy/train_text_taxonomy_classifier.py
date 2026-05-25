from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS
from text_vision.data_utils import normalize_transcript
from text_vision.calibrate_taxonomy_thresholds import threshold_report_rows, thresholds_for_predictions
from text_vision.train_taxonomy_head import TAX_COLUMNS, taxonomy_metrics


TEXT_COLUMNS = [
    "text_for_taxonomy",
    "Scene",
    "Action",
    "Subcategories",
    "ParentLabels",
    "action_clean",
    "transcription",
    "ocr_text",
    "visual_description",
]


def row_text(row: Mapping[str, Any]) -> str:
    parts = []
    for column in TEXT_COLUMNS:
        text = normalize_transcript(row.get(column, ""))
        if text:
            parts.append(f"[{column.upper()}]\n{text}")
    return "\n\n".join(parts)


def load_synthetic_examples(path: str | Path | None, ratio: float, real_count: int, seed: int) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    synthetic_path = Path(path)
    if not synthetic_path.exists():
        raise FileNotFoundError(synthetic_path)
    rows: List[Dict[str, Any]] = []
    with synthetic_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                row = {"split": "train", "text_for_taxonomy": item.get("text", "")}
                labels = set(item.get("labels", []))
                for label in TAXONOMY_LABELS:
                    row[f"tax_{label}"] = int(label in labels)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    keep = max(1, int(round(real_count * float(ratio))))
    return pd.DataFrame(rows[:keep])


def prepare_frames(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(args.manifest)
    missing = [column for column in TAX_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing taxonomy columns: {missing}")
    synthetic = load_synthetic_examples(args.synthetic_jsonl, args.synthetic_ratio, len(manifest), args.seed)
    if not synthetic.empty:
        manifest = pd.concat([manifest, synthetic], ignore_index=True)
    manifest["text_input"] = [row_text(row) for _, row in manifest.iterrows()]
    manifest = manifest[manifest["text_input"].str.len() > 0].copy()
    if args.limit is not None:
        manifest = manifest.head(int(args.limit)).copy()
    if manifest.empty:
        raise ValueError("No text rows are available for text taxonomy training")
    train = manifest[manifest["split"].astype(str).eq("train")].copy()
    eval_frame = manifest[manifest["split"].astype(str).eq(args.eval_split)].copy()
    if eval_frame.empty:
        eval_frame = manifest[manifest["split"].astype(str).ne("train")].copy()
    if eval_frame.empty:
        eval_frame = train.copy()
    return train, eval_frame


def build_model(class_weight: str | None = "balanced"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight=class_weight))),
        ]
    )


def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(frame["text_input"].tolist())
    if isinstance(probabilities, list):
        probabilities = np.vstack([prob[:, 1] if prob.ndim == 2 else prob for prob in probabilities]).T
    rows = []
    y_true = frame[TAX_COLUMNS].astype(int).to_numpy()
    for row_index, (_, row) in enumerate(frame.iterrows()):
        payload = {
            "source_video_id": row.get("source_video_id", ""),
            "segment_id": row.get("segment_id", ""),
            "split": row.get("split", ""),
        }
        for index, label in enumerate(TAXONOMY_LABELS):
            payload[f"true_{label}"] = int(y_true[row_index, index])
            payload[f"prob_{label}"] = float(probabilities[row_index, index])
        rows.append(payload)
    return pd.DataFrame(rows)


def run_training(args: argparse.Namespace) -> None:
    train, eval_frame = prepare_frames(args)
    model = build_model(class_weight=None if args.class_weight == "none" else args.class_weight)
    y_train = train[TAX_COLUMNS].astype(int).to_numpy()
    model.fit(train["text_input"].tolist(), y_train)
    predictions = predict_frame(model, eval_frame)
    thresholds_by_mode = thresholds_for_predictions(predictions)
    thresholds = thresholds_by_mode["production"]
    summary, per_label = taxonomy_metrics(predictions, thresholds)

    output_path = Path(args.output)
    report_dir = Path(args.report_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(
        {
            "model": model,
            "taxonomy_labels": TAXONOMY_LABELS,
            "thresholds": thresholds,
            "thresholds_by_mode": thresholds_by_mode,
            "text_columns": TEXT_COLUMNS,
        },
        output_path,
    )
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    per_label.to_csv(report_dir / "per_label_metrics.csv", index=False)
    pd.DataFrame(threshold_report_rows(predictions, thresholds_by_mode)).to_csv(report_dir / "threshold_report.csv", index=False)
    pd.DataFrame([{**summary, "train_rows": len(train), "eval_rows": len(eval_frame)}]).to_csv(report_dir / "metrics.csv", index=False)
    (report_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (report_dir / "calibrated_thresholds.json").write_text(
        json.dumps(
            {
                "production_thresholds": thresholds,
                "thresholds_by_mode": thresholds_by_mode,
                "production_mode": "precision_target_80_or_top_k_with_rare_overrides",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "report_dir": str(report_dir), **summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight text-only multi-label taxonomy classifier.")
    parser.add_argument("--manifest", default="data/multilabel_manifest.csv")
    parser.add_argument("--output", default="checkpoints/text_taxonomy_classifier.joblib")
    parser.add_argument("--report-dir", default="reports/text_taxonomy_classifier")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--synthetic-jsonl", default=None)
    parser.add_argument("--synthetic-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()

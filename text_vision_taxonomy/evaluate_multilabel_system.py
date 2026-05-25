from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, evaluate_probabilities, load_thresholds, write_json


def default_thresholds(value: float = 0.5) -> Dict[str, float]:
    return {label: float(value) for label in LABEL_COLUMNS}


def filter_eval_rows(frame: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    out = frame.copy()
    if "synthetic" in out.columns:
        out = out[pd.to_numeric(out["synthetic"], errors="coerce").fillna(0).astype(int).eq(0)].copy()
    if "split" not in out.columns or eval_split == "all":
        return out
    if eval_split == "non_train":
        return out[~out["split"].astype(str).eq("train")].copy()
    return out[out["split"].astype(str).eq(eval_split)].copy()


def load_prediction_file(path: str | Path) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    return pd.read_csv(path)


def transformer_best_predictions(report_dir: str | Path) -> tuple[str, str] | None:
    metrics_path = Path(report_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    blob = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = blob.get("best_model", {})
    prediction_path = Path(best.get("report_dir", "")) / "predictions.csv"
    threshold_path = Path(best.get("report_dir", "")) / "thresholds.json"
    if prediction_path.exists():
        return str(prediction_path), str(threshold_path)
    return None


def add_system(
    systems: List[Tuple[str, pd.DataFrame, Mapping[str, float], bool]],
    name: str,
    predictions_path: str,
    thresholds_path: str | None,
    use_final_labels: bool = False,
    default_threshold: float = 0.5,
) -> None:
    predictions = load_prediction_file(predictions_path)
    if predictions is None:
        return
    thresholds = load_thresholds(thresholds_path, default=default_threshold) if thresholds_path else default_thresholds(default_threshold)
    systems.append((name, predictions, thresholds, use_final_labels))


def run(args: argparse.Namespace) -> None:
    systems: List[Tuple[str, pd.DataFrame, Mapping[str, float], bool]] = []
    add_system(
        systems,
        "tfidf_only",
        args.tfidf_predictions,
        None,
        default_threshold=0.5,
    )
    add_system(
        systems,
        "tfidf_calibrated",
        args.tfidf_predictions,
        args.tfidf_thresholds,
    )
    transformer = transformer_best_predictions(args.transformer_report_dir)
    transformer_predictions = args.transformer_predictions
    transformer_thresholds = args.transformer_thresholds
    if transformer and not Path(transformer_predictions).exists():
        transformer_predictions, transformer_thresholds = transformer
    add_system(systems, "transformer_only", transformer_predictions, None, default_threshold=0.5)
    add_system(systems, "transformer_calibrated", transformer_predictions, transformer_thresholds)
    add_system(systems, "ensemble", args.ensemble_predictions, args.ensemble_thresholds)
    add_system(systems, "ensemble_rare_rules", args.ensemble_predictions, args.ensemble_thresholds, use_final_labels=True)

    summaries: List[Dict[str, Any]] = []
    per_label_frames: List[pd.DataFrame] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, predictions, thresholds, use_final_labels in systems:
        frame = filter_eval_rows(predictions, args.eval_split)
        if frame.empty:
            continue
        missing = [f"true_{label}" for label in LABEL_COLUMNS if f"true_{label}" not in frame.columns]
        if missing:
            continue
        summary, per_label = evaluate_probabilities(frame, thresholds, system=name, use_final_labels=use_final_labels)
        summaries.append(summary)
        per_label_frames.append(per_label)

    metrics = pd.DataFrame(summaries)
    per_label_all = pd.concat(per_label_frames, ignore_index=True) if per_label_frames else pd.DataFrame()
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    per_label_all.to_csv(output_dir / "per_label_metrics.csv", index=False)
    write_json(output_dir / "metrics.json", {"metrics": summaries})
    print(metrics.to_string(index=False) if not metrics.empty else "No systems evaluated")
    if not per_label_all.empty:
        best_system = args.table_system or (summaries[-1]["system"] if summaries else "")
        table = per_label_all[per_label_all["system"].eq(best_system)]
        if table.empty:
            table = per_label_all
        table = table[["label", "true_count", "predicted_count", "precision", "recall", "f1"]]
        print("\nlabel | true_count | predicted_count | precision | recall | f1")
        print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare multi-label taxonomy systems.")
    parser.add_argument("--tfidf-predictions", default="reports/text_multilabel_classifier/predictions.csv")
    parser.add_argument("--tfidf-thresholds", default="reports/text_multilabel_classifier/calibrated_thresholds.json")
    parser.add_argument("--transformer-predictions", default="reports/transformer_multilabel_classifier/predictions.csv")
    parser.add_argument("--transformer-thresholds", default="reports/transformer_multilabel_classifier/thresholds.json")
    parser.add_argument("--transformer-report-dir", default="reports/transformer_multilabel_classifier")
    parser.add_argument("--ensemble-predictions", default="reports/multilabel_ensemble/predictions.csv")
    parser.add_argument("--ensemble-thresholds", default="reports/multilabel_calibration/thresholds.json")
    parser.add_argument("--output-dir", default="reports/multilabel_system_eval")
    parser.add_argument("--eval-split", choices=["test", "val", "non_train", "all"], default="test")
    parser.add_argument("--table-system", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import (
    DEFAULT_THRESHOLD_FLOORS,
    LABEL_COLUMNS,
    RARE_SUPPORT_CUTOFF,
    calibrate_thresholds,
    evaluate_probabilities,
    load_thresholds,
    write_json,
)


def system_name(path: str | Path) -> str:
    p = Path(path)
    if p.parent.name:
        return p.parent.name
    return p.stem


def calibration_rows(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = predictions.copy()
    if "synthetic" in frame.columns:
        frame = frame[pd.to_numeric(frame["synthetic"], errors="coerce").fillna(0).astype(int).eq(0)].copy()
    if split and split != "all" and "split" in frame.columns:
        if split == "non_train":
            frame = frame[~frame["split"].astype(str).eq("train")].copy()
        else:
            frame = frame[frame["split"].astype(str).eq(split)].copy()
    if frame.empty:
        raise ValueError(f"No rows available for calibration split: {split}")
    return frame


def calibrate_file(path: str | Path, output_dir: Path, split: str) -> Dict[str, Any]:
    predictions = pd.read_csv(path)
    calibration_frame = calibration_rows(predictions, split)
    thresholds_by_mode, report = calibrate_thresholds(calibration_frame)
    thresholds = thresholds_by_mode["production"]
    summary, per_label = evaluate_probabilities(calibration_frame, thresholds, system=system_name(path))
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "threshold_report.csv", index=False)
    per_label.to_csv(output_dir / "per_label_metrics.csv", index=False)
    payload = {
        "labels": LABEL_COLUMNS,
        "source_predictions": str(path),
        "calibration_split": split,
        "calibration_rows": int(len(calibration_frame)),
        "production_thresholds": thresholds,
        "thresholds_by_mode": thresholds_by_mode,
        "rare_support_cutoff": RARE_SUPPORT_CUTOFF,
        "default_threshold_floors": DEFAULT_THRESHOLD_FLOORS,
        "production_mode": "precision_first_with_rare_and_overprediction_floors",
        "metrics": summary,
    }
    write_json(output_dir / "thresholds.json", payload)
    write_json(output_dir / "metrics.json", summary)
    return {"system": system_name(path), "thresholds": thresholds, "metrics": summary, "output_dir": str(output_dir)}


def run(args: argparse.Namespace) -> None:
    base_output = Path(args.output_dir)
    summaries: List[Dict[str, Any]] = []
    for prediction_file in args.predictions:
        if len(args.predictions) == 1:
            output_dir = base_output
        else:
            output_dir = base_output / system_name(prediction_file)
        summaries.append(calibrate_file(prediction_file, output_dir, args.split))
    base_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "system": item["system"],
                "output_dir": item["output_dir"],
                **{key: value for key, value in item["metrics"].items() if isinstance(value, (int, float, str))},
            }
            for item in summaries
        ]
    ).to_csv(base_output / "calibration_summary.csv", index=False)
    if summaries:
        write_json(
            base_output / "thresholds.json",
            {
                "labels": LABEL_COLUMNS,
                "production_thresholds": summaries[0]["thresholds"],
                "models": summaries,
                "production_mode": "first_prediction_file_thresholds",
            },
        )
    print(json.dumps({"calibrated": summaries}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate multi-label taxonomy thresholds from validation predictions.")
    parser.add_argument("--predictions", nargs="+", default=["reports/text_multilabel_classifier/predictions.csv"])
    parser.add_argument("--output-dir", default="reports/multilabel_calibration")
    parser.add_argument("--split", choices=["val", "test", "non_train", "all"], default="val")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

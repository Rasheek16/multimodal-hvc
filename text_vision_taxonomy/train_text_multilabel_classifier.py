from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import (
    LABEL_COLUMNS,
    TEXT_FIELDS,
    add_prediction_columns,
    calibrate_thresholds,
    default_dataset_path,
    evaluate_probabilities,
    load_multilabel_dataset,
    target_matrix,
    write_json,
)


def _calibrated_svc(estimator, cv: int):
    from sklearn.calibration import CalibratedClassifierCV

    try:
        return CalibratedClassifierCV(estimator=estimator, cv=cv, method="sigmoid")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, cv=cv, method="sigmoid")


def build_candidate_models(min_positive: int, seed: int) -> List[Tuple[str, Any]]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC

    logreg = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=int(seed),
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        max_features=80000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=120000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    combined = FeatureUnion([("word", word), ("char", char)])
    models: List[Tuple[str, Any]] = [
        ("tfidf_word_logreg_ovr", Pipeline([("features", word), ("clf", OneVsRestClassifier(logreg))])),
        ("tfidf_char_logreg_ovr", Pipeline([("features", char), ("clf", OneVsRestClassifier(logreg))])),
        ("tfidf_word_char_logreg_ovr", Pipeline([("features", combined), ("clf", OneVsRestClassifier(logreg))])),
    ]
    cv = min(3, int(min_positive))
    if cv >= 2:
        svc = LinearSVC(C=1.0, class_weight="balanced", max_iter=8000, random_state=int(seed))
        models.append(
            (
                "tfidf_word_char_linearsvc_calibrated_ovr",
                Pipeline([("features", combined), ("clf", OneVsRestClassifier(_calibrated_svc(svc, cv=cv)))]),
            )
        )
    return models


def predict_probabilities(model: Any, texts: List[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
    else:
        decision = model.decision_function(texts)
        probs = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))
    if isinstance(probs, list):
        probs = np.vstack([item[:, 1] if getattr(item, "ndim", 0) == 2 else item for item in probs]).T
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    return np.clip(probs, 0.0, 1.0)


def selection_score(summary: Dict[str, Any], per_label: pd.DataFrame) -> float:
    predicted = per_label.set_index("label")["predicted_count"].to_dict()
    penalty = 0.0
    for label, cap in {"fear": 0.35, "threat": 0.08, "illegal": 0.05, "online_harm": 0.06}.items():
        if label in predicted:
            rate = float(predicted[label]) / max(int(summary["rows"]), 1)
            penalty += max(0.0, rate - cap)
    return float(summary.get("micro_f1", 0.0)) + 0.35 * float(summary.get("micro_precision", 0.0)) - penalty


def train_and_compare(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = report_dir / "splits.csv"
    frame = load_multilabel_dataset(
        args.dataset,
        seed=args.seed,
        split_path=args.splits if args.splits else None,
        save_split_path=split_path if not args.splits else None,
    )
    if args.limit is not None:
        frame = frame.head(int(args.limit)).copy()
    frame = frame[frame["text_input"].astype(str).str.len() > 0].copy()
    if frame.empty:
        raise ValueError("No text rows are available for training")
    train = frame[frame["split"].eq("train")].copy()
    val = frame[(frame["split"].eq("val")) & frame["synthetic"].eq(0)].copy()
    test = frame[(frame["split"].eq("test")) & frame["synthetic"].eq(0)].copy()
    if train.empty or val.empty:
        raise ValueError("Train and validation splits must be non-empty")

    y_train = target_matrix(train)
    min_positive = max(1, int(y_train.sum(axis=0).min()))
    candidates = build_candidate_models(min_positive=min_positive, seed=args.seed)
    comparison: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for name, model in candidates:
        try:
            model.fit(train["text_input"].tolist(), y_train)
            val_probs = predict_probabilities(model, val["text_input"].tolist())
            val_raw = add_prediction_columns(val, val_probs)
            thresholds_by_mode, threshold_report = calibrate_thresholds(val_raw)
            thresholds = thresholds_by_mode["production"]
            val_predictions = add_prediction_columns(val, val_probs, thresholds)
            val_summary, val_per_label = evaluate_probabilities(val_predictions, thresholds, system=name)
            score = selection_score(val_summary, val_per_label)
            record = {"model": name, "selection_score": score, **{f"val_{k}": v for k, v in val_summary.items() if k != "system"}}
            comparison.append(record)
            joblib.dump(model, output_dir / f"{name}.joblib")
            threshold_report.to_csv(report_dir / f"{name}_threshold_report.csv", index=False)
            if best is None or score > float(best["score"]):
                best = {
                    "name": name,
                    "model": model,
                    "score": score,
                    "thresholds": thresholds,
                    "thresholds_by_mode": thresholds_by_mode,
                    "threshold_report": threshold_report,
                    "val_summary": val_summary,
                }
        except Exception as exc:
            comparison.append({"model": name, "error": str(exc)})

    if best is None:
        raise RuntimeError("No text baseline model trained successfully")

    best_model = best["model"]
    all_probs = predict_probabilities(best_model, frame["text_input"].tolist())
    predictions = add_prediction_columns(frame, all_probs, best["thresholds"])
    eval_predictions = predictions[predictions["synthetic"].eq(0)].copy()
    eval_predictions = eval_predictions[eval_predictions["split"].isin(["val", "test"])].copy()
    val_predictions = eval_predictions[eval_predictions["split"].eq("val")].copy()
    test_predictions = eval_predictions[eval_predictions["split"].eq("test")].copy()
    val_summary, val_per_label = evaluate_probabilities(val_predictions, best["thresholds"], system=best["name"]) if not val_predictions.empty else ({}, pd.DataFrame())
    test_summary, test_per_label = evaluate_probabilities(test_predictions, best["thresholds"], system=best["name"]) if not test_predictions.empty else ({}, pd.DataFrame())
    all_summary, per_label = evaluate_probabilities(eval_predictions, best["thresholds"], system=best["name"])

    joblib.dump(
        {
            "model": best_model,
            "model_name": best["name"],
            "taxonomy_labels": LABEL_COLUMNS,
            "thresholds": best["thresholds"],
            "thresholds_by_mode": best["thresholds_by_mode"],
            "text_fields": TEXT_FIELDS,
            "selection_score": best["score"],
        },
        report_dir / "best_model.joblib",
    )
    joblib.dump(
        {
            "model": best_model,
            "model_name": best["name"],
            "taxonomy_labels": LABEL_COLUMNS,
            "thresholds": best["thresholds"],
            "thresholds_by_mode": best["thresholds_by_mode"],
            "text_fields": TEXT_FIELDS,
            "selection_score": best["score"],
        },
        output_dir / "best_model.joblib",
    )
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    per_label.to_csv(report_dir / "per_label_metrics.csv", index=False)
    best["threshold_report"].to_csv(report_dir / "threshold_report.csv", index=False)
    pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)
    metrics = {
        "best_model": best["name"],
        "selection_score": best["score"],
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "thresholds": best["thresholds"],
        "all_eval": all_summary,
        "val": val_summary,
        "test": test_summary,
        "model_comparison": comparison,
    }
    write_json(report_dir / "metrics.json", metrics)
    write_json(report_dir / "thresholds.json", best["thresholds"])
    write_json(
        report_dir / "calibrated_thresholds.json",
        {
            "production_thresholds": best["thresholds"],
            "thresholds_by_mode": best["thresholds_by_mode"],
            "production_mode": "precision_first_with_rare_and_overprediction_floors",
        },
    )
    print(json.dumps({"best_model": best["name"], "report_dir": str(report_dir), "metrics": all_summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strong classical text multi-label taxonomy baselines.")
    parser.add_argument("--dataset", default=str(default_dataset_path()))
    parser.add_argument("--output-dir", default="checkpoints/text_multilabel_classifier")
    parser.add_argument("--report-dir", default="reports/text_multilabel_classifier")
    parser.add_argument("--splits", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    train_and_compare(parse_args())


if __name__ == "__main__":
    main()

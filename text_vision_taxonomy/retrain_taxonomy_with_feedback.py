from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, TEXT_FIELDS, default_dataset_path, parse_label_list
from text_vision.train_text_multilabel_classifier import train_and_compare
from text_vision.train_transformer_multilabel_classifier import run_training as train_transformer


def _empty_text(value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _normalize_training_rows(frame: pd.DataFrame, source: str, synthetic: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    for field in TEXT_FIELDS:
        if field not in frame.columns:
            frame[field] = ""
    if "File Name" not in frame.columns:
        if "video" in frame.columns:
            frame["File Name"] = frame["video"]
        elif "frame_dir" in frame.columns:
            frame["File Name"] = frame["frame_dir"]
        else:
            frame["File Name"] = [f"{source}_{index}" for index in range(len(frame))]
    if "final_labels" in frame.columns:
        parsed = [set(parse_label_list(value)) for value in frame["final_labels"]]
    elif "suggested_labels" in frame.columns:
        parsed = [set(parse_label_list(value)) for value in frame["suggested_labels"]]
        frame["final_labels"] = ["|".join(sorted(labels)) for labels in parsed]
    else:
        parsed = [set() for _ in range(len(frame))]
        frame["final_labels"] = ""
    for label in LABEL_COLUMNS:
        if label in frame.columns:
            frame[label] = pd.to_numeric(frame[label], errors="coerce").fillna(0).astype(int)
        else:
            frame[label] = [int(label in labels) for labels in parsed]
    if "neutral" not in frame.columns:
        frame["neutral"] = frame[LABEL_COLUMNS].sum(axis=1).eq(0).astype(int)
    frame["synthetic"] = int(synthetic)
    frame["feedback_source"] = source
    return frame


def _read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _approved_review_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    if "approved" in frame.columns:
        return frame[pd.to_numeric(frame["approved"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    if "review_status" in frame.columns:
        status = frame["review_status"].fillna("").astype(str).str.lower().str.strip()
        return frame[status.isin(["approved", "accepted", "corrected"])].copy()
    return pd.DataFrame(columns=frame.columns)


def _hard_negative_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    for label in LABEL_COLUMNS:
        frame[label] = 0
    frame["final_labels"] = ""
    frame["neutral"] = 1
    return frame


def build_training_dataset(args: argparse.Namespace) -> pd.DataFrame:
    original = pd.read_csv(args.dataset)
    original = _normalize_training_rows(original, "original", synthetic=0)

    review = _approved_review_rows(_read_csv_if_exists(args.approved_review))
    review = _normalize_training_rows(review, "human_review", synthetic=1)

    pseudo = _read_csv_if_exists(args.pseudo_labels)
    pseudo = _normalize_training_rows(pseudo, "pseudo_label", synthetic=1)

    hard_negatives = _hard_negative_rows(_read_csv_if_exists(args.hard_negatives))
    hard_negatives = _normalize_training_rows(hard_negatives, "hard_negative", synthetic=1)

    combined = pd.concat([original, review, pseudo, hard_negatives], ignore_index=True, sort=False)
    has_text = combined[TEXT_FIELDS].fillna("").astype(str).apply(lambda row: any(not _empty_text(value) for value in row), axis=1)
    combined = combined[has_text].copy()
    combined["neutral"] = combined[LABEL_COLUMNS].sum(axis=1).eq(0).astype(int)
    return combined


def run(args: argparse.Namespace) -> None:
    output_dataset = Path(args.output_dataset)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    combined = build_training_dataset(args)
    combined.to_csv(output_dataset, index=False)
    source_counts = combined.get("feedback_source", pd.Series(dtype=str)).fillna("").value_counts().to_dict()
    summary = {
        "output_dataset": str(output_dataset),
        "rows": int(len(combined)),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "synthetic_rows": int(pd.to_numeric(combined["synthetic"], errors="coerce").fillna(0).astype(int).sum()),
    }
    if args.prepare_only:
        print(json.dumps(summary, indent=2))
        return

    if args.trainer == "transformer":
        train_args = argparse.Namespace(
            dataset=str(output_dataset),
            model_names=args.model_names,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            splits=args.splits,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            max_length=args.max_length,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss=args.loss,
            gamma=args.gamma,
            use_pos_weight=args.use_pos_weight,
            max_pos_weight=args.max_pos_weight,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            local_files_only=args.local_files_only,
            limit=args.limit,
        )
        train_transformer(train_args)
    else:
        train_args = argparse.Namespace(
            dataset=str(output_dataset),
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            splits=args.splits,
            seed=args.seed,
            limit=args.limit,
        )
        train_and_compare(train_args)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine feedback rows and retrain the taxonomy classifier.")
    parser.add_argument("--dataset", default=str(default_dataset_path()))
    parser.add_argument("--approved-review", default="data/taxonomy_review_queue.csv")
    parser.add_argument("--pseudo-labels", default="data/taxonomy_pseudo_labeled.csv")
    parser.add_argument("--hard-negatives", default="")
    parser.add_argument("--output-dataset", default="data/taxonomy_feedback_training.csv")
    parser.add_argument("--trainer", choices=["transformer", "tfidf"], default="transformer")
    parser.add_argument("--output-dir", default="checkpoints/transformer_multilabel_classifier_feedback")
    parser.add_argument("--report-dir", default="reports/transformer_multilabel_classifier_feedback")
    parser.add_argument("--splits", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model-names", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

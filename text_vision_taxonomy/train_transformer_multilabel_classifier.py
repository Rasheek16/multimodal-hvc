from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.targets = target_matrix(self.frame).astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "text": str(row["text_input"]),
            "labels": torch.tensor(self.targets[index], dtype=torch.float32),
            "row_index": int(index),
        }


def collate_batch(tokenizer, max_length: int):
    def _collate(items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        encoded["labels"] = torch.stack([item["labels"] for item in items])
        return encoded

    return _collate


def compute_pos_weight(train: pd.DataFrame, max_pos_weight: float) -> torch.Tensor:
    y = target_matrix(train).astype(np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = neg / np.maximum(pos, 1.0)
    return torch.tensor(np.clip(weights, 1.0, float(max_pos_weight)), dtype=torch.float32)


def multilabel_loss(logits: torch.Tensor, labels: torch.Tensor, loss_name: str, pos_weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    if loss_name == "bce":
        return bce.mean()
    if loss_name != "focal":
        raise ValueError(f"Unsupported loss: {loss_name}")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    return ((1.0 - p_t).pow(float(gamma)) * bce).mean()


def predict(model, tokenizer, frame: pd.DataFrame, device: torch.device, batch_size: int, max_length: int, num_workers: int) -> np.ndarray:
    dataset = TextDataset(frame)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_batch(tokenizer, max_length),
    )
    model.eval()
    probs: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="predict", leave=False):
            labels = batch.pop("labels")
            encoded = {key: value.to(device) for key, value in batch.items()}
            logits = model(**encoded).logits
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, len(LABEL_COLUMNS)), dtype=np.float32)


def selection_score(summary: Dict[str, Any], per_label: pd.DataFrame) -> float:
    predicted = per_label.set_index("label")["predicted_count"].to_dict()
    penalty = 0.0
    for label, cap in {"fear": 0.35, "threat": 0.08, "illegal": 0.05, "online_harm": 0.06}.items():
        if label in predicted:
            rate = float(predicted[label]) / max(int(summary["rows"]), 1)
            penalty += max(0.0, rate - cap)
    return float(summary.get("micro_f1", 0.0)) + 0.35 * float(summary.get("micro_precision", 0.0)) - penalty


def train_one_model(
    model_name: str,
    args: argparse.Namespace,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    all_rows: pd.DataFrame,
    output_dir: Path,
    report_dir: Path,
) -> Dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    slug = model_name.replace("/", "__")
    model_output_dir = output_dir / slug
    model_report_dir = report_dir / slug
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_report_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(args.local_files_only))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
        local_files_only=bool(args.local_files_only),
        ignore_mismatched_sizes=True,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    pos_weight = compute_pos_weight(train, args.max_pos_weight).to(device) if args.use_pos_weight else None
    train_loader = DataLoader(
        TextDataset(train),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch(tokenizer, args.max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, len(train_loader) * int(args.epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.08), num_training_steps=total_steps)

    best: Dict[str, Any] | None = None
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for batch in tqdm(train_loader, desc=f"train {slug} epoch {epoch}", leave=False):
            labels = batch.pop("labels").to(device)
            encoded = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits
            loss = multilabel_loss(logits, labels, args.loss, pos_weight, args.gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))

        val_probs = predict(model, tokenizer, val, device, args.batch_size, args.max_length, args.num_workers)
        val_raw = add_prediction_columns(val, val_probs)
        thresholds_by_mode, threshold_report = calibrate_thresholds(val_raw)
        thresholds = thresholds_by_mode["production"]
        val_predictions = add_prediction_columns(val, val_probs, thresholds)
        val_summary, val_per_label = evaluate_probabilities(val_predictions, thresholds, system=slug)
        score = selection_score(val_summary, val_per_label)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, "selection_score": score, **val_summary}
        history.append(row)
        pd.DataFrame(history).to_csv(model_report_dir / "history.csv", index=False)
        if best is None or score > float(best["score"]):
            best = {
                "epoch": epoch,
                "score": score,
                "thresholds": thresholds,
                "thresholds_by_mode": thresholds_by_mode,
                "threshold_report": threshold_report,
                "val_summary": val_summary,
            }
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            write_json(model_output_dir / "taxonomy_config.json", {"labels": LABEL_COLUMNS, "thresholds": thresholds, "text_fields": TEXT_FIELDS, "model_name": model_name})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break

    if best is None:
        raise RuntimeError(f"No successful epoch for {model_name}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_output_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_output_dir, local_files_only=True).to(device)
    all_probs = predict(model, tokenizer, all_rows, device, args.batch_size, args.max_length, args.num_workers)
    predictions = add_prediction_columns(all_rows, all_probs, best["thresholds"])
    eval_predictions = predictions[predictions["synthetic"].eq(0) & predictions["split"].isin(["val", "test"])].copy()
    summary, per_label = evaluate_probabilities(eval_predictions, best["thresholds"], system=slug)
    test_predictions = predictions[predictions["synthetic"].eq(0) & predictions["split"].eq("test")].copy()
    test_summary, _ = evaluate_probabilities(test_predictions, best["thresholds"], system=slug) if not test_predictions.empty else ({}, pd.DataFrame())

    predictions.to_csv(model_report_dir / "predictions.csv", index=False)
    per_label.to_csv(model_report_dir / "per_label_metrics.csv", index=False)
    best["threshold_report"].to_csv(model_report_dir / "threshold_report.csv", index=False)
    write_json(model_report_dir / "thresholds.json", best["thresholds"])
    write_json(
        model_report_dir / "metrics.json",
        json_safe({"model_name": model_name, "best": best, "eval": summary, "test": test_summary}),
    )
    return {
        "model_name": model_name,
        "slug": slug,
        "score": best["score"],
        "output_dir": str(model_output_dir),
        "report_dir": str(model_report_dir),
        "metrics": summary,
    }


def run_training(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_multilabel_dataset(
        args.dataset,
        seed=args.seed,
        split_path=args.splits if args.splits else None,
        save_split_path=report_dir / "splits.csv" if not args.splits else None,
    )
    if args.limit is not None:
        frame = frame.head(int(args.limit)).copy()
    frame = frame[frame["text_input"].astype(str).str.len() > 0].copy()
    train = frame[frame["split"].eq("train")].copy()
    val = frame[(frame["split"].eq("val")) & frame["synthetic"].eq(0)].copy()
    test = frame[(frame["split"].eq("test")) & frame["synthetic"].eq(0)].copy()
    if train.empty or val.empty:
        raise ValueError("Train and validation splits must be non-empty")

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for model_name in model_names:
        try:
            results.append(train_one_model(model_name, args, train, val, test, frame, output_dir, report_dir))
        except Exception as exc:
            errors.append({"model_name": model_name, "error": str(exc)})
    if not results:
        raise RuntimeError(f"No transformer models trained successfully: {errors}")
    best = max(results, key=lambda item: float(item["score"]))
    best_report_dir = Path(best["report_dir"])
    best_output_dir = Path(best["output_dir"])
    for name in ["predictions.csv", "per_label_metrics.csv", "threshold_report.csv", "thresholds.json"]:
        source = best_report_dir / name
        if source.exists():
            shutil.copyfile(source, report_dir / name)
    best_dir = output_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    if best_output_dir.exists():
        shutil.copytree(best_output_dir, best_dir)
    write_json(report_dir / "metrics.json", {"best_model": best, "models": results, "errors": errors})
    print(json.dumps({"best_model": best, "errors": errors}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer multi-label taxonomy classifiers.")
    parser.add_argument("--dataset", default=str(default_dataset_path()))
    parser.add_argument("--model-names", default="microsoft/deberta-v3-base,roberta-base,distilroberta-base")
    parser.add_argument("--output-dir", default="checkpoints/transformer_multilabel_classifier")
    parser.add_argument("--report-dir", default="reports/transformer_multilabel_classifier")
    parser.add_argument("--splits", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()

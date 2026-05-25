from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS, as_path, get_config
from text_vision.data_utils import coerce_bool, safe_torch_load
from text_vision.train_frame_text_vision import load_model_from_checkpoint
from text_vision.train_taxonomy_head import (
    TAX_COLUMNS,
    best_thresholds,
    filter_trainable_frame_rows,
    make_loaders,
    move_tensor_batch,
    predict_loader,
    taxonomy_metrics,
)


def configure_trainable_parameters(
    model: torch.nn.Module,
    unfreeze_severity_head: bool = False,
    unfreeze_binary_head: bool = False,
) -> Dict[str, int]:
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)
    trainable: Dict[str, int] = {}
    prefixes = [
        "fusion.trunk",
        "fusion.gate",
        "fusion.visual_branch",
        "fusion.text_branch",
        "fusion.metadata_branch",
        "fusion.taxonomy_head",
    ]
    if unfreeze_severity_head:
        prefixes.append("fusion.severity_head")
    if unfreeze_binary_head:
        prefixes.append("fusion.binary_head")
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
            group = name.split(".")[1] if name.startswith("fusion.") and len(name.split(".")) > 1 else "other"
            trainable[group] = trainable.get(group, 0) + parameter.numel()
    if trainable.get("taxonomy_head", 0) == 0:
        raise RuntimeError("No taxonomy_head parameters were found to fine-tune")
    return trainable


def build_optimizer(model: torch.nn.Module, fusion_lr: float, taxonomy_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    taxonomy_params = []
    fusion_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "taxonomy_head" in name:
            taxonomy_params.append(parameter)
        else:
            fusion_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": fusion_params, "lr": float(fusion_lr), "weight_decay": float(weight_decay)},
            {"params": taxonomy_params, "lr": float(taxonomy_lr), "weight_decay": float(weight_decay)},
        ]
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    taxonomy_criterion: torch.nn.Module,
    lambda_taxonomy: float,
    lambda_binary: float,
    lambda_severity: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    count = 0
    for step, raw_batch in enumerate(tqdm(loader, desc="fusion-train", leave=False), start=1):
        if max_batches is not None and step > int(max_batches):
            break
        batch = move_tensor_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        taxonomy_loss = taxonomy_criterion(outputs["taxonomy_logits"], batch["taxonomy_target"])
        binary_loss = F.binary_cross_entropy_with_logits(outputs["binary_logits"], batch["label"].float())
        severity_loss = F.cross_entropy(outputs["severity_logits"], batch["severity_target"].long())
        loss = (
            float(lambda_taxonomy) * taxonomy_loss
            + float(lambda_binary) * binary_loss
            + float(lambda_severity) * severity_loss
        )
        loss.backward()
        optimizer.step()
        parts = {
            "loss": loss,
            "taxonomy_loss": taxonomy_loss,
            "binary_loss": binary_loss,
            "severity_loss": severity_loss,
        }
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    source_checkpoint: Path,
    base_threshold: float,
    cfg,
    text_embedding_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "architecture": "frame_text_vision_taxonomy_fusion_finetuned",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
            "threshold": float(base_threshold),
            "taxonomy_labels": list(TAXONOMY_LABELS),
            "taxonomy_thresholds": dict(thresholds),
            "source_checkpoint": str(source_checkpoint),
            "text_embedding_dim": int(text_embedding_dim),
            "config": cfg.to_jsonable(),
        },
        path,
    )


def run_training(args: argparse.Namespace) -> None:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.frame_checkpoint_path)
    cfg.num_frames = int(args.num_frames)
    cfg.batch_size = int(args.batch_size)
    cfg.num_workers = int(args.num_workers)
    cfg.taxonomy_labels = list(TAXONOMY_LABELS)
    cfg.num_taxonomy_labels = len(TAXONOMY_LABELS)

    manifest = pd.read_csv(cfg.manifest_path)
    missing = [column for column in TAX_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing taxonomy columns: {missing}")
    manifest = filter_trainable_frame_rows(manifest)
    if args.limit is not None:
        manifest = manifest.head(int(args.limit)).copy()
    if manifest.empty:
        raise ValueError("No manifest rows with usable frame_dir are available")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = safe_torch_load(cfg.checkpoint_path, map_location="cpu")
    base_threshold = float(checkpoint.get("threshold", 0.5))
    text_embedding_dim = int(checkpoint.get("text_embedding_dim", cfg.text_embedding_dim))
    model, _ = load_model_from_checkpoint(cfg.checkpoint_path, cfg, device=device, local_files_only=args.local_files_only)
    trainable = configure_trainable_parameters(
        model,
        unfreeze_severity_head=bool(args.unfreeze_severity_head),
        unfreeze_binary_head=bool(args.unfreeze_binary_head),
    )
    print("trainable parameters:", trainable)

    train_frame = manifest[manifest["split"] == "train"]
    pos_weight = None
    if args.use_pos_weight:
        targets = train_frame[TAX_COLUMNS].astype(float).to_numpy()
        pos = targets.sum(axis=0)
        neg = targets.shape[0] - pos
        pos_weight = torch.tensor(np.divide(neg, np.maximum(pos, 1.0)), dtype=torch.float32, device=device)
    taxonomy_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(model, args.fusion_lr, args.taxonomy_lr, args.weight_decay)
    loaders = make_loaders(manifest, cfg, text_embedding_dim, args.batch_size, args.num_workers)
    if "train" not in loaders:
        raise ValueError("Manifest has no train split")

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0
    best_threshold_map = {label: 0.5 for label in TAXONOMY_LABELS}
    for epoch in range(1, int(args.num_epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            taxonomy_criterion,
            args.lambda_taxonomy,
            args.lambda_binary,
            args.lambda_severity,
            max_batches=args.max_train_batches,
        )
        eval_split = "val" if "val" in loaders else "train"
        predictions = predict_loader(model, loaders[eval_split], device, max_batches=args.max_eval_batches)
        if len(predictions):
            y_true = predictions[[f"true_{label}" for label in TAXONOMY_LABELS]].astype(int).to_numpy()
            y_prob = predictions[[f"prob_{label}" for label in TAXONOMY_LABELS]].astype(float).to_numpy()
            threshold_map = best_thresholds(y_true, y_prob)
            summary, per_label = taxonomy_metrics(predictions, threshold_map)
        else:
            threshold_map = best_threshold_map
            summary, per_label = {}, pd.DataFrame()
        score = float(summary.get("micro_f1", 0.0))
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in summary.items()}})
        save_checkpoint(
            output_dir / "last_taxonomy_fusion_finetuned.pt",
            model,
            optimizer,
            epoch,
            summary,
            threshold_map,
            cfg.checkpoint_path,
            base_threshold,
            cfg,
            text_embedding_dim,
        )
        if score > best_score:
            best_score = score
            best_threshold_map = threshold_map
            save_checkpoint(
                output_dir / "best_taxonomy_fusion_finetuned.pt",
                model,
                optimizer,
                epoch,
                summary,
                threshold_map,
                cfg.checkpoint_path,
                base_threshold,
                cfg,
                text_embedding_dim,
            )
            predictions.to_csv(report_dir / "predictions.csv", index=False)
            per_label.to_csv(report_dir / "per_label_metrics.csv", index=False)
            (report_dir / "thresholds.json").write_text(json.dumps(threshold_map, indent=2), encoding="utf-8")
        pd.DataFrame(history).to_csv(report_dir / "metrics.csv", index=False)
        print(f"epoch {epoch}: train_loss={train_metrics.get('loss', 0.0):.4f} val_micro_f1={score:.4f}")


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Fine-tune fusion layers plus taxonomy head while keeping the image backbone frozen.")
    parser.add_argument("--manifest", default="data/multilabel_manifest.csv")
    parser.add_argument("--checkpoint", default=str(cfg.checkpoint_dir / "best_taxonomy_head.pt"))
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--report-dir", default="reports/taxonomy_fusion_finetune")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fusion-lr", type=float, default=3e-5)
    parser.add_argument("--taxonomy-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-taxonomy", type=float, default=1.0)
    parser.add_argument("--lambda-binary", type=float, default=0.2)
    parser.add_argument("--lambda-severity", type=float, default=0.1)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--unfreeze-severity-head", action="store_true")
    parser.add_argument("--unfreeze-binary-head", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()

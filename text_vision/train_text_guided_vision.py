from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TextVisionConfig, as_path, get_config
from text_vision.data_utils import (
    VideoTextClipDataset,
    aggregate_predictions,
    build_balanced_sampler,
    compute_binary_metrics,
    compute_severity_metrics,
    load_teacher_cache,
    safe_torch_load,
    tune_threshold,
    write_confusion_matrix,
)
from text_vision.models.fusion_model import FusionClassifier
from text_vision.models.text_guided_vision import TextGuidedVisionModel, cosine_alignment_loss


class TextGuidedClassifier(nn.Module):
    def __init__(self, vision: TextGuidedVisionModel, fusion: FusionClassifier) -> None:
        super().__init__()
        self.vision = vision
        self.fusion = fusion

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vision_out = self.vision(batch["pixel_values"])
        fusion_out = self.fusion(
            visual_embedding=vision_out["visual_embedding"],
            text_embedding=batch["text_embedding"],
            text_binary_prob=batch["text_binary_prob"],
            text_severity_probs=batch["text_severity_probs"],
            has_text=batch["has_text"],
            transcript_source_id=batch["transcript_source_id"],
            visual_severity_logits=vision_out["severity_logits"],
        )
        return {
            "binary_logits": fusion_out["binary_logits"],
            "severity_logits": fusion_out["severity_logits"],
            "vision_binary_logits": vision_out["binary_logits"],
            "vision_severity_logits": vision_out["severity_logits"],
            "projected_text_embedding": vision_out["projected_text_embedding"],
            "visual_embedding": vision_out["visual_embedding"],
            "fusion_embedding": fusion_out["fusion_embedding"],
        }


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cfg_from_args(args: argparse.Namespace) -> TextVisionConfig:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.text_teacher_cache_path = as_path(args.teacher_cache, cfg.text_teacher_cache_path)
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.checkpoint_path)
    cfg.last_checkpoint_path = as_path(args.last_checkpoint, cfg.last_checkpoint_path)
    cfg.report_dir = as_path(args.report_dir, cfg.report_dir)
    cfg.vision_model_name = args.model_name
    cfg.batch_size = int(args.batch_size)
    cfg.accum_steps = max(1, int(args.accum_steps))
    cfg.epochs = int(args.epochs)
    cfg.num_workers = int(args.num_workers)
    cfg.seed = int(args.seed)
    cfg.head_lr = float(args.head_lr)
    cfg.backbone_lr = float(args.backbone_lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.grad_clip = float(args.grad_clip)
    cfg.freeze_epochs = int(args.freeze_epochs)
    cfg.unfreeze_last_n_blocks = int(args.unfreeze_last_n_blocks)
    cfg.head_dropout = float(args.head_dropout)
    cfg.text_dropout = float(args.text_dropout)
    cfg.class_weight_power = float(args.class_weight_power)
    cfg.loss_weights["severity_ce"] = float(args.severity_ce_weight)
    cfg.ensure_directories()
    return cfg


def move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def masked_weighted_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    values = values.float()
    weight = mask.float()
    if weights is not None:
        weight = weight * weights.float()
    denom = weight.sum().clamp_min(1e-6)
    return (values * weight).sum() / denom


def compute_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    labels = batch["label"].float()
    teacher_mask = batch["has_teacher"].float()
    teacher_conf = batch["teacher_confidence"].float().clamp(0.0, 1.0)

    binary_loss_fusion = F.binary_cross_entropy_with_logits(outputs["binary_logits"], labels, reduction="none")
    binary_loss_vision = F.binary_cross_entropy_with_logits(outputs["vision_binary_logits"], labels, reduction="none")
    binary_label_loss = binary_loss_fusion.mean() + 0.30 * binary_loss_vision.mean()

    binary_distill_values = F.binary_cross_entropy_with_logits(
        outputs["binary_logits"],
        batch["text_binary_prob"].float(),
        reduction="none",
    )
    text_binary_distill = masked_weighted_mean(binary_distill_values, teacher_mask, teacher_conf)

    text_target = batch["text_severity_probs"].float().clamp_min(1e-8)
    text_target = text_target / text_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    severity_log_probs = F.log_softmax(outputs["severity_logits"], dim=-1)
    vision_severity_log_probs = F.log_softmax(outputs["vision_severity_logits"], dim=-1)
    severity_distill_values = -(text_target * severity_log_probs).sum(dim=-1)
    vision_severity_distill_values = -(text_target * vision_severity_log_probs).sum(dim=-1)
    text_severity_distill = masked_weighted_mean(
        severity_distill_values + 0.25 * vision_severity_distill_values,
        teacher_mask,
        teacher_conf,
    )

    alignment_values = cosine_alignment_loss(outputs["projected_text_embedding"], batch["text_embedding"].float())
    embedding_alignment = masked_weighted_mean(alignment_values, teacher_mask, teacher_conf)

    severity_positive_prob = F.softmax(outputs["severity_logits"], dim=-1)[:, 1:].sum(dim=-1).clamp(0.0, 1.0)
    hate_prob = torch.sigmoid(outputs["binary_logits"])
    severity_consistency = F.mse_loss(hate_prob, severity_positive_prob, reduction="mean")

    severity_ce = F.cross_entropy(outputs["severity_logits"], batch["severity_target"].long(), reduction="mean")

    total = (
        float(loss_weights.get("video_binary", 1.0)) * binary_label_loss
        + float(loss_weights.get("text_binary_distill", 0.35)) * text_binary_distill
        + float(loss_weights.get("text_severity_distill", 0.45)) * text_severity_distill
        + float(loss_weights.get("embedding_alignment", 0.25)) * embedding_alignment
        + float(loss_weights.get("severity_consistency", 0.20)) * severity_consistency
        + float(loss_weights.get("severity_ce", 0.0)) * severity_ce
    )
    parts = {
        "loss": float(total.detach().cpu().item()),
        "binary_label_loss": float(binary_label_loss.detach().cpu().item()),
        "text_binary_distill": float(text_binary_distill.detach().cpu().item()),
        "text_severity_distill": float(text_severity_distill.detach().cpu().item()),
        "embedding_alignment": float(embedding_alignment.detach().cpu().item()),
        "severity_consistency": float(severity_consistency.detach().cpu().item()),
        "severity_ce": float(severity_ce.detach().cpu().item()),
    }
    return total, parts


def make_dataloaders(
    manifest: pd.DataFrame,
    cfg: TextVisionConfig,
    teacher_cache: Mapping[str, Mapping],
    text_embedding_dim: int,
    use_weighted_sampler: bool,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, frame in manifest.groupby("split"):
        train = split == "train"
        dataset = VideoTextClipDataset(
            frame,
            cfg=cfg,
            teacher_cache=teacher_cache,
            text_embedding_dim=text_embedding_dim,
            train=train,
        )
        sampler = build_balanced_sampler(frame, cfg.class_weight_power) if train and use_weighted_sampler else None
        loaders[str(split)] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def build_model(
    cfg: TextVisionConfig,
    text_embedding_dim: int,
    local_files_only: bool = False,
) -> TextGuidedClassifier:
    vision = TextGuidedVisionModel(
        model_name=cfg.vision_model_name,
        text_embedding_dim=text_embedding_dim,
        dropout=cfg.head_dropout,
        local_files_only=local_files_only,
    )
    fusion = FusionClassifier(
        visual_dim=vision.visual_dim,
        text_embedding_dim=text_embedding_dim,
        dropout=cfg.head_dropout,
        text_dropout=cfg.text_dropout,
    )
    return TextGuidedClassifier(vision=vision, fusion=fusion)


def configure_training_mode(model: TextGuidedClassifier, cfg: TextVisionConfig, epoch: int, fine_tune: bool) -> None:
    if fine_tune:
        model.vision.freeze_backbone()
        return
    if epoch <= cfg.freeze_epochs:
        model.vision.freeze_backbone()
    else:
        model.vision.unfreeze_last_encoder_blocks(cfg.unfreeze_last_n_blocks)


def build_optimizer(model: TextGuidedClassifier, cfg: TextVisionConfig) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("vision.backbone"):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    groups = [
        {"params": backbone_params, "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay},
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
    ]
    return torch.optim.AdamW(groups)


def train_one_epoch(
    model: TextGuidedClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TextVisionConfig,
    device: torch.device,
    max_batches: int | None = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    totals: Dict[str, float] = {}
    count = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc="train", leave=False)
    for step, raw_batch in enumerate(progress, start=1):
        if max_batches is not None and step > int(max_batches):
            break
        batch = move_batch(raw_batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            outputs = model(batch)
            loss, parts = compute_loss(outputs, batch, cfg.loss_weights)
            scaled_loss = loss / cfg.accum_steps
        scaler.scale(scaled_loss).backward()
        if step % cfg.accum_steps == 0:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        progress.set_postfix(loss=f"{parts['loss']:.4f}")

    if count and count % cfg.accum_steps != 0:
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.inference_mode()
def predict_loader(
    model: TextGuidedClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict] = []
    for step, raw_batch in enumerate(tqdm(loader, desc="predict", leave=False), start=1):
        if max_batches is not None and step > int(max_batches):
            break
        batch = move_batch(raw_batch, device)
        outputs = model(batch)
        probs = torch.sigmoid(outputs["binary_logits"]).detach().cpu().numpy()
        logits = outputs["binary_logits"].detach().cpu().numpy()
        severity_probs = F.softmax(outputs["severity_logits"], dim=-1).detach().cpu().numpy()
        labels = batch["label_long"].detach().cpu().numpy()
        severity_targets = batch["severity_target"].detach().cpu().numpy()
        source_ids = raw_batch["source_video_id"]
        clip_paths = raw_batch["clip_path"]
        splits = raw_batch["split"]
        transcript_sources = raw_batch["transcript_source"]
        has_text = batch["has_text"].detach().cpu().numpy()
        for index, source_video_id in enumerate(source_ids):
            row = {
                "source_video_id": str(source_video_id),
                "clip_path": str(clip_paths[index]),
                "split": str(splits[index]),
                "label": int(labels[index]),
                "prob_hate": float(probs[index]),
                "binary_logit": float(logits[index]),
                "has_text": bool(has_text[index] > 0.5),
                "transcript_source": str(transcript_sources[index]),
                "severity_target": int(severity_targets[index]),
                "pred_severity_clip": int(severity_probs[index].argmax()),
            }
            for severity in range(4):
                row[f"severity_prob_{severity}"] = float(severity_probs[index, severity])
            rows.append(row)
    return pd.DataFrame(rows)


def metric_rows_for_predictions(video_predictions: pd.DataFrame, threshold: float, split: str) -> List[Dict]:
    rows: List[Dict] = []
    groups = [("all", video_predictions)]
    for source in ["human", "asr", "missing"]:
        groups.append((source, video_predictions[video_predictions["transcript_source"] == source]))
    for group_name, frame in groups:
        if frame.empty:
            continue
        binary = compute_binary_metrics(frame["label"], frame["prob_hate"], threshold)
        severity = compute_severity_metrics(frame["severity_target"], frame["pred_severity"])
        rows.append({"split": split, "group": group_name, **binary, **severity, "n": int(len(frame))})
    return rows


def save_reports(
    report_dir: Path,
    clip_predictions: pd.DataFrame,
    threshold: float,
    cfg: TextVisionConfig,
    include_baseline: bool = True,
) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    video_predictions = aggregate_predictions(clip_predictions, threshold=threshold)
    video_predictions.to_csv(report_dir / "predictions.csv", index=False)

    metrics_rows: List[Dict] = []
    for split, frame in video_predictions.groupby("split"):
        metrics_rows.extend(metric_rows_for_predictions(frame, threshold, str(split)))
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(report_dir / "metrics.csv", index=False)

    comparison_rows = []
    test_all = metrics[(metrics["split"] == "test") & (metrics["group"] == "all")]
    val_all = metrics[(metrics["split"] == "val") & (metrics["group"] == "all")]
    chosen = test_all if not test_all.empty else val_all
    if not chosen.empty:
        row = chosen.iloc[0].to_dict()
        comparison_rows.append({"model": "severity_boosted_text_guided_vision", **row})
    if include_baseline:
        baseline_path = cfg.project_root / "hate_non_hate_vision_classification" / "reports_diversified" / "video_transformer_metrics.csv"
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            baseline_test = baseline[baseline["split"].astype(str).eq("test")].tail(1)
            if not baseline_test.empty:
                base = baseline_test.iloc[0].to_dict()
                comparison_rows.append(
                    {
                        "model": "existing_vision_only_baseline",
                        "split": "test",
                        "group": "all",
                        "threshold": base.get("test_tuned_threshold", base.get("test_default_threshold", 0.5)),
                        "accuracy": base.get("test_tuned_accuracy", base.get("test_default_accuracy", np.nan)),
                        "f1": base.get("test_tuned_f1", base.get("test_default_f1", np.nan)),
                        "precision": base.get("test_tuned_precision", base.get("test_default_precision", np.nan)),
                        "recall": base.get("test_tuned_recall", base.get("test_default_recall", np.nan)),
                        "roc_auc": base.get("test_tuned_roc_auc", base.get("test_default_roc_auc", np.nan)),
                        "pr_auc": base.get("test_tuned_pr_auc", base.get("test_default_pr_auc", np.nan)),
                    }
                )
    pd.DataFrame(comparison_rows).to_csv(report_dir / "model_comparison.csv", index=False)

    confusion_split = "test" if (video_predictions["split"] == "test").any() else "val"
    confusion_frame = video_predictions[video_predictions["split"] == confusion_split]
    if not confusion_frame.empty:
        write_confusion_matrix(
            confusion_frame["label"],
            confusion_frame["pred_binary"],
            labels=[0, 1],
            path=report_dir / "confusion_binary.png",
            title=f"Binary Confusion ({confusion_split})",
        )
        write_confusion_matrix(
            confusion_frame["severity_target"],
            confusion_frame["pred_severity"],
            labels=[0, 1, 2, 3],
            path=report_dir / "confusion_severity.png",
            title=f"Pseudo Severity Confusion ({confusion_split})",
        )
        hard = confusion_frame.copy()
        hard["binary_error"] = (hard["label"] != hard["pred_binary"]).astype(int)
        hard["margin"] = (hard["prob_hate"] - float(threshold)).abs()
        hard = hard.sort_values(["binary_error", "margin"], ascending=[False, True])
        hard.head(200).to_csv(report_dir / "hard_examples.csv", index=False)
    return metrics


def save_checkpoint(
    path: Path,
    model: TextGuidedClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    cfg: TextVisionConfig,
    text_embedding_dim: int,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
            "threshold": float(threshold),
            "text_embedding_dim": int(text_embedding_dim),
            "config": cfg.to_jsonable(),
        },
        path,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: TextVisionConfig,
    device: torch.device,
    local_files_only: bool = False,
) -> Tuple[TextGuidedClassifier, Dict]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model_cfg = dict(checkpoint.get("config", {}))
    if model_cfg.get("vision_model_name"):
        cfg.vision_model_name = str(model_cfg["vision_model_name"])
    text_embedding_dim = int(checkpoint.get("text_embedding_dim", model_cfg.get("text_embedding_dim", cfg.text_embedding_dim)))
    model = build_model(cfg, text_embedding_dim=text_embedding_dim, local_files_only=local_files_only)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def run_evaluation(
    checkpoint_path: Path,
    cfg: TextVisionConfig,
    device: torch.device,
    threshold: float | None = None,
    max_eval_batches: int | None = None,
    local_files_only: bool = False,
) -> pd.DataFrame:
    teacher_cache, text_embedding_dim = load_teacher_cache(cfg.text_teacher_cache_path)
    manifest = pd.read_csv(cfg.manifest_path)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, cfg, device, local_files_only=local_files_only)
    if threshold is None:
        threshold = float(checkpoint.get("threshold", 0.5))
    loaders = make_dataloaders(
        manifest,
        cfg,
        teacher_cache=teacher_cache,
        text_embedding_dim=text_embedding_dim,
        use_weighted_sampler=False,
    )
    predictions = []
    for split in ["val", "test", "train"]:
        if split in loaders:
            predictions.append(predict_loader(model, loaders[split], device, max_batches=max_eval_batches))
    clip_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return save_reports(cfg.report_dir, clip_predictions, threshold, cfg)


def run_training(args: argparse.Namespace, fine_tune: bool = False) -> None:
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    teacher_cache, text_embedding_dim = load_teacher_cache(cfg.text_teacher_cache_path)
    manifest = pd.read_csv(cfg.manifest_path)
    loaders = make_dataloaders(
        manifest,
        cfg,
        teacher_cache=teacher_cache,
        text_embedding_dim=text_embedding_dim,
        use_weighted_sampler=not args.no_weighted_sampler,
    )
    if "train" not in loaders:
        raise ValueError("Manifest does not contain a train split")

    model = build_model(cfg, text_embedding_dim=text_embedding_dim, local_files_only=args.local_files_only)
    model.to(device)
    optimizer = build_optimizer(model, cfg)

    start_epoch = 1
    best_score = -1.0
    best_threshold = 0.5
    if args.resume and cfg.last_checkpoint_path.exists():
        checkpoint = safe_torch_load(cfg.last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_score = float(checkpoint.get("metrics", {}).get("f1", -1.0))
        best_threshold = float(checkpoint.get("threshold", 0.5))

    if fine_tune:
        cfg.loss_weights["severity_ce"] = max(float(cfg.loss_weights.get("severity_ce", 0.0)), 0.20)

    history: List[Dict] = []
    for epoch in range(start_epoch, cfg.epochs + 1):
        configure_training_mode(model, cfg, epoch=epoch, fine_tune=fine_tune)
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            cfg,
            device,
            max_batches=args.max_train_batches,
            use_amp=not args.no_amp,
        )
        val_loader = loaders.get("val")
        if val_loader is not None:
            val_clip = predict_loader(model, val_loader, device, max_batches=args.max_eval_batches)
            val_video_default = aggregate_predictions(val_clip, threshold=0.5)
            threshold, val_metrics = tune_threshold(
                val_video_default["label"],
                val_video_default["prob_hate"],
                cfg.threshold_min,
                cfg.threshold_max,
                cfg.threshold_steps,
            )
            val_video = aggregate_predictions(val_clip, threshold=threshold)
            val_metrics = compute_binary_metrics(val_video["label"], val_video["prob_hate"], threshold)
            val_metrics.update(compute_severity_metrics(val_video["severity_target"], val_video["pred_severity"]))
        else:
            threshold = 0.5
            val_metrics = {"f1": 0.0, "accuracy": 0.0, "pr_auc": 0.0}

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(cfg.report_dir / "history.csv", index=False)
        print(json.dumps(row, indent=2))

        score = float(val_metrics.get(cfg.selection_metric, val_metrics.get("f1", 0.0)))
        save_checkpoint(cfg.last_checkpoint_path, model, optimizer, epoch, val_metrics, cfg, text_embedding_dim, threshold)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            save_checkpoint(cfg.checkpoint_path, model, optimizer, epoch, val_metrics, cfg, text_embedding_dim, threshold)

    print(f"best validation {cfg.selection_metric}: {best_score:.4f} @ threshold={best_threshold:.3f}")
    if cfg.checkpoint_path.exists():
        run_evaluation(
            cfg.checkpoint_path,
            cfg,
            device=device,
            threshold=best_threshold,
            max_eval_batches=args.max_eval_batches,
            local_files_only=args.local_files_only,
        )


def build_arg_parser(description: str = "Train a severity-boosted text-guided vision classifier.") -> argparse.ArgumentParser:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--teacher-cache", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--checkpoint", type=str, default=str(cfg.checkpoint_path))
    parser.add_argument("--last-checkpoint", type=str, default=str(cfg.last_checkpoint_path))
    parser.add_argument("--report-dir", type=str, default=str(cfg.report_dir))
    parser.add_argument("--model-name", type=str, default=cfg.vision_model_name)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--accum-steps", type=int, default=cfg.accum_steps)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--head-lr", type=float, default=cfg.head_lr)
    parser.add_argument("--backbone-lr", type=float, default=cfg.backbone_lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--freeze-epochs", type=int, default=cfg.freeze_epochs)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=cfg.unfreeze_last_n_blocks)
    parser.add_argument("--head-dropout", type=float, default=cfg.head_dropout)
    parser.add_argument("--text-dropout", type=float, default=cfg.text_dropout)
    parser.add_argument("--class-weight-power", type=float, default=cfg.class_weight_power)
    parser.add_argument("--severity-ce-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-optimizer", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    run_training(parser.parse_args(), fine_tune=False)


if __name__ == "__main__":
    main()


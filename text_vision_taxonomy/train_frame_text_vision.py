from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

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
    VideoTextFrameDataset,
    aggregate_predictions,
    build_balanced_sampler,
    compute_binary_metrics,
    compute_severity_metrics,
    load_teacher_cache,
    safe_torch_load,
    tune_threshold,
    video_level_manifest,
    write_confusion_matrix,
)
from text_vision.models.frame_text_vision import FrameTextVisionModel
from text_vision.models.fusion_model import FusionClassifier, GatedFusionClassifier
from text_vision.train_text_guided_vision import compute_loss, move_batch, set_seed, train_one_epoch


class FrameTextGuidedClassifier(nn.Module):
    def __init__(self, vision: FrameTextVisionModel, fusion: FusionClassifier) -> None:
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
        result = {
            "binary_logits": fusion_out["binary_logits"],
            "severity_logits": fusion_out["severity_logits"],
            "vision_binary_logits": vision_out["binary_logits"],
            "vision_severity_logits": vision_out["severity_logits"],
            "projected_text_embedding": vision_out["projected_text_embedding"],
            "visual_embedding": vision_out["visual_embedding"],
            "fusion_embedding": fusion_out["fusion_embedding"],
        }
        if "taxonomy_logits" in fusion_out:
            result["taxonomy_logits"] = fusion_out["taxonomy_logits"]
        return result


def cfg_from_args(args: argparse.Namespace) -> TextVisionConfig:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.text_teacher_cache_path = as_path(args.teacher_cache, cfg.text_teacher_cache_path)
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.frame_checkpoint_path)
    cfg.last_checkpoint_path = as_path(args.last_checkpoint, cfg.frame_last_checkpoint_path)
    cfg.report_dir = as_path(args.report_dir, cfg.frame_report_dir)
    cfg.frame_vision_model_name = args.model_name
    cfg.vision_model_name = args.model_name
    cfg.num_frames = int(args.num_frames)
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
    cfg.temporal_layers = int(args.temporal_layers)
    cfg.temporal_heads = int(args.temporal_heads)
    cfg.temporal_dropout = float(args.temporal_dropout)
    cfg.temporal_pooling = str(args.temporal_pooling)
    cfg.frame_chunk_size = int(args.frame_chunk_size)
    cfg.fusion_architecture = str(args.fusion_architecture)
    cfg.selection_metric = str(args.selection_metric)
    cfg.loss_weights["severity_ce"] = float(args.severity_ce_weight)
    cfg.ensure_directories()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def make_dataloaders(
    manifest: pd.DataFrame,
    cfg: TextVisionConfig,
    teacher_cache: Mapping[str, Mapping],
    text_embedding_dim: int,
    use_weighted_sampler: bool,
) -> Dict[str, DataLoader]:
    videos = video_level_manifest(manifest)
    loaders: Dict[str, DataLoader] = {}
    for split, frame in videos.groupby("split"):
        train = split == "train"
        dataset = VideoTextFrameDataset(
            frame,
            cfg=cfg,
            teacher_cache=teacher_cache,
            text_embedding_dim=text_embedding_dim,
            train=train,
        )
        sampler = build_balanced_sampler(dataset.frame, cfg.class_weight_power) if train and use_weighted_sampler else None
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
) -> FrameTextGuidedClassifier:
    vision = FrameTextVisionModel(
        model_name=cfg.frame_vision_model_name,
        text_embedding_dim=text_embedding_dim,
        num_frames=cfg.num_frames,
        temporal_layers=cfg.temporal_layers,
        temporal_heads=cfg.temporal_heads,
        temporal_dropout=cfg.temporal_dropout,
        temporal_pooling=cfg.temporal_pooling,
        frame_chunk_size=cfg.frame_chunk_size,
        dropout=cfg.head_dropout,
        local_files_only=local_files_only,
    )
    fusion_class = GatedFusionClassifier if cfg.fusion_architecture == "gated" else FusionClassifier
    fusion = fusion_class(
        visual_dim=vision.visual_dim,
        text_embedding_dim=text_embedding_dim,
        dropout=cfg.head_dropout,
        text_dropout=cfg.text_dropout,
        num_taxonomy_labels=int(getattr(cfg, "num_taxonomy_labels", 0)),
    )
    return FrameTextGuidedClassifier(vision=vision, fusion=fusion)


def load_checkpoint_state(
    model: FrameTextGuidedClassifier,
    checkpoint: Mapping,
    partial: bool = False,
) -> None:
    state = checkpoint["model_state_dict"]
    if not partial:
        model.load_state_dict(state)
        return

    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    result = model.load_state_dict(compatible, strict=False)
    print(
        "partial resume:",
        f"loaded={len(compatible)}",
        f"skipped={len(skipped)}",
        f"missing={len(result.missing_keys)}",
        f"unexpected={len(result.unexpected_keys)}",
    )


def configure_training_mode(model: FrameTextGuidedClassifier, cfg: TextVisionConfig, epoch: int) -> None:
    if epoch <= cfg.freeze_epochs:
        model.vision.freeze_backbone()
    else:
        model.vision.unfreeze_last_encoder_blocks(cfg.unfreeze_last_n_blocks)


def build_optimizer(model: FrameTextGuidedClassifier, cfg: TextVisionConfig) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("vision.backbone"):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        ]
    )


@torch.inference_mode()
def predict_loader(
    model: FrameTextGuidedClassifier,
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
        has_text = batch["has_text"].detach().cpu().numpy()
        source_ids = raw_batch["source_video_id"]
        frame_dirs = raw_batch["frame_dir"]
        splits = raw_batch["split"]
        transcript_sources = raw_batch["transcript_source"]
        for index, source_video_id in enumerate(source_ids):
            row = {
                "source_video_id": str(source_video_id),
                "frame_dir": str(frame_dirs[index]),
                "split": str(splits[index]),
                "label": int(labels[index]),
                "prob_hate": float(probs[index]),
                "binary_logit": float(logits[index]),
                "has_text": bool(has_text[index] > 0.5),
                "transcript_source": str(transcript_sources[index]),
                "severity_target": int(severity_targets[index]),
                "pred_severity_frame": int(severity_probs[index].argmax()),
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
    predictions: pd.DataFrame,
    threshold: float,
    cfg: TextVisionConfig,
) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    video_predictions = aggregate_predictions(predictions, threshold=threshold)
    video_predictions.to_csv(report_dir / "predictions.csv", index=False)

    metrics_rows: List[Dict] = []
    for split, frame in video_predictions.groupby("split"):
        metrics_rows.extend(metric_rows_for_predictions(frame, threshold, str(split)))
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(report_dir / "metrics.csv", index=False)

    comparison_rows = []
    test_all = metrics[(metrics["split"] == "test") & (metrics["group"] == "all")] if not metrics.empty else pd.DataFrame()
    val_all = metrics[(metrics["split"] == "val") & (metrics["group"] == "all")] if not metrics.empty else pd.DataFrame()
    chosen = test_all if not test_all.empty else val_all
    if not chosen.empty:
        comparison_rows.append({"model": "frame_text_vision", **chosen.iloc[0].to_dict()})
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

    if not video_predictions.empty:
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
    model: FrameTextGuidedClassifier,
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
            "architecture": "frame_text_vision",
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
) -> Tuple[FrameTextGuidedClassifier, Dict]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model_cfg = dict(checkpoint.get("config", {}))
    model_name = model_cfg.get("frame_vision_model_name") or model_cfg.get("vision_model_name")
    if model_name:
        cfg.frame_vision_model_name = str(model_name)
        cfg.vision_model_name = str(model_name)
    for key in ["num_frames", "temporal_layers", "temporal_heads", "frame_chunk_size"]:
        if key in model_cfg:
            setattr(cfg, key, int(model_cfg[key]))
    if "temporal_dropout" in model_cfg:
        cfg.temporal_dropout = float(model_cfg["temporal_dropout"])
    cfg.temporal_pooling = str(model_cfg.get("temporal_pooling", "cls"))
    cfg.fusion_architecture = str(model_cfg.get("fusion_architecture", "concat"))
    text_embedding_dim = int(checkpoint.get("text_embedding_dim", model_cfg.get("text_embedding_dim", cfg.text_embedding_dim)))
    model = build_model(cfg, text_embedding_dim=text_embedding_dim, local_files_only=local_files_only)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in checkpoint["model_state_dict"].items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        elif key in current and key.endswith("frame_pos_embedding") and value.ndim == 3 and current[key].ndim == 3:
            resized = F.interpolate(
                value.transpose(1, 2),
                size=int(current[key].shape[1]),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            if tuple(resized.shape) == tuple(current[key].shape):
                compatible[key] = resized
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    result = model.load_state_dict(compatible, strict=False)
    if skipped or result.missing_keys or result.unexpected_keys:
        print(
            "checkpoint compatibility:",
            f"loaded={len(compatible)}",
            f"skipped={len(skipped)}",
            f"missing={len(result.missing_keys)}",
            f"unexpected={len(result.unexpected_keys)}",
            file=sys.stderr,
        )
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
    frame_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return save_reports(cfg.report_dir, frame_predictions, threshold, cfg)


def run_training(args: argparse.Namespace) -> None:
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    teacher_cache, text_embedding_dim = load_teacher_cache(cfg.text_teacher_cache_path)
    manifest = pd.read_csv(cfg.manifest_path)
    videos = video_level_manifest(manifest)
    print("video-level rows:", len(videos))
    print(videos["transcript_source"].value_counts(dropna=False).sort_index().to_string())
    loaders = make_dataloaders(
        videos,
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
    resume_path = as_path(args.resume_checkpoint, cfg.last_checkpoint_path) if args.resume_checkpoint else cfg.last_checkpoint_path
    if args.resume and resume_path.exists():
        print(f"resuming from: {resume_path}")
        checkpoint = safe_torch_load(resume_path, map_location=device)
        load_checkpoint_state(model, checkpoint, partial=args.partial_resume)
        if args.resume_optimizer and not args.partial_resume and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_score = float(checkpoint.get("metrics", {}).get(cfg.selection_metric, checkpoint.get("metrics", {}).get("f1", -1.0)))
        best_threshold = float(checkpoint.get("threshold", 0.5))

    history: List[Dict] = []
    for epoch in range(start_epoch, cfg.epochs + 1):
        configure_training_mode(model, cfg, epoch=epoch)
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
            val_pred = predict_loader(model, val_loader, device, max_batches=args.max_eval_batches)
            if not val_pred.empty:
                val_video_default = aggregate_predictions(val_pred, threshold=0.5)
                threshold, _ = tune_threshold(
                    val_video_default["label"],
                    val_video_default["prob_hate"],
                    cfg.threshold_min,
                    cfg.threshold_max,
                    cfg.threshold_steps,
                )
                val_video = aggregate_predictions(val_pred, threshold=threshold)
                val_metrics = compute_binary_metrics(val_video["label"], val_video["prob_hate"], threshold)
                val_metrics.update(compute_severity_metrics(val_video["severity_target"], val_video["pred_severity"]))
            else:
                threshold = 0.5
                val_metrics = {"f1": 0.0, "accuracy": 0.0, "pr_auc": 0.0}
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


def build_arg_parser(description: str = "Train a frame-directory text-vision classifier.") -> argparse.ArgumentParser:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--teacher-cache", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--checkpoint", type=str, default=str(cfg.frame_checkpoint_path))
    parser.add_argument("--last-checkpoint", type=str, default=str(cfg.frame_last_checkpoint_path))
    parser.add_argument("--report-dir", type=str, default=str(cfg.frame_report_dir))
    parser.add_argument("--model-name", type=str, default=cfg.frame_vision_model_name)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=16)
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
    parser.add_argument("--temporal-layers", type=int, default=cfg.temporal_layers)
    parser.add_argument("--temporal-heads", type=int, default=cfg.temporal_heads)
    parser.add_argument("--temporal-dropout", type=float, default=cfg.temporal_dropout)
    parser.add_argument("--temporal-pooling", choices=["cls", "attn_mean_max"], default=cfg.temporal_pooling)
    parser.add_argument("--frame-chunk-size", type=int, default=cfg.frame_chunk_size)
    parser.add_argument("--fusion-architecture", choices=["concat", "gated"], default=cfg.fusion_architecture)
    parser.add_argument("--selection-metric", type=str, default=cfg.selection_metric)
    parser.add_argument("--severity-ce-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--partial-resume", action="store_true")
    parser.add_argument("--resume-optimizer", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()

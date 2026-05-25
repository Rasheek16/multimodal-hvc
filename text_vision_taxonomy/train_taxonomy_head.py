from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import SOURCE_TO_ID, TAXONOMY_LABELS, as_path, get_config
from text_vision.data_utils import (
    coerce_bool,
    list_frame_paths,
    load_frame_image_tensor,
    resolve_frame_dir,
    safe_torch_load,
    sample_frame_indices,
)
from text_vision.segment_frame_utils import sample_segment_frames_from_dir
from text_vision.train_frame_text_vision import load_model_from_checkpoint


TAX_COLUMNS = [f"tax_{label}" for label in TAXONOMY_LABELS]
SUMMARY_METRIC_ORDER = [
    "loss",
    "micro_f1",
    "macro_f1",
    "samples_f1",
    "micro_precision",
    "macro_precision",
    "samples_precision",
    "micro_recall",
    "macro_recall",
    "samples_recall",
    "micro_map",
    "macro_map",
    "micro_auroc",
    "macro_auroc",
    "subset_accuracy",
    "hamming_loss",
]
PER_LABEL_METRIC_COLUMNS = [
    "label",
    "support",
    "predicted_positive_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "threshold",
    "precision",
    "recall",
    "f1",
    "ap",
    "auroc",
]
INTEGER_PER_LABEL_COLUMNS = {
    "support",
    "predicted_positive_count",
    "true_positives",
    "false_positives",
    "false_negatives",
}
RARE_LABELS = {"threat", "illegal", "online_harm"}
CONSERVATIVE_THRESHOLD_DEFAULTS = {
    "hate_speech": 0.60,
    "discrimination": 0.65,
    "contextual_hate": 0.60,
    "threat": 0.90,
    "violence": 0.50,
    "fear": 0.75,
    "sexual": 0.75,
    "illegal": 0.95,
    "online_harm": 0.90,
}


class TaxonomyFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg, text_embedding_dim: int, train: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.text_embedding_dim = int(text_embedding_dim)
        self.train = bool(train)
        self._frame_path_cache: Dict[str, List[Path]] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _paths_for_row(self, row: pd.Series) -> List[Path]:
        frame_dir = resolve_frame_dir(row["frame_dir"], self.cfg)
        cache_key = str(frame_dir)
        frame_dir_is_segment = coerce_bool(row.get("frame_dir_is_segment", False))
        has_segment_bounds = "start_sec" in row.index and "end_sec" in row.index
        if has_segment_bounds and not frame_dir_is_segment:
            fps_value = pd.to_numeric(row.get("fps", np.nan), errors="coerce")
            fps = None if pd.isna(fps_value) else float(fps_value)
            return sample_segment_frames_from_dir(
                frame_dir,
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                num_frames=self.cfg.num_frames,
                fps=fps,
                warn_on_fallback=False,
            )
        cached = self._frame_path_cache.get(cache_key)
        if cached is None:
            cached = list_frame_paths(frame_dir)
            self._frame_path_cache[cache_key] = cached
        return cached

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        paths = self._paths_for_row(row)
        if not paths:
            raise ValueError(f"No image frames found for segment {row.get('segment_id', index)} in {row.get('frame_dir', '')}")
        indices = sample_frame_indices(len(paths), self.cfg.num_frames, train=self.train)
        pixel_values = torch.stack(
            [
                load_frame_image_tensor(
                    paths[int(frame_index)],
                    image_size=self.cfg.image_size,
                    image_mean=self.cfg.image_mean,
                    image_std=self.cfg.image_std,
                    train=self.train,
                )
                for frame_index in indices
            ],
            dim=0,
        )
        taxonomy_target = torch.tensor([float(row[column]) for column in TAX_COLUMNS], dtype=torch.float32)
        label = float(row.get("binary_label", row.get("label", 0)))
        severity = int(row.get("pseudo_severity", row.get("fallback_severity", 0)))
        text_severity = torch.zeros(4, dtype=torch.float32)
        text_severity[max(0, min(3, severity))] = 1.0
        return {
            "pixel_values": pixel_values,
            "taxonomy_target": taxonomy_target,
            "label": torch.tensor(label, dtype=torch.float32),
            "label_long": torch.tensor(int(label), dtype=torch.long),
            "severity_target": torch.tensor(severity, dtype=torch.long),
            "text_embedding": torch.zeros(self.text_embedding_dim, dtype=torch.float32),
            "text_binary_prob": torch.tensor(0.5, dtype=torch.float32),
            "text_severity_probs": text_severity,
            "has_text": torch.tensor(0.0, dtype=torch.float32),
            "has_teacher": torch.tensor(0.0, dtype=torch.float32),
            "teacher_confidence": torch.tensor(0.0, dtype=torch.float32),
            "transcript_source_id": torch.tensor(SOURCE_TO_ID["missing"], dtype=torch.long),
            "source_video_id": str(row.get("source_video_id", "")),
            "segment_id": str(row.get("segment_id", "")),
            "split": str(row.get("split", "")),
        }


def move_tensor_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def make_amp_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def amp_autocast(use_amp: bool):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


TAXONOMY_TEMPORAL_PREFIXES = (
    "vision.cls_token",
    "vision.frame_pos_embedding",
    "vision.temporal_encoder",
    "vision.norm",
    "vision.frame_attention",
    "vision.temporal_projection",
)


def _is_taxonomy_path_parameter(name: str, train_scope: str) -> bool:
    if "taxonomy_head" in name:
        return True
    if train_scope == "taxonomy_head":
        return False
    if name.startswith(TAXONOMY_TEMPORAL_PREFIXES):
        return True
    if not name.startswith("fusion."):
        return False
    if ".binary_head" in name or ".severity_head" in name:
        return False
    return True


def configure_taxonomy_trainable(
    model: torch.nn.Module,
    train_scope: str,
    unfreeze_last_n_blocks: int = 0,
) -> int:
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)

    trainable = 0
    for name, parameter in model.named_parameters():
        if _is_taxonomy_path_parameter(name, train_scope):
            parameter.requires_grad_(True)
            trainable += parameter.numel()

    if train_scope == "video_finetune" and int(unfreeze_last_n_blocks) > 0:
        if hasattr(model, "vision") and hasattr(model.vision, "unfreeze_last_encoder_blocks"):
            model.vision.unfreeze_last_encoder_blocks(int(unfreeze_last_n_blocks))
            for name, parameter in model.named_parameters():
                if name.startswith("vision.backbone") and parameter.requires_grad:
                    trainable += parameter.numel()

    if trainable == 0:
        raise RuntimeError("No trainable taxonomy parameters were found")
    return int(trainable)


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_params = []
    taxonomy_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("vision.backbone"):
            backbone_params.append(parameter)
        else:
            taxonomy_params.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": taxonomy_params,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            },
            {
                "params": backbone_params,
                "lr": float(args.backbone_lr),
                "weight_decay": float(args.weight_decay),
            },
        ]
    )


def compute_pos_weight(frame: pd.DataFrame) -> torch.Tensor:
    targets = frame[TAX_COLUMNS].astype(float).to_numpy()
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    weights = np.divide(neg, np.maximum(pos, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def build_pos_weight(frame: pd.DataFrame, mode: str, cap: float = 5.0) -> torch.Tensor | None:
    mode = str(mode).lower()
    if mode in {"none", "false", "0"}:
        return None
    weights = compute_pos_weight(frame)
    if mode == "capped":
        return torch.clamp(weights, max=float(cap))
    if mode == "full":
        return weights
    raise ValueError(f"Unsupported pos_weight mode: {mode}")


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = focal * alpha_t
        return (focal * bce).mean()


def build_criterion(args: argparse.Namespace, train_frame: pd.DataFrame, device: torch.device) -> torch.nn.Module:
    pos_weight_mode = args.pos_weight_mode
    if args.use_pos_weight and pos_weight_mode == "none":
        pos_weight_mode = "full"
    pos_weight = build_pos_weight(train_frame, pos_weight_mode, cap=args.pos_weight_cap)
    pos_weight = pos_weight.to(device) if pos_weight is not None else None
    if args.loss == "focal":
        return FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha, pos_weight=pos_weight)
    if args.loss == "bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Unsupported loss: {args.loss}")


def filter_trainable_frame_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    original = len(manifest)
    frame = manifest[manifest["frame_dir"].fillna("").astype(str).str.len() > 0].copy()
    if "frame_dir_exists" in frame.columns:
        frame = frame[frame["frame_dir_exists"].map(coerce_bool)].copy()
    if "num_frames" in frame.columns:
        numeric_frames = pd.to_numeric(frame["num_frames"], errors="coerce").fillna(0)
        frame = frame[numeric_frames > 0].copy()
    dropped = original - len(frame)
    if dropped:
        print(f"filtered {dropped} manifest rows without usable frames")
    return frame


def print_frame_sampling_summary(manifest: pd.DataFrame) -> None:
    if "start_sec" not in manifest.columns or "end_sec" not in manifest.columns:
        return
    frame_dir_is_segment = (
        manifest.get("frame_dir_is_segment", pd.Series(False, index=manifest.index))
        .map(coerce_bool)
        .astype(bool)
    )
    fps_values = pd.to_numeric(manifest.get("fps", pd.Series(np.nan, index=manifest.index)), errors="coerce")
    start_values = pd.to_numeric(manifest["start_sec"], errors="coerce")
    end_values = pd.to_numeric(manifest["end_sec"], errors="coerce")
    has_bounds = start_values.notna() & end_values.notna() & (end_values > start_values)
    fallback_rows = has_bounds & ~frame_dir_is_segment & (fps_values.isna() | (fps_values <= 0))
    fallback_count = int(fallback_rows.sum())
    if fallback_count == 0:
        return
    split_counts = manifest.loc[fallback_rows].groupby("split").size() if "split" in manifest.columns else pd.Series(dtype=int)
    split_text = ", ".join(f"{split}={int(count)}" for split, count in split_counts.items())
    suffix = f" ({split_text})" if split_text else ""
    print(
        "frame sampling fallback: "
        f"{fallback_count} non-segment rows with start/end bounds have no fps{suffix}; "
        "using uniform available-frame sampling for those rows."
    )


def make_loaders(manifest: pd.DataFrame, cfg, text_embedding_dim: int, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, frame in manifest.groupby("split"):
        dataset = TaxonomyFrameDataset(frame, cfg=cfg, text_embedding_dim=text_embedding_dim, train=split == "train")
        loaders[str(split)] = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=split == "train",
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    accum_steps: int = 1,
    grad_clip: float = 0.0,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.train()
    losses = []
    accum_steps = max(1, int(accum_steps))
    use_amp = bool(use_amp and device.type == "cuda")
    scaler = make_amp_scaler(use_amp)
    optimizer.zero_grad(set_to_none=True)
    processed_steps = 0
    for step, raw_batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        if max_batches is not None and step > max_batches:
            break
        processed_steps += 1
        batch = move_tensor_batch(raw_batch, device)
        with amp_autocast(use_amp):
            outputs = model(batch)
            loss = criterion(outputs["taxonomy_logits"], batch["taxonomy_target"])
            scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()
        should_step = processed_steps % accum_steps == 0
        if should_step:
            if grad_clip and float(grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(grad_clip),
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item()))
    if losses and processed_steps % accum_steps != 0:
        if grad_clip and float(grad_clip) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(grad_clip),
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": float(np.mean(losses)) if losses else 0.0}


@torch.inference_mode()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    criterion: torch.nn.Module | None = None,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    losses = []
    for step, raw_batch in enumerate(tqdm(loader, desc="predict", leave=False), start=1):
        if max_batches is not None and step > max_batches:
            break
        batch = move_tensor_batch(raw_batch, device)
        logits = model(batch)["taxonomy_logits"]
        if criterion is not None:
            loss = criterion(logits, batch["taxonomy_target"])
            losses.append(float(loss.detach().cpu().item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = raw_batch["taxonomy_target"].detach().cpu().numpy()
        for index in range(probs.shape[0]):
            row = {
                "source_video_id": raw_batch["source_video_id"][index],
                "segment_id": raw_batch["segment_id"][index],
                "split": raw_batch["split"][index],
            }
            for label_index, label in enumerate(TAXONOMY_LABELS):
                row[f"prob_{label}"] = float(probs[index, label_index])
                row[f"true_{label}"] = int(targets[index, label_index])
            rows.append(row)
    predictions = pd.DataFrame(rows)
    if losses:
        predictions.attrs["loss"] = float(np.mean(losses))
    return predictions


def _safe_metric(fn, *args, default: float = float("nan"), **kwargs) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(fn(*args, **kwargs))
        return value if np.isfinite(value) else default
    except Exception:
        return default


def best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.60,
    rare_support_min: int = 20,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_steps: int = 19,
) -> Dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds: Dict[str, float] = {}
    grid = np.linspace(float(threshold_min), float(threshold_max), int(threshold_steps))
    for index, label in enumerate(TAXONOMY_LABELS):
        true_label = y_true[:, index]
        support = int(true_label.sum())
        default_threshold = float(CONSERVATIVE_THRESHOLD_DEFAULTS.get(label, 0.50))
        if support == 0:
            thresholds[label] = max(default_threshold, float(threshold_max))
            continue

        precision_candidates = []
        count_candidates = []
        for threshold in grid:
            pred = (y_prob[:, index] >= float(threshold)).astype(int)
            pred_count = int(pred.sum())
            precision = _safe_metric(precision_score, true_label, pred, zero_division=0, default=0.0)
            recall = _safe_metric(recall_score, true_label, pred, zero_division=0, default=0.0)
            f1 = _safe_metric(f1_score, true_label, pred, zero_division=0, default=0.0)
            count_error = abs(pred_count - support)
            item = {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count_error": count_error,
            }
            count_candidates.append(item)
            if precision >= float(min_precision):
                precision_candidates.append(item)

        if precision_candidates:
            chosen = max(
                precision_candidates,
                key=lambda item: (item["f1"], item["recall"], -item["count_error"], item["threshold"]),
            )
        else:
            chosen = max(
                count_candidates,
                key=lambda item: (-item["count_error"], item["precision"], item["recall"], item["f1"], item["threshold"]),
            )
        threshold = float(chosen["threshold"])
        if label in RARE_LABELS or support < int(rare_support_min):
            threshold = max(threshold, default_threshold)
        thresholds[label] = threshold
    return thresholds


def taxonomy_metrics(predictions: pd.DataFrame, thresholds: Mapping[str, float]) -> Tuple[Dict[str, float], pd.DataFrame]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = predictions[[f"true_{label}" for label in TAXONOMY_LABELS]].astype(int).to_numpy()
    y_prob = predictions[[f"prob_{label}" for label in TAXONOMY_LABELS]].astype(float).to_numpy()
    threshold_values = np.array([float(thresholds.get(label, 0.5)) for label in TAXONOMY_LABELS]).reshape(1, -1)
    y_pred = (y_prob >= threshold_values).astype(int)
    summary = {
        "micro_f1": _safe_metric(f1_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": _safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": _safe_metric(f1_score, y_true, y_pred, average="samples", zero_division=0),
        "micro_precision": _safe_metric(precision_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": _safe_metric(precision_score, y_true, y_pred, average="macro", zero_division=0),
        "samples_precision": _safe_metric(precision_score, y_true, y_pred, average="samples", zero_division=0),
        "micro_recall": _safe_metric(recall_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_recall": _safe_metric(recall_score, y_true, y_pred, average="macro", zero_division=0),
        "samples_recall": _safe_metric(recall_score, y_true, y_pred, average="samples", zero_division=0),
        "micro_map": _safe_metric(average_precision_score, y_true, y_prob, average="micro"),
        "macro_map": _safe_metric(average_precision_score, y_true, y_prob, average="macro"),
        "micro_auroc": _safe_metric(roc_auc_score, y_true, y_prob, average="micro"),
        "macro_auroc": _safe_metric(roc_auc_score, y_true, y_prob, average="macro"),
        "subset_accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "hamming_loss": _safe_metric(hamming_loss, y_true, y_pred),
    }
    rows = []
    for index, label in enumerate(TAXONOMY_LABELS):
        true_label = y_true[:, index]
        pred_label = y_pred[:, index]
        true_positives = int(((true_label == 1) & (pred_label == 1)).sum())
        false_positives = int(((true_label == 0) & (pred_label == 1)).sum())
        false_negatives = int(((true_label == 1) & (pred_label == 0)).sum())
        rows.append(
            {
                "label": label,
                "threshold": float(thresholds.get(label, 0.5)),
                "support": int(true_label.sum()),
                "predicted_positive_count": int(pred_label.sum()),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": _safe_metric(precision_score, true_label, pred_label, zero_division=0),
                "recall": _safe_metric(recall_score, true_label, pred_label, zero_division=0),
                "f1": _safe_metric(f1_score, true_label, pred_label, zero_division=0),
                "ap": _safe_metric(average_precision_score, y_true[:, index], y_prob[:, index]),
                "auroc": _safe_metric(roc_auc_score, true_label, y_prob[:, index]),
            }
        )
    return summary, pd.DataFrame(rows)


def format_metric_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def ordered_metric_items(metrics: Mapping[str, float]) -> List[Tuple[str, float]]:
    seen = set()
    ordered: List[Tuple[str, float]] = []
    for key in SUMMARY_METRIC_ORDER:
        if key in metrics:
            ordered.append((key, metrics[key]))
            seen.add(key)
    for key in sorted(metrics):
        if key not in seen:
            ordered.append((key, metrics[key]))
    return ordered


def print_epoch_metrics(
    epoch: int,
    train_metrics: Mapping[str, float],
    eval_split: str,
    summary: Mapping[str, float],
    per_label: pd.DataFrame,
) -> None:
    train_text = " ".join(f"train_{key}={format_metric_value(value)}" for key, value in ordered_metric_items(train_metrics))
    eval_text = " ".join(f"{eval_split}_{key}={format_metric_value(value)}" for key, value in ordered_metric_items(summary))
    print(f"epoch {epoch}: {train_text}")
    if eval_text:
        print(f"  {eval_text}")
    if per_label.empty:
        return
    columns = [column for column in PER_LABEL_METRIC_COLUMNS if column in per_label.columns]
    display = per_label[columns].copy()
    for column in display.columns:
        if column in INTEGER_PER_LABEL_COLUMNS:
            display[column] = display[column].map(lambda value: str(int(value)) if pd.notna(value) else "nan")
        elif column != "label":
            display[column] = display[column].map(format_metric_value)
    print("  per-label metrics:")
    for line in display.to_string(index=False).splitlines():
        print(f"    {line}")


def save_taxonomy_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    base_checkpoint: Path,
    base_threshold: float,
    cfg,
    text_embedding_dim: int,
    training_config: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "architecture": "frame_text_vision_taxonomy",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
            "threshold": float(base_threshold),
            "taxonomy_labels": list(TAXONOMY_LABELS),
            "taxonomy_thresholds": dict(thresholds),
            "base_checkpoint": str(base_checkpoint),
            "text_embedding_dim": int(text_embedding_dim),
            "training_config": dict(training_config or {}),
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
    cfg.frame_chunk_size = int(args.frame_chunk_size)
    cfg.taxonomy_labels = list(TAXONOMY_LABELS)
    cfg.num_taxonomy_labels = len(TAXONOMY_LABELS)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg.manifest_path)
    missing = [column for column in TAX_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing taxonomy columns: {missing}")
    manifest = filter_trainable_frame_rows(manifest)
    if args.limit is not None:
        manifest = manifest.head(int(args.limit)).copy()
    if manifest.empty:
        raise ValueError("No manifest rows with frame_dir are available for taxonomy training")
    print_frame_sampling_summary(manifest)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = safe_torch_load(cfg.checkpoint_path, map_location="cpu")
    base_threshold = float(checkpoint.get("threshold", 0.5))
    text_embedding_dim = int(checkpoint.get("text_embedding_dim", cfg.text_embedding_dim))
    model, _ = load_model_from_checkpoint(cfg.checkpoint_path, cfg, device=device, local_files_only=args.local_files_only)
    cfg.num_frames = int(args.num_frames)
    cfg.frame_chunk_size = int(args.frame_chunk_size)
    if hasattr(model, "vision"):
        model.vision.num_frames = int(args.num_frames)
        model.vision.frame_chunk_size = int(args.frame_chunk_size)
    trainable = configure_taxonomy_trainable(
        model,
        train_scope=args.train_scope,
        unfreeze_last_n_blocks=0,
    )
    print(
        "video taxonomy training:",
        f"scope={args.train_scope}",
        f"num_frames={cfg.num_frames}",
        f"frame_chunk_size={cfg.frame_chunk_size}",
        f"trainable_parameters={trainable:,}",
    )

    train_frame = manifest[manifest["split"] == "train"]
    criterion = build_criterion(args, train_frame, device)
    optimizer = build_optimizer(model, args)
    loaders = make_loaders(manifest, cfg, text_embedding_dim, args.batch_size, args.num_workers)
    if "train" not in loaders:
        raise ValueError("Manifest has no train split")

    history = []
    best_score = -1.0
    best_threshold_map = {label: 0.5 for label in TAXONOMY_LABELS}
    training_config = {
        "loss": args.loss,
        "pos_weight_mode": "full" if args.use_pos_weight and args.pos_weight_mode == "none" else args.pos_weight_mode,
        "pos_weight_cap": args.pos_weight_cap,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "train_scope": args.train_scope,
        "num_frames": cfg.num_frames,
        "frame_chunk_size": cfg.frame_chunk_size,
        "accum_steps": args.accum_steps,
        "amp": bool(args.amp and device.type == "cuda"),
        "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "unfreeze_backbone_after_epoch": args.unfreeze_backbone_after_epoch,
        "threshold_min_precision": args.threshold_min_precision,
        "rare_support_min": args.rare_support_min,
    }
    for epoch in range(1, int(args.num_epochs) + 1):
        unfreeze_blocks = (
            int(args.unfreeze_last_n_blocks)
            if args.train_scope == "video_finetune" and epoch >= int(args.unfreeze_backbone_after_epoch)
            else 0
        )
        trainable = configure_taxonomy_trainable(
            model,
            train_scope=args.train_scope,
            unfreeze_last_n_blocks=unfreeze_blocks,
        )
        print(f"epoch {epoch}: trainable_parameters={trainable:,}")
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            max_batches=args.max_train_batches,
            accum_steps=args.accum_steps,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
        )
        eval_split = "val" if "val" in loaders else "train"
        predictions = predict_loader(model, loaders[eval_split], device, max_batches=args.max_eval_batches, criterion=criterion)
        y_true = predictions[[f"true_{label}" for label in TAXONOMY_LABELS]].astype(int).to_numpy()
        y_prob = predictions[[f"prob_{label}" for label in TAXONOMY_LABELS]].astype(float).to_numpy()
        threshold_map = (
            best_thresholds(
                y_true,
                y_prob,
                min_precision=args.threshold_min_precision,
                rare_support_min=args.rare_support_min,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_steps=args.threshold_steps,
            )
            if len(predictions)
            else best_threshold_map
        )
        summary, per_label = taxonomy_metrics(predictions, threshold_map) if len(predictions) else ({}, pd.DataFrame())
        if "loss" in predictions.attrs:
            summary = {"loss": predictions.attrs["loss"], **summary}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in summary.items()}}
        history.append(row)
        score = float(summary.get("micro_f1", 0.0))
        save_taxonomy_checkpoint(
            output_dir / "last_taxonomy_head.pt",
            model,
            optimizer,
            epoch,
            summary,
            threshold_map,
            cfg.checkpoint_path,
            base_threshold,
            cfg,
            text_embedding_dim,
            training_config,
        )
        if score > best_score:
            best_score = score
            best_threshold_map = threshold_map
            save_taxonomy_checkpoint(
                output_dir / "best_taxonomy_head.pt",
                model,
                optimizer,
                epoch,
                summary,
                threshold_map,
                cfg.checkpoint_path,
                base_threshold,
                cfg,
                text_embedding_dim,
                training_config,
            )
            per_label.to_csv(report_dir / "per_label_metrics.csv", index=False)
            predictions.to_csv(report_dir / "predictions.csv", index=False)
            (report_dir / "thresholds.json").write_text(json.dumps(threshold_map, indent=2), encoding="utf-8")
        pd.DataFrame(history).to_csv(report_dir / "metrics.csv", index=False)
        print_epoch_metrics(epoch, train_metrics, eval_split, summary, per_label)


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Train a deep video taxonomy model on sampled video frames.")
    parser.add_argument("--manifest", type=str, default="data/multilabel_manifest.csv")
    parser.add_argument("--checkpoint", type=str, default=str(cfg.checkpoint_dir / "best_frame_text_vision_relabel_all_splits.pt"))
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--report-dir", type=str, default="reports/taxonomy_head")
    parser.add_argument("--freeze-backbone", action="store_true", help="Accepted for CLI compatibility; the ViT/image backbone stays frozen unless --train-scope video_finetune is used.")
    parser.add_argument(
        "--train-scope",
        choices=["taxonomy_head", "temporal_fusion", "video_finetune"],
        default="temporal_fusion",
        help=(
            "taxonomy_head trains only the final linear layer; temporal_fusion trains temporal video pooling "
            "and fusion taxonomy layers with the image backbone frozen; video_finetune also unfreezes last ViT blocks."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--frame-chunk-size", type=int, default=8, help="Number of flattened frames passed through ViT at once; lower this for high --num-frames.")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps for large frame counts.")
    parser.add_argument("--grad-clip", type=float, default=0.8)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision to fit more frames.")
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-backbone-after-epoch", type=int, default=2)
    parser.add_argument("--threshold-min-precision", type=float, default=0.60)
    parser.add_argument("--rare-support-min", type=int, default=20)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--pos-weight-mode", choices=["none", "full", "capped"], default="none")
    parser.add_argument("--pos-weight-cap", type=float, default=5.0)
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()

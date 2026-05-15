from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import ID_TO_SOURCE, SOURCE_TO_ID, TextVisionConfig, parse_source_id


EMPTY_TEXT_VALUES = {"", "nan", "none", "null", "n/a", "na"}
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def coerce_bool(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def normalize_transcript(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if text.lower() in EMPTY_TEXT_VALUES:
        return ""
    return text


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def load_teacher_cache(path: str | Path | None) -> Tuple[Dict[str, Dict[str, Any]], int]:
    if path is None:
        return {}, 768
    cache_path = Path(path)
    if not cache_path.exists():
        return {}, 768
    blob = safe_torch_load(cache_path, map_location="cpu")
    if isinstance(blob, Mapping) and "items" in blob:
        items = dict(blob["items"])
        metadata = dict(blob.get("metadata", {}))
    elif isinstance(blob, Mapping):
        items = dict(blob)
        metadata = {}
    else:
        raise TypeError(f"Unsupported teacher cache format: {type(blob)!r}")
    embedding_dim = int(metadata.get("embedding_dim", 768))
    return items, embedding_dim


def tensor_from_cache(value: Any, length: int | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    if length is not None and tensor.numel() != length:
        fixed = torch.zeros(length, dtype=torch.float32)
        fixed[: min(length, tensor.numel())] = tensor.flatten()[: min(length, tensor.numel())]
        return fixed
    return tensor.flatten().float()


def video_level_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Collapse a clip-level multimodal manifest to one row per source video."""
    if "source_video_id" not in manifest.columns:
        raise ValueError("Manifest must contain source_video_id")
    sort_columns = [column for column in ["split", "source_video_id", "clip_idx"] if column in manifest.columns]
    videos = manifest.sort_values(sort_columns).drop_duplicates("source_video_id", keep="first").copy()
    if "num_frames_video" in videos.columns:
        videos["num_frames"] = pd.to_numeric(videos["num_frames_video"], errors="coerce").fillna(
            pd.to_numeric(videos.get("num_frames", 0), errors="coerce")
        )
    required = [
        "source_video_id",
        "label",
        "split",
        "frame_dir",
        "num_frames",
        "transcription",
        "has_text",
        "transcript_source",
        "transcript_confidence",
    ]
    missing = [column for column in required if column not in videos.columns]
    if missing:
        raise ValueError(f"Manifest is missing required video-level columns: {missing}")
    videos["label"] = videos["label"].astype(int)
    videos["split"] = videos["split"].astype(str)
    videos["transcription"] = videos["transcription"].fillna("").map(normalize_transcript)
    videos["has_text"] = videos["has_text"].map(coerce_bool) & videos["transcription"].ne("")
    videos["transcript_source"] = videos["transcript_source"].fillna("missing").astype(str)
    videos["transcript_confidence"] = pd.to_numeric(
        videos["transcript_confidence"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    return videos.reset_index(drop=True)


def resolve_frame_dir(value: Any, cfg: TextVisionConfig) -> Path:
    text = "" if value is None else str(value).strip()
    if text == "":
        return Path("")
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [
        cfg.project_root / path,
        cfg.workspace_root / path,
        cfg.dataset_root / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def list_frame_paths(frame_dir: str | Path) -> List[Path]:
    path = Path(frame_dir)
    if not path.exists() or not path.is_dir():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in FRAME_EXTENSIONS
    )


def sample_frame_indices(num_available: int, target_frames: int, train: bool = False) -> np.ndarray:
    num_available = int(num_available)
    target_frames = int(target_frames)
    if num_available <= 0:
        raise ValueError("Cannot sample frames from an empty frame directory")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if train:
        segment = float(num_available) / float(target_frames)
        offsets = np.random.random(target_frames)
        positions = (np.arange(target_frames, dtype=np.float64) + offsets) * segment
        indices = np.floor(positions).astype(np.int64)
    else:
        indices = np.rint(np.linspace(0, num_available - 1, num=target_frames)).astype(np.int64)
    return np.clip(indices, 0, num_available - 1)


def _resampling_filter() -> int:
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def _center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions")
    scale = float(image_size) / float(min(width, height))
    resized = image.resize((max(image_size, round(width * scale)), max(image_size, round(height * scale))), _resampling_filter())
    width, height = resized.size
    left = max(0, (width - image_size) // 2)
    top = max(0, (height - image_size) // 2)
    return resized.crop((left, top, left + image_size, top + image_size))


def _random_resized_crop(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    area = width * height
    for _ in range(10):
        target_area = area * random.uniform(0.85, 1.0)
        aspect = random.uniform(0.9, 1.1)
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            cropped = image.crop((left, top, left + crop_w, top + crop_h))
            return cropped.resize((image_size, image_size), _resampling_filter())
    return _center_crop_resize(image, image_size)


def _apply_light_augmentation(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        transpose = getattr(Image, "Transpose", Image)
        image = image.transpose(getattr(transpose, "FLIP_LEFT_RIGHT"))
    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.92, 1.08))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.94, 1.06))
    return image


def load_frame_image_tensor(
    path: str | Path,
    image_size: int,
    image_mean: Iterable[float],
    image_std: Iterable[float],
    train: bool = False,
) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if train:
            image = _apply_light_augmentation(_random_resized_crop(image, int(image_size)))
        else:
            image = _center_crop_resize(image, int(image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(tuple(float(x) for x in image_mean), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(tuple(float(x) for x in image_std), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def load_frame_tensor(
    frame_dir: str | Path,
    cfg: TextVisionConfig,
    num_frames: int | None = None,
    train: bool = False,
) -> torch.Tensor:
    paths = list_frame_paths(frame_dir)
    if not paths:
        raise ValueError(f"No image frames found in frame_dir: {frame_dir}")
    indices = sample_frame_indices(len(paths), int(num_frames or cfg.num_frames), train=train)
    frames = [
        load_frame_image_tensor(
            paths[int(index)],
            image_size=cfg.image_size,
            image_mean=cfg.image_mean,
            image_std=cfg.image_std,
            train=train,
        )
        for index in indices
    ]
    return torch.stack(frames, dim=0)


def adapt_clip_to_model_frames(
    clip: torch.Tensor,
    target_frames: int,
    image_size: int,
    image_mean: Iterable[float],
    image_std: Iterable[float],
) -> torch.Tensor:
    """Return a normalized [T,C,H,W] float tensor for VideoMAE."""
    if not isinstance(clip, torch.Tensor):
        clip = torch.as_tensor(clip)

    if clip.ndim != 4:
        raise ValueError(f"Expected a 4D clip tensor, got shape {tuple(clip.shape)}")

    # Supported saved layouts: [C,T,H,W], [T,C,H,W], and [T,H,W,C].
    if clip.shape[0] in (1, 3):
        clip_tchw = clip.permute(1, 0, 2, 3).contiguous()
    elif clip.shape[1] in (1, 3):
        clip_tchw = clip.contiguous()
    elif clip.shape[-1] in (1, 3):
        clip_tchw = clip.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Cannot infer channel dimension from clip shape {tuple(clip.shape)}")

    if clip_tchw.dtype == torch.uint8:
        clip_tchw = clip_tchw.float().div(255.0)
    else:
        clip_tchw = clip_tchw.float()
        if clip_tchw.max().item() > 2.0:
            clip_tchw = clip_tchw.div(255.0)

    time_dim = int(clip_tchw.shape[0])
    if time_dim <= 0:
        raise ValueError("Clip has no frames")
    if time_dim != int(target_frames):
        positions = torch.linspace(0, time_dim - 1, steps=int(target_frames)).round().long()
        positions = positions.clamp(0, time_dim - 1)
        clip_tchw = clip_tchw.index_select(0, positions)

    if clip_tchw.shape[-2:] != (int(image_size), int(image_size)):
        clip_tchw = F.interpolate(
            clip_tchw,
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
        )

    mean = torch.tensor(tuple(float(x) for x in image_mean), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(tuple(float(x) for x in image_std), dtype=torch.float32).view(1, 3, 1, 1)
    return (clip_tchw - mean) / std


def load_clip_tensor(row: Mapping[str, Any], cfg: TextVisionConfig) -> torch.Tensor:
    payload = safe_torch_load(row["clip_path"], map_location="cpu")
    clip = payload["clip"] if isinstance(payload, Mapping) and "clip" in payload else payload
    return adapt_clip_to_model_frames(
        clip,
        target_frames=cfg.num_frames,
        image_size=cfg.image_size,
        image_mean=cfg.image_mean,
        image_std=cfg.image_std,
    )


class VideoTextClipDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        cfg: TextVisionConfig,
        teacher_cache: Optional[Mapping[str, Mapping[str, Any]]] = None,
        text_embedding_dim: int = 768,
        train: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.cfg = cfg
        self.teacher_cache = dict(teacher_cache or {})
        self.text_embedding_dim = int(text_embedding_dim)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.frame)

    def _teacher_entry(self, source_video_id: str) -> Optional[Mapping[str, Any]]:
        entry = self.teacher_cache.get(str(source_video_id))
        return entry if isinstance(entry, Mapping) else None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        source_video_id = str(row["source_video_id"])
        label = int(row["label"])
        transcript_source = str(row.get("transcript_source", "missing"))
        source_id = parse_source_id(transcript_source)

        teacher = self._teacher_entry(source_video_id)
        has_teacher = teacher is not None
        text_embedding = torch.zeros(self.text_embedding_dim, dtype=torch.float32)
        text_binary_prob = torch.tensor(0.5, dtype=torch.float32)
        text_severity_probs = torch.zeros(4, dtype=torch.float32)
        text_severity_probs[0 if label == 0 else 2] = 1.0
        teacher_confidence = torch.tensor(0.0, dtype=torch.float32)
        severity_target = 0 if label == 0 else 2

        if has_teacher:
            text_embedding = tensor_from_cache(teacher.get("embedding", text_embedding), self.text_embedding_dim)
            text_binary_prob = torch.tensor(float(teacher.get("binary_prob", 0.5)), dtype=torch.float32)
            text_severity_probs = tensor_from_cache(teacher.get("severity_probs", text_severity_probs), 4)
            text_severity_probs = text_severity_probs.clamp_min(1e-8)
            text_severity_probs = text_severity_probs / text_severity_probs.sum()
            teacher_confidence = torch.tensor(float(teacher.get("teacher_confidence", 1.0)), dtype=torch.float32)
            severity_target = int(teacher.get("pseudo_severity", teacher.get("predicted_severity", severity_target)))
            if label == 0:
                severity_target = 0
            elif severity_target == 0:
                severity_target = 2

        return {
            "pixel_values": load_clip_tensor(row, self.cfg),
            "label": torch.tensor(label, dtype=torch.float32),
            "label_long": torch.tensor(label, dtype=torch.long),
            "severity_target": torch.tensor(severity_target, dtype=torch.long),
            "text_embedding": text_embedding,
            "text_binary_prob": text_binary_prob,
            "text_severity_probs": text_severity_probs,
            "has_text": torch.tensor(float(bool(row.get("has_text", False))), dtype=torch.float32),
            "has_teacher": torch.tensor(float(has_teacher), dtype=torch.float32),
            "teacher_confidence": teacher_confidence,
            "transcript_source_id": torch.tensor(source_id, dtype=torch.long),
            "source_video_id": source_video_id,
            "clip_path": str(row["clip_path"]),
            "split": str(row.get("split", "")),
            "transcript_source": ID_TO_SOURCE.get(source_id, "missing"),
        }


class VideoTextFrameDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        cfg: TextVisionConfig,
        teacher_cache: Optional[Mapping[str, Mapping[str, Any]]] = None,
        text_embedding_dim: int = 768,
        train: bool = False,
    ) -> None:
        self.frame = video_level_manifest(frame).reset_index(drop=True)
        self.cfg = cfg
        self.teacher_cache = dict(teacher_cache or {})
        self.text_embedding_dim = int(text_embedding_dim)
        self.train = bool(train)
        self._frame_path_cache: Dict[str, List[Path]] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _teacher_entry(self, source_video_id: str) -> Optional[Mapping[str, Any]]:
        entry = self.teacher_cache.get(str(source_video_id))
        return entry if isinstance(entry, Mapping) else None

    def _frame_paths(self, source_video_id: str, frame_dir: Path) -> List[Path]:
        cached = self._frame_path_cache.get(source_video_id)
        if cached is not None:
            return cached
        paths = list_frame_paths(frame_dir)
        self._frame_path_cache[source_video_id] = paths
        return paths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        source_video_id = str(row["source_video_id"])
        label = int(row["label"])
        transcript_source = str(row.get("transcript_source", "missing"))
        source_id = parse_source_id(transcript_source)

        teacher = self._teacher_entry(source_video_id)
        has_teacher = teacher is not None
        text_embedding = torch.zeros(self.text_embedding_dim, dtype=torch.float32)
        text_binary_prob = torch.tensor(0.5, dtype=torch.float32)
        text_severity_probs = torch.zeros(4, dtype=torch.float32)
        text_severity_probs[0 if label == 0 else 2] = 1.0
        teacher_confidence = torch.tensor(0.0, dtype=torch.float32)
        severity_target = 0 if label == 0 else 2

        if has_teacher:
            text_embedding = tensor_from_cache(teacher.get("embedding", text_embedding), self.text_embedding_dim)
            text_binary_prob = torch.tensor(float(teacher.get("binary_prob", 0.5)), dtype=torch.float32)
            text_severity_probs = tensor_from_cache(teacher.get("severity_probs", text_severity_probs), 4)
            text_severity_probs = text_severity_probs.clamp_min(1e-8)
            text_severity_probs = text_severity_probs / text_severity_probs.sum()
            teacher_confidence = torch.tensor(float(teacher.get("teacher_confidence", 1.0)), dtype=torch.float32)
            severity_target = int(teacher.get("pseudo_severity", teacher.get("predicted_severity", severity_target)))
            if label == 0:
                severity_target = 0
            elif severity_target == 0:
                severity_target = 2

        frame_dir = resolve_frame_dir(row["frame_dir"], self.cfg)
        paths = self._frame_paths(source_video_id, frame_dir)
        if not paths:
            raise ValueError(f"No image frames found for {source_video_id} in frame_dir: {frame_dir}")
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

        usable_text = coerce_bool(row.get("has_text", False)) and has_teacher
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.float32),
            "label_long": torch.tensor(label, dtype=torch.long),
            "severity_target": torch.tensor(severity_target, dtype=torch.long),
            "text_embedding": text_embedding,
            "text_binary_prob": text_binary_prob,
            "text_severity_probs": text_severity_probs,
            "has_text": torch.tensor(float(usable_text), dtype=torch.float32),
            "has_teacher": torch.tensor(float(has_teacher), dtype=torch.float32),
            "teacher_confidence": teacher_confidence,
            "transcript_source_id": torch.tensor(source_id, dtype=torch.long),
            "source_video_id": source_video_id,
            "frame_dir": str(frame_dir),
            "num_source_frames": torch.tensor(len(paths), dtype=torch.long),
            "split": str(row.get("split", "")),
            "transcript_source": ID_TO_SOURCE.get(source_id, "missing"),
        }


def build_balanced_sampler(frame: pd.DataFrame, class_weight_power: float = 0.5) -> WeightedRandomSampler:
    labels = frame["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0.0] = 1.0
    class_weights = (len(labels) / counts) ** float(class_weight_power)
    sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def safe_metric(default: float, fn, *args, **kwargs) -> float:
    try:
        value = fn(*args, **kwargs)
    except Exception:
        return default
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def compute_binary_metrics(y_true: Iterable[int], prob_hate: Iterable[float], threshold: float) -> Dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(prob_hate), dtype=float)
    pred = (p >= float(threshold)).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": safe_metric(0.0, accuracy_score, y, pred),
        "f1": safe_metric(0.0, f1_score, y, pred, zero_division=0),
        "precision": safe_metric(0.0, precision_score, y, pred, zero_division=0),
        "recall": safe_metric(0.0, recall_score, y, pred, zero_division=0),
        "roc_auc": safe_metric(0.5, roc_auc_score, y, p),
        "pr_auc": safe_metric(float(y.mean()) if len(y) else 0.0, average_precision_score, y, p),
    }


def tune_threshold(
    y_true: Iterable[int],
    prob_hate: Iterable[float],
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(threshold_steps))
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, prob_hate, best_threshold)
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, prob_hate, float(threshold))
        if (metrics["f1"], metrics["pr_auc"], metrics["accuracy"]) > (
            best_metrics["f1"],
            best_metrics["pr_auc"],
            best_metrics["accuracy"],
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def compute_severity_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    return {
        "severity_accuracy_pseudo": safe_metric(0.0, accuracy_score, y, pred),
        "severity_macro_f1_pseudo": safe_metric(0.0, f1_score, y, pred, average="macro", zero_division=0),
    }


def aggregate_predictions(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    severity_cols = [column for column in predictions.columns if column.startswith("severity_prob_")]
    aggregations: Dict[str, Any] = {
        "label": "first",
        "prob_hate": "mean",
        "binary_logit": "mean",
        "split": "first",
        "has_text": "first",
        "transcript_source": "first",
        "severity_target": "first",
    }
    for optional_column in ["frame_dir", "clip_path"]:
        if optional_column in predictions.columns:
            aggregations[optional_column] = "first"
    for column in severity_cols:
        aggregations[column] = "mean"
    grouped = predictions.groupby("source_video_id", as_index=False).agg(aggregations)
    grouped["pred_binary"] = (grouped["prob_hate"] >= float(threshold)).astype(int)
    if severity_cols:
        grouped["pred_severity"] = grouped[severity_cols].to_numpy().argmax(axis=1).astype(int)
    else:
        grouped["pred_severity"] = 0
    grouped["confidence"] = np.maximum(grouped["prob_hate"], 1.0 - grouped["prob_hate"])
    return grouped


def write_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: List[int],
    path: str | Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

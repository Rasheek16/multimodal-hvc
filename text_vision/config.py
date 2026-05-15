from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_workspace_root() -> Path:
    project_root = _default_project_root()
    return Path(os.environ.get("HVC_WORKSPACE_ROOT", project_root.parent)).resolve()


PROJECT_ROOT = Path(os.environ.get("HVC_PROJECT_ROOT", _default_project_root())).resolve()
WORKSPACE_ROOT = _default_workspace_root()
TEXT_VISION_ROOT = PROJECT_ROOT / "text_vision"
DATASET_ROOT = Path(os.environ.get("HVC_DATASET_ROOT", WORKSPACE_ROOT / "datasets" / "diversified")).resolve()


SOURCE_TO_ID = {"missing": 0, "human": 1, "asr": 2}
ID_TO_SOURCE = {value: key for key, value in SOURCE_TO_ID.items()}


@dataclass
class TextVisionConfig:
    project_root: Path = PROJECT_ROOT
    workspace_root: Path = WORKSPACE_ROOT
    text_vision_root: Path = TEXT_VISION_ROOT
    dataset_root: Path = DATASET_ROOT

    splits_path: Path = DATASET_ROOT / "splits.csv"
    materialized_clip_index_path: Path = DATASET_ROOT / "materialized_clip_index.csv"
    clips_dir: Path = DATASET_ROOT / "clips"
    manifest_path: Path = TEXT_VISION_ROOT / "data" / "multimodal_manifest.csv"
    asr_cache_path: Path = TEXT_VISION_ROOT / "data" / "asr_transcripts.csv"

    text_model_path: Path = PROJECT_ROOT / "text_classification" / "roberta_base_finetuned_dualhead.pt"
    text_teacher_cache_path: Path = TEXT_VISION_ROOT / "data" / "text_teacher_cache.pt"
    text_teacher_summary_path: Path = TEXT_VISION_ROOT / "data" / "text_teacher_cache.csv"

    checkpoint_dir: Path = TEXT_VISION_ROOT / "checkpoints"
    report_dir: Path = TEXT_VISION_ROOT / "reports"
    checkpoint_path: Path = TEXT_VISION_ROOT / "checkpoints" / "best_text_guided_vision.pt"
    last_checkpoint_path: Path = TEXT_VISION_ROOT / "checkpoints" / "last_text_guided_vision.pt"
    frame_checkpoint_path: Path = TEXT_VISION_ROOT / "checkpoints" / "best_frame_text_vision.pt"
    frame_last_checkpoint_path: Path = TEXT_VISION_ROOT / "checkpoints" / "last_frame_text_vision.pt"
    frame_report_dir: Path = TEXT_VISION_ROOT / "reports" / "frame_text_vision"

    vision_model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics"
    frame_vision_model_name: str = "google/vit-base-patch16-224-in21k"
    text_model_name: str = "roberta-base"
    num_frames: int = 16
    image_size: int = 224
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    text_max_length: int = 160
    text_embedding_dim: int = 768

    batch_size: int = 4
    accum_steps: int = 8
    epochs: int = 8
    num_workers: int = 0
    seed: int = 42
    head_lr: float = 1e-5
    backbone_lr: float = 3e-7
    weight_decay: float = 5e-4
    grad_clip: float = 0.8
    freeze_epochs: int = 1
    unfreeze_last_n_blocks: int = 2
    head_dropout: float = 0.35
    text_dropout: float = 0.35
    class_weight_power: float = 0.5
    temporal_layers: int = 2
    temporal_heads: int = 8
    temporal_dropout: float = 0.10
    frame_chunk_size: int = 16
    temporal_pooling: str = "attn_mean_max"
    fusion_architecture: str = "gated"

    threshold_min: float = 0.10
    threshold_max: float = 0.90
    threshold_steps: int = 81
    selection_metric: str = "f1"

    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "video_binary": 1.00,
            "text_binary_distill": 0.35,
            "text_severity_distill": 0.45,
            "embedding_alignment": 0.25,
            "severity_consistency": 0.20,
            "severity_ce": 0.00,
        }
    )

    def ensure_directories(self) -> None:
        for path in [
            self.text_vision_root / "data",
            self.checkpoint_dir,
            self.report_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def to_jsonable(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


def get_config() -> TextVisionConfig:
    cfg = TextVisionConfig()
    cfg.ensure_directories()
    return cfg


def as_path(value: str | Path | None, fallback: Path) -> Path:
    if value is None or str(value).strip() == "":
        return fallback
    return Path(value).expanduser().resolve()


def parse_source_id(value: str | int | float | None) -> int:
    if value is None:
        return SOURCE_TO_ID["missing"]
    if isinstance(value, (int, float)) and int(value) in ID_TO_SOURCE:
        return int(value)
    return SOURCE_TO_ID.get(str(value).strip().lower(), SOURCE_TO_ID["missing"])


def source_vocab_size() -> int:
    return max(ID_TO_SOURCE) + 1

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import SOURCE_TO_ID, as_path, get_config
from text_vision.data_utils import adapt_clip_to_model_frames, load_clip_tensor, normalize_transcript, safe_torch_load
from text_vision.models.text_teacher import FrozenTextTeacher
from text_vision.train_text_guided_vision import load_model_from_checkpoint


def load_raw_video(path: Path, cfg) -> torch.Tensor:
    try:
        from torchvision.io import read_video
    except Exception as exc:
        raise RuntimeError("Raw video inference requires torchvision video IO. Use --clip-path for .pt clips.") from exc
    video, _, _ = read_video(str(path), pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"No frames could be read from {path}")
    return adapt_clip_to_model_frames(video, cfg.num_frames, cfg.image_size, cfg.image_mean, cfg.image_std)


def clips_from_args(args: argparse.Namespace, cfg) -> List[torch.Tensor]:
    if args.clip_path:
        row = {"clip_path": str(as_path(args.clip_path, Path(args.clip_path)))}
        return [load_clip_tensor(row, cfg)]
    if args.video_path:
        video_path = as_path(args.video_path, Path(args.video_path))
        if video_path.suffix.lower() == ".pt":
            row = {"clip_path": str(video_path)}
            return [load_clip_tensor(row, cfg)]
        return [load_raw_video(video_path, cfg)]
    if args.source_video_id:
        manifest = pd.read_csv(cfg.manifest_path)
        rows = manifest[manifest["source_video_id"].astype(str) == str(args.source_video_id)]
        if rows.empty:
            raise ValueError(f"source_video_id not found in manifest: {args.source_video_id}")
        rows = rows.sort_values("clip_idx").head(int(args.max_clips))
        return [load_clip_tensor(row, cfg) for row in rows.to_dict("records")]
    raise ValueError("Provide --clip-path, --video-path, or --source-video-id")


def teacher_features(args: argparse.Namespace, cfg, device: torch.device, text_embedding_dim: int) -> Dict[str, torch.Tensor]:
    transcript = normalize_transcript(args.transcript)
    if transcript:
        teacher = FrozenTextTeacher(
            checkpoint_path=cfg.text_model_path,
            device=device,
            local_files_only=args.local_files_only,
        )
        output = teacher.predict_one(transcript, max_length=cfg.text_max_length)
        return {
            "text_embedding": output["embedding"].float().to(device).view(1, -1),
            "text_binary_prob": output["binary_prob"].float().to(device).view(1),
            "text_severity_probs": output["severity_probs"].float().to(device).view(1, 4),
            "has_text": torch.ones(1, device=device),
            "transcript_source_id": torch.tensor([SOURCE_TO_ID["human"]], dtype=torch.long, device=device),
        }
    return {
        "text_embedding": torch.zeros(1, text_embedding_dim, dtype=torch.float32, device=device),
        "text_binary_prob": torch.full((1,), 0.5, dtype=torch.float32, device=device),
        "text_severity_probs": torch.full((1, 4), 0.25, dtype=torch.float32, device=device),
        "has_text": torch.zeros(1, dtype=torch.float32, device=device),
        "transcript_source_id": torch.tensor([SOURCE_TO_ID["missing"]], dtype=torch.long, device=device),
    }


@torch.inference_mode()
def classify(args: argparse.Namespace) -> Dict:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.text_teacher_cache_path = as_path(args.teacher_cache, cfg.text_teacher_cache_path)
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.checkpoint_path)
    cfg.text_model_path = as_path(args.text_model, cfg.text_model_path)
    cfg.vision_model_name = args.model_name or cfg.vision_model_name

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = safe_torch_load(cfg.checkpoint_path, map_location="cpu")
    text_embedding_dim = int(checkpoint.get("text_embedding_dim", cfg.text_embedding_dim))
    model, loaded_checkpoint = load_model_from_checkpoint(
        cfg.checkpoint_path,
        cfg,
        device=device,
        local_files_only=args.local_files_only,
    )
    threshold = float(args.threshold if args.threshold is not None else loaded_checkpoint.get("threshold", 0.5))
    text = teacher_features(args, cfg, device, text_embedding_dim)
    clips = clips_from_args(args, cfg)

    binary_logits = []
    severity_probs = []
    for clip in clips:
        batch = {
            "pixel_values": clip.unsqueeze(0).to(device),
            **text,
        }
        outputs = model(batch)
        binary_logits.append(outputs["binary_logits"].detach().cpu())
        severity_probs.append(F.softmax(outputs["severity_logits"], dim=-1).detach().cpu())

    mean_logit = torch.cat(binary_logits).mean().item()
    prob_hate = float(torch.sigmoid(torch.tensor(mean_logit)).item())
    mean_severity_probs = torch.cat(severity_probs, dim=0).mean(dim=0)
    severity = int(mean_severity_probs.argmax().item())
    pred_binary = int(prob_hate >= threshold)
    return {
        "is_hate": bool(pred_binary),
        "binary_label": pred_binary,
        "prob_hate": prob_hate,
        "confidence": float(max(prob_hate, 1.0 - prob_hate)),
        "threshold": threshold,
        "severity": severity,
        "severity_probs": [float(value) for value in mean_severity_probs.tolist()],
        "num_clips": len(clips),
    }


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Run text-guided vision inference on a clip or video.")
    parser.add_argument("--checkpoint", type=str, default=str(cfg.checkpoint_path))
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--teacher-cache", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--text-model", type=str, default=str(cfg.text_model_path))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--clip-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--source-video-id", type=str, default=None)
    parser.add_argument("--transcript", type=str, default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(classify(parse_args()), indent=2))


if __name__ == "__main__":
    main()


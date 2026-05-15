from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import SOURCE_TO_ID, as_path, get_config
from text_vision.data_utils import (
    load_frame_tensor,
    normalize_transcript,
    resolve_frame_dir,
    safe_torch_load,
    video_level_manifest,
)
from text_vision.models.text_teacher import FrozenTextTeacher
from text_vision.train_frame_text_vision import load_model_from_checkpoint


def frame_tensor_from_args(args: argparse.Namespace, cfg) -> Tuple[torch.Tensor, Dict]:
    if args.frame_dir:
        frame_dir = resolve_frame_dir(args.frame_dir, cfg)
        return load_frame_tensor(frame_dir, cfg, num_frames=cfg.num_frames, train=False), {"frame_dir": str(frame_dir)}
    if args.source_video_id:
        manifest = video_level_manifest(pd.read_csv(cfg.manifest_path))
        rows = manifest[manifest["source_video_id"].astype(str) == str(args.source_video_id)]
        if rows.empty:
            raise ValueError(f"source_video_id not found in manifest: {args.source_video_id}")
        row = rows.iloc[0]
        frame_dir = resolve_frame_dir(row["frame_dir"], cfg)
        tensor = load_frame_tensor(frame_dir, cfg, num_frames=cfg.num_frames, train=False)
        return tensor, {
            "source_video_id": str(row["source_video_id"]),
            "frame_dir": str(frame_dir),
            "manifest_label": int(row["label"]),
            "manifest_split": str(row["split"]),
            "manifest_transcript_source": str(row.get("transcript_source", "missing")),
        }
    raise ValueError("Provide --source-video-id or --frame-dir")


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
            "has_text": torch.ones(1, dtype=torch.float32, device=device),
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
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.frame_checkpoint_path)
    cfg.text_model_path = as_path(args.text_model, cfg.text_model_path)
    cfg.frame_vision_model_name = args.model_name or cfg.frame_vision_model_name
    cfg.vision_model_name = cfg.frame_vision_model_name
    cfg.num_frames = int(args.num_frames)

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
    frames, metadata = frame_tensor_from_args(args, cfg)

    batch = {
        "pixel_values": frames.unsqueeze(0).to(device),
        **text,
    }
    outputs = model(batch)
    prob_hate = float(torch.sigmoid(outputs["binary_logits"])[0].detach().cpu().item())
    severity_probs = F.softmax(outputs["severity_logits"], dim=-1)[0].detach().cpu()
    severity = int(severity_probs.argmax().item())
    pred_binary = int(prob_hate >= threshold)
    return {
        **metadata,
        "is_hate": bool(pred_binary),
        "binary_label": pred_binary,
        "prob_hate": prob_hate,
        "confidence": float(max(prob_hate, 1.0 - prob_hate)),
        "threshold": threshold,
        "severity": severity,
        "severity_probs": [float(value) for value in severity_probs.tolist()],
        "num_frames": int(cfg.num_frames),
        "has_transcript_input": bool(normalize_transcript(args.transcript)),
    }


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Run frame-directory text-vision inference.")
    parser.add_argument("--checkpoint", type=str, default=str(cfg.frame_checkpoint_path))
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--teacher-cache", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--text-model", type=str, default=str(cfg.text_model_path))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--source-video-id", type=str, default=None)
    parser.add_argument("--frame-dir", type=str, default=None)
    parser.add_argument("--transcript", type=str, default="")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(classify(parse_args()), indent=2))


if __name__ == "__main__":
    main()

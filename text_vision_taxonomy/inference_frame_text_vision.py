from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import SOURCE_TO_ID, TAXONOMY_LABELS, as_path, get_config
from text_vision.data_utils import (
    coerce_bool,
    load_frame_tensor,
    normalize_transcript,
    resolve_frame_dir,
    safe_torch_load,
    video_level_manifest,
)
from text_vision.moderation_reasoner import reason_about_moderation
from text_vision.models.text_teacher import FrozenTextTeacher
from text_vision.taxonomy_postprocessing import evidence_text_blob, postprocess_taxonomy
from text_vision.train_frame_text_vision import load_model_from_checkpoint
from text_vision.video_description import VideoDescriptionResult, build_video_description


def frame_tensor_from_args(args: argparse.Namespace, cfg) -> Tuple[torch.Tensor, Dict[str, Any]]:
    start_sec = float(args.start_sec) if args.start_sec is not None else None
    end_sec = float(args.end_sec) if args.end_sec is not None else None
    if args.frame_dir:
        frame_dir = resolve_frame_dir(args.frame_dir, cfg)
        tensor = load_frame_tensor(
            frame_dir,
            cfg,
            num_frames=cfg.num_frames,
            train=False,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        metadata = {"frame_dir": str(frame_dir)}
        if start_sec is not None and end_sec is not None:
            metadata["segment"] = {"start_sec": start_sec, "end_sec": end_sec}
        return tensor, metadata
    if args.source_video_id:
        raw_manifest = pd.read_csv(cfg.manifest_path)
        if "segment_id" in raw_manifest.columns:
            rows = raw_manifest[raw_manifest["source_video_id"].astype(str) == str(args.source_video_id)].copy()
            if start_sec is not None and end_sec is not None and not rows.empty:
                row_start = pd.to_numeric(rows["start_sec"], errors="coerce")
                row_end = pd.to_numeric(rows["end_sec"], errors="coerce")
                rows = rows[(row_start.sub(start_sec).abs() < 1e-3) & (row_end.sub(end_sec).abs() < 1e-3)]
        else:
            manifest = video_level_manifest(raw_manifest)
            rows = manifest[manifest["source_video_id"].astype(str) == str(args.source_video_id)]
        if rows.empty:
            raise ValueError(f"source_video_id not found in manifest: {args.source_video_id}")
        row = rows.iloc[0]
        frame_dir = resolve_frame_dir(row["frame_dir"], cfg)
        row_start = float(row["start_sec"]) if "start_sec" in row.index and pd.notna(row["start_sec"]) else start_sec
        row_end = float(row["end_sec"]) if "end_sec" in row.index and pd.notna(row["end_sec"]) else end_sec
        fps_value = pd.to_numeric(row.get("fps", None), errors="coerce") if "fps" in row.index else None
        fps = None if fps_value is None or pd.isna(fps_value) else float(fps_value)
        tensor = load_frame_tensor(
            frame_dir,
            cfg,
            num_frames=cfg.num_frames,
            train=False,
            start_sec=row_start,
            end_sec=row_end,
            fps=fps,
            frame_dir_is_segment=coerce_bool(row.get("frame_dir_is_segment", False)),
        )
        metadata = {
            "source_video_id": str(row["source_video_id"]),
            "frame_dir": str(frame_dir),
            "manifest_label": int(row.get("label", row.get("binary_label", 0))),
            "manifest_split": str(row["split"]),
            "manifest_transcript_source": str(row.get("transcript_source", "missing")),
            "scene_text": normalize_transcript(row.get("Scene", row.get("text_for_taxonomy", ""))),
            "frame_dir_is_segment": coerce_bool(row.get("frame_dir_is_segment", False)),
        }
        if "segment_id" in row.index:
            metadata["segment_id"] = str(row["segment_id"])
        if row_start is not None and row_end is not None:
            metadata["segment"] = {"start_sec": row_start, "end_sec": row_end}
        return tensor, metadata
    raise ValueError("Provide --source-video-id or --frame-dir")


def teacher_features(
    transcript_value: str,
    args: argparse.Namespace,
    cfg,
    device: torch.device,
    text_embedding_dim: int,
    source: str = "human",
) -> Dict[str, torch.Tensor]:
    transcript = normalize_transcript(transcript_value)
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
            "transcript_source_id": torch.tensor([SOURCE_TO_ID.get(source, SOURCE_TO_ID["asr"])], dtype=torch.long, device=device),
        }
    return {
        "text_embedding": torch.zeros(1, text_embedding_dim, dtype=torch.float32, device=device),
        "text_binary_prob": torch.full((1,), 0.5, dtype=torch.float32, device=device),
        "text_severity_probs": torch.full((1, 4), 0.25, dtype=torch.float32, device=device),
        "has_text": torch.zeros(1, dtype=torch.float32, device=device),
        "transcript_source_id": torch.tensor([SOURCE_TO_ID["missing"]], dtype=torch.long, device=device),
    }


def _checkpoint_has_taxonomy(checkpoint: Mapping) -> bool:
    if checkpoint.get("taxonomy_labels") or checkpoint.get("taxonomy_thresholds"):
        return True
    config = checkpoint.get("config", {})
    if isinstance(config, Mapping) and config.get("taxonomy_labels"):
        return True
    state = checkpoint.get("model_state_dict", {})
    return isinstance(state, Mapping) and any("taxonomy_head" in key for key in state)


def _taxonomy_labels(checkpoint: Mapping, cfg) -> list[str]:
    labels = checkpoint.get("taxonomy_labels")
    if not labels and isinstance(checkpoint.get("config", {}), Mapping):
        labels = checkpoint["config"].get("taxonomy_labels")
    return [str(label) for label in (labels or getattr(cfg, "taxonomy_labels", TAXONOMY_LABELS))]


def _load_taxonomy_thresholds(path: str | None, checkpoint: Mapping, labels: list[str]) -> Dict[str, float]:
    thresholds = checkpoint.get("taxonomy_thresholds", {})
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            blob = json.load(handle)
        if isinstance(blob, Mapping):
            thresholds = blob.get("production_thresholds", blob.get("thresholds", blob))
        else:
            thresholds = {}
    return {label: float(thresholds.get(label, 0.5)) for label in labels}


def _description_to_output(description: VideoDescriptionResult | None) -> Dict[str, Any] | None:
    if description is None:
        return None
    return {
        "transcript": description.transcript,
        "ocr_text": description.ocr_text,
        "visual_description": description.visual_description,
        "text_for_classifier": description.text_for_classifier,
        "warnings": list(description.warnings),
    }


@torch.inference_mode()
def classify(args: argparse.Namespace) -> Dict:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.text_teacher_cache_path = as_path(args.teacher_cache, cfg.text_teacher_cache_path)
    selected_checkpoint = args.taxonomy_checkpoint or args.checkpoint
    cfg.checkpoint_path = as_path(selected_checkpoint, cfg.frame_checkpoint_path)
    cfg.text_model_path = as_path(args.text_model, cfg.text_model_path)
    cfg.frame_vision_model_name = args.model_name or cfg.frame_vision_model_name
    cfg.vision_model_name = cfg.frame_vision_model_name
    cfg.num_frames = int(args.num_frames)
    cfg.taxonomy_labels = list(TAXONOMY_LABELS)
    cfg.num_taxonomy_labels = len(TAXONOMY_LABELS)

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
    frames, metadata = frame_tensor_from_args(args, cfg)

    description: VideoDescriptionResult | None = None
    if args.describe_video or args.use_description_as_transcript:
        segment = metadata.get("segment", {}) if isinstance(metadata.get("segment", {}), Mapping) else {}
        description = build_video_description(
            metadata["frame_dir"],
            transcript=args.transcript,
            num_frames=int(args.description_frames),
            start_sec=segment.get("start_sec"),
            end_sec=segment.get("end_sec"),
            frame_dir_is_segment=bool(metadata.get("frame_dir_is_segment", False)),
            enable_vlm=bool(args.enable_vlm),
            vlm_model_name=args.vlm_model_name,
        )

    teacher_transcript = args.transcript
    teacher_source = "human"
    if args.use_description_as_transcript and description is not None:
        teacher_transcript = description.text_for_classifier
        teacher_source = "asr"
    text = teacher_features(teacher_transcript, args, cfg, device, text_embedding_dim, source=teacher_source)

    batch = {
        "pixel_values": frames.unsqueeze(0).to(device),
        **text,
    }
    outputs = model(batch)
    prob_hate = float(torch.sigmoid(outputs["binary_logits"])[0].detach().cpu().item())
    severity_probs = F.softmax(outputs["severity_logits"], dim=-1)[0].detach().cpu()
    severity = int(severity_probs.argmax().item())
    pred_binary = int(prob_hate >= threshold)
    result = {
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
        "has_description_input": bool(description and description.text_for_classifier),
    }
    taxonomy_probs: Dict[str, float] = {}
    taxonomy_thresholds: Dict[str, float] = {}
    should_return_taxonomy = bool(args.taxonomy_checkpoint or args.taxonomy_thresholds or _checkpoint_has_taxonomy(loaded_checkpoint))
    if should_return_taxonomy and "taxonomy_logits" in outputs:
        labels = _taxonomy_labels(loaded_checkpoint, cfg)
        thresholds = _load_taxonomy_thresholds(args.taxonomy_thresholds, loaded_checkpoint, labels)
        probs_tensor = torch.sigmoid(outputs["taxonomy_logits"])[0].detach().cpu()
        probs = {label: float(probs_tensor[index].item()) for index, label in enumerate(labels)}
        taxonomy_probs = probs
        taxonomy_thresholds = thresholds
        evidence_blob = evidence_text_blob(
            args.transcript,
            metadata.get("scene_text", ""),
            description.to_dict() if description is not None else None,
        )
        processed = postprocess_taxonomy(
            probs,
            thresholds,
            prob_hate=prob_hate,
            evidence_text=evidence_blob,
            labels=labels,
        )
        result["taxonomy"] = {
            "labels": processed["labels"],
            "candidates": processed["candidates"],
            "suppressed": processed["suppressed"],
            "probs": probs,
            "thresholds_used": processed["thresholds_used"],
        }
    if description is not None:
        result["description"] = _description_to_output(description)
    if args.include_evidence or description is not None or taxonomy_probs:
        reasoned = reason_about_moderation(
            prob_hate=prob_hate,
            taxonomy_probs=taxonomy_probs,
            taxonomy_labels=list(taxonomy_probs) if taxonomy_probs else TAXONOMY_LABELS,
            thresholds=taxonomy_thresholds,
            transcript=args.transcript,
            ocr_text=description.ocr_text if description is not None else "",
            visual_description=description.visual_description if description is not None else "",
            scene_text=str(metadata.get("scene_text", "")),
            evidence=[item.to_dict() for item in description.evidence] if description is not None else None,
            include_evidence=bool(args.include_evidence),
        )
        result["target_group"] = reasoned.target_group
        result["moderation_explanation"] = reasoned.explanation
        result["moderation_summary"] = reasoned.category_summary
        if args.include_evidence:
            result["evidence"] = reasoned.evidence
    if args.return_embeddings:
        result["embeddings"] = {
            "visual_embedding": [float(value) for value in outputs["visual_embedding"][0].detach().cpu().tolist()],
            "fusion_embedding": [float(value) for value in outputs["fusion_embedding"][0].detach().cpu().tolist()],
        }
    return result


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Run frame-directory text-vision inference.")
    parser.add_argument("--checkpoint", type=str, default=str(cfg.checkpoint_dir / "best_frame_text_vision_relabel_all_splits.pt"))
    parser.add_argument("--taxonomy-checkpoint", type=str, default=None)
    parser.add_argument("--taxonomy-thresholds", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--teacher-cache", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--text-model", type=str, default=str(cfg.text_model_path))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--source-video-id", type=str, default=None)
    parser.add_argument("--frame-dir", type=str, default=None)
    parser.add_argument("--transcript", type=str, default="")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--describe-video", action="store_true")
    parser.add_argument("--description-frames", type=int, default=8)
    parser.add_argument("--use-description-as-transcript", action="store_true")
    parser.add_argument("--include-evidence", action="store_true")
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--vlm-model-name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--return-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(classify(parse_args()), indent=2))


if __name__ == "__main__":
    main()

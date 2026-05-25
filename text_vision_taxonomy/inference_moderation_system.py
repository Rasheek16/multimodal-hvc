from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import get_config
from text_vision.data_utils import normalize_transcript
from text_vision.inference_frame_text_vision import classify as classify_vision
from text_vision.inference_multilabel_taxonomy import tfidf_probs, transformer_probs
from text_vision.models.text_teacher import FrozenTextTeacher
from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, load_thresholds, text_block
from text_vision.rare_label_rules import (
    RARE_LABELS,
    THREAT_TERMS,
    apply_rare_label_rules,
    contains_any,
)
from text_vision.video_metadata_generator import (
    VideoMetadata,
    build_video_metadata,
    metadata_runtime_status,
    sample_frames,
)


HATE_SPECIFIC_LABELS = {
    "hate_speech",
    "discrimination",
    "contextual_hate",
    "threat",
    "illegal",
    "online_harm",
}
ACTION_LABELS = {"violence", "sexual", "fear"}


def _model_log_context(show_logs: bool):
    stack = contextlib.ExitStack()
    if not show_logs:
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _manual_metadata_provided(args: argparse.Namespace) -> bool:
    return any(
        normalize_transcript(value)
        for value in [
            args.scene,
            args.action,
            args.action_clean,
            args.subcategories,
            args.parent_labels,
        ]
    )


def _metadata_from_manual_args(args: argparse.Namespace) -> VideoMetadata:
    return VideoMetadata(
        scene=normalize_transcript(args.scene),
        action=normalize_transcript(args.action),
        action_clean=normalize_transcript(args.action_clean),
        subcategories=normalize_transcript(args.subcategories),
        parent_labels=normalize_transcript(args.parent_labels),
        transcript=normalize_transcript(args.transcript),
        ocr_text=normalize_transcript(args.ocr_text),
        evidence=[
            {
                "source": "manual_metadata",
                "text": "manual taxonomy metadata fields were provided; generation was skipped",
                "status": "provided",
            }
        ],
    )


def _metadata_dict(metadata: VideoMetadata, generated: bool, manual: bool) -> dict[str, Any]:
    payload = {
        "scene": metadata.scene,
        "action": metadata.action,
        "action_clean": metadata.action_clean,
        "subcategories": metadata.subcategories,
        "parent_labels": metadata.parent_labels,
        "transcript": metadata.transcript,
        "ocr_text": metadata.ocr_text,
        "manual_metadata_used": bool(manual),
    }
    status = metadata_runtime_status(metadata)
    payload["metadata_generation_available"] = bool(generated and status["metadata_generation_available"])
    payload["available_fields"] = status["available_fields"]
    return payload


def _taxonomy_text(metadata: VideoMetadata) -> str:
    row = {
        "Scene": metadata.scene,
        "Action": metadata.action,
        "Subcategories": metadata.subcategories,
        "ParentLabels": metadata.parent_labels,
        "action_clean": metadata.action_clean,
    }
    base = text_block(row)
    extra_parts: list[str] = []
    if metadata.transcript:
        extra_parts.append(f"[TRANSCRIPT]\n{metadata.transcript}")
    if metadata.ocr_text:
        extra_parts.append(f"[OCR_TEXT]\n{metadata.ocr_text}")
    if extra_parts:
        return "\n\n".join(part for part in [base, *extra_parts] if part)
    return base


def _load_taxonomy_probs(args: argparse.Namespace, text: str, evidence: list[dict]) -> tuple[dict[str, float], bool, str]:
    source = (
        "transformer_multilabel_classifier"
        if args.taxonomy_model_type == "transformer"
        else "text_multilabel_classifier"
    )
    path = Path(args.transformer_taxonomy_model if args.taxonomy_model_type == "transformer" else args.taxonomy_classifier)
    if not path.exists():
        evidence.append(
            {
                "source": "taxonomy_classifier",
                "status": "missing",
                "text": f"{source} not found: {path}",
            }
        )
        return {label: 0.0 for label in LABEL_COLUMNS}, False, source
    try:
        with _model_log_context(bool(args.show_model_logs)):
            if args.taxonomy_model_type == "transformer":
                probs = transformer_probs(
                    path,
                    text,
                    local_files_only=bool(args.local_files_only),
                    device_name=args.device,
                    max_length=int(args.taxonomy_max_length),
                )
            else:
                probs = tfidf_probs(path, text)
    except Exception as exc:
        evidence.append(
            {
                "source": "taxonomy_classifier",
                "status": "failed",
                "text": f"{source} failed: {type(exc).__name__}: {exc}",
            }
        )
        return {label: 0.0 for label in LABEL_COLUMNS}, False, source
    evidence.append(
        {
            "source": "taxonomy_classifier",
            "status": "ok",
            "text": f"{source} probabilities generated from {path}",
        }
    )
    return {label: float(probs.get(label, 0.0)) for label in LABEL_COLUMNS}, True, source


def _combine_deep_taxonomy_probs(
    text_probs: Mapping[str, float],
    video_probs: Mapping[str, float],
) -> dict[str, float]:
    has_text = any(float(text_probs.get(label, 0.0)) > 0.0 for label in LABEL_COLUMNS)
    has_video = any(float(video_probs.get(label, 0.0)) > 0.0 for label in LABEL_COLUMNS)
    if has_text and not has_video:
        return {label: float(text_probs.get(label, 0.0)) for label in LABEL_COLUMNS}
    if has_video and not has_text:
        return {label: float(video_probs.get(label, 0.0)) for label in LABEL_COLUMNS}
    if not has_text and not has_video:
        return {label: 0.0 for label in LABEL_COLUMNS}

    combined: dict[str, float] = {}
    for label in LABEL_COLUMNS:
        text_value = float(text_probs.get(label, 0.0))
        video_value = float(video_probs.get(label, 0.0))
        if label in {"violence", "sexual", "fear"}:
            text_weight, video_weight = 0.55, 0.45
        elif label == "threat":
            text_weight, video_weight = 0.60, 0.40
        else:
            text_weight, video_weight = 0.80, 0.20
        combined[label] = float(text_weight * text_value + video_weight * video_value)
    return combined


def _severity_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _max_severity(*values: Any) -> int | None:
    parsed = [_severity_value(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return max(parsed) if parsed else None


def _hate_prob_values(signals: Mapping[str, Any]) -> list[float]:
    values = []
    for key in ["text_hate_prob", "vision_hate_prob"]:
        value = signals.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _append_unique_item(items: list[dict], item: Mapping[str, Any]) -> None:
    label = str(item.get("label", ""))
    reason = str(item.get("reason", ""))
    for existing in items:
        if str(existing.get("label", "")) == label and str(existing.get("reason", "")) == reason:
            return
    items.append({key: _jsonable(value) for key, value in item.items()})


def _apply_base_model_gating(
    probs: Mapping[str, float],
    thresholds: Mapping[str, float],
    text: str,
    rare_result: Mapping[str, Any],
    base_signals: Mapping[str, Any],
    low_hate_threshold: float = 0.35,
    support_hate_threshold: float = 0.70,
    boost_margin: float = 0.10,
) -> dict[str, Any]:
    raw_labels = [
        label
        for label in LABEL_COLUMNS
        if float(probs.get(label, 0.0)) >= float(thresholds.get(label, 0.5))
    ]
    final_labels = list(rare_result.get("final_labels", []))
    candidates = [dict(item) for item in rare_result.get("candidate_labels", [])]
    suppressed = [dict(item) for item in rare_result.get("suppressed_labels", [])]
    thresholds_used = {label: float(thresholds.get(label, 0.5)) for label in LABEL_COLUMNS}

    hate_probs = _hate_prob_values(base_signals)
    both_hate_probs_available = (
        base_signals.get("text_hate_prob") is not None and base_signals.get("vision_hate_prob") is not None
    )
    low_both_hate = both_hate_probs_available and all(value < float(low_hate_threshold) for value in hate_probs)
    supportive_hate = any(value >= float(support_hate_threshold) for value in hate_probs)
    max_severity = _max_severity(base_signals.get("text_severity"), base_signals.get("vision_severity"))

    if low_both_hate:
        for label in list(final_labels):
            if label in HATE_SPECIFIC_LABELS:
                final_labels.remove(label)
                _append_unique_item(
                    suppressed,
                    {
                        "label": label,
                        "probability": float(probs.get(label, 0.0)),
                        "threshold": thresholds_used[label],
                        "reason": "both_base_hate_probabilities_low",
                        "text_hate_prob": base_signals.get("text_hate_prob"),
                        "vision_hate_prob": base_signals.get("vision_hate_prob"),
                    },
                )
        for label in raw_labels:
            if label in HATE_SPECIFIC_LABELS and label not in final_labels:
                _append_unique_item(
                    suppressed,
                    {
                        "label": label,
                        "probability": float(probs.get(label, 0.0)),
                        "threshold": thresholds_used[label],
                        "reason": "both_base_hate_probabilities_low",
                        "text_hate_prob": base_signals.get("text_hate_prob"),
                        "vision_hate_prob": base_signals.get("vision_hate_prob"),
                    },
                )

    if supportive_hate and not low_both_hate:
        for label in ["hate_speech", "discrimination", "contextual_hate"]:
            probability = float(probs.get(label, 0.0))
            threshold = thresholds_used[label]
            if label not in final_labels and probability >= max(0.50, threshold - float(boost_margin)):
                final_labels.append(label)
                _append_unique_item(
                    candidates,
                    {
                        "label": label,
                        "probability": probability,
                        "threshold": threshold,
                        "reason": "base_hate_support_boost",
                    },
                )
        for label in ["threat", "illegal", "online_harm"]:
            probability = float(probs.get(label, 0.0))
            threshold = thresholds_used[label]
            if label not in final_labels and probability >= max(0.50, threshold - float(boost_margin)):
                _append_unique_item(
                    candidates,
                    {
                        "label": label,
                        "probability": probability,
                        "threshold": threshold,
                        "reason": "base_hate_support_candidate_rare_label",
                    },
                )

    if max_severity is not None and max_severity >= 3:
        violence_prob = float(probs.get("violence", 0.0))
        violence_threshold = thresholds_used["violence"]
        if "violence" not in final_labels and violence_prob >= max(0.35, violence_threshold - 0.25):
            _append_unique_item(
                candidates,
                {
                    "label": "violence",
                    "probability": violence_prob,
                    "threshold": violence_threshold,
                    "reason": "severity_high_candidate",
                    "severity": max_severity,
                },
            )
        threat_prob = float(probs.get("threat", 0.0))
        threat_threshold = thresholds_used["threat"]
        has_threat_text = contains_any(text, THREAT_TERMS)
        if (
            "threat" not in final_labels
            and threat_prob >= max(0.70, threat_threshold - 0.20)
            and (has_threat_text or threat_prob >= 0.95)
        ):
            _append_unique_item(
                candidates,
                {
                    "label": "threat",
                    "probability": threat_prob,
                    "threshold": threat_threshold,
                    "reason": "severity_high_threat_candidate_with_evidence",
                    "severity": max_severity,
                },
            )

    if "threat" in final_labels:
        threat_prob = float(probs.get("threat", 0.0))
        has_threat_text = contains_any(text, THREAT_TERMS)
        if not has_threat_text and threat_prob < 0.95:
            final_labels.remove("threat")
            item = {
                "label": "threat",
                "probability": threat_prob,
                "threshold": thresholds_used["threat"],
                "reason": "threat_requires_text_evidence_or_very_high_taxonomy_probability",
            }
            _append_unique_item(suppressed, item)
            _append_unique_item(candidates, item)

    # Keep deterministic label order.
    final_set = set(final_labels)
    final_labels = [label for label in LABEL_COLUMNS if label in final_set]
    return {
        "taxonomy_raw_labels": raw_labels,
        "taxonomy_final_labels": final_labels,
        "candidate_labels": candidates,
        "suppressed_labels": suppressed,
        "thresholds_used": thresholds_used,
    }


def _frame_dir_from_metadata(args: argparse.Namespace, metadata: VideoMetadata) -> str:
    if args.frame_dir:
        return str(args.frame_dir)
    for item in metadata.evidence:
        if item.get("source") == "frame_sampler" and item.get("output_dir"):
            output_dir = Path(str(item["output_dir"]))
            if output_dir.exists():
                return str(output_dir)
    return ""


def _run_vision_base_model(args: argparse.Namespace, metadata: VideoMetadata, evidence: list[dict]) -> dict[str, Any]:
    frame_dir = _frame_dir_from_metadata(args, metadata)
    if not frame_dir and args.video:
        frames = sample_frames(video_path=args.video, num_frames=int(args.metadata_frames))
        if frames:
            frame_dir = str(Path(frames[0]).parent)
            evidence.append(
                {
                    "source": "frame_sampler",
                    "status": "ok",
                    "text": f"sampled {len(frames)} frame(s) for vision base model",
                    "video_path": args.video,
                    "output_dir": frame_dir,
                }
            )
    if not frame_dir:
        evidence.append({"source": "vision_base_model", "status": "skipped", "text": "no frame directory available"})
        return {"available": False, "prob_hate": None, "severity": None}
    taxonomy_checkpoint = Path(args.video_taxonomy_checkpoint)
    use_video_taxonomy = bool(args.use_video_taxonomy and taxonomy_checkpoint.exists())
    if args.use_video_taxonomy and not taxonomy_checkpoint.exists():
        evidence.append(
            {
                "source": "video_taxonomy_head",
                "status": "missing",
                "text": f"video taxonomy checkpoint not found: {taxonomy_checkpoint}",
            }
        )
    checkpoint = taxonomy_checkpoint if use_video_taxonomy else Path(args.vision_checkpoint)
    if not checkpoint.exists():
        evidence.append(
            {
                "source": "vision_base_model",
                "status": "missing",
                "text": f"vision checkpoint not found: {checkpoint}",
            }
        )
        return {"available": False, "prob_hate": None, "severity": None}
    try:
        vision_args = argparse.Namespace(
            checkpoint=str(checkpoint),
            taxonomy_checkpoint=str(taxonomy_checkpoint) if use_video_taxonomy else None,
            taxonomy_thresholds=args.video_taxonomy_thresholds if use_video_taxonomy else None,
            manifest=args.manifest,
            teacher_cache=args.teacher_cache,
            text_model=args.text_model,
            model_name=args.vision_model_name,
            source_video_id=None,
            frame_dir=frame_dir,
            transcript="",
            num_frames=int(args.vision_num_frames),
            start_sec=None,
            end_sec=None,
            describe_video=False,
            description_frames=int(args.metadata_frames),
            use_description_as_transcript=False,
            include_evidence=False,
            enable_vlm=False,
            vlm_model_name=args.vlm_model_name,
            threshold=args.hate_threshold,
            device=args.device,
            local_files_only=bool(args.local_files_only),
            return_embeddings=False,
        )
        with _model_log_context(bool(args.show_model_logs)):
            result = classify_vision(vision_args)
        evidence.append(
            {
                "source": "vision_base_model",
                "status": "ok",
                "text": f"vision model ran on {frame_dir}",
            }
        )
        return {
            "available": True,
            "prob_hate": float(result.get("prob_hate", 0.0)),
            "severity": _severity_value(result.get("severity")),
            "threshold": float(result.get("threshold", args.hate_threshold or 0.5)),
            "severity_probs": result.get("severity_probs", []),
            "taxonomy_available": bool(result.get("taxonomy", {}).get("probs")),
            "taxonomy_probs": {
                label: float(result.get("taxonomy", {}).get("probs", {}).get(label, 0.0))
                for label in LABEL_COLUMNS
            },
            "taxonomy_checkpoint": str(taxonomy_checkpoint) if use_video_taxonomy else "",
        }
    except Exception as exc:
        evidence.append(
            {
                "source": "vision_base_model",
                "status": "failed",
                "text": f"{type(exc).__name__}: {exc}",
            }
        )
        return {"available": False, "prob_hate": None, "severity": None}


def _text_for_base_model(metadata: VideoMetadata) -> str:
    parts = [
        metadata.transcript,
        metadata.ocr_text,
        metadata.scene,
        metadata.action,
        metadata.subcategories,
        metadata.parent_labels,
    ]
    return normalize_transcript(" ".join(part for part in parts if part))


def _run_text_base_model(args: argparse.Namespace, metadata: VideoMetadata, evidence: list[dict]) -> dict[str, Any]:
    text = _text_for_base_model(metadata)
    if not text:
        evidence.append({"source": "text_base_model", "status": "skipped", "text": "no text input available"})
        return {"available": False, "prob_hate": None, "severity": None}
    checkpoint = Path(args.text_model)
    if not checkpoint.exists():
        evidence.append(
            {
                "source": "text_base_model",
                "status": "missing",
                "text": f"text model checkpoint not found: {checkpoint}",
            }
        )
        return {"available": False, "prob_hate": None, "severity": None}
    try:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with _model_log_context(bool(args.show_model_logs)):
            teacher = FrozenTextTeacher(checkpoint, device=device, local_files_only=bool(args.local_files_only))
            output = teacher.predict_one(text, max_length=int(args.text_max_length))
        severity_probs = output["severity_probs"].detach().cpu()
        severity = int(severity_probs.argmax().item())
        prob_hate = float(output["binary_prob"].detach().cpu().item())
        evidence.append({"source": "text_base_model", "status": "ok", "text": f"text model ran on {checkpoint}"})
        return {
            "available": True,
            "prob_hate": prob_hate,
            "severity": severity,
            "severity_probs": [float(value) for value in severity_probs.tolist()],
        }
    except Exception as exc:
        evidence.append(
            {
                "source": "text_base_model",
                "status": "failed",
                "text": f"{type(exc).__name__}: {exc}",
            }
        )
        return {"available": False, "prob_hate": None, "severity": None}


def _base_model_signals(text_result: Mapping[str, Any], vision_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text_hate_prob": text_result.get("prob_hate") if text_result.get("available") else None,
        "vision_hate_prob": vision_result.get("prob_hate") if vision_result.get("available") else None,
        "text_severity": text_result.get("severity") if text_result.get("available") else None,
        "vision_severity": vision_result.get("severity") if vision_result.get("available") else None,
    }


def _apply_base_signal_overrides(
    args: argparse.Namespace,
    text_result: dict[str, Any],
    vision_result: dict[str, Any],
    evidence: list[dict],
) -> dict[str, Any]:
    if args.text_hate_prob is not None:
        text_result["available"] = True
        text_result["prob_hate"] = float(args.text_hate_prob)
        text_result["provided_signal"] = True
    if args.text_severity is not None:
        text_result["available"] = True
        text_result["severity"] = int(args.text_severity)
        text_result["provided_signal"] = True
    if args.vision_hate_prob is not None:
        vision_result["available"] = True
        vision_result["prob_hate"] = float(args.vision_hate_prob)
        vision_result["provided_signal"] = True
    if args.vision_severity is not None:
        vision_result["available"] = True
        vision_result["severity"] = int(args.vision_severity)
        vision_result["provided_signal"] = True
    if any(
        value is not None
        for value in [args.text_hate_prob, args.text_severity, args.vision_hate_prob, args.vision_severity]
    ):
        evidence.append(
            {
                "source": "base_model_signals",
                "status": "provided",
                "text": "one or more base-model signals were supplied by CLI override",
            }
        )
    return _base_model_signals(text_result, vision_result)


def _final_hate_probability(base_signals: Mapping[str, Any], taxonomy_probs: Mapping[str, float]) -> float:
    hate_probs = _hate_prob_values(base_signals)
    if hate_probs:
        return float(sum(hate_probs) / len(hate_probs))
    hate_taxonomy = [
        float(taxonomy_probs.get(label, 0.0))
        for label in ["hate_speech", "discrimination", "contextual_hate", "threat", "illegal", "online_harm"]
    ]
    return float(max(hate_taxonomy) if hate_taxonomy else 0.0)


def _append_feedback_log(output: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video": output.get("input", {}).get("video", ""),
        "frame_dir": output.get("input", {}).get("frame_dir", ""),
        "metadata": json.dumps(output.get("metadata", {}), ensure_ascii=False),
        "taxonomy_probs": json.dumps(output.get("taxonomy", {}).get("probs", {}), ensure_ascii=False),
        "taxonomy_raw_labels": json.dumps(output.get("taxonomy", {}).get("taxonomy_raw_labels", []), ensure_ascii=False),
        "taxonomy_final_labels": json.dumps(output.get("taxonomy", {}).get("taxonomy_final_labels", []), ensure_ascii=False),
        "suppressed_labels": json.dumps(output.get("taxonomy", {}).get("suppressed_labels", []), ensure_ascii=False),
        "candidate_labels": json.dumps(output.get("taxonomy", {}).get("candidate_labels", []), ensure_ascii=False),
        "base_model_signals": json.dumps(output.get("base_model_signals", {}), ensure_ascii=False),
        "thresholds_used": json.dumps(output.get("taxonomy", {}).get("thresholds_used", {}), ensure_ascii=False),
        "evidence": json.dumps(output.get("evidence", []), ensure_ascii=False),
    }
    write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manual_metadata = _manual_metadata_provided(args)
    generated_metadata = False
    if manual_metadata:
        metadata = _metadata_from_manual_args(args)
    elif args.generate_metadata or args.video or args.frame_dir:
        auto_asr = bool(args.video) and not bool(args.disable_asr)
        metadata = build_video_metadata(
            video_path=args.video,
            frame_dir=args.frame_dir,
            transcript=args.transcript,
            num_frames=int(args.metadata_frames),
            enable_asr=bool(args.enable_asr or auto_asr),
            enable_ocr=not bool(args.disable_ocr),
            enable_vlm=bool(args.enable_vlm and not args.disable_vlm),
            enable_rule_metadata=bool(args.enable_rule_metadata),
            vlm_model_name=args.vlm_model_name,
        )
        generated_metadata = True
    else:
        metadata = VideoMetadata(
            transcript=normalize_transcript(args.transcript),
            ocr_text=normalize_transcript(args.ocr_text),
            evidence=[
                {
                    "source": "metadata",
                    "status": "empty",
                    "text": "no manual metadata or raw video/frame input supplied",
                }
            ],
        )

    evidence = list(metadata.evidence)
    taxonomy_text = _taxonomy_text(metadata)
    thresholds = load_thresholds(args.taxonomy_thresholds)
    vision = _run_vision_base_model(args, metadata, evidence)
    text = _run_text_base_model(args, metadata, evidence)
    text_taxonomy_probs, text_taxonomy_available, text_taxonomy_source = _load_taxonomy_probs(args, taxonomy_text, evidence)
    video_taxonomy_probs = {
        label: float(vision.get("taxonomy_probs", {}).get(label, 0.0)) for label in LABEL_COLUMNS
    }
    video_taxonomy_available = bool(vision.get("taxonomy_available"))
    taxonomy_probs = _combine_deep_taxonomy_probs(text_taxonomy_probs, video_taxonomy_probs)
    taxonomy_available = bool(text_taxonomy_available or video_taxonomy_available)
    taxonomy_source = (
        "text_transformer_plus_video_taxonomy"
        if text_taxonomy_available and video_taxonomy_available
        else text_taxonomy_source
        if text_taxonomy_available
        else "video_taxonomy_head"
        if video_taxonomy_available
        else text_taxonomy_source
    )
    base_signals = _apply_base_signal_overrides(args, text, vision, evidence)
    max_severity = _max_severity(base_signals["text_severity"], base_signals["vision_severity"])

    rare_result = apply_rare_label_rules(
        taxonomy_probs,
        thresholds,
        text=taxonomy_text,
        text_hate_prob=base_signals["text_hate_prob"],
        vision_hate_prob=base_signals["vision_hate_prob"],
        severity=max_severity,
    )
    gated = _apply_base_model_gating(
        taxonomy_probs,
        thresholds,
        taxonomy_text,
        rare_result,
        base_signals,
        low_hate_threshold=float(args.low_hate_threshold),
        support_hate_threshold=float(args.support_hate_threshold),
        boost_margin=float(args.base_boost_margin),
    )
    prob_hate = _final_hate_probability(base_signals, taxonomy_probs)
    final = {
        "is_hate": bool(prob_hate >= float(args.hate_threshold)),
        "prob_hate": prob_hate,
        "severity": max_severity,
        "taxonomy_labels": gated["taxonomy_final_labels"],
    }
    output = {
        "input": {
            "video": args.video or "",
            "frame_dir": args.frame_dir or "",
        },
        "metadata": _metadata_dict(metadata, generated=generated_metadata, manual=manual_metadata),
        "base_model_signals": base_signals,
        "vision": vision,
        "text": text,
        "taxonomy": {
            "available": taxonomy_available,
            "taxonomy_raw_labels": gated["taxonomy_raw_labels"],
            "taxonomy_final_labels": gated["taxonomy_final_labels"],
            "suppressed_labels": gated["suppressed_labels"],
            "candidate_labels": gated["candidate_labels"],
            "probs": taxonomy_probs,
            "text_probs": text_taxonomy_probs,
            "video_probs": video_taxonomy_probs,
            "thresholds_used": gated["thresholds_used"],
            "source": taxonomy_source,
            "model_sources": {
                "text_transformer": bool(text_taxonomy_available),
                "video_taxonomy": bool(video_taxonomy_available),
            },
        },
        "final": final,
        "evidence": evidence,
    }
    if not args.no_feedback_log:
        _append_feedback_log(output, args.feedback_log)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    default_transformer_taxonomy_model = (
        "checkpoints/transformer_multilabel_classifier_feedback/best"
        if Path("checkpoints/transformer_multilabel_classifier_feedback/best").exists()
        else "checkpoints/transformer_multilabel_classifier/best"
    )
    default_transformer_thresholds = (
        "reports/transformer_multilabel_feedback_calibration/thresholds.json"
        if Path("reports/transformer_multilabel_feedback_calibration/thresholds.json").exists()
        else "reports/transformer_multilabel_calibration/thresholds.json"
    )
    parser = argparse.ArgumentParser(
        description="Run the final metadata-guided moderation system.",
        epilog=(
            "Examples:\n"
            "  python inference_moderation_system.py --scene \"A man gives an aggressive speech targeting a group\" "
            "--action \"verbal abuse\" --subcategories \"hate speech, discrimination\"\n\n"
            "  python inference_moderation_system.py --video path\\to\\video.mp4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", default="")
    parser.add_argument("--frame-dir", default="")
    parser.add_argument("--generate-metadata", action="store_true")
    parser.add_argument("--metadata-frames", type=int, default=8)
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--disable-asr", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--disable-vlm", action="store_true")
    parser.add_argument("--enable-rule-metadata", action="store_true")
    parser.add_argument("--vlm-model-name", default="Salesforce/blip-image-captioning-base")

    parser.add_argument("--scene", default="")
    parser.add_argument("--action", default="")
    parser.add_argument("--action-clean", default="")
    parser.add_argument("--subcategories", default="")
    parser.add_argument("--parent-labels", default="")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--ocr-text", default="")

    parser.add_argument("--taxonomy-model-type", choices=["transformer", "tfidf"], default="transformer")
    parser.add_argument("--transformer-taxonomy-model", default=default_transformer_taxonomy_model)
    parser.add_argument("--taxonomy-classifier", default="checkpoints/text_multilabel_classifier/best_model.joblib")
    parser.add_argument("--taxonomy-thresholds", default=default_transformer_thresholds)
    parser.add_argument("--taxonomy-max-length", type=int, default=256)
    parser.add_argument("--use-video-taxonomy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-taxonomy-checkpoint", default="checkpoints/best_taxonomy_head.pt")
    parser.add_argument("--video-taxonomy-thresholds", default="reports/taxonomy_head/thresholds.json")
    parser.add_argument("--vision-checkpoint", default="checkpoints/best_frame_text_vision_relabel_all_splits.pt")
    parser.add_argument("--vision-model-name", default=None)
    parser.add_argument("--vision-num-frames", type=int, default=64)
    parser.add_argument("--manifest", default=str(cfg.manifest_path))
    parser.add_argument("--teacher-cache", default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--text-model", default=str(cfg.text_model_path))
    parser.add_argument("--text-max-length", type=int, default=int(cfg.text_max_length))
    parser.add_argument("--text-hate-prob", type=float, default=None)
    parser.add_argument("--vision-hate-prob", type=float, default=None)
    parser.add_argument("--text-severity", type=int, default=None)
    parser.add_argument("--vision-severity", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--show-model-logs", action="store_true")

    parser.add_argument("--hate-threshold", type=float, default=0.5)
    parser.add_argument("--low-hate-threshold", type=float, default=0.35)
    parser.add_argument("--support-hate-threshold", type=float, default=0.70)
    parser.add_argument("--base-boost-margin", type=float, default=0.10)

    parser.add_argument("--feedback-log", default="reports/runtime_feedback/inference_logs.csv")
    parser.add_argument("--no-feedback-log", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()

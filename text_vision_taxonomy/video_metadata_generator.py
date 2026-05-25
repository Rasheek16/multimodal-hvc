from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.data_utils import FRAME_EXTENSIONS, list_frame_paths, normalize_transcript
from text_vision.video_description import (
    EvidenceItem,
    extract_ocr_evidence,
    generate_visual_description,
    sample_description_frame_paths,
)


@dataclass
class VideoMetadata:
    scene: str = ""
    action: str = ""
    action_clean: str = ""
    subcategories: str = ""
    parent_labels: str = ""
    transcript: str = ""
    ocr_text: str = ""
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


HATE_TERMS = [
    "hate",
    "hateful",
    "slur",
    "racist",
    "racism",
    "bigot",
    "bigotry",
    "dehumanize",
    "dehumanizing",
]
DISCRIMINATION_TERMS = [
    "discrimination",
    "discriminate",
    "religion",
    "race",
    "ethnic",
    "ethnicity",
    "caste",
    "gender",
    "minority",
    "immigrant",
]
CONTEXTUAL_HATE_TERMS = [
    "extremist",
    "extremism",
    "propaganda",
    "supremacist",
    "supremacy",
    "radical",
    "genocide",
]
THREAT_TERMS = [
    "threat",
    "threaten",
    "kill",
    "murder",
    "hurt",
    "attack",
    "bomb",
    "shoot",
    "stab",
    "beat",
    "destroy",
    "hang",
]
VIOLENCE_TERMS = [
    "violence",
    "violent",
    "fight",
    "fighting",
    "attack",
    "assault",
    "blood",
    "weapon",
    "gun",
    "knife",
    "shoot",
    "stab",
]
FEAR_TERMS = ["fear", "afraid", "scared", "panic", "terrified", "terror", "intimidate"]
SEXUAL_TERMS = ["sexual", "sex", "nude", "nudity", "porn", "explicit"]
ILLEGAL_TERMS = [
    "illegal",
    "weapon",
    "gun",
    "knife",
    "drug",
    "crime",
    "criminal",
    "abuse",
    "exploit",
    "fraud",
    "scam",
    "hack",
    "stolen",
    "counterfeit",
    "trafficking",
]
ONLINE_HARM_TERMS = [
    "online",
    "social media",
    "chat",
    "comment",
    "comments",
    "meme",
    "post",
    "tweet",
    "message",
    "harass",
    "harassment",
    "dox",
    "doxx",
    "leak",
    "private information",
    "cyberbully",
]

LABEL_TERMS: Mapping[str, Sequence[str]] = {
    "hate_speech": HATE_TERMS,
    "discrimination": DISCRIMINATION_TERMS,
    "contextual_hate": CONTEXTUAL_HATE_TERMS,
    "threat": THREAT_TERMS,
    "violence": VIOLENCE_TERMS,
    "fear": FEAR_TERMS,
    "sexual": SEXUAL_TERMS,
    "illegal": ILLEGAL_TERMS,
    "online_harm": ONLINE_HARM_TERMS,
}

PARENT_BY_LABEL = {
    "hate_speech": "hate",
    "discrimination": "hate",
    "contextual_hate": "hate",
    "threat": "safety",
    "violence": "safety",
    "fear": "safety",
    "sexual": "sexual",
    "illegal": "illegal",
    "online_harm": "online_harm",
}


def evidence(source: str, text: str = "", **extra: Any) -> dict[str, Any]:
    item = {"source": source, "text": normalize_transcript(text)}
    item.update({key: value for key, value in extra.items() if value not in (None, "")})
    return {key: value for key, value in item.items() if value not in (None, "")}


def _linspace_indices(count: int, target: int) -> list[int]:
    count = int(count)
    target = max(1, int(target))
    if count <= 0:
        return []
    if count <= target:
        return list(range(count))
    if target == 1:
        return [count // 2]
    return [int(round(value)) for value in _linspace(0, count - 1, target)]


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + step * index for index in range(count)]


def _safe_stem(path: str | Path) -> str:
    base = Path(path).stem or "video"
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base).strip("_") or "video"
    digest = hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _extract_video_frames(
    video_path: str | Path,
    num_frames: int,
    output_root: str | Path = "reports/runtime_metadata/frames",
) -> tuple[list[Path], list[dict]]:
    events: list[dict] = []
    path = Path(video_path)
    if not path.exists():
        return [], [evidence("frame_sampler", f"video file not found: {path}", status="missing")]
    try:
        import cv2
    except Exception as exc:
        return [], [
            evidence(
                "frame_sampler",
                f"OpenCV unavailable; cannot sample raw video frames ({type(exc).__name__}: {exc})",
                status="unavailable",
            )
        ]

    output_dir = Path(output_root) / _safe_stem(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], [evidence("frame_sampler", f"could not open video: {path}", status="failed")]

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sampled: list[Path] = []
    try:
        if frame_count > 0:
            for index in _linspace_indices(frame_count, num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                output = output_dir / f"frame_{int(index):06d}.jpg"
                cv2.imwrite(str(output), frame)
                sampled.append(output)
        else:
            stride = 30
            index = 0
            kept = 0
            while kept < int(num_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if index % stride == 0:
                    output = output_dir / f"frame_{int(index):06d}.jpg"
                    cv2.imwrite(str(output), frame)
                    sampled.append(output)
                    kept += 1
                index += 1
    finally:
        cap.release()

    events.append(
        evidence(
            "frame_sampler",
            f"sampled {len(sampled)} frame(s) from raw video",
            status="ok" if sampled else "empty",
            video_path=str(path),
            output_dir=str(output_dir),
            frame_count=frame_count,
        )
    )
    return sampled, events


def sample_frames(
    video_path: str | Path | None = None,
    frame_dir: str | Path | None = None,
    num_frames: int = 8,
) -> list[Path]:
    if frame_dir:
        return sample_description_frame_paths(frame_dir, num_frames=num_frames)
    if video_path:
        frames, _ = _extract_video_frames(video_path, num_frames=num_frames)
        return frames
    return []


def run_asr(video_path: str | Path | None) -> str:
    if not video_path:
        return ""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return ""
    path = Path(video_path)
    if not path.exists():
        return ""
    try:
        model = WhisperModel("small", device="auto", compute_type="int8")
        segments, _info = model.transcribe(str(path), beam_size=5, vad_filter=True)
        return normalize_transcript(" ".join(segment.text for segment in segments))
    except Exception:
        return ""


def run_ocr(frame_paths: Iterable[str | Path]) -> str:
    items = extract_ocr_evidence(frame_paths)
    return normalize_transcript(" ".join(item.text for item in items))


def _visual_description(
    frame_paths: Sequence[str | Path],
    transcript: str = "",
    ocr_text: str = "",
    enable_vlm: bool = False,
    vlm_model_name: str = "Salesforce/blip-image-captioning-base",
) -> EvidenceItem:
    return generate_visual_description(
        frame_paths,
        transcript=transcript,
        ocr_text=ocr_text,
        enable_vlm=enable_vlm,
        model_name=vlm_model_name,
    )


def generate_scene_description(
    frame_paths: Sequence[str | Path],
    transcript: str = "",
    ocr_text: str = "",
    enable_vlm: bool = False,
    vlm_model_name: str = "Salesforce/blip-image-captioning-base",
) -> str:
    vlm_item = _visual_description(
        frame_paths,
        transcript=transcript,
        ocr_text=ocr_text,
        enable_vlm=enable_vlm,
        vlm_model_name=vlm_model_name,
    )
    visual = normalize_transcript(vlm_item.text)
    if visual:
        return visual
    transcript_text = normalize_transcript(transcript)
    ocr = normalize_transcript(ocr_text)
    if transcript_text and ocr:
        return f"Transcript and on-screen text are available. Transcript: {transcript_text} OCR: {ocr}"
    if transcript_text:
        return f"Transcript content is available: {transcript_text}"
    if ocr:
        return f"On-screen text is available: {ocr}"
    return ""


def _clean_action_text(value: str) -> str:
    text = normalize_transcript(value).lower()
    text = re.sub(r"[^a-z0-9\s,_-]+", " ", text)
    return " ".join(text.split())


def generate_action_description(
    frame_paths: Sequence[str | Path],
    transcript: str = "",
    ocr_text: str = "",
    scene: str = "",
) -> tuple[str, str]:
    del frame_paths
    evidence_text = " ".join(
        text
        for text in [
            normalize_transcript(scene),
            normalize_transcript(transcript),
            normalize_transcript(ocr_text),
        ]
        if text
    ).lower()
    if not evidence_text:
        return "", ""
    action_hints = [
        ("verbal abuse", ["abuse", "slur", "insult", "harass"]),
        ("threatening speech", ["threat", "kill", "hurt", "attack", "shoot", "stab"]),
        ("violent action", ["fight", "attack", "assault", "beat", "shoot", "stab"]),
        ("sexual content", ["sexual", "nude", "nudity", "porn"]),
        ("online harassment", ["comment", "post", "message", "tweet", "harass", "dox"]),
    ]
    for action, terms in action_hints:
        if _contains_any(evidence_text, terms):
            return action, _clean_action_text(action)
    if transcript or ocr_text:
        action = "spoken or written content is present"
        return action, _clean_action_text(action)
    return "", ""


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    clean = normalize_transcript(text).lower()
    for term in terms:
        escaped = re.escape(term.lower())
        if re.search(rf"(?<!\w){escaped}(?!\w)", clean):
            return True
    return False


def derive_subcategories_and_parent_labels(
    scene: str,
    action: str,
    transcript: str,
    ocr_text: str,
) -> tuple[str, str]:
    text = " ".join(
        value
        for value in [
            normalize_transcript(scene),
            normalize_transcript(action),
            normalize_transcript(transcript),
            normalize_transcript(ocr_text),
        ]
        if value
    )
    if not text:
        return "", ""
    labels = [label for label, terms in LABEL_TERMS.items() if _contains_any(text, terms)]
    parents = sorted({PARENT_BY_LABEL[label] for label in labels if label in PARENT_BY_LABEL})
    return ", ".join(labels), ", ".join(parents)


def _metadata_available_fields(metadata: VideoMetadata) -> dict[str, bool]:
    return {
        "scene": bool(metadata.scene),
        "action": bool(metadata.action),
        "action_clean": bool(metadata.action_clean),
        "subcategories": bool(metadata.subcategories),
        "parent_labels": bool(metadata.parent_labels),
        "transcript": bool(metadata.transcript),
        "ocr_text": bool(metadata.ocr_text),
    }


def metadata_runtime_status(metadata: VideoMetadata) -> dict[str, Any]:
    available_fields = _metadata_available_fields(metadata)
    return {
        "metadata_generation_available": any(available_fields.values()),
        "available_fields": available_fields,
    }


def build_video_metadata(
    video_path: str | Path | None = None,
    frame_dir: str | Path | None = None,
    transcript: str = "",
    num_frames: int = 8,
    enable_asr: bool = False,
    enable_ocr: bool = True,
    enable_vlm: bool = False,
    enable_rule_metadata: bool = False,
    vlm_model_name: str = "Salesforce/blip-image-captioning-base",
) -> VideoMetadata:
    events: list[dict] = []
    frames: list[Path] = []
    if frame_dir:
        frames = sample_description_frame_paths(frame_dir, num_frames=num_frames)
        events.append(
            evidence(
                "frame_sampler",
                f"sampled {len(frames)} frame(s) from frame directory",
                status="ok" if frames else "empty",
                frame_dir=str(frame_dir),
            )
        )
    elif video_path:
        frames, sample_events = _extract_video_frames(video_path, num_frames=num_frames)
        events.extend(sample_events)
    else:
        events.append(evidence("metadata", "no video_path or frame_dir supplied", status="missing_input"))

    transcript_text = normalize_transcript(transcript)
    if transcript_text:
        events.append(evidence("transcript", transcript_text, status="provided"))
    elif enable_asr and video_path:
        asr_text = run_asr(video_path)
        transcript_text = normalize_transcript(asr_text)
        events.append(
            evidence(
                "asr",
                transcript_text or "ASR returned no transcript",
                status="ok" if transcript_text else "empty_or_unavailable",
            )
        )
    else:
        events.append(evidence("asr", "ASR not enabled or no video path supplied", status="skipped"))

    ocr_text = ""
    if enable_ocr and frames:
        ocr_items = extract_ocr_evidence(frames)
        ocr_text = normalize_transcript(" ".join(item.text for item in ocr_items))
        events.extend(item.to_dict() for item in ocr_items)
        if not ocr_text:
            events.append(evidence("ocr", "OCR returned no text or OCR is unavailable", status="empty_or_unavailable"))
    elif enable_ocr:
        events.append(evidence("ocr", "OCR skipped because no frames were available", status="skipped"))
    else:
        events.append(evidence("ocr", "OCR disabled", status="skipped"))

    vlm_item = _visual_description(
        frames,
        transcript=transcript_text,
        ocr_text=ocr_text,
        enable_vlm=enable_vlm,
        vlm_model_name=vlm_model_name,
    )
    if normalize_transcript(vlm_item.text):
        events.append(vlm_item.to_dict())
    elif enable_vlm:
        events.append(evidence("vlm", "VLM captioning returned no description", status="empty_or_unavailable"))
    else:
        events.append(evidence("vlm", "VLM disabled", status="skipped"))

    scene = normalize_transcript(vlm_item.text) or generate_scene_description(
        frames,
        transcript=transcript_text,
        ocr_text=ocr_text,
        enable_vlm=False,
        vlm_model_name=vlm_model_name,
    )
    if enable_rule_metadata:
        action, action_clean = generate_action_description(
            frames,
            transcript=transcript_text,
            ocr_text=ocr_text,
            scene=scene,
        )
        subcategories, parent_labels = derive_subcategories_and_parent_labels(scene, action, transcript_text, ocr_text)
        events.append(evidence("metadata_rules", "rule-derived action, subcategories, and parent labels enabled", status="enabled"))
    else:
        action, action_clean = "", ""
        subcategories, parent_labels = "", ""
        events.append(
            evidence(
                "metadata_rules",
                "rule-derived action, subcategories, and parent labels disabled; taxonomy labels come from the trained model",
                status="disabled",
            )
        )
    metadata = VideoMetadata(
        scene=scene,
        action=action,
        action_clean=action_clean,
        subcategories=subcategories,
        parent_labels=parent_labels,
        transcript=transcript_text,
        ocr_text=ocr_text,
        evidence=events,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conservative metadata for taxonomy inference.")
    parser.add_argument("--video", default=None)
    parser.add_argument("--frame-dir", default=None)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--enable-rule-metadata", action="store_true")
    parser.add_argument("--vlm-model-name", default="Salesforce/blip-image-captioning-base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_video_metadata(
        video_path=args.video,
        frame_dir=args.frame_dir,
        transcript=args.transcript,
        num_frames=args.num_frames,
        enable_asr=args.enable_asr,
        enable_ocr=not args.disable_ocr,
        enable_vlm=args.enable_vlm,
        enable_rule_metadata=args.enable_rule_metadata,
        vlm_model_name=args.vlm_model_name,
    )
    output = metadata.to_dict()
    output.update(metadata_runtime_status(metadata))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

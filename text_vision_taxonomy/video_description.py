from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.data_utils import FRAME_EXTENSIONS, list_frame_paths, normalize_transcript
from text_vision.segment_frame_utils import sample_segment_frames_from_dir


@dataclass
class EvidenceItem:
    source: str
    text: str
    confidence: Optional[float] = None
    frame_path: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "")}


@dataclass
class VideoDescriptionResult:
    transcript: str = ""
    ocr_text: str = ""
    visual_description: str = ""
    text_for_classifier: str = ""
    evidence: List[EvidenceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "ocr_text": self.ocr_text,
            "visual_description": self.visual_description,
            "text_for_classifier": self.text_for_classifier,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
        }


def _warn(message: str, warnings_out: Optional[List[str]] = None) -> None:
    message = str(message)
    if len(message) > 600:
        message = f"{message[:600]}..."
    if warnings_out is not None:
        warnings_out.append(message)
    warnings.warn(message, RuntimeWarning)


def sample_description_frame_paths(
    frame_dir: str | Path,
    num_frames: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    fps: float | None = None,
    frame_dir_is_segment: bool = False,
) -> List[Path]:
    path = Path(frame_dir)
    if start_sec is not None and end_sec is not None and not frame_dir_is_segment:
        return sample_segment_frames_from_dir(path, start_sec=start_sec, end_sec=end_sec, num_frames=num_frames, fps=fps)
    frames = list_frame_paths(path)
    if not frames:
        return []
    if len(frames) <= int(num_frames):
        return frames
    positions = [round(value) for value in _linspace(0, len(frames) - 1, int(num_frames))]
    return [frames[int(index)] for index in positions]


def _linspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + step * index for index in range(count)]


def extract_ocr_evidence(frame_paths: Iterable[str | Path]) -> List[EvidenceItem]:
    paths = [Path(path) for path in frame_paths]
    if not paths:
        return []
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:
        _warn(f"OCR unavailable; install pytesseract/Pillow OCR support to enable it ({exc})")
        return []

    _configure_tesseract(pytesseract)
    evidence: List[EvidenceItem] = []
    for path in paths:
        try:
            if path.suffix.lower() not in FRAME_EXTENSIONS:
                continue
            with Image.open(path) as image:
                text = normalize_transcript(pytesseract.image_to_string(image.convert("RGB")))
            if text:
                evidence.append(EvidenceItem(source="ocr", text=text, frame_path=str(path)))
        except Exception as exc:
            _warn(f"OCR failed for {path}: {exc}")
    return evidence


def _configure_tesseract(pytesseract_module) -> None:
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract_module.pytesseract.tesseract_cmd = str(candidate)
            return


def _caption_text_from_output(output) -> str:
    if isinstance(output, list):
        texts = []
        for item in output:
            text = _caption_text_from_output(item)
            if text:
                texts.append(text)
        return normalize_transcript(" ".join(texts))
    if isinstance(output, dict):
        for key in ["generated_text", "caption", "text"]:
            text = normalize_transcript(output.get(key, ""))
            if text:
                return text
    return normalize_transcript(output)


def _pipeline_captioner(model_name: str, device: int | str | None):
    from transformers import pipeline

    errors: list[str] = []
    for task in ["image-to-text", "image-text-to-text"]:
        try:
            return pipeline(task, model=model_name, device=device), task
        except Exception as exc:
            errors.append(f"{task}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _direct_blip_caption(frame_paths: Sequence[str | Path], model_name: str, device: int | str | None) -> str:
    from PIL import Image
    import torch
    from transformers import AutoProcessor, BlipForConditionalGeneration

    torch_device = torch.device(str(device) if device is not None and str(device) not in {"-1", "cpu"} else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    model.eval()
    captions: List[str] = []
    for path in frame_paths:
        with Image.open(path) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=30)
        text = normalize_transcript(processor.decode(generated[0], skip_special_tokens=True))
        if text:
            captions.append(text)
    return normalize_transcript(" ".join(captions))


def generate_visual_description(
    frame_paths: Sequence[str | Path],
    transcript: str = "",
    ocr_text: str = "",
    enable_vlm: bool = False,
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: int | str | None = None,
) -> EvidenceItem:
    if not frame_paths:
        return EvidenceItem(source="vlm", text="", confidence=None)
    if not enable_vlm:
        del transcript, ocr_text
        return EvidenceItem(source="vlm", text="", confidence=None, frame_path=str(frame_paths[0]))

    try:
        captioner, task = _pipeline_captioner(model_name, device)
    except Exception as exc:
        _warn(f"VLM pipeline unavailable for {model_name}; trying direct BLIP fallback ({type(exc).__name__}: {exc})")
        try:
            text = _direct_blip_caption(frame_paths, model_name, device)
            return EvidenceItem(source="vlm", text=text, confidence=None, frame_path=str(frame_paths[0]))
        except Exception as direct_exc:
            _warn(f"VLM direct BLIP captioning failed for {model_name}: {type(direct_exc).__name__}: {direct_exc}")
            return EvidenceItem(source="vlm", text="", confidence=None, frame_path=str(frame_paths[0]))

    try:
        captions: List[str] = []
        for path in frame_paths:
            try:
                output = captioner(str(path))
            except TypeError:
                output = captioner(images=str(path), text="Describe this image.")
            text = _caption_text_from_output(output)
            if text:
                captions.append(text)
        return EvidenceItem(
            source="vlm",
            text=normalize_transcript(" ".join(captions)),
            confidence=None,
            frame_path=str(frame_paths[0]),
        )
    except Exception as exc:
        try:
            text = _direct_blip_caption(frame_paths, model_name, device)
            return EvidenceItem(source="vlm", text=text, confidence=None, frame_path=str(frame_paths[0]))
        except Exception as direct_exc:
            _warn(
                f"VLM captioning failed for {model_name} with pipeline task {task}; "
                f"direct BLIP failed too ({type(direct_exc).__name__}: {direct_exc})"
            )
        return EvidenceItem(source="vlm", text="", confidence=None, frame_path=str(frame_paths[0]))


def _combine_classifier_text(transcript: str, ocr_text: str, visual_description: str) -> str:
    parts = []
    for label, text in [
        ("TRANSCRIPT", transcript),
        ("OCR", ocr_text),
        ("VISUAL_DESCRIPTION", visual_description),
    ]:
        normalized = normalize_transcript(text)
        if normalized:
            parts.append(f"[{label}]\n{normalized}")
    return "\n\n".join(parts)


def build_video_description(
    frame_dir: str | Path,
    video_path: str | Path | None = None,
    transcript: str = "",
    num_frames: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
    fps: float | None = None,
    frame_dir_is_segment: bool = False,
    enable_vlm: bool = False,
    vlm_model_name: str = "Salesforce/blip-image-captioning-base",
) -> VideoDescriptionResult:
    del video_path
    warnings_out: List[str] = []
    frame_paths = sample_description_frame_paths(
        frame_dir,
        num_frames=num_frames,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        frame_dir_is_segment=frame_dir_is_segment,
    )
    if not frame_paths:
        _warn(f"No raw frames available for description: {frame_dir}", warnings_out)

    transcript_text = normalize_transcript(transcript)
    evidence: List[EvidenceItem] = []
    if transcript_text:
        evidence.append(EvidenceItem(source="asr", text=transcript_text, confidence=1.0))

    ocr_items = extract_ocr_evidence(frame_paths)
    evidence.extend(ocr_items)
    ocr_text = normalize_transcript(" ".join(item.text for item in ocr_items))

    vlm_item = generate_visual_description(
        frame_paths,
        transcript=transcript_text,
        ocr_text=ocr_text,
        enable_vlm=enable_vlm,
        model_name=vlm_model_name,
    )
    visual_description = normalize_transcript(vlm_item.text)
    if visual_description:
        evidence.append(vlm_item)

    return VideoDescriptionResult(
        transcript=transcript_text,
        ocr_text=ocr_text,
        visual_description=visual_description,
        text_for_classifier=_combine_classifier_text(transcript_text, ocr_text, visual_description),
        evidence=evidence,
        warnings=warnings_out,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR/VLM-ready text evidence for a frame directory.")
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--frame-dir-is-segment", action="store_true")
    parser.add_argument("--enable-vlm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_video_description(
        args.frame_dir,
        transcript=args.transcript,
        num_frames=args.num_frames,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        fps=args.fps,
        frame_dir_is_segment=args.frame_dir_is_segment,
        enable_vlm=args.enable_vlm,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

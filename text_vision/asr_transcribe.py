from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import as_path, get_config
from text_vision.data_utils import normalize_transcript


ASR_COLUMNS = [
    "source_video_id",
    "video_path",
    "transcript",
    "confidence",
    "language",
    "source",
    "status",
    "error",
]


def write_asr_rows(rows: List[Dict], output_path: Path) -> None:
    frame = pd.DataFrame(rows)
    for column in ASR_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame[ASR_COLUMNS].to_csv(output_path, index=False)


def transcribe_missing_videos(
    manifest_path: Path,
    output_path: Path,
    model_size: str = "small",
    device: str = "auto",
    limit: int | None = None,
    language: str | None = None,
    overwrite: bool = False,
    retry_failed: bool = False,
) -> pd.DataFrame:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "ASR requires faster-whisper. Install text_vision/requirements.txt or run without ASR."
        ) from exc

    manifest = pd.read_csv(manifest_path)
    videos = manifest.drop_duplicates("source_video_id").copy()
    videos = videos[~videos["has_text"].astype(bool)].copy()
    videos = videos[videos.get("video_path", "").fillna("").map(lambda value: Path(str(value)).exists())]
    if limit is not None:
        videos = videos.head(int(limit)).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    if output_path.exists() and not overwrite:
        existing = pd.read_csv(output_path)
        if "status" not in existing.columns:
            transcript_text = existing.get("transcript", pd.Series([""] * len(existing))).fillna("").map(normalize_transcript)
            existing["status"] = np.where(transcript_text != "", "ok", "empty")
        if "error" not in existing.columns:
            existing["error"] = ""
        rows = existing.to_dict("records")
        write_asr_rows(rows, output_path)
        processed = existing.copy()
        if retry_failed and "status" in processed.columns:
            processed = processed[processed["status"].fillna("") != "failed"]
        processed_ids = set(processed["source_video_id"].astype(str))
        videos = videos[~videos["source_video_id"].astype(str).isin(processed_ids)].copy()
        print(f"resume: loaded {len(existing):,} existing ASR rows; remaining videos={len(videos):,}")

    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    for record in tqdm(videos.to_dict("records"), desc="asr"):
        video_path = str(record["video_path"])
        row = {
            "source_video_id": str(record["source_video_id"]),
            "video_path": video_path,
            "transcript": "",
            "confidence": 0.0,
            "language": language or "",
            "source": "asr",
            "status": "failed",
            "error": "",
        }
        try:
            segments, info = model.transcribe(video_path, beam_size=5, vad_filter=True, language=language)
            segment_list = list(segments)
            text = normalize_transcript(" ".join(segment.text for segment in segment_list))
            if segment_list:
                confidences = []
                for segment in segment_list:
                    avg_logprob = float(getattr(segment, "avg_logprob", -4.0))
                    no_speech = float(getattr(segment, "no_speech_prob", 0.0))
                    confidences.append(max(0.0, min(1.0, math.exp(avg_logprob) * (1.0 - no_speech))))
                confidence = float(np.mean(confidences))
            else:
                confidence = 0.0
            row.update(
                {
                    "transcript": text,
                    "confidence": confidence,
                    "language": getattr(info, "language", language or ""),
                    "status": "ok" if text else "empty",
                    "error": "",
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            tqdm.write(f"ASR failed for {row['source_video_id']}: {row['error']}")

        rows.append(row)
        write_asr_rows(rows, output_path)

    result = pd.DataFrame(rows)
    write_asr_rows(rows, output_path)
    return result


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Generate ASR transcripts for manifest rows without text.")
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--output", type=str, default=str(cfg.asr_cache_path))
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    cfg = get_config()
    args = parse_args()
    result = transcribe_missing_videos(
        manifest_path=as_path(args.manifest, cfg.manifest_path),
        output_path=as_path(args.output, cfg.asr_cache_path),
        model_size=args.model_size,
        device=args.device,
        limit=args.limit,
        language=args.language,
        overwrite=args.overwrite,
        retry_failed=args.retry_failed,
    )
    status_counts = result.get("status", pd.Series(dtype=str)).fillna("").value_counts().to_dict()
    print(f"ASR rows saved: {len(result):,}")
    print(f"status counts: {status_counts}")
    print(f"output: {as_path(args.output, cfg.asr_cache_path)}")


if __name__ == "__main__":
    main()

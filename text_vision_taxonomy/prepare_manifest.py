from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import as_path, get_config
from text_vision.data_utils import normalize_transcript


def load_asr_cache(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["source_video_id", "asr_transcription", "asr_confidence"])
    asr = pd.read_csv(path)
    rename = {}
    if "transcription" in asr.columns:
        rename["transcription"] = "asr_transcription"
    if "transcript" in asr.columns:
        rename["transcript"] = "asr_transcription"
    if "confidence" in asr.columns:
        rename["confidence"] = "asr_confidence"
    asr = asr.rename(columns=rename)
    keep = [column for column in ["source_video_id", "asr_transcription", "asr_confidence"] if column in asr.columns]
    if "source_video_id" not in keep or "asr_transcription" not in keep:
        raise ValueError(f"ASR cache must contain source_video_id and transcript text columns: {path}")
    asr = asr[keep].copy()
    if "asr_confidence" not in asr.columns:
        asr["asr_confidence"] = 0.75
    asr["asr_transcription"] = asr["asr_transcription"].map(normalize_transcript)
    asr["asr_confidence"] = pd.to_numeric(asr["asr_confidence"], errors="coerce").fillna(0.75).clip(0.0, 1.0)
    asr = asr[asr["asr_transcription"] != ""]
    return asr.drop_duplicates("source_video_id", keep="last")


def build_manifest(
    splits_path: Path,
    materialized_clip_index_path: Path,
    output_path: Path,
    asr_cache_path: Optional[Path] = None,
    validate_paths: bool = True,
) -> pd.DataFrame:
    splits = pd.read_csv(splits_path)
    clips = pd.read_csv(materialized_clip_index_path)

    required_clip_columns = {"source_video_id", "label", "split", "clip_idx", "clip_path"}
    missing = required_clip_columns - set(clips.columns)
    if missing:
        raise ValueError(f"materialized clip index is missing required columns: {sorted(missing)}")

    video_columns = [
        "source_video_id",
        "video_file_name",
        "label",
        "label_source",
        "source",
        "transcription",
        "video_path",
        "has_video_file",
        "frame_dir",
        "num_frames",
    ]
    video_columns = [column for column in video_columns if column in splits.columns]
    videos = splits[video_columns].copy().drop_duplicates("source_video_id", keep="first")
    videos["human_transcription"] = videos.get("transcription", "").map(normalize_transcript)

    asr = load_asr_cache(asr_cache_path)
    if not asr.empty:
        videos = videos.merge(asr, on="source_video_id", how="left")
    else:
        videos["asr_transcription"] = ""
        videos["asr_confidence"] = 0.0

    videos["asr_transcription"] = videos["asr_transcription"].fillna("").map(normalize_transcript)
    videos["asr_confidence"] = pd.to_numeric(videos["asr_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    human_mask = videos["human_transcription"] != ""
    asr_mask = (~human_mask) & (videos["asr_transcription"] != "")
    videos["transcription"] = np.where(
        human_mask,
        videos["human_transcription"],
        np.where(asr_mask, videos["asr_transcription"], ""),
    )
    videos["transcript_source"] = np.where(human_mask, "human", np.where(asr_mask, "asr", "missing"))
    videos["transcript_confidence"] = np.where(human_mask, 1.0, np.where(asr_mask, videos["asr_confidence"], 0.0))
    videos["has_text"] = videos["transcription"] != ""

    join_columns = [
        "source_video_id",
        "video_file_name",
        "label_source",
        "source",
        "transcription",
        "video_path",
        "has_video_file",
        "frame_dir",
        "num_frames",
        "has_text",
        "transcript_source",
        "transcript_confidence",
    ]
    join_columns = [column for column in join_columns if column in videos.columns]
    manifest = clips.merge(videos[join_columns], on="source_video_id", how="left", suffixes=("", "_video"))
    manifest["label"] = manifest["label"].astype(int)
    manifest["split"] = manifest["split"].astype(str)
    manifest["transcription"] = manifest["transcription"].fillna("").map(normalize_transcript)
    manifest["has_text"] = manifest["transcription"] != ""
    manifest["transcript_source"] = manifest["transcript_source"].fillna("missing")
    manifest["transcript_confidence"] = pd.to_numeric(
        manifest["transcript_confidence"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    manifest["fallback_severity"] = np.where(manifest["label"].astype(int) == 0, 0, 2)
    manifest["clip_path_exists"] = manifest["clip_path"].map(lambda value: Path(str(value)).exists())
    if "video_path" in manifest.columns:
        manifest["video_path_exists"] = manifest["video_path"].fillna("").map(lambda value: Path(str(value)).exists())
    else:
        manifest["video_path_exists"] = False
    if validate_paths:
        manifest = manifest[manifest["clip_path_exists"]].copy()

    sort_columns = [column for column in ["split", "source_video_id", "clip_idx"] if column in manifest.columns]
    manifest = manifest.sort_values(sort_columns).reset_index(drop=True)
    manifest.insert(0, "sample_id", np.arange(len(manifest), dtype=int))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def print_summary(manifest: pd.DataFrame, output_path: Path) -> None:
    videos = manifest.drop_duplicates("source_video_id")
    print(f"saved manifest: {output_path}")
    print(f"clip rows: {len(manifest):,}")
    print(f"video rows: {len(videos):,}")
    print("split clip counts:")
    print(manifest["split"].value_counts().sort_index().to_string())
    print("transcript source video counts:")
    print(videos["transcript_source"].value_counts(dropna=False).sort_index().to_string())
    print(f"text coverage by video: {int(videos['has_text'].sum()):,} / {len(videos):,}")
    print(f"missing clip paths after filtering: {int((~manifest['clip_path_exists']).sum()):,}")


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Build the text-guided vision multimodal manifest.")
    parser.add_argument("--splits", type=str, default=str(cfg.splits_path))
    parser.add_argument("--clip-index", type=str, default=str(cfg.materialized_clip_index_path))
    parser.add_argument("--output", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--asr-cache", type=str, default=str(cfg.asr_cache_path))
    parser.add_argument("--no-validate-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    cfg = get_config()
    args = parse_args()
    asr_cache = as_path(args.asr_cache, cfg.asr_cache_path)
    if not asr_cache.exists():
        asr_cache = None
    manifest = build_manifest(
        splits_path=as_path(args.splits, cfg.splits_path),
        materialized_clip_index_path=as_path(args.clip_index, cfg.materialized_clip_index_path),
        output_path=as_path(args.output, cfg.manifest_path),
        asr_cache_path=asr_cache,
        validate_paths=not args.no_validate_paths,
    )
    print_summary(manifest, as_path(args.output, cfg.manifest_path))


if __name__ == "__main__":
    main()


from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, List


FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def time_to_seconds(value: Any) -> float:
    if value is None:
        raise ValueError("missing time value")
    text = str(value).strip()
    if not text:
        raise ValueError("empty time value")
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    if len(parts) == 2:
        hours = 0.0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"unsupported time value: {value!r}")
    return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)


def _uniform_indices(start: int, stop: int, num_frames: int) -> List[int]:
    start = int(max(0, start))
    stop = int(max(start + 1, stop))
    num_frames = int(max(1, num_frames))
    if stop - start <= 1:
        return [start] * num_frames
    if num_frames == 1:
        return [start]
    step = float((stop - 1) - start) / float(num_frames - 1)
    values = [int(round(start + step * index)) for index in range(num_frames)]
    return [min(max(value, start), stop - 1) for value in values]


def segment_frame_indices(
    num_frames_video: int,
    fps: float | None,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    warn_on_fallback: bool = True,
) -> List[int]:
    total = int(num_frames_video)
    if total <= 0:
        return []
    try:
        start = float(start_sec)
        end = float(end_sec)
        if end <= start:
            raise ValueError("end_sec must be greater than start_sec")
        if fps is None or float(fps) <= 0:
            if warn_on_fallback:
                warnings.warn("fps is unknown; sampling uniformly from available frames", RuntimeWarning, stacklevel=2)
            return _uniform_indices(0, total, num_frames)
        start_idx = int(math.floor(start * float(fps)))
        end_idx = int(math.ceil(end * float(fps)))
        start_idx = max(0, min(total - 1, start_idx))
        end_idx = max(start_idx + 1, min(total, end_idx))
        return _uniform_indices(start_idx, end_idx, num_frames)
    except Exception as exc:
        if warn_on_fallback:
            warnings.warn(f"segment frame mapping failed ({exc}); sampling whole frame_dir", RuntimeWarning, stacklevel=2)
        return _uniform_indices(0, total, num_frames)


def _list_frame_paths(frame_dir: Path) -> List[Path]:
    if not frame_dir.exists() or not frame_dir.is_dir():
        return []
    return sorted(
        child
        for child in frame_dir.iterdir()
        if child.is_file() and child.suffix.lower() in FRAME_EXTENSIONS
    )


def sample_segment_frames_from_dir(
    frame_dir: Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    fps: float | None = None,
    warn_on_fallback: bool = True,
) -> List[Path]:
    paths = _list_frame_paths(Path(frame_dir))
    if not paths:
        return []
    indices = segment_frame_indices(
        num_frames_video=len(paths),
        fps=fps,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        num_frames=int(num_frames),
        warn_on_fallback=warn_on_fallback,
    )
    if not indices:
        return []
    return [paths[int(index)] for index in indices]

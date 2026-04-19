from pathlib import Path
import shutil

import cv2
import numpy as np


def uniform_indices(frame_count, num_frames):
    if frame_count <= 1:
        return [0] * num_frames
    positions = np.linspace(0, frame_count - 1, num=num_frames)
    return np.clip(np.round(positions).astype(int), 0, frame_count - 1).tolist()


def clear_directory(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_frame_outputs(frame, output_dir, output_indices, image_size, jpeg_quality):
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    saved_paths = []

    for output_index in output_indices:
        frame_path = output_dir / f"{output_index:04d}.jpg"
        write_ok = cv2.imwrite(
            str(frame_path),
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if write_ok:
            saved_paths.append(frame_path)

    return saved_paths


def extract_uniform_frames_cpu_task(task):
    source_video_id = str(task["source_video_id"])
    video_path = Path(task["video_path"])
    output_dir = Path(task["frame_dir"])
    num_frames = int(task["num_frames"])
    image_size = int(task["image_size"])
    jpeg_quality = int(task["jpeg_quality"])
    overwrite = bool(task["overwrite"])

    if overwrite:
        clear_directory(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(output_dir.glob("*.jpg"))
        if len(existing) >= num_frames:
            return {
                "ok": True,
                "reason": "ok",
                "saved_frames": len(existing),
                "used_cache": True,
                "source_video_id": source_video_id,
                "video_path": str(video_path),
                "frame_dir": str(output_dir),
            }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "ok": False,
            "reason": "open_failed",
            "source_video_id": source_video_id,
            "video_path": str(video_path),
            "frame_dir": str(output_dir),
        }

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if frame_count <= 0:
        frame_count = 1

    target_indices = uniform_indices(frame_count, num_frames)
    pending_outputs = {}
    for output_index, frame_index in enumerate(target_indices):
        pending_outputs.setdefault(int(frame_index), []).append(output_index)

    saved_paths = []
    current_index = 0
    ok, frame = capture.read()

    if not ok or frame is None:
        capture.release()
        return {
            "ok": False,
            "reason": "first_frame_failed",
            "source_video_id": source_video_id,
            "video_path": str(video_path),
            "frame_dir": str(output_dir),
        }

    output_indices = pending_outputs.pop(0, None)
    if output_indices is not None:
        saved_paths.extend(
            _write_frame_outputs(frame, output_dir, output_indices, image_size, jpeg_quality)
        )

    last_target_index = max(target_indices) if target_indices else 0
    while pending_outputs and current_index < last_target_index:
        ok, frame = capture.read()
        current_index += 1

        if not ok or frame is None:
            break

        output_indices = pending_outputs.pop(current_index, None)
        if output_indices is not None:
            saved_paths.extend(
                _write_frame_outputs(frame, output_dir, output_indices, image_size, jpeg_quality)
            )

    capture.release()

    if not saved_paths:
        return {
            "ok": False,
            "reason": "no_frames_saved",
            "source_video_id": source_video_id,
            "video_path": str(video_path),
            "frame_dir": str(output_dir),
        }

    return {
        "ok": True,
        "reason": "ok",
        "saved_frames": len(saved_paths),
        "used_cache": False,
        "source_video_id": source_video_id,
        "video_path": str(video_path),
        "frame_dir": str(output_dir),
        "frame_count": int(frame_count),
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": (frame_count / fps) if fps and fps > 0 else np.nan,
    }

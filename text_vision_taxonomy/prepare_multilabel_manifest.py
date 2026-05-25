from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from text_vision.segment_frame_utils import time_to_seconds
except Exception:
    from segment_frame_utils import time_to_seconds


TAXONOMY_LABELS = [
    "hate_speech",
    "discrimination",
    "contextual_hate",
    "threat",
    "violence",
    "fear",
    "sexual",
    "illegal",
    "online_harm",
]
TAX_COLUMNS = [f"tax_{label}" for label in TAXONOMY_LABELS]
FALLBACK_LABEL_COLUMNS = [
    "discrimination",
    "fear",
    "hate_speech",
    "illegal",
    "sexual",
    "threat",
    "violence",
]
BINARY_POSITIVE_DEFAULT = {
    "hate_speech",
    "discrimination",
    "contextual_hate",
    "threat",
    "violence",
    "illegal",
    "online_harm",
}
SEVERITY_BY_LABEL = {
    "neutral": 0,
    "fear": 1,
    "sexual": 1,
    "online_harm": 1,
    "hate_speech": 2,
    "discrimination": 2,
    "contextual_hate": 2,
    "illegal": 2,
    "threat": 3,
    "violence": 3,
}
PRESERVED_TEXT_COLUMNS = ["Action", "Subcategories", "Scene", "ParentLabels", "action_clean", "final_labels"]


def _clean_label(value: Any) -> str:
    return str(value).strip().strip("\"'").lower()


def _parse_boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_final_labels(value: Any, row: Dict[str, Any]) -> Tuple[List[str], bool]:
    text = "" if value is None else str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, str):
            labels = [_clean_label(parsed)]
        elif isinstance(parsed, Iterable):
            labels = [_clean_label(item) for item in parsed]
        else:
            labels = []
        labels = [label for label in labels if label]
        if labels:
            return labels, True
    except Exception:
        pass

    labels = []
    for column in FALLBACK_LABEL_COLUMNS:
        if _parse_boolish(row.get(column, "")):
            labels.append(column)
    if not labels and _parse_boolish(row.get("neutral", "")):
        labels.append("neutral")
    return labels, False


def _format_segment_part(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-6:
        return str(int(round(float(value))))
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _segment_dir_name(start_sec: float, end_sec: float) -> str:
    return f"_{int(round(start_sec * 1000.0)):010d}_{int(round(end_sec * 1000.0)):010d}"


def find_segment_frame_dir(frame_root: Path, source_video_id: str, start_sec: float, end_sec: float) -> Path | None:
    video_root = frame_root / source_video_id
    if not video_root.exists() or not video_root.is_dir():
        return None
    suffix = _segment_dir_name(start_sec, end_sec)
    matches = [child for child in video_root.iterdir() if child.is_dir() and child.name.endswith(suffix)]
    if matches:
        return sorted(matches)[0]
    return None


def count_frames(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in exts)


def build_text_for_taxonomy(row: Dict[str, Any]) -> str:
    parts = []
    for label, column in [
        ("ACTION", "Action"),
        ("SUBCATEGORIES", "Subcategories"),
        ("SCENE", "Scene"),
        ("PARENT_LABELS", "ParentLabels"),
    ]:
        text = str(row.get(column, "") or "").strip()
        if text:
            parts.append(f"[{label}]\n{text}")
    return "\n\n".join(parts)


def split_source_ids(source_ids: List[str], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, str]:
    ids = sorted(set(source_ids))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    n = len(ids)
    train_end = int(round(n * float(train_ratio)))
    val_end = train_end + int(round(n * float(val_ratio)))
    mapping = {}
    for index, source_id in enumerate(ids):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "val"
        else:
            split = "test"
        mapping[source_id] = split
    return mapping


def prepare_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.summary)
    frame_root = Path(args.frame_root) if args.frame_root else None
    whole_frame_root = Path(args.whole_frame_root) if args.whole_frame_root else None
    video_root = Path(args.video_root) if args.video_root else None
    binary_positive = set(BINARY_POSITIVE_DEFAULT)
    if args.sexual_as_positive:
        binary_positive.add("sexual")
    if args.fear_as_positive:
        binary_positive.add("fear")

    rows: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    parse_source_counts = Counter()
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_index, row in enumerate(reader):
            file_name = str(row.get("File Name", "") or "").strip()
            start_value = str(row.get("Start Time", "") or "").strip()
            end_value = str(row.get("End Time", "") or "").strip()
            try:
                if not file_name or not start_value or not end_value:
                    raise ValueError("missing File Name/Start Time/End Time")
                start_sec = float(time_to_seconds(start_value))
                end_sec = float(time_to_seconds(end_value))
                if end_sec <= start_sec:
                    raise ValueError("end_sec <= start_sec")
            except Exception as exc:
                dropped.append({"row_index": raw_index, "reason": str(exc), "File Name": file_name})
                continue

            source_video_id = Path(file_name).stem
            segment_id = f"{source_video_id}_{_format_segment_part(start_sec)}_{_format_segment_part(end_sec)}"
            labels, parsed_ok = parse_final_labels(row.get("final_labels", ""), row)
            parse_source_counts["literal_eval" if parsed_ok else "fallback_columns"] += 1
            label_set = set(labels)
            if label_set == {"neutral"}:
                active_taxonomy = set()
            else:
                active_taxonomy = {label for label in label_set if label in TAXONOMY_LABELS}

            output_row: Dict[str, Any] = {
                "sample_id": len(rows),
                "source_video_id": source_video_id,
                "segment_id": segment_id,
                "file_name": file_name,
                "video_file_name": file_name,
                "start_sec": f"{start_sec:.6f}".rstrip("0").rstrip("."),
                "end_sec": f"{end_sec:.6f}".rstrip("0").rstrip("."),
                "duration_sec": f"{(end_sec - start_sec):.6f}".rstrip("0").rstrip("."),
                "final_labels_parsed": "|".join(sorted(label_set)),
                "label_parse_source": "literal_eval" if parsed_ok else "fallback_columns",
                "text_for_taxonomy": build_text_for_taxonomy(row),
                "source": "final_dataset",
                "label_source": "final_dataset.csv",
                "fps": "",
            }
            for column in PRESERVED_TEXT_COLUMNS:
                output_row[column] = row.get(column, "")
            for label in TAXONOMY_LABELS:
                output_row[f"tax_{label}"] = 1 if label in active_taxonomy else 0

            binary_label = int(bool(active_taxonomy & binary_positive))
            pseudo_severity = max([SEVERITY_BY_LABEL.get(label, 0) for label in label_set], default=0)
            output_row.update(
                {
                    "binary_label": binary_label,
                    "label": binary_label,
                    "pseudo_severity": pseudo_severity,
                    "fallback_severity": pseudo_severity,
                    "transcription": output_row["text_for_taxonomy"],
                    "has_text": bool(output_row["text_for_taxonomy"].strip()),
                    "transcript_source": "scene" if output_row["text_for_taxonomy"].strip() else "missing",
                    "transcript_confidence": 1.0 if output_row["text_for_taxonomy"].strip() else 0.0,
                }
            )

            frame_dir = None
            frame_dir_is_segment = False
            if frame_root is not None:
                candidate = find_segment_frame_dir(frame_root, source_video_id, start_sec, end_sec)
                if candidate is not None and count_frames(candidate) > 0:
                    frame_dir = candidate
                    frame_dir_is_segment = True
            if frame_dir is None and whole_frame_root is not None:
                candidate = whole_frame_root / source_video_id
                if candidate.exists() and candidate.is_dir():
                    frame_dir = candidate
            output_row["frame_dir"] = str(frame_dir) if frame_dir is not None else ""
            output_row["frame_dir_is_segment"] = frame_dir_is_segment
            output_row["num_frames"] = count_frames(frame_dir) if frame_dir is not None else 0
            output_row["num_frames_video"] = output_row["num_frames"]
            output_row["frame_dir_exists"] = bool(frame_dir and frame_dir.exists())

            video_path = video_root / file_name if video_root is not None else Path("")
            output_row["video_path"] = str(video_path) if video_root is not None and video_path.exists() else ""
            output_row["video_path_exists"] = bool(video_root is not None and video_path.exists())
            rows.append(output_row)

    split_map = split_source_ids(
        [row["source_video_id"] for row in rows],
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    for row in rows:
        row["split"] = split_map[row["source_video_id"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_video_id",
        "segment_id",
        "file_name",
        "video_file_name",
        "split",
        "start_sec",
        "end_sec",
        "duration_sec",
        "frame_dir",
        "frame_dir_is_segment",
        "frame_dir_exists",
        "num_frames",
        "num_frames_video",
        "video_path",
        "video_path_exists",
        "binary_label",
        "label",
        "pseudo_severity",
        "fallback_severity",
        *TAX_COLUMNS,
        "text_for_taxonomy",
        "transcription",
        "has_text",
        "transcript_source",
        "transcript_confidence",
        "final_labels_parsed",
        "label_parse_source",
        "source",
        "label_source",
        "fps",
        *PRESERVED_TEXT_COLUMNS,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(row["split"] for row in rows)
    taxonomy_counts = {column: sum(int(row[column]) for row in rows) for column in TAX_COLUMNS}
    binary_counts = Counter(str(row["binary_label"]) for row in rows)
    severity_counts = Counter(str(row["pseudo_severity"]) for row in rows)
    source_to_splits = defaultdict(set)
    for row in rows:
        source_to_splits[row["source_video_id"]].add(row["split"])
    leakage = {source: sorted(splits) for source, splits in source_to_splits.items() if len(splits) > 1}
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "total_input_rows": len(rows) + len(dropped),
        "total_rows": len(rows),
        "dropped_rows": len(dropped),
        "dropped_examples": dropped[:20],
        "split_counts": dict(split_counts),
        "taxonomy_label_counts": taxonomy_counts,
        "binary_label_counts": dict(binary_counts),
        "pseudo_severity_counts": dict(severity_counts),
        "label_parse_source_counts": dict(parse_source_counts),
        "source_video_count": len(source_to_splits),
        "split_leakage_source_ids": leakage,
        "frame_dir_exists_rows": sum(1 for row in rows if row["frame_dir_exists"]),
        "segment_frame_dir_rows": sum(1 for row in rows if row["frame_dir_is_segment"]),
        "video_path_exists_rows": sum(1 for row in rows if row["video_path_exists"]),
        "taxonomy_labels": TAXONOMY_LABELS,
        "neutral_is_bce_label": False,
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"total rows: {len(rows)}")
    print(f"dropped rows: {len(dropped)}")
    print(f"split counts: {dict(split_counts)}")
    print(f"taxonomy label counts: {taxonomy_counts}")
    print(f"binary label counts: {dict(binary_counts)}")
    print(f"pseudo_severity counts: {dict(severity_counts)}")
    print(f"output: {output_path}")
    print(f"summary: {report_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare segment-level multi-label taxonomy manifest.")
    parser.add_argument("--input", type=str, default=r"D:\hvc\datasets\action\archieve\final_dataset.csv")
    parser.add_argument("--output", type=str, default="data/multilabel_manifest.csv")
    parser.add_argument("--summary", type=str, default="reports/multilabel_manifest_summary.json")
    parser.add_argument("--frame-root", type=str, default=r"D:\hvc\datasets\action\timestamp_frames")
    parser.add_argument("--whole-frame-root", type=str, default=r"D:\hvc\datasets\frames")
    parser.add_argument("--video-root", type=str, default=r"D:\hvc\datasets\action\videos")
    parser.add_argument("--sexual-as-positive", action="store_true")
    parser.add_argument("--fear-as-positive", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    prepare_manifest(parse_args())


if __name__ == "__main__":
    main()

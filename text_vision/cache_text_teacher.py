from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import as_path, get_config
from text_vision.models.text_teacher import FrozenTextTeacher


def cache_teacher_outputs(
    manifest_path: Path,
    model_path: Path,
    output_path: Path,
    summary_path: Path,
    batch_size: int,
    device: str,
    max_length: int,
    limit: int | None = None,
    local_files_only: bool = False,
) -> Dict[str, Dict]:
    manifest = pd.read_csv(manifest_path)
    videos = manifest.drop_duplicates("source_video_id").copy()
    videos = videos[videos["has_text"].astype(bool)].copy()
    videos["transcript_confidence"] = pd.to_numeric(
        videos.get("transcript_confidence", 1.0),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    if limit is not None:
        videos = videos.head(int(limit)).copy()

    teacher = FrozenTextTeacher(
        checkpoint_path=model_path,
        device=device,
        local_files_only=local_files_only,
    )

    items: Dict[str, Dict] = {}
    summary_rows: List[Dict] = []
    records = videos.to_dict("records")
    for start in tqdm(range(0, len(records), int(batch_size)), desc="teacher"):
        batch_records = records[start : start + int(batch_size)]
        texts = [str(record["transcription"]) for record in batch_records]
        outputs = teacher.predict_batch(texts, max_length=max_length)
        for offset, record in enumerate(batch_records):
            source_video_id = str(record["source_video_id"])
            label = int(record["label"])
            predicted_severity = int(outputs["predicted_severity"][offset].item())
            pseudo_severity = 0 if label == 0 else predicted_severity
            if label == 1 and pseudo_severity == 0:
                pseudo_severity = 2
            confidence = float(outputs["teacher_confidence"][offset].item())
            confidence *= float(record.get("transcript_confidence", 1.0))
            item = {
                "source_video_id": source_video_id,
                "transcript_source": str(record.get("transcript_source", "human")),
                "transcript_confidence": float(record.get("transcript_confidence", 1.0)),
                "text": str(record["transcription"]),
                "embedding": outputs["embedding"][offset].cpu(),
                "binary_logit": float(outputs["binary_logit"][offset].item()),
                "binary_prob": float(outputs["binary_prob"][offset].item()),
                "severity_logits": outputs["severity_logits"][offset].cpu(),
                "severity_probs": outputs["severity_probs"][offset].cpu(),
                "predicted_severity": predicted_severity,
                "pseudo_severity": int(pseudo_severity),
                "teacher_confidence": confidence,
                "label": label,
            }
            items[source_video_id] = item
            summary_rows.append(
                {
                    "source_video_id": source_video_id,
                    "label": label,
                    "transcript_source": item["transcript_source"],
                    "binary_prob": item["binary_prob"],
                    "predicted_severity": predicted_severity,
                    "pseudo_severity": int(pseudo_severity),
                    "teacher_confidence": confidence,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "metadata": {
            "model_path": str(model_path),
            "model_name": teacher.model_name,
            "embedding_dim": teacher.embedding_dim,
            "count": len(items),
            "max_length": int(max_length),
        },
        "items": items,
    }
    torch.save(blob, output_path)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return items


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Cache frozen text teacher embeddings and logits.")
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--model", type=str, default=str(cfg.text_model_path))
    parser.add_argument("--output", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--summary", type=str, default=str(cfg.text_teacher_summary_path))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=cfg.text_max_length)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    cfg = get_config()
    args = parse_args()
    items = cache_teacher_outputs(
        manifest_path=as_path(args.manifest, cfg.manifest_path),
        model_path=as_path(args.model, cfg.text_model_path),
        output_path=as_path(args.output, cfg.text_teacher_cache_path),
        summary_path=as_path(args.summary, cfg.text_teacher_summary_path),
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        limit=args.limit,
        local_files_only=args.local_files_only,
    )
    print(f"cached teacher records: {len(items):,}")
    print(f"output: {as_path(args.output, cfg.text_teacher_cache_path)}")


if __name__ == "__main__":
    main()


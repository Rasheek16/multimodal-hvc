from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS, get_config
from text_vision.moderation_reasoner import reason_about_moderation
from text_vision.prepare_multilabel_manifest import parse_final_labels
from text_vision.train_taxonomy_head import TAX_COLUMNS
from text_vision.video_description import build_video_description


def check_manifest(path: Path) -> List[str]:
    failures: List[str] = []
    frame = pd.read_csv(path)
    missing = [column for column in TAX_COLUMNS + ["source_video_id", "split", "binary_label", "pseudo_severity"] if column not in frame.columns]
    if missing:
        failures.append(f"manifest missing columns: {missing}")
        return failures
    neutral_rows = frame[frame["final_labels_parsed"].astype(str).eq("neutral")] if "final_labels_parsed" in frame.columns else pd.DataFrame()
    if not neutral_rows.empty and int(neutral_rows[TAX_COLUMNS].sum(axis=1).sum()) != 0:
        failures.append("neutral rows have non-zero taxonomy targets")
    source_splits = frame.groupby("source_video_id")["split"].nunique()
    leaked = source_splits[source_splits > 1]
    if not leaked.empty:
        failures.append(f"source_video_id split leakage: {leaked.index[:5].tolist()}")
    parsed, ok = parse_final_labels("['contextual_hate', 'discrimination']", {})
    if not ok or set(parsed) != {"contextual_hate", "discrimination"}:
        failures.append("final_labels literal parsing failed")
    parsed, ok = parse_final_labels("not a list", {"hate_speech": "1"})
    if ok or parsed != ["hate_speech"]:
        failures.append("final_labels fallback parsing failed")
    return failures


def check_optional_layers() -> List[str]:
    failures: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        result = build_video_description(tmp, transcript="", num_frames=2)
        if result is None or not isinstance(result.to_dict(), dict):
            failures.append("video_description did not return a structured result")
    reasoned = reason_about_moderation(
        prob_hate=0.9,
        taxonomy_probs={"discrimination": 0.9},
        transcript="A scene targets women with discriminatory language.",
        include_evidence=True,
    )
    if reasoned.target_group != "Women":
        failures.append("target group extraction failed for a grounded gender term")
    return failures


def check_model_smoke(args: argparse.Namespace) -> List[str]:
    failures: List[str] = []
    try:
        import torch

        from text_vision.data_utils import safe_torch_load
        from text_vision.train_frame_text_vision import load_model_from_checkpoint

        cfg = get_config()
        cfg.checkpoint_path = Path(args.checkpoint)
        cfg.taxonomy_labels = list(TAXONOMY_LABELS)
        cfg.num_taxonomy_labels = len(TAXONOMY_LABELS)
        device = torch.device(args.device or "cpu")
        checkpoint = safe_torch_load(cfg.checkpoint_path, map_location="cpu")
        text_embedding_dim = int(checkpoint.get("text_embedding_dim", cfg.text_embedding_dim))
        model, _ = load_model_from_checkpoint(cfg.checkpoint_path, cfg, device=device, local_files_only=args.local_files_only)
        batch_size = 2
        batch = {
            "pixel_values": torch.zeros(batch_size, cfg.num_frames, 3, cfg.image_size, cfg.image_size, device=device),
            "text_embedding": torch.zeros(batch_size, text_embedding_dim, device=device),
            "text_binary_prob": torch.full((batch_size,), 0.5, device=device),
            "text_severity_probs": torch.full((batch_size, 4), 0.25, device=device),
            "has_text": torch.zeros(batch_size, device=device),
            "transcript_source_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            outputs = model(batch)
        if "taxonomy_logits" not in outputs or tuple(outputs["taxonomy_logits"].shape) != (batch_size, len(TAXONOMY_LABELS)):
            failures.append(f"taxonomy logits shape mismatch: {tuple(outputs.get('taxonomy_logits', torch.empty(0)).shape)}")
        if tuple(outputs["binary_logits"].shape) != (batch_size,):
            failures.append(f"binary logits shape mismatch: {tuple(outputs['binary_logits'].shape)}")
    except Exception as exc:
        failures.append(f"model smoke failed: {exc}")
    return failures


def run(args: argparse.Namespace) -> None:
    failures = []
    failures.extend(check_manifest(Path(args.manifest)))
    failures.extend(check_optional_layers())
    if args.include_model:
        failures.extend(check_model_smoke(args))
    payload: Dict[str, Any] = {
        "manifest": args.manifest,
        "include_model": bool(args.include_model),
        "ok": not failures,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run light moderation smoke checks.")
    parser.add_argument("--manifest", default="data/multilabel_manifest.csv")
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/best_frame_text_vision_relabel_all_splits.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

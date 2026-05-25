from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS, as_path, get_config
from text_vision.data_utils import compute_binary_metrics, compute_severity_metrics, safe_torch_load
from text_vision.train_frame_text_vision import load_model_from_checkpoint
from text_vision.train_taxonomy_head import (
    TAX_COLUMNS,
    TaxonomyFrameDataset,
    filter_trainable_frame_rows,
    move_tensor_batch,
    print_frame_sampling_summary,
    taxonomy_metrics,
)


def _taxonomy_labels(checkpoint: Mapping[str, Any]) -> List[str]:
    labels = checkpoint.get("taxonomy_labels")
    if not labels and isinstance(checkpoint.get("config", {}), Mapping):
        labels = checkpoint["config"].get("taxonomy_labels")
    return [str(label) for label in (labels or TAXONOMY_LABELS)]


def predictions_from_inference(args: argparse.Namespace, manifest: pd.DataFrame) -> pd.DataFrame:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.frame_checkpoint_path)
    cfg.frame_vision_model_name = args.model_name or cfg.frame_vision_model_name
    cfg.vision_model_name = cfg.frame_vision_model_name
    cfg.num_frames = int(args.num_frames)
    cfg.taxonomy_labels = list(TAXONOMY_LABELS)
    cfg.num_taxonomy_labels = len(TAXONOMY_LABELS)

    missing = [column for column in TAX_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing taxonomy columns: {missing}")
    working_manifest = manifest.copy()
    working_manifest["__row_index"] = working_manifest.index.astype(int)
    working_manifest = filter_trainable_frame_rows(working_manifest)
    if working_manifest.empty:
        raise ValueError("No manifest rows with usable frame_dir are available for evaluation")
    print_frame_sampling_summary(working_manifest)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = safe_torch_load(cfg.checkpoint_path, map_location="cpu")
    labels = _taxonomy_labels(checkpoint)
    text_embedding_dim = int(checkpoint.get("text_embedding_dim", cfg.text_embedding_dim))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    model, _ = load_model_from_checkpoint(
        cfg.checkpoint_path,
        cfg,
        device=device,
        local_files_only=bool(args.local_files_only),
    )
    dataset = TaxonomyFrameDataset(
        working_manifest,
        cfg=cfg,
        text_embedding_dim=text_embedding_dim,
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    rows: List[Dict[str, Any]] = []
    model.eval()
    cursor = 0
    with torch.inference_mode():
        for step, raw_batch in enumerate(tqdm(loader, desc="evaluate", leave=False), start=1):
            if args.max_batches is not None and step > int(args.max_batches):
                break
            batch = move_tensor_batch(raw_batch, device)
            outputs = model(batch)
            taxonomy_probs = torch.sigmoid(outputs["taxonomy_logits"]).detach().cpu()
            prob_hate = torch.sigmoid(outputs["binary_logits"]).view(-1).detach().cpu()
            severity = F.softmax(outputs["severity_logits"], dim=-1).argmax(dim=-1).detach().cpu()
            batch_rows = working_manifest.iloc[cursor : cursor + int(taxonomy_probs.shape[0])]
            cursor += int(taxonomy_probs.shape[0])
            for batch_index, (_, row) in enumerate(batch_rows.iterrows()):
                output = {
                    "row_index": int(row["__row_index"]),
                    "source_video_id": row.get("source_video_id", ""),
                    "segment_id": row.get("segment_id", ""),
                    "split": row.get("split", ""),
                    "true_binary": int(row.get("binary_label", row.get("label", 0))),
                    "prob_hate": float(prob_hate[batch_index].item()),
                    "pred_binary": int(float(prob_hate[batch_index].item()) >= threshold),
                    "true_severity": int(row.get("pseudo_severity", row.get("fallback_severity", 0))),
                    "pred_severity": int(severity[batch_index].item()),
                    "target_group": "",
                }
                for label_index, label in enumerate(labels):
                    output[f"true_{label}"] = int(row.get(f"tax_{label}", 0))
                    output[f"prob_{label}"] = float(taxonomy_probs[batch_index, label_index].item())
                rows.append(output)
    return pd.DataFrame(rows)


def load_or_create_predictions(args: argparse.Namespace, manifest: pd.DataFrame) -> pd.DataFrame:
    if args.predictions and Path(args.predictions).exists():
        return pd.read_csv(args.predictions)
    if args.limit is not None:
        manifest = manifest.head(int(args.limit)).copy()
    return predictions_from_inference(args, manifest)


def build_error_analysis(predictions: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    threshold_values = {label: float(thresholds.get(label, 0.5)) for label in TAXONOMY_LABELS}
    for _, row in predictions.iterrows():
        true_labels = {label for label in TAXONOMY_LABELS if int(row.get(f"true_{label}", 0)) == 1}
        pred_labels = {label for label in TAXONOMY_LABELS if float(row.get(f"prob_{label}", 0.0)) >= threshold_values[label]}
        categories = []
        if not true_labels and pred_labels:
            categories.append("false positive neutral")
        for label in ["hate_speech", "discrimination", "threat", "violence"]:
            if label in true_labels and label not in pred_labels:
                categories.append(f"missed {label}")
        if {"sexual", "fear"} & pred_labels and not ({"sexual", "fear"} & true_labels):
            categories.append("overpredicted sexual/fear")
        if int(row.get("true_binary", 0)) == 1 and int(row.get("pred_binary", 0)) == 0:
            categories.append("hate false negative")
        if int(row.get("true_binary", 0)) == 0 and int(row.get("pred_binary", 0)) == 1:
            categories.append("neutral false positive")
        if categories:
            rows.append({**row.to_dict(), "error_categories": "|".join(sorted(set(categories)))})
    return pd.DataFrame(rows)


def evaluate(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    predictions = load_or_create_predictions(args, manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {label: 0.5 for label in TAXONOMY_LABELS}
    if args.taxonomy_thresholds and Path(args.taxonomy_thresholds).exists():
        blob = json.loads(Path(args.taxonomy_thresholds).read_text(encoding="utf-8"))
        thresholds.update({label: float(value) for label, value in (blob.get("thresholds", blob) if isinstance(blob, Mapping) else {}).items()})

    binary = compute_binary_metrics(predictions["true_binary"], predictions["prob_hate"], args.threshold or 0.5)
    severity = compute_severity_metrics(predictions["true_severity"], predictions["pred_severity"])
    taxonomy_summary, per_label = taxonomy_metrics(predictions, thresholds)
    neutral = predictions[[f"true_{label}" for label in TAXONOMY_LABELS]].sum(axis=1).eq(0)
    hate = predictions["true_binary"].astype(int).eq(1)
    metrics = {
        "binary": binary,
        "severity": severity,
        "taxonomy": taxonomy_summary,
        "neutral_false_positive_rate": float(((predictions["pred_binary"].astype(int).eq(1)) & neutral).sum() / max(neutral.sum(), 1)),
        "hate_false_negative_rate": float(((predictions["pred_binary"].astype(int).eq(0)) & hate).sum() / max(hate.sum(), 1)),
        "rows": int(len(predictions)),
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    per_label.to_csv(output_dir / "per_label_metrics.csv", index=False)
    build_error_analysis(predictions, thresholds).to_csv(output_dir / "error_analysis.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full moderation system outputs.")
    parser.add_argument("--manifest", default="data/multilabel_manifest.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_taxonomy_fusion_finetuned.pt")
    parser.add_argument("--taxonomy-thresholds", default="reports/taxonomy_head/thresholds.json")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--output-dir", default="reports/final_moderation_eval")
    parser.add_argument("--describe-video", action="store_true")
    parser.add_argument("--description-frames", type=int, default=8)
    parser.add_argument("--use-description-as-transcript", action="store_true")
    parser.add_argument("--include-evidence", action="store_true")
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--vlm-model-name", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

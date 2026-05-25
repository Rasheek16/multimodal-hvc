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
from text_vision.data_utils import normalize_transcript, safe_torch_load
from text_vision.moderation_reasoner import reason_about_moderation
from text_vision.taxonomy_postprocessing import evidence_text_blob, postprocess_taxonomy
from text_vision.train_frame_text_vision import load_model_from_checkpoint
from text_vision.train_taxonomy_head import (
    TAX_COLUMNS,
    TaxonomyFrameDataset,
    filter_trainable_frame_rows,
    move_tensor_batch,
    print_frame_sampling_summary,
)


def _load_taxonomy_thresholds(path: str | None, checkpoint: Mapping[str, Any], labels: List[str]) -> Dict[str, float]:
    thresholds = checkpoint.get("taxonomy_thresholds", {})
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            blob = json.load(handle)
        if isinstance(blob, Mapping):
            thresholds = blob.get("production_thresholds", blob.get("thresholds", blob))
        else:
            thresholds = {}
    return {label: float(thresholds.get(label, 0.5)) for label in labels}


def _taxonomy_labels(checkpoint: Mapping[str, Any]) -> List[str]:
    labels = checkpoint.get("taxonomy_labels")
    if not labels and isinstance(checkpoint.get("config", {}), Mapping):
        labels = checkpoint["config"].get("taxonomy_labels")
    return [str(label) for label in (labels or TAXONOMY_LABELS)]


def _row_text(row: Mapping[str, Any]) -> str:
    return normalize_transcript(row.get("transcription", row.get("text_for_taxonomy", "")))


def _row_scene_text(row: Mapping[str, Any]) -> str:
    return normalize_transcript(row.get("Scene", row.get("text_for_taxonomy", "")))


def _review_reasons(row: Mapping[str, Any], result: Mapping[str, Any], taxonomy_labels: List[str]) -> List[str]:
    reasons: List[str] = []
    prob_hate = float(result.get("prob_hate", 0.0))
    binary_label = int(row.get("binary_label", row.get("label", 0)))
    taxonomy = result.get("taxonomy", {})
    predicted_labels = set(taxonomy.get("labels", [])) if isinstance(taxonomy, Mapping) else set()
    true_labels = {label for label in taxonomy_labels if int(row.get(f"tax_{label}", 0)) == 1}
    neutral = not true_labels
    if neutral and prob_hate >= 0.65:
        reasons.append("neutral label but high hate probability")
    if binary_label == 1 and prob_hate <= 0.35:
        reasons.append("hate label but low hate probability")
    if int(result.get("severity", 0)) >= 3 and not predicted_labels:
        reasons.append("high severity but no taxonomy labels")
    if predicted_labels and true_labels and predicted_labels.isdisjoint(true_labels):
        reasons.append("taxonomy labels conflict with final_labels")
    if {"discrimination", "contextual_hate"} & predicted_labels and not result.get("target_group"):
        reasons.append("target_group missing for discrimination/contextual_hate")
    max_prob = max([float(value) for value in taxonomy.get("probs", {}).values()], default=0.0) if isinstance(taxonomy, Mapping) else 0.0
    if max_prob < 0.45 and prob_hate < 0.55:
        reasons.append("very low confidence")
    return reasons


def _auto_accept(row: Mapping[str, Any], result: Mapping[str, Any], review_reasons: List[str], taxonomy_labels: List[str]) -> bool:
    if review_reasons:
        return False
    prob_hate = float(result.get("prob_hate", 0.0))
    taxonomy = result.get("taxonomy", {})
    predicted_labels = set(taxonomy.get("labels", [])) if isinstance(taxonomy, Mapping) else set()
    true_labels = {label for label in taxonomy_labels if int(row.get(f"tax_{label}", 0)) == 1}
    neutral = not true_labels
    if neutral and predicted_labels:
        return False
    if predicted_labels and true_labels and not predicted_labels.issubset(true_labels | {"online_harm"}):
        return False
    max_prob = max([float(value) for value in taxonomy.get("probs", {}).values()], default=0.0) if isinstance(taxonomy, Mapping) else 0.0
    return prob_hate >= 0.75 or max_prob >= 0.75 or (neutral and prob_hate <= 0.25)


def run_generation(args: argparse.Namespace) -> None:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.frame_checkpoint_path)
    cfg.frame_vision_model_name = args.model_name or cfg.frame_vision_model_name
    cfg.vision_model_name = cfg.frame_vision_model_name
    cfg.num_frames = int(args.num_frames)
    cfg.taxonomy_labels = list(TAXONOMY_LABELS)
    cfg.num_taxonomy_labels = len(TAXONOMY_LABELS)

    manifest = pd.read_csv(cfg.manifest_path)
    if args.limit is not None:
        manifest = manifest.head(int(args.limit)).copy()
    missing = [column for column in TAX_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing taxonomy columns: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched = manifest.copy()
    working_manifest = manifest.copy()
    working_manifest["__row_index"] = working_manifest.index.astype(int)
    working_manifest = filter_trainable_frame_rows(working_manifest)
    if working_manifest.empty:
        raise ValueError("No manifest rows with usable frame_dir are available for pseudo-label generation")
    print_frame_sampling_summary(working_manifest)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = safe_torch_load(cfg.checkpoint_path, map_location="cpu")
    labels = _taxonomy_labels(checkpoint)
    thresholds = _load_taxonomy_thresholds(args.taxonomy_thresholds, checkpoint, labels)
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    text_embedding_dim = int(checkpoint.get("text_embedding_dim", cfg.text_embedding_dim))
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

    prediction_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    accepted_rows: List[Dict[str, Any]] = []

    model.eval()
    cursor = 0
    with torch.inference_mode():
        for step, raw_batch in enumerate(tqdm(loader, desc="pseudo-label", leave=False), start=1):
            if args.max_batches is not None and step > int(args.max_batches):
                break
            batch = move_tensor_batch(raw_batch, device)
            outputs = model(batch)
            taxonomy_probs_tensor = torch.sigmoid(outputs["taxonomy_logits"]).detach().cpu()
            prob_hate_tensor = torch.sigmoid(outputs["binary_logits"]).view(-1).detach().cpu()
            severity_probs_tensor = F.softmax(outputs["severity_logits"], dim=-1).detach().cpu()
            batch_rows = working_manifest.iloc[cursor : cursor + int(taxonomy_probs_tensor.shape[0])]
            cursor += int(taxonomy_probs_tensor.shape[0])

            for batch_index, (_, row) in enumerate(batch_rows.iterrows()):
                row_index = int(row["__row_index"])
                row_map = row.to_dict()
                prob_hate = float(prob_hate_tensor[batch_index].item())
                severity_probs = severity_probs_tensor[batch_index]
                severity = int(severity_probs.argmax().item())
                probs = {
                    label: float(taxonomy_probs_tensor[batch_index, label_index].item())
                    for label_index, label in enumerate(labels)
                }
                transcript = _row_text(row_map)
                scene_text = _row_scene_text(row_map)
                processed = postprocess_taxonomy(
                    probs,
                    thresholds,
                    prob_hate=prob_hate,
                    evidence_text=evidence_text_blob(transcript, scene_text, None),
                    labels=labels,
                )
                reasoned = reason_about_moderation(
                    prob_hate=prob_hate,
                    taxonomy_probs=probs,
                    taxonomy_labels=labels,
                    thresholds=thresholds,
                    transcript=transcript,
                    scene_text=scene_text,
                    include_evidence=bool(args.include_evidence),
                )
                result = {
                    "binary_label": int(prob_hate >= threshold),
                    "prob_hate": prob_hate,
                    "severity": severity,
                    "severity_probs": [float(value) for value in severity_probs.tolist()],
                    "target_group": reasoned.target_group,
                    "moderation_explanation": reasoned.explanation,
                    "moderation_summary": reasoned.category_summary,
                    "taxonomy": {
                        "labels": processed["labels"],
                        "candidates": processed["candidates"],
                        "suppressed": processed["suppressed"],
                        "probs": probs,
                        "thresholds_used": processed["thresholds_used"],
                    },
                }
                if args.include_evidence:
                    result["evidence"] = reasoned.evidence
                taxonomy = result["taxonomy"]
                prediction = {
                    "row_index": row_index,
                    "source_video_id": row.get("source_video_id", ""),
                    "segment_id": row.get("segment_id", ""),
                    "prob_hate": result.get("prob_hate"),
                    "binary_label_pred": result.get("binary_label"),
                    "severity_pred": result.get("severity"),
                    "target_group": result.get("target_group"),
                    "taxonomy_labels_pred": "|".join(taxonomy.get("labels", [])) if isinstance(taxonomy, Mapping) else "",
                    "raw_json": json.dumps(result, ensure_ascii=False),
                }
                if isinstance(taxonomy, Mapping):
                    for label in labels:
                        prediction[f"pred_tax_{label}"] = float(taxonomy.get("probs", {}).get(label, 0.0))
                        enriched.loc[row_index, f"pred_tax_{label}"] = prediction[f"pred_tax_{label}"]
                reasons = _review_reasons(row_map, result, labels)
                accepted = _auto_accept(row_map, result, reasons, labels)
                prediction["review_reasons"] = "|".join(reasons)
                prediction["auto_accept"] = bool(accepted)
                prediction_rows.append(prediction)
                if accepted:
                    accepted_rows.append(prediction)
                else:
                    review_rows.append(prediction)

    predictions = pd.DataFrame(prediction_rows)
    review_queue = pd.DataFrame(review_rows)
    auto_accept = pd.DataFrame(accepted_rows)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    review_queue.to_csv(output_dir / "review_queue.csv", index=False)
    auto_accept.to_csv(output_dir / "auto_accept.csv", index=False)
    enriched_path = Path(args.enriched_output)
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(enriched_path, index=False)
    summary = {
        "rows": int(len(predictions)),
        "auto_accept_rows": int(len(auto_accept)),
        "review_queue_rows": int(len(review_queue)),
        "output_dir": str(output_dir),
        "enriched_manifest": str(enriched_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate taxonomy pseudo-labels and a review queue.")
    parser.add_argument("--manifest", default="data/multilabel_manifest.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_taxonomy_fusion_finetuned.pt")
    parser.add_argument("--taxonomy-thresholds", default="reports/taxonomy_head/thresholds.json")
    parser.add_argument("--output-dir", default="reports/taxonomy_pseudo_labels")
    parser.add_argument("--enriched-output", default="data/multilabel_manifest_enriched.csv")
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
    run_generation(parse_args())


if __name__ == "__main__":
    main()

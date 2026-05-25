from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.ensemble_multilabel_classifier import weighted_prob
from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, load_thresholds, parse_json_map, text_block
from text_vision.rare_label_rules import apply_rare_label_rules


def tfidf_probs(path: str | Path, text: str) -> Dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    payload = joblib.load(path)
    model = payload.get("model", payload) if isinstance(payload, dict) else payload
    probs = model.predict_proba([text])
    if isinstance(probs, list):
        probs = np.vstack([item[:, 1] if getattr(item, "ndim", 0) == 2 else item for item in probs]).T
    values = np.asarray(probs, dtype=float)[0]
    labels = payload.get("taxonomy_labels", LABEL_COLUMNS) if isinstance(payload, dict) else LABEL_COLUMNS
    return {label: float(values[index]) for index, label in enumerate(labels)}


def transformer_probs(path: str | Path, text: str, local_files_only: bool, device_name: str | None = None, max_length: int = 256) -> Dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=local_files_only).to(device)
    model.eval()
    encoded = tokenizer([text], padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        probs = torch.sigmoid(model(**encoded).logits)[0].detach().cpu().numpy()
    config_path = Path(path) / "taxonomy_config.json"
    labels = LABEL_COLUMNS
    if config_path.exists():
        blob = json.loads(config_path.read_text(encoding="utf-8"))
        labels = [str(label) for label in blob.get("labels", labels)]
    return {label: float(probs[index]) for index, label in enumerate(labels)}


def run(args: argparse.Namespace) -> None:
    row = {
        "Scene": args.scene,
        "Action": args.action,
        "Subcategories": args.subcategories,
        "ParentLabels": args.parent_labels,
        "action_clean": args.action_clean,
    }
    text = text_block(row)
    if args.transcript:
        text = f"{text}\n\n[TRANSCRIPT]\n{args.transcript}" if text else f"[TRANSCRIPT]\n{args.transcript}"
    tfidf = tfidf_probs(args.tfidf_model, text)
    transformer = transformer_probs(args.transformer_model, text, args.local_files_only, args.device, args.max_length)
    video = parse_json_map(args.vision_taxonomy_probs)
    thresholds = load_thresholds(args.thresholds)
    probs = {
        label: weighted_prob(
            label,
            tfidf.get(label) if tfidf else None,
            transformer.get(label) if transformer else None,
            video.get(label) if video else None,
        )
        for label in LABEL_COLUMNS
    }
    result = apply_rare_label_rules(
        probs,
        thresholds,
        text=text,
        text_hate_prob=args.text_hate_prob,
        vision_hate_prob=args.vision_hate_prob,
        severity=args.severity,
    )
    output = {
        "taxonomy_labels": result["final_labels"],
        "taxonomy_candidates": result["candidate_labels"],
        "taxonomy_suppressed": result["suppressed_labels"],
        "taxonomy_probs": probs,
        "thresholds_used": result["thresholds_used"],
        "model_sources": {
            "tfidf": bool(tfidf),
            "transformer": bool(transformer),
            "video_taxonomy": bool(video),
        },
    }
    print(json.dumps(output, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final multi-label taxonomy inference.")
    parser.add_argument("--scene", default="")
    parser.add_argument("--action", default="")
    parser.add_argument("--subcategories", default="")
    parser.add_argument("--parent-labels", default="")
    parser.add_argument("--action-clean", default="")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--vision-taxonomy-probs", default="")
    parser.add_argument("--text-hate-prob", type=float, default=None)
    parser.add_argument("--vision-hate-prob", type=float, default=None)
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument("--tfidf-model", default="checkpoints/text_multilabel_classifier/best_model.joblib")
    parser.add_argument("--transformer-model", default="checkpoints/transformer_multilabel_classifier/best")
    parser.add_argument("--thresholds", default="reports/multilabel_calibration/thresholds.json")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

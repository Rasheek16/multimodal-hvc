from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS
from text_vision.moderation_reasoner import reason_about_moderation


def _as_prob_map(values: Mapping[str, Any] | None, labels: Sequence[str]) -> Dict[str, float]:
    values = values or {}
    return {label: float(values.get(label, 0.0)) for label in labels}


def rule_boosts_from_evidence(reasoner_output: Mapping[str, Any] | None, labels: Sequence[str]) -> Dict[str, float]:
    output = reasoner_output or {}
    selected = set(str(label) for label in output.get("hate_types", []))
    boosts = {label: 0.0 for label in labels}
    for label in selected:
        if label in boosts:
            boosts[label] = 1.0
    if output.get("target_group"):
        for label in ("hate_speech", "discrimination", "contextual_hate"):
            if label in boosts:
                boosts[label] = max(boosts[label], 0.7)
    return boosts


def combine_taxonomy_probs(
    video_taxonomy_probs: Mapping[str, Any] | None = None,
    text_taxonomy_probs: Mapping[str, Any] | None = None,
    rule_boosts: Mapping[str, Any] | None = None,
    labels: Sequence[str] = TAXONOMY_LABELS,
    text_evidence_available: bool = False,
) -> Dict[str, float]:
    video = _as_prob_map(video_taxonomy_probs, labels)
    text = _as_prob_map(text_taxonomy_probs, labels)
    rules = _as_prob_map(rule_boosts, labels)
    if text_evidence_available:
        weights = (0.35, 0.55, 0.10)
    else:
        weights = (0.80, 0.10, 0.10)
    return {
        label: min(1.0, max(0.0, weights[0] * video[label] + weights[1] * text[label] + weights[2] * rules[label]))
        for label in labels
    }


def ensemble_moderation(
    video_result: Mapping[str, Any],
    text_taxonomy_probs: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    labels: Sequence[str] = TAXONOMY_LABELS,
) -> Dict[str, Any]:
    taxonomy = dict(video_result.get("taxonomy", {}))
    video_probs = taxonomy.get("probs", {})
    description = dict(video_result.get("description", {}))
    reasoner = reason_about_moderation(
        prob_hate=float(video_result.get("prob_hate", 0.0)),
        taxonomy_probs=video_probs,
        taxonomy_labels=labels,
        thresholds=thresholds or taxonomy.get("thresholds", {}),
        transcript=description.get("transcript", ""),
        ocr_text=description.get("ocr_text", ""),
        visual_description=description.get("visual_description", ""),
        evidence=video_result.get("evidence", []),
    )
    combined_probs = combine_taxonomy_probs(
        video_taxonomy_probs=video_probs,
        text_taxonomy_probs=text_taxonomy_probs,
        rule_boosts=rule_boosts_from_evidence(reasoner.to_dict(), labels),
        labels=labels,
        text_evidence_available=bool(description.get("text_for_classifier") or text_taxonomy_probs),
    )
    threshold_map = {label: float((thresholds or taxonomy.get("thresholds", {})).get(label, 0.5)) for label in labels}
    labels_out = [label for label in labels if combined_probs[label] >= threshold_map[label]]
    result = dict(video_result)
    result["taxonomy_ensemble"] = {
        "labels": labels_out,
        "probs": combined_probs,
        "thresholds": threshold_map,
        "weights": "text-dominant" if bool(description.get("text_for_classifier") or text_taxonomy_probs) else "video-dominant",
    }
    result["target_group"] = reasoner.target_group
    result["moderation_explanation"] = reasoner.explanation
    if "evidence" not in result:
        result["evidence"] = reasoner.evidence
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine video, text, and rule moderation signals.")
    parser.add_argument("--video-json", required=True, help="Path to an inference JSON file.")
    parser.add_argument("--text-taxonomy-json", default="{}", help="Inline JSON probability map or path to JSON.")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _load_json_arg(value: str) -> Dict[str, Any]:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def main() -> None:
    args = parse_args()
    video_result = json.loads(Path(args.video_json).read_text(encoding="utf-8"))
    result = ensemble_moderation(video_result, text_taxonomy_probs=_load_json_arg(args.text_taxonomy_json))
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, RARE_LABELS


HATE_SPECIFIC_LABELS = {
    "hate_speech",
    "discrimination",
    "contextual_hate",
    "threat",
    "illegal",
    "online_harm",
}


def _parse_json(value: Any, fallback: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def _labels(value: Any) -> list[str]:
    parsed = _parse_json(value, [])
    if isinstance(parsed, list):
        out = []
        for item in parsed:
            if isinstance(item, Mapping):
                label = str(item.get("label", "")).strip()
            else:
                label = str(item).strip()
            if label:
                out.append(label)
        return out
    if isinstance(parsed, str):
        return [label for label in parsed.split("|") if label]
    return []


def _label_names(items: Any) -> list[str]:
    parsed = _parse_json(items, [])
    names: list[str] = []
    if not isinstance(parsed, list):
        return names
    for item in parsed:
        if isinstance(item, Mapping):
            label = str(item.get("label", "")).strip()
            if label:
                names.append(label)
    return names


def _max_base_hate(signals: Mapping[str, Any]) -> float:
    values = []
    for key in ["text_hate_prob", "vision_hate_prob"]:
        value = signals.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _max_severity(signals: Mapping[str, Any]) -> int:
    values = []
    for key in ["text_severity", "vision_severity"]:
        value = signals.get(key)
        if value is None:
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0


def _metadata_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "Scene": metadata.get("scene", ""),
        "Action": metadata.get("action", ""),
        "Subcategories": metadata.get("subcategories", ""),
        "ParentLabels": metadata.get("parent_labels", ""),
        "action_clean": metadata.get("action_clean", ""),
        "transcript": metadata.get("transcript", ""),
        "ocr_text": metadata.get("ocr_text", ""),
    }


def _review_reason(
    raw_labels: list[str],
    final_labels: list[str],
    suppressed_labels: list[str],
    candidate_labels: list[str],
    signals: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    max_hate = _max_base_hate(signals)
    max_severity = _max_severity(signals)
    hate_final = [label for label in final_labels if label in HATE_SPECIFIC_LABELS]
    hate_raw = [label for label in raw_labels if label in HATE_SPECIFIC_LABELS]
    if suppressed_labels:
        reasons.append("suppressed_taxonomy_labels")
    if max_hate >= 0.75 and not hate_final:
        reasons.append("base_models_high_hate_taxonomy_missed_hate")
    if max_hate < 0.35 and hate_raw:
        reasons.append("base_models_low_hate_taxonomy_predicted_hate")
    if max_severity >= 3 and not {"violence", "threat"}.intersection(final_labels):
        reasons.append("high_severity_taxonomy_missed_violence_or_threat")
    if RARE_LABELS.intersection(set(raw_labels + candidate_labels + suppressed_labels)):
        reasons.append("rare_label_candidate")
    available = metadata.get("available_fields", {})
    if isinstance(available, Mapping) and not any(bool(value) for value in available.values()):
        reasons.append("weak_or_empty_metadata")
    return reasons


def _pseudo_allowed(
    final_labels: list[str],
    suppressed_labels: list[str],
    probs: Mapping[str, Any],
    signals: Mapping[str, Any],
    min_prob: float,
    support_hate_threshold: float,
) -> bool:
    if not final_labels:
        return False
    if set(final_labels).intersection(suppressed_labels):
        return False
    max_hate = _max_base_hate(signals)
    for label in final_labels:
        try:
            prob = float(probs.get(label, 0.0))
        except (TypeError, ValueError):
            prob = 0.0
        if prob < min_prob:
            return False
        if label in HATE_SPECIFIC_LABELS and label not in RARE_LABELS and max_hate < support_hate_threshold:
            return False
        if label in RARE_LABELS and prob < max(0.90, min_prob):
            return False
    return True


def mine_cases(args: argparse.Namespace) -> None:
    input_path = Path(args.inference_log)
    if not input_path.exists():
        raise FileNotFoundError(f"inference log not found: {input_path}")
    logs = pd.read_csv(input_path)
    review_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []

    for index, row in logs.iterrows():
        metadata = _parse_json(row.get("metadata", ""), {})
        probs = _parse_json(row.get("taxonomy_probs", ""), {})
        signals = _parse_json(row.get("base_model_signals", ""), {})
        raw_labels = _labels(row.get("taxonomy_raw_labels", ""))
        final_labels = _labels(row.get("taxonomy_final_labels", ""))
        suppressed_label_names = _label_names(row.get("suppressed_labels", ""))
        candidate_label_names = _label_names(row.get("candidate_labels", ""))
        reasons = _review_reason(
            raw_labels,
            final_labels,
            suppressed_label_names,
            candidate_label_names,
            signals if isinstance(signals, Mapping) else {},
            metadata if isinstance(metadata, Mapping) else {},
        )
        fields = _metadata_fields(metadata if isinstance(metadata, Mapping) else {})
        base_payload = {
            "row_id": int(index),
            "video": row.get("video", ""),
            "frame_dir": row.get("frame_dir", ""),
            **fields,
            "taxonomy_probs": json.dumps(probs, ensure_ascii=False),
            "raw_labels": "|".join(raw_labels),
            "final_labels": "|".join(final_labels),
            "suppressed_labels": "|".join(suppressed_label_names),
            "candidate_labels": "|".join(candidate_label_names),
            "base_model_signals": json.dumps(signals, ensure_ascii=False),
            "suggested_labels": "|".join(final_labels or raw_labels or candidate_label_names),
            "needs_review_reason": "|".join(reasons),
        }
        if reasons:
            review_rows.append(base_payload)
        if _pseudo_allowed(
            final_labels,
            suppressed_label_names,
            probs if isinstance(probs, Mapping) else {},
            signals if isinstance(signals, Mapping) else {},
            min_prob=float(args.min_pseudo_prob),
            support_hate_threshold=float(args.support_hate_threshold),
        ):
            pseudo_payload = {
                "File Name": row.get("video", "") or row.get("frame_dir", "") or f"runtime_{index}",
                **fields,
                "final_labels": "|".join(final_labels),
                "synthetic": 1,
                "pseudo_label_source": "runtime_feedback",
            }
            for label in LABEL_COLUMNS:
                pseudo_payload[label] = int(label in final_labels)
            pseudo_rows.append(pseudo_payload)

    review_output = Path(args.review_output)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    review_columns = [
        "row_id",
        "video",
        "frame_dir",
        "Scene",
        "Action",
        "Subcategories",
        "ParentLabels",
        "action_clean",
        "transcript",
        "ocr_text",
        "taxonomy_probs",
        "raw_labels",
        "final_labels",
        "suppressed_labels",
        "candidate_labels",
        "base_model_signals",
        "suggested_labels",
        "needs_review_reason",
    ]
    pd.DataFrame(review_rows, columns=review_columns).to_csv(review_output, index=False)

    pseudo_output = Path(args.pseudo_output)
    pseudo_output.parent.mkdir(parents=True, exist_ok=True)
    pseudo_columns = [
        "File Name",
        "Scene",
        "Action",
        "Subcategories",
        "ParentLabels",
        "action_clean",
        "transcript",
        "ocr_text",
        "final_labels",
        "synthetic",
        "pseudo_label_source",
        *LABEL_COLUMNS,
    ]
    pd.DataFrame(pseudo_rows, columns=pseudo_columns).to_csv(pseudo_output, index=False)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "rows": int(len(logs)),
                "review_rows": int(len(review_rows)),
                "pseudo_rows": int(len(pseudo_rows)),
                "review_output": str(review_output),
                "pseudo_output": str(pseudo_output),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine runtime taxonomy logs into review and pseudo-label queues.")
    parser.add_argument("--inference-log", default="reports/runtime_feedback/inference_logs.csv")
    parser.add_argument("--review-output", default="data/taxonomy_review_queue.csv")
    parser.add_argument("--pseudo-output", default="data/taxonomy_pseudo_labeled.csv")
    parser.add_argument("--min-pseudo-prob", type=float, default=0.85)
    parser.add_argument("--support-hate-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    mine_cases(parse_args())


if __name__ == "__main__":
    main()

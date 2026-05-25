from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .config import TAXONOMY_LABELS
from .data_utils import normalize_transcript


HATE_GATED_LABELS = {"hate_speech", "discrimination", "contextual_hate", "threat", "illegal", "online_harm"}
TEXT_REQUIRED_LABELS = {"illegal", "online_harm"}
RARE_LABELS = {"threat", "illegal", "online_harm", "sexual"}

THREAT_TERMS = ["threat", "kill", "hurt", "attack", "bomb", "shoot", "stab", "beat", "destroy", "violence"]
ILLEGAL_TERMS = ["illegal", "weapon", "drugs", "fraud", "scam", "hack", "stolen", "counterfeit", "buy prohibited"]
ONLINE_HARM_TERMS = ["dox", "doxx", "harass", "brigade", "mass report", "leak address", "private information"]


def load_threshold_file(path: str | Path | None, labels: Sequence[str] = TAXONOMY_LABELS) -> Dict[str, float]:
    if not path:
        return {label: 0.5 for label in labels}
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(blob, Mapping):
        if "production_thresholds" in blob:
            values = blob["production_thresholds"]
        elif "thresholds" in blob:
            values = blob["thresholds"]
        else:
            values = blob
    else:
        values = {}
    return {label: float(values.get(label, 0.5)) for label in labels}


def evidence_text_blob(*parts: Any) -> str:
    flattened: List[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            flattened.append(part)
        elif isinstance(part, Mapping):
            flattened.extend(str(value) for value in part.values() if isinstance(value, str))
        elif isinstance(part, Iterable):
            for item in part:
                if isinstance(item, Mapping):
                    flattened.extend(str(value) for value in item.values() if isinstance(value, str))
                elif isinstance(item, str):
                    flattened.append(item)
    return normalize_transcript(" ".join(flattened)).lower()


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    for term in terms:
        escaped = re.escape(term.lower())
        if re.search(rf"(?<!\w){escaped}(?!\w)", text, flags=re.IGNORECASE):
            return True
    return False


def label_has_required_evidence(label: str, text: str, probability: float) -> bool:
    if label == "threat":
        return probability >= 0.95 or _contains_any(text, THREAT_TERMS)
    if label == "illegal":
        return _contains_any(text, ILLEGAL_TERMS)
    if label == "online_harm":
        return _contains_any(text, ONLINE_HARM_TERMS)
    return True


def postprocess_taxonomy(
    probs: Mapping[str, float],
    thresholds: Mapping[str, float],
    prob_hate: float,
    evidence_text: str = "",
    labels: Sequence[str] = TAXONOMY_LABELS,
    candidate_margin: float = 0.10,
) -> Dict[str, Any]:
    final_labels: List[str] = []
    candidates: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    text = normalize_transcript(evidence_text).lower()
    thresholds_used = {label: float(thresholds.get(label, 0.5)) for label in labels}

    for label in labels:
        probability = float(probs.get(label, 0.0))
        threshold = thresholds_used[label]
        candidate_floor = max(0.0, min(0.50, threshold - float(candidate_margin)))
        if probability < candidate_floor:
            continue

        item = {"label": label, "probability": probability, "threshold": threshold}
        if probability < threshold:
            item["reason"] = "below_threshold_candidate"
            candidates.append(item)
            continue
        if label in HATE_GATED_LABELS and float(prob_hate) < 0.35:
            item["reason"] = "suppressed_low_binary_hate_probability"
            suppressed.append(item)
            candidates.append(item)
            continue
        if label in TEXT_REQUIRED_LABELS and not text:
            item["reason"] = "suppressed_missing_text_or_metadata_evidence"
            suppressed.append(item)
            candidates.append(item)
            continue
        if label in RARE_LABELS and not label_has_required_evidence(label, text, probability):
            item["reason"] = "suppressed_missing_required_rare_label_evidence"
            suppressed.append(item)
            candidates.append(item)
            continue
        final_labels.append(label)

    return {
        "labels": final_labels,
        "candidates": candidates,
        "suppressed": suppressed,
        "thresholds_used": thresholds_used,
    }

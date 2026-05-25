from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.data_utils import normalize_transcript
from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, load_thresholds, parse_json_map, write_json


RARE_LABELS = {"threat", "illegal", "online_harm"}
THREAT_TERMS = [
    "threat",
    "threaten",
    "kill",
    "murder",
    "hurt",
    "attack",
    "bomb",
    "shoot",
    "stab",
    "beat",
    "destroy",
    "hang",
]
ILLEGAL_TERMS = [
    "illegal",
    "weapon",
    "gun",
    "knife",
    "drug",
    "crime",
    "criminal",
    "abuse",
    "exploit",
    "fraud",
    "scam",
    "hack",
    "stolen",
    "counterfeit",
    "trafficking",
]
ONLINE_HARM_TERMS = [
    "online",
    "social media",
    "chat",
    "comment",
    "comments",
    "meme",
    "post",
    "tweet",
    "message",
    "harass",
    "harassment",
    "dox",
    "doxx",
    "leak",
    "private information",
    "cyberbully",
]


def contains_any(text: str, terms: Sequence[str]) -> bool:
    clean = normalize_transcript(text).lower()
    for term in terms:
        escaped = re.escape(term.lower())
        if re.search(rf"(?<!\w){escaped}(?!\w)", clean, flags=re.IGNORECASE):
            return True
    return False


def evidence_for_label(
    label: str,
    text: str,
    probs: Mapping[str, float],
    text_hate_prob: float | None = None,
    vision_hate_prob: float | None = None,
    severity: int | None = None,
) -> tuple[bool, str]:
    probability = float(probs.get(label, 0.0))
    if label == "threat":
        if contains_any(text, THREAT_TERMS):
            return True, "explicit_threat_text"
        if float(probs.get("violence", 0.0)) >= 0.80:
            return True, "violence_probability_high"
        if severity is not None and int(severity) >= 3:
            return True, "severity_high"
        if probability >= 0.98:
            return True, "probability_extremely_high"
        return False, "missing_threat_evidence"
    if label == "illegal":
        if contains_any(text, ILLEGAL_TERMS):
            return True, "explicit_illegal_text"
        return False, "missing_illegal_evidence"
    if label == "online_harm":
        if contains_any(text, ONLINE_HARM_TERMS):
            return True, "explicit_online_context"
        return False, "missing_online_harm_context"
    return True, "not_rare_label"


def apply_rare_label_rules(
    probs: Mapping[str, float],
    thresholds: Mapping[str, float],
    text: str = "",
    text_hate_prob: float | None = None,
    vision_hate_prob: float | None = None,
    severity: int | None = None,
    labels: Sequence[str] = LABEL_COLUMNS,
    candidate_margin: float = 0.15,
) -> Dict[str, Any]:
    final_labels: List[str] = []
    candidates: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    thresholds_used = {label: float(thresholds.get(label, 0.5)) for label in labels}
    for label in labels:
        probability = float(probs.get(label, 0.0))
        threshold = thresholds_used[label]
        item = {"label": label, "probability": probability, "threshold": threshold}
        candidate_floor = max(0.0, threshold - float(candidate_margin))
        if probability < candidate_floor:
            continue
        if probability < threshold:
            item["reason"] = "below_threshold_candidate"
            candidates.append(item)
            continue
        if label in RARE_LABELS:
            ok, reason = evidence_for_label(label, text, probs, text_hate_prob, vision_hate_prob, severity)
            if not ok:
                item["reason"] = reason
                suppressed.append(item)
                candidates.append(item)
                continue
            item["reason"] = reason
        final_labels.append(label)
    return {
        "final_labels": final_labels,
        "candidate_labels": candidates,
        "suppressed_labels": suppressed,
        "thresholds_used": thresholds_used,
    }


def apply_rules_to_predictions(predictions: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    rows = []
    for _, row in predictions.iterrows():
        probs = {label: float(row.get(f"prob_{label}", 0.0) or 0.0) for label in LABEL_COLUMNS}
        result = apply_rare_label_rules(
            probs,
            thresholds,
            text=str(row.get("text_input", "")),
            text_hate_prob=float(row["text_hate_prob"]) if "text_hate_prob" in row and pd.notna(row["text_hate_prob"]) else None,
            vision_hate_prob=float(row["vision_hate_prob"]) if "vision_hate_prob" in row and pd.notna(row["vision_hate_prob"]) else None,
            severity=int(row["severity"]) if "severity" in row and pd.notna(row["severity"]) else None,
        )
        payload = row.to_dict()
        payload["final_labels"] = "|".join(result["final_labels"])
        payload["candidate_labels"] = "|".join(item["label"] for item in result["candidate_labels"])
        payload["suppressed_labels"] = "|".join(item["label"] for item in result["suppressed_labels"])
        payload["rare_rule_json"] = json.dumps(result, ensure_ascii=False)
        rows.append(payload)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply conservative rare-label taxonomy rules.")
    parser.add_argument("--predictions", default="reports/multilabel_ensemble/predictions.csv")
    parser.add_argument("--thresholds", default="reports/multilabel_calibration/thresholds.json")
    parser.add_argument("--output", default="reports/multilabel_ensemble/predictions_with_rules.csv")
    parser.add_argument("--probs-json", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--severity", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probs_json:
        thresholds = load_thresholds(args.thresholds)
        result = apply_rare_label_rules(parse_json_map(args.probs_json), thresholds, text=args.text, severity=args.severity)
        print(json.dumps(result, indent=2))
        return
    predictions = pd.read_csv(args.predictions)
    thresholds = load_thresholds(args.thresholds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    apply_rules_to_predictions(predictions, thresholds).to_csv(output, index=False)
    print(json.dumps({"output": str(output), "rows": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()

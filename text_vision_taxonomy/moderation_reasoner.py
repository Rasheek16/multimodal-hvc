from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TAXONOMY_LABELS
from text_vision.data_utils import normalize_transcript


SENSITIVE_TAXONOMY = {"hate_speech", "discrimination", "contextual_hate"}


@dataclass
class ModerationReasoningResult:
    hate_types: List[str] = field(default_factory=list)
    target_group: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    category_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", [], {})}


def default_target_group_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "target_groups.json"


def load_target_group_config(path: str | Path | None = None) -> Dict[str, Dict[str, List[str]]]:
    config_path = Path(path) if path else default_target_group_config_path()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        blob = json.load(handle)
    return {
        str(category): {str(group): [str(term) for term in terms] for group, terms in groups.items()}
        for category, groups in dict(blob).items()
    }


def _documents(
    transcript: str = "",
    ocr_text: str = "",
    visual_description: str = "",
    scene_text: str = "",
    evidence: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for source, text in [
        ("asr", transcript),
        ("ocr", ocr_text),
        ("vlm", visual_description),
        ("scene", scene_text),
    ]:
        normalized = normalize_transcript(text)
        if normalized:
            docs.append({"source": source, "text": normalized})
    for item in evidence or []:
        text = normalize_transcript(item.get("text", ""))
        if not text:
            continue
        docs.append({**dict(item), "source": str(item.get("source", "evidence")), "text": text})
    return docs


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip().lower())
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def extract_target_group(
    documents: Sequence[Mapping[str, Any]],
    target_groups: Mapping[str, Mapping[str, Iterable[str]]],
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    matches: Dict[str, List[Dict[str, Any]]] = {}
    for category, groups in target_groups.items():
        for group, terms in groups.items():
            for document in documents:
                text = normalize_transcript(document.get("text", ""))
                if not text:
                    continue
                hit_terms = []
                for term in terms:
                    if term and _term_pattern(str(term)).search(text):
                        hit_terms.append(str(term))
                if hit_terms:
                    matches.setdefault(str(group), []).append(
                        {
                            "source": document.get("source", "unknown"),
                            "text": text,
                            "target_group": str(group),
                            "target_group_category": str(category),
                            "matched_terms": sorted(set(hit_terms)),
                            **{key: value for key, value in document.items() if key not in {"source", "text"}},
                        }
                    )
    if not matches:
        return None, []
    target_group, evidence = max(matches.items(), key=lambda item: sum(len(row["matched_terms"]) for row in item[1]))
    return target_group, evidence


def _selected_taxonomy(
    taxonomy_probs: Mapping[str, float] | None,
    taxonomy_labels: Optional[Sequence[str]] = None,
    thresholds: Mapping[str, float] | None = None,
) -> List[str]:
    labels = list(taxonomy_labels or TAXONOMY_LABELS)
    probs = taxonomy_probs or {}
    selected = []
    for label in labels:
        score = float(probs.get(label, 0.0))
        threshold = float((thresholds or {}).get(label, 0.5))
        if score >= threshold:
            selected.append(label)
    return selected


def reason_about_moderation(
    prob_hate: float | None = None,
    taxonomy_probs: Mapping[str, float] | None = None,
    taxonomy_labels: Optional[Sequence[str]] = None,
    thresholds: Mapping[str, float] | None = None,
    transcript: str = "",
    ocr_text: str = "",
    visual_description: str = "",
    scene_text: str = "",
    evidence: Optional[Sequence[Mapping[str, Any]]] = None,
    target_group_config: str | Path | None = None,
    include_evidence: bool = True,
) -> ModerationReasoningResult:
    selected = _selected_taxonomy(taxonomy_probs, taxonomy_labels, thresholds)
    max_taxonomy = max([float(value) for value in (taxonomy_probs or {}).values()], default=0.0)
    hate_score = float(prob_hate) if prob_hate is not None else 0.0
    if hate_score < 0.35 and max_taxonomy < 0.35:
        return ModerationReasoningResult(
            hate_types=[],
            target_group=None,
            evidence=[],
            explanation="No high-confidence moderation signal was found.",
            confidence=max(hate_score, max_taxonomy),
            category_summary={"selected_taxonomy_count": 0, "max_taxonomy_prob": max_taxonomy},
        )

    docs = _documents(transcript, ocr_text, visual_description, scene_text, evidence)
    target_group = None
    group_evidence: List[Dict[str, Any]] = []
    if selected and (set(selected) & SENSITIVE_TAXONOMY):
        target_group, group_evidence = extract_target_group(docs, load_target_group_config(target_group_config))

    model_evidence = [
        {
            "source": "model",
            "text": f"{label} probability {float((taxonomy_probs or {}).get(label, 0.0)):.3f}",
            "label": label,
            "confidence": float((taxonomy_probs or {}).get(label, 0.0)),
        }
        for label in selected
    ]
    output_evidence: List[Dict[str, Any]] = []
    if include_evidence:
        output_evidence.extend(group_evidence)
        output_evidence.extend(model_evidence)

    if selected:
        explanation = "Selected taxonomy labels: " + ", ".join(selected) + "."
        if target_group:
            explanation += f" Target group found from evidence: {target_group}."
        elif set(selected) & SENSITIVE_TAXONOMY:
            explanation += " No target group was returned because none was grounded in text evidence."
    else:
        explanation = "Binary hate score was elevated, but no taxonomy label passed threshold."

    return ModerationReasoningResult(
        hate_types=selected,
        target_group=target_group,
        evidence=output_evidence,
        explanation=explanation,
        confidence=max(hate_score, max_taxonomy),
        category_summary={
            "selected_taxonomy_count": len(selected),
            "max_taxonomy_prob": max_taxonomy,
            "prob_hate": hate_score,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule-grounded moderation reasoning over text/model outputs.")
    parser.add_argument("--taxonomy-probs-json", default="{}")
    parser.add_argument("--prob-hate", type=float, default=0.0)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--ocr-text", default="")
    parser.add_argument("--visual-description", default="")
    parser.add_argument("--scene-text", default="")
    parser.add_argument("--target-groups", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = reason_about_moderation(
        prob_hate=args.prob_hate,
        taxonomy_probs=json.loads(args.taxonomy_probs_json),
        transcript=args.transcript,
        ocr_text=args.ocr_text,
        visual_description=args.visual_description,
        scene_text=args.scene_text,
        target_group_config=args.target_groups,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

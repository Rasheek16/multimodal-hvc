from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


LABEL_TEMPLATES: Dict[str, List[str]] = {
    "hate_speech": [
        "A post attacks [TARGET_GROUP] using [SLUR] and says they do not belong here.",
        "A speaker uses dehumanizing language toward [TARGET_GROUP] without condemning it.",
    ],
    "discrimination": [
        "A workplace scene shows [TARGET_GROUP] being denied service because of group identity.",
        "A policy discussion advocates excluding [TARGET_GROUP] from housing or jobs.",
    ],
    "contextual_hate": [
        "A video quotes hateful language toward [TARGET_GROUP] and amplifies it approvingly.",
        "A coded meme targets [TARGET_GROUP] through repeated hostile stereotypes.",
    ],
    "threat": [
        "A user directs [THREAT_PHRASE] at [TARGET_GROUP].",
        "A caption threatens to hurt [TARGET_GROUP] at a public event.",
    ],
    "violence": [
        "A clip depicts people encouraging physical violence against [TARGET_GROUP].",
        "A scene celebrates an assault and asks viewers to repeat it.",
    ],
    "fear": [
        "A scene shows a frightened person hiding during a tense confrontation.",
        "A narrator describes panic and intimidation without naming a protected group.",
    ],
    "sexual": [
        "A scene contains sexualized discussion without hate or threats.",
        "A caption describes adult sexual content in non-hateful terms.",
    ],
    "illegal": [
        "A tutorial encourages illegal activity and gives operational steps.",
        "A user promotes buying prohibited goods through an online contact.",
    ],
    "online_harm": [
        "A post encourages coordinated harassment of a private person.",
        "A user asks followers to mass-report and dox a target.",
    ],
}

HARD_NEGATIVES = [
    {
        "text": "A news report quotes a hateful phrase using [SLUR] only to condemn it and explain the harm.",
        "labels": [],
        "kind": "news_reporting",
    },
    {
        "text": "A counterspeech video tells viewers not to attack [TARGET_GROUP] and rejects discrimination.",
        "labels": [],
        "kind": "counterspeech",
    },
    {
        "text": "An educational lecture discusses the history of hate speech with placeholders like [SLUR].",
        "labels": [],
        "kind": "educational",
    },
    {
        "text": "A satire scene uses exaggerated ambiguity but does not target a real group.",
        "labels": [],
        "kind": "satire",
    },
]


def examples_per_label(count_per_label: int) -> Iterable[dict]:
    for label, templates in LABEL_TEMPLATES.items():
        for index in range(int(count_per_label)):
            template = templates[index % len(templates)]
            yield {
                "text": template,
                "labels": [label],
                "source": "synthetic_template",
                "kind": "positive",
            }


def generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(examples_per_label(args.count_per_label))
    rows.extend(HARD_NEGATIVES * max(1, int(args.hard_negative_repeats)))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate safe placeholder-based text taxonomy examples.")
    parser.add_argument("--output", default="data/synthetic_taxonomy_examples.jsonl")
    parser.add_argument("--count-per-label", type=int, default=20)
    parser.add_argument("--hard-negative-repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()

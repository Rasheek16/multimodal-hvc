from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import LABEL_COLUMNS, default_dataset_path, ensure_label_columns


RARE_LABELS = ["threat", "illegal", "online_harm"]
PREFIXES = [
    "The scene shows",
    "The clip depicts",
    "The description indicates",
    "The segment contains",
    "The action appears to include",
]
NEUTRAL_HARD_NEGATIVES = {
    "threat": [
        "The scene discusses threats in a news or educational context without anyone making a direct threat.",
        "Characters mention danger as part of fictional exposition, but no explicit threat is made.",
    ],
    "illegal": [
        "The scene mentions crime in a historical or educational context without depicting illegal action.",
        "A legal discussion references weapons or drugs without promoting illegal behavior.",
    ],
    "online_harm": [
        "The scene shows people using social media without harassment or harmful targeting.",
        "A comment thread is visible, but it does not include abuse, doxxing, or coordinated harassment.",
    ],
}


def row_text(row: pd.Series) -> str:
    parts = []
    for column in ["Scene", "Action", "Subcategories", "ParentLabels", "action_clean"]:
        value = str(row.get(column, "") or "").strip()
        if value:
            parts.append(value)
    return " ".join(parts)


def paraphrase_row(row: pd.Series, label: str, index: int, rng: random.Random) -> Dict[str, Any]:
    payload = row.to_dict()
    base = row_text(row)
    prefix = rng.choice(PREFIXES)
    payload["Scene"] = f"{prefix} {base}".strip()
    payload["Action"] = str(row.get("Action", "") or "")
    payload["Subcategories"] = str(row.get("Subcategories", "") or "")
    payload["ParentLabels"] = str(row.get("ParentLabels", "") or "")
    payload["action_clean"] = str(row.get("action_clean", "") or "")
    payload["final_labels"] = str([name for name in LABEL_COLUMNS if int(row.get(name, 0)) == 1])
    payload["File Name"] = f"synthetic_{label}_{int(row.name)}_{index}.mp4"
    payload["synthetic"] = 1
    payload["synthetic_source_label"] = label
    payload["synthetic_source_row"] = int(row.name)
    return payload


def hard_negative(label: str, index: int, text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "Start Time": "00:00:00",
        "End Time": "00:00:05",
        "Action": "neutral context",
        "Subcategories": "['hard negative']",
        "Scene": text,
        "File Name": f"synthetic_hard_negative_{label}_{index}.mp4",
        "ParentLabels": "['neutral']",
        "action_clean": "neutral context",
        "final_labels": "['neutral']",
        "neutral": 1,
        "synthetic": 1,
        "synthetic_source_label": label,
        "synthetic_source_row": -1,
    }
    for name in LABEL_COLUMNS:
        payload[name] = 0
    return payload


def augment(args: argparse.Namespace) -> None:
    rng = random.Random(int(args.seed))
    frame = ensure_label_columns(pd.read_csv(args.dataset))
    synthetic_values = frame["synthetic"] if "synthetic" in frame.columns else pd.Series(0, index=frame.index)
    frame["synthetic"] = pd.to_numeric(synthetic_values, errors="coerce").fillna(0).astype(int)
    rows: List[Dict[str, Any]] = []
    for label in RARE_LABELS:
        positives = frame[frame[label].astype(int).eq(1)]
        for _, row in positives.iterrows():
            count = rng.randint(int(args.min_paraphrases), int(args.max_paraphrases))
            for index in range(count):
                rows.append(paraphrase_row(row, label, index, rng))
        for index, text in enumerate(NEUTRAL_HARD_NEGATIVES[label]):
            rows.append(hard_negative(label, index, text))
    augmented = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output, index=False)
    summary = {
        "input_rows": int(len(frame)),
        "synthetic_rows_added": int(len(rows)),
        "output_rows": int(len(augmented)),
        "output": str(output),
    }
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative rare-label augmentation for taxonomy training.")
    parser.add_argument("--dataset", default=str(default_dataset_path()))
    parser.add_argument("--output", default="data/multilabel_augmented.csv")
    parser.add_argument("--min-paraphrases", type=int, default=5)
    parser.add_argument("--max-paraphrases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    augment(parse_args())


if __name__ == "__main__":
    main()

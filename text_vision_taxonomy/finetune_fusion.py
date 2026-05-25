from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.train_text_guided_vision import build_arg_parser, run_training


def main() -> None:
    parser = build_arg_parser("Fine-tune fusion and heads with severity-boosted pseudo labels.")
    parser.set_defaults(epochs=3, freeze_epochs=999, severity_ce_weight=0.20)
    args = parser.parse_args()
    run_training(args, fine_tune=True)


if __name__ == "__main__":
    main()


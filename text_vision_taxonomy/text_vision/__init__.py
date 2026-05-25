from __future__ import annotations

from pathlib import Path

# This isolated workspace is named text_vision_taxonomy, while the existing
# scripts import text_vision.*. Make this local package resolve submodules from
# the workspace root so commands do not fall through to E:\text_vision.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in __path__:
    __path__.append(str(_ROOT))

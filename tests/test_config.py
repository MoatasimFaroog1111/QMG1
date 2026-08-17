from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import HORIZONS_HOURS  # noqa: E402


def test_requested_forecast_horizons_are_exact() -> None:
    assert HORIZONS_HOURS == (2, 4, 8, 12, 24, 72, 168, 360, 720)

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.ml.targets import CalendarHorizonTargetBuilder  # noqa: E402


def test_calendar_horizon_uses_elapsed_time_and_next_available_quote() -> None:
    index = pd.to_datetime(
        [
            "2025-01-03T20:00:00Z",
            "2025-01-03T21:00:00Z",
            "2025-01-05T22:00:00Z",
            "2025-01-05T23:00:00Z",
        ],
        utc=True,
    )
    hourly = pd.DataFrame(
        {
            "open": [100.0, 101.0, 110.0, 111.0],
            "high": [101.0, 102.0, 111.0, 112.0],
            "low": [99.0, 100.0, 109.0, 110.0],
            "close": [100.5, 101.5, 110.5, 111.5],
            "volume": [1.0, 1.0, 1.0, 1.0],
            "minute_count": [60, 60, 60, 60],
        },
        index=index,
    )

    result = CalendarHorizonTargetBuilder(max_forward_tolerance_hours=72).attach(
        hourly.copy(), hourly, horizon_hours=2
    )

    # Friday 21:00 + 2 elapsed hours is Friday 23:00. The market data has no
    # weekend quote, so the first available quote at/after the requested time
    # is Sunday 22:00, not merely the second following dataframe row.
    row = result.loc[pd.Timestamp("2025-01-03T21:00:00Z")]
    assert row["future_close_2h"] == 110.5
    assert row["target_timestamp_2h"] == pd.Timestamp("2025-01-05T22:00:00Z")

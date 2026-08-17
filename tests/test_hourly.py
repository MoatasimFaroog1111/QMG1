from __future__ import annotations

from pathlib import Path

import pandas as pd

from qmg1.data.hourly import HourlyUsdPerKgBuilder
from qmg1.ml.dataset import ForecastDatasetBuilder


def test_hourly_builder_compacts_m1_and_preserves_minute_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1735689600000,31.1034768,31.414511568,30.792442032,31.1034768,2\n"
        "1735689660000,31.1034768,31.725546336,30.481407264,31.414511568,3\n",
        encoding="utf-8",
    )

    hourly = HourlyUsdPerKgBuilder.build(raw)
    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["minute_count"] == 2
    assert row["open"] == 1000.0
    assert row["close"] == 1010.0
    assert row["volume"] == 5.0


def test_hourly_feature_base_loads_without_second_resample(tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01", periods=900, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 1000.0,
            "high": 1002.0,
            "low": 998.0,
            "close": 1001.0,
            "volume": 60.0,
            "minute_count": 60,
        },
        index=index,
    )
    path = tmp_path / "hourly.csv"
    frame.to_csv(path, index_label="timestamp_utc")

    base = ForecastDatasetBuilder().load_hourly_feature_base(str(path))
    assert len(base.hourly) == 900
    assert base.hourly.iloc[0]["minute_count"] == 60
    assert "volatility_720h" in base.features.columns

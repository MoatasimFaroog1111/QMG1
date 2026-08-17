from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_constant_zero_volume_does_not_erase_training_dataset(tmp_path: Path) -> None:
    index = pd.date_range("2020-01-01", periods=6_500, freq="h", tz="UTC")
    trend = 1000.0 + np.linspace(0.0, 250.0, len(index))
    frame = pd.DataFrame(
        {
            "open": trend,
            "high": trend + 2.0,
            "low": trend - 2.0,
            "close": trend + np.sin(np.arange(len(index)) / 11.0),
            "volume": 0.0,
            "minute_count": 60,
        },
        index=index,
    )
    path = tmp_path / "zero_volume_hourly.csv"
    frame.to_csv(path, index_label="timestamp_utc")

    builder = ForecastDatasetBuilder()
    base = builder.load_hourly_feature_base(str(path))
    after_warmup = base.features.iloc[800:]
    assert after_warmup["volume_zscore_24h"].eq(0.0).all()
    assert after_warmup["volume_zscore_168h"].eq(0.0).all()

    prepared = builder.build_from_base(base, horizon_hours=2)
    assert len(prepared.frame) > 5_000

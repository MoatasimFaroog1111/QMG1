from __future__ import annotations

from pathlib import Path

from qmg1.data.market_series import HourlyNativeSeriesBuilder, US_DOLLAR_INDEX


def test_native_market_builder_does_not_apply_metal_unit_conversion(tmp_path: Path) -> None:
    raw = tmp_path / "udx_m1.csv"
    raw.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1735689600000,100.0,100.2,99.8,100.1,0\n"
        "1735689660000,100.1,100.3,99.9,100.2,0\n",
        encoding="utf-8",
    )

    hourly = HourlyNativeSeriesBuilder.build(raw)

    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 100.3
    assert row["low"] == 99.8
    assert row["close"] == 100.2
    assert row["minute_count"] == 2
    assert US_DOLLAR_INDEX.histdata_pair == "UDXUSD"
    assert US_DOLLAR_INDEX.unit == "index_points"

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from qmg1.ml.dataset import ForecastDatasetBuilder
from qmg1.ml.exogenous import (
    GoldSilverFeatureProvider,
    SpxFeatureProvider,
    UsdIndexFeatureProvider,
    WtiFeatureProvider,
)


def _hourly_frame(index: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [0.0] * len(close),
            "minute_count": [60] * len(close),
        },
        index=index,
    )


def test_gold_alignment_is_backward_only() -> None:
    silver_index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    gold_index = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T02:00:00Z")]
    )
    silver = _hourly_frame(silver_index, [20.0, 21.0, 22.0])
    gold = _hourly_frame(gold_index, [2000.0, 2200.0])

    provider = GoldSilverFeatureProvider(gold)
    augmented = provider.augment(pd.DataFrame(index=silver.index), silver)

    assert augmented.loc[silver_index[1], "gold_close_usd_per_kg"] == 2000.0
    assert augmented.loc[silver_index[2], "gold_close_usd_per_kg"] == 2200.0
    assert augmented.loc[silver_index[1], "gold_source_age_hours"] == 1.0
    assert augmented.loc[silver_index[2], "gold_source_age_hours"] == 0.0
    assert math.isclose(
        augmented.loc[silver_index[1], "gold_silver_ratio"], 2000.0 / 21.0
    )


def test_udx_alignment_is_backward_only() -> None:
    silver_index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    source_index = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T02:00:00Z")]
    )
    silver = _hourly_frame(silver_index, [20.0, 21.0, 22.0])
    udx = _hourly_frame(source_index, [100.0, 101.0])

    augmented = UsdIndexFeatureProvider(udx).augment(
        pd.DataFrame(index=silver.index), silver
    )

    assert augmented.loc[silver_index[1], "udx_close"] == 100.0
    assert augmented.loc[silver_index[2], "udx_close"] == 101.0
    assert augmented.loc[silver_index[1], "udx_source_age_hours"] == 1.0
    assert augmented.loc[silver_index[2], "udx_source_age_hours"] == 0.0


def test_spx_and_wti_alignment_are_backward_only() -> None:
    silver_index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    source_index = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T02:00:00Z")]
    )
    silver = _hourly_frame(silver_index, [20.0, 21.0, 22.0])
    spx = _hourly_frame(source_index, [5000.0, 5100.0])
    wti = _hourly_frame(source_index, [70.0, 75.0])

    spx_augmented = SpxFeatureProvider(spx).augment(
        pd.DataFrame(index=silver.index), silver
    )
    wti_augmented = WtiFeatureProvider(wti).augment(
        pd.DataFrame(index=silver.index), silver
    )

    assert spx_augmented.loc[silver_index[1], "spx_close"] == 5000.0
    assert spx_augmented.loc[silver_index[2], "spx_close"] == 5100.0
    assert spx_augmented.loc[silver_index[1], "spx_source_age_hours"] == 1.0
    assert wti_augmented.loc[silver_index[1], "wti_close"] == 70.0
    assert wti_augmented.loc[silver_index[2], "wti_close"] == 75.0
    assert wti_augmented.loc[silver_index[1], "wti_source_age_hours"] == 1.0


def test_hourly_dataset_builder_adds_cross_market_features(tmp_path: Path) -> None:
    index = pd.date_range("2023-01-01", periods=1_000, freq="h", tz="UTC")
    silver = _hourly_frame(index, [20.0 + i * 0.002 for i in range(len(index))])
    gold = _hourly_frame(index, [1800.0 + i * 0.02 for i in range(len(index))])
    udx = _hourly_frame(index, [100.0 + i * 0.001 for i in range(len(index))])
    spx = _hourly_frame(index, [4000.0 + i * 0.1 for i in range(len(index))])
    wti = _hourly_frame(index, [70.0 + i * 0.001 for i in range(len(index))])

    silver_path = tmp_path / "silver.csv"
    silver.to_csv(silver_path, index_label="timestamp_utc")
    providers = [
        GoldSilverFeatureProvider(gold, source_file="gold.csv"),
        UsdIndexFeatureProvider(udx, source_file="udx.csv"),
        SpxFeatureProvider(spx, source_file="spx.csv"),
        WtiFeatureProvider(wti, source_file="wti.csv"),
    ]

    base = ForecastDatasetBuilder(
        exogenous_providers=providers
    ).load_hourly_feature_base(str(silver_path))

    for column in (
        "gold_log_return_24h",
        "gold_silver_ratio",
        "udx_log_return_24h",
        "usd_pressure_24h",
        "spx_log_return_24h",
        "silver_minus_spx_return_168h",
        "wti_log_return_24h",
        "silver_minus_wti_return_168h",
    ):
        assert column in base.features.columns

    for column in (
        "gold_source_age_hours",
        "udx_source_age_hours",
        "spx_source_age_hours",
        "wti_source_age_hours",
    ):
        assert base.features[column].dropna().max() == 0.0

    metadata = [provider.metadata() for provider in providers]
    assert all(item["alignment"] == "backward_asof" for item in metadata)
    assert all(item["future_quotes_allowed"] is False for item in metadata)
    assert metadata[2]["source_symbol"] == "SPX/USD"
    assert metadata[3]["source_unit"] == "usd_per_barrel"

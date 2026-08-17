from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from qmg1.ml.dataset import ForecastDatasetBuilder
from qmg1.ml.exogenous import GoldSilverFeatureProvider


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
    base_features = pd.DataFrame(index=silver.index)
    augmented = provider.augment(base_features, silver)

    assert augmented.loc[silver_index[1], "gold_close_usd_per_kg"] == 2000.0
    assert augmented.loc[silver_index[2], "gold_close_usd_per_kg"] == 2200.0
    assert augmented.loc[silver_index[1], "gold_source_age_hours"] == 1.0
    assert augmented.loc[silver_index[2], "gold_source_age_hours"] == 0.0
    assert math.isclose(
        augmented.loc[silver_index[1], "gold_silver_ratio"],
        2000.0 / 21.0,
    )


def test_hourly_dataset_builder_adds_gold_silver_features(tmp_path: Path) -> None:
    index = pd.date_range("2023-01-01", periods=1_000, freq="h", tz="UTC")
    silver_close = [20.0 + i * 0.002 for i in range(len(index))]
    gold_close = [1800.0 + i * 0.02 for i in range(len(index))]
    silver = _hourly_frame(index, silver_close)
    gold = _hourly_frame(index, gold_close)

    silver_path = tmp_path / "silver.csv"
    silver.to_csv(silver_path, index_label="timestamp_utc")
    provider = GoldSilverFeatureProvider(gold, source_file="gold.csv")

    base = ForecastDatasetBuilder(
        exogenous_providers=[provider]
    ).load_hourly_feature_base(str(silver_path))

    assert "gold_log_return_24h" in base.features.columns
    assert "gold_silver_ratio" in base.features.columns
    assert "gold_minus_silver_return_168h" in base.features.columns
    assert base.features["gold_source_age_hours"].dropna().max() == 0.0
    metadata = provider.metadata()
    assert metadata["alignment"] == "backward_asof"
    assert metadata["future_quotes_allowed"] is False

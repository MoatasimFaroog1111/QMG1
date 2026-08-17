from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.features import build_features, feature_columns, resample_to_hourly  # noqa: E402


def _sample_hourly() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=60 * 900, freq="min", tz="UTC")
    base = np.linspace(1000.0, 1200.0, len(idx))
    m1 = pd.DataFrame(
        {
            "open_usd_per_kg": base,
            "high_usd_per_kg": base + 2.0,
            "low_usd_per_kg": base - 2.0,
            "close_usd_per_kg": base + 0.5,
            "volume_source_units": 1.0,
        },
        index=idx,
    )
    return resample_to_hourly(m1)


def test_resample_and_features_are_shape_safe() -> None:
    hourly = _sample_hourly()
    features = build_features(hourly)
    assert len(hourly) == 900
    assert "rsi_14" in features.columns
    assert "atr_14_pct" in features.columns
    assert "volatility_720h" in features.columns
    assert features.index.is_monotonic_increasing


def test_future_and_target_columns_never_enter_feature_set() -> None:
    features = build_features(_sample_hourly())
    features["target_24h"] = 0.1
    features["future_close_24h"] = 1234.0
    cols = feature_columns(features)
    assert "target_24h" not in cols
    assert "future_close_24h" not in cols

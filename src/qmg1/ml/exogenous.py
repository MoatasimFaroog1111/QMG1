from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


GOLD_SILVER_PROVIDER_NAME = "gold_silver_cross_market"
USD_INDEX_PROVIDER_NAME = "usd_index_cross_market"
SPX_PROVIDER_NAME = "spx_cross_market"
WTI_PROVIDER_NAME = "wti_cross_market"
CROSS_MARKET_HORIZONS: tuple[int, ...] = (1, 4, 24, 72, 168)


class ExogenousFeatureProvider(Protocol):
    name: str

    def augment(
        self,
        features: pd.DataFrame,
        target_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        ...

    def metadata(self) -> dict[str, object]:
        ...


def _calendar_log_return(series: pd.Series, horizon_hours: int) -> pd.Series:
    """Causal log return versus the last observation at/before t - horizon."""
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(index=series.index, dtype=float)

    left = pd.DataFrame(
        {
            "_position": np.arange(len(clean)),
            "_query_time": clean.index - pd.Timedelta(int(horizon_hours), unit="h"),
        }
    ).sort_values("_query_time")
    right = pd.DataFrame(
        {
            "_source_time": clean.index,
            "_lag_price": clean.to_numpy(dtype=float),
        }
    ).sort_values("_source_time")
    matched = pd.merge_asof(
        left,
        right,
        left_on="_query_time",
        right_on="_source_time",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_position")

    current = clean.to_numpy(dtype=float)
    lagged = matched["_lag_price"].to_numpy(dtype=float)
    values = np.log(current) - np.log(lagged)
    result = pd.Series(values, index=clean.index, dtype=float)
    return result.reindex(series.index)


def _rolling_zscore(series: pd.Series, window_hours: int) -> pd.Series:
    window = f"{window_hours}h"
    minimum = max(4, min(24, window_hours // 2))
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std()
    zscore = (series - mean) / std.replace(0.0, np.nan)
    ready = mean.notna() & std.eq(0.0)
    return zscore.mask(ready, 0.0)


def _normalize_hourly(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "close" not in frame.columns:
        raise ValueError(f"Missing {label} hourly close column")
    normalized = frame.copy().sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    return normalized


class GoldSilverFeatureProvider:
    name = GOLD_SILVER_PROVIDER_NAME

    def __init__(self, gold_hourly: pd.DataFrame) -> None:
        self.gold_hourly = _normalize_hourly(gold_hourly, "gold")

    @classmethod
    def from_hourly_csv(cls, path: str) -> "GoldSilverFeatureProvider":
        frame = pd.read_csv(path)
        if "timestamp_utc" not in frame.columns:
            raise ValueError("Gold hourly dataset is missing timestamp_utc")
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        close_column = "close_usd_per_kg" if "close_usd_per_kg" in frame.columns else "close"
        frame = frame.set_index("timestamp_utc").rename(columns={close_column: "close"})
        return cls(frame[["close"]])

    def augment(
        self,
        features: pd.DataFrame,
        target_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        result = features.copy()
        gold = self.gold_hourly["close"].reindex(result.index, method="ffill")
        silver = pd.to_numeric(target_hourly["close"], errors="coerce").reindex(result.index)

        result["gold_close"] = gold
        ratio = gold / silver.replace(0.0, np.nan)
        result["gold_silver_ratio"] = ratio

        for horizon in CROSS_MARKET_HORIZONS:
            gold_return = _calendar_log_return(gold, horizon)
            ratio_return = _calendar_log_return(ratio, horizon)
            silver_return = _calendar_log_return(silver, horizon)
            result[f"gold_return_{horizon}h"] = gold_return
            result[f"gold_silver_ratio_return_{horizon}h"] = ratio_return
            result[f"gold_minus_silver_return_{horizon}h"] = gold_return - silver_return

        result["gold_silver_ratio_z_168h"] = _rolling_zscore(ratio, 168)
        return result

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "source": "gold H1", "causal_alignment": "backward"}


class _SingleMarketFeatureProvider:
    name = "single_market"
    prefix = "market"

    def __init__(self, hourly: pd.DataFrame) -> None:
        self.hourly = _normalize_hourly(hourly, self.prefix)

    def augment(
        self,
        features: pd.DataFrame,
        target_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        result = features.copy()
        context = self.hourly["close"].reindex(result.index, method="ffill")
        silver = pd.to_numeric(target_hourly["close"], errors="coerce").reindex(result.index)
        result[f"{self.prefix}_close"] = context
        for horizon in CROSS_MARKET_HORIZONS:
            context_return = _calendar_log_return(context, horizon)
            silver_return = _calendar_log_return(silver, horizon)
            result[f"{self.prefix}_return_{horizon}h"] = context_return
            result[f"{self.prefix}_minus_silver_return_{horizon}h"] = (
                context_return - silver_return
            )
        result[f"{self.prefix}_return_z_168h"] = _rolling_zscore(
            result[f"{self.prefix}_return_24h"], 168
        )
        return result

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "source": self.prefix, "causal_alignment": "backward"}

    @classmethod
    def _from_hourly_csv(cls, path: str):
        frame = pd.read_csv(path)
        if "timestamp_utc" not in frame.columns:
            raise ValueError(f"{cls.prefix} hourly dataset is missing timestamp_utc")
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        close_column = "close_native" if "close_native" in frame.columns else "close"
        frame = frame.set_index("timestamp_utc").rename(columns={close_column: "close"})
        return cls(frame[["close"]])


class UsdIndexFeatureProvider(_SingleMarketFeatureProvider):
    name = USD_INDEX_PROVIDER_NAME
    prefix = "udx"

    @classmethod
    def from_hourly_csv(cls, path: str) -> "UsdIndexFeatureProvider":
        return cls._from_hourly_csv(path)


class SpxFeatureProvider(_SingleMarketFeatureProvider):
    name = SPX_PROVIDER_NAME
    prefix = "spx"

    @classmethod
    def from_hourly_csv(cls, path: str) -> "SpxFeatureProvider":
        return cls._from_hourly_csv(path)


class WtiFeatureProvider(_SingleMarketFeatureProvider):
    name = WTI_PROVIDER_NAME
    prefix = "wti"

    @classmethod
    def from_hourly_csv(cls, path: str) -> "WtiFeatureProvider":
        return cls._from_hourly_csv(path)

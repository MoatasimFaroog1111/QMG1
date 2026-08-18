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
    normalized = normalized.dropna(subset=["close"])
    if normalized.empty:
        raise ValueError(f"{label} hourly dataset has no usable close prices")
    if normalized.index.tz is None:
        normalized.index = pd.DatetimeIndex(normalized.index).tz_localize("UTC")
    else:
        normalized.index = pd.DatetimeIndex(normalized.index).tz_convert("UTC")
    return normalized


def _align_backward(
    target_index: pd.DatetimeIndex,
    source_features: pd.DataFrame,
    source_time_column: str,
) -> pd.DataFrame:
    left = pd.DataFrame({"_target_time": target_index})
    right = source_features.copy()
    right[source_time_column] = right.index
    right = right.reset_index(drop=True)
    aligned = pd.merge_asof(
        left.sort_values("_target_time"),
        right.sort_values(source_time_column),
        left_on="_target_time",
        right_on=source_time_column,
        direction="backward",
        allow_exact_matches=True,
    )
    aligned.index = target_index
    return aligned


def _source_age_hours(
    target_index: pd.DatetimeIndex,
    aligned: pd.DataFrame,
    source_time_column: str,
) -> pd.Series:
    source_time = pd.to_datetime(
        aligned[source_time_column], utc=True, errors="coerce"
    )
    source_time.index = target_index
    target_time = pd.Series(target_index, index=target_index)
    return (target_time - source_time).dt.total_seconds() / 3600.0


def _load_hourly_csv(path: str | Path) -> tuple[pd.DataFrame, str]:
    source = Path(path)
    frame = pd.read_csv(source, parse_dates=["timestamp_utc"])
    return frame.set_index("timestamp_utc").sort_index(), source.name


class GoldSilverFeatureProvider:
    """Causal XAU/USD context for a silver target series."""

    name = GOLD_SILVER_PROVIDER_NAME

    def __init__(self, gold_hourly: pd.DataFrame, source_file: str = "") -> None:
        self.gold_hourly = _normalize_hourly(gold_hourly, "Gold")
        self.source_file = source_file

    @classmethod
    def from_hourly_csv(cls, path: str | Path) -> "GoldSilverFeatureProvider":
        frame, source_file = _load_hourly_csv(path)
        return cls(frame, source_file=source_file)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_symbol": "XAU/USD",
            "source_file": self.source_file,
            "alignment": "backward_asof",
            "future_quotes_allowed": False,
        }

    def _gold_features(self) -> pd.DataFrame:
        close = self.gold_hourly["close"]
        gold = pd.DataFrame(index=close.index)
        gold["gold_close_usd_per_kg"] = close
        for horizon in CROSS_MARKET_HORIZONS:
            gold[f"gold_log_return_{horizon}h"] = _calendar_log_return(close, horizon)
        gold["gold_close_zscore_24h"] = _rolling_zscore(close, 24)
        gold["gold_close_zscore_168h"] = _rolling_zscore(close, 168)
        return gold

    def augment(
        self,
        features: pd.DataFrame,
        target_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        if len(features.index) == 0 or target_hourly.empty:
            return features.copy()

        target = _normalize_hourly(target_hourly, "Target")
        aligned = _align_backward(
            pd.DatetimeIndex(target.index), self._gold_features(), "_gold_time"
        )
        result = features.reindex(target.index).copy()
        for column in aligned.columns:
            if column not in {"_target_time", "_gold_time"}:
                result[column] = pd.to_numeric(aligned[column], errors="coerce").to_numpy()
        result["gold_source_age_hours"] = _source_age_hours(
            pd.DatetimeIndex(target.index), aligned, "_gold_time"
        )

        silver_close = pd.to_numeric(target["close"], errors="coerce")
        ratio = result["gold_close_usd_per_kg"] / silver_close
        result["gold_silver_ratio"] = ratio
        result["gold_silver_ratio_zscore_24h"] = _rolling_zscore(ratio, 24)
        result["gold_silver_ratio_zscore_168h"] = _rolling_zscore(ratio, 168)
        for horizon in CROSS_MARKET_HORIZONS:
            ratio_return = _calendar_log_return(ratio, horizon)
            silver_return = _calendar_log_return(silver_close, horizon)
            result[f"gold_silver_ratio_log_return_{horizon}h"] = ratio_return
            result[f"gold_minus_silver_return_{horizon}h"] = (
                result[f"gold_log_return_{horizon}h"] - silver_return
            )
        return result.replace([np.inf, -np.inf], np.nan)


class UsdIndexFeatureProvider:
    """Causal UDX/USD (US Dollar Index) context for silver forecasts."""

    name = USD_INDEX_PROVIDER_NAME

    def __init__(self, udx_hourly: pd.DataFrame, source_file: str = "") -> None:
        self.udx_hourly = _normalize_hourly(udx_hourly, "US Dollar Index")
        self.source_file = source_file

    @classmethod
    def from_hourly_csv(cls, path: str | Path) -> "UsdIndexFeatureProvider":
        frame, source_file = _load_hourly_csv(path)
        return cls(frame, source_file=source_file)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_symbol": "UDX/USD",
            "source_file": self.source_file,
            "source_unit": "index_points",
            "alignment": "backward_asof",
            "future_quotes_allowed": False,
        }

    def _udx_features(self) -> pd.DataFrame:
        close = self.udx_hourly["close"]
        udx = pd.DataFrame(index=close.index)
        udx["udx_close"] = close
        for horizon in CROSS_MARKET_HORIZONS:
            ret = _calendar_log_return(close, horizon)
            udx[f"udx_log_return_{horizon}h"] = ret
            udx[f"usd_pressure_{horizon}h"] = -ret
        udx["udx_close_zscore_24h"] = _rolling_zscore(close, 24)
        udx["udx_close_zscore_168h"] = _rolling_zscore(close, 168)
        return udx

    def augment(
        self,
        features: pd.DataFrame,
        target_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        if len(features.index) == 0 or target_hourly.empty:
            return features.copy()
        target = _normalize_hourly(target_hourly, "Target")
        aligned = _align_backward(
            pd.DatetimeIndex(target.index), self._udx_features(), "_udx_time"
        )
        result = features.reindex(target.index).copy()
        for column in aligned.columns:
            if column not in {"_target_time", "_udx_time"}:
                result[column] = pd.to_numeric(aligned[column], errors="coerce").to_numpy()
        result["udx_source_age_hours"] = _source_age_hours(
            pd.DatetimeIndex(target.index), aligned, "_udx_time"
        )
        silver_close = pd.to_numeric(target["close"], errors="coerce")
        for horizon in CROSS_MARKET_HORIZONS:
            silver_return = _calendar_log_return(silver_close, horizon)
            result[f"silver_minus_udx_return_{horizon}h"] = (
                silver_return - result[f"udx_log_return_{horizon}h"]
            )
        return result.replace([np.inf, -np.inf], np.nan)


class NativeMarketFeatureProvider:
    """Reusable causal native-market context aligned backward to the target clock."""

    def __init__(
        self,
        hourly: pd.DataFrame,
        *,
        name: str,
        prefix: str,
        source_symbol: str,
        source_unit: str,
        label: str,
        source_file: str = "",
    ) -> None:
        self.hourly = _normalize_hourly(hourly, label)
        self.name = name
        self.prefix = prefix
        self.source_symbol = source_symbol
        self.source_unit = source_unit
        self.label = label
        self.source_file = source_file

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_symbol": self.source_symbol,
            "source_file": self.source_file,
            "source_unit": self.source_unit,
            "alignment": "backward_asof",
            "future_quotes_allowed": False,
        }

    def _source_features(self) -> pd.DataFrame:
        close = self.hourly["close"]
        frame = pd.DataFrame(index=close.index)
        frame[f"{self.prefix}_close"] = close
        for horizon in CROSS_MARKET_HORIZONS:
            frame[f"{self.prefix}_log_return_{horizon}h"] = _calendar_log_return(
                close, horizon
            )
        frame[f"{self.prefix}_close_zscore_24h"] = _rolling_zscore(close, 24)
        frame[f"{self.prefix}_close_zscore_168h"] = _rolling_zscore(close, 168)
        return frame

    def augment(
        self,
        features: pd.DataFrame,
        target_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        if len(features.index) == 0 or target_hourly.empty:
            return features.copy()
        target = _normalize_hourly(target_hourly, "Target")
        source_time = f"_{self.prefix}_time"
        aligned = _align_backward(
            pd.DatetimeIndex(target.index), self._source_features(), source_time
        )
        result = features.reindex(target.index).copy()
        for column in aligned.columns:
            if column not in {"_target_time", source_time}:
                result[column] = pd.to_numeric(aligned[column], errors="coerce").to_numpy()
        result[f"{self.prefix}_source_age_hours"] = _source_age_hours(
            pd.DatetimeIndex(target.index), aligned, source_time
        )
        target_close = pd.to_numeric(target["close"], errors="coerce")
        for horizon in CROSS_MARKET_HORIZONS:
            target_return = _calendar_log_return(target_close, horizon)
            result[f"silver_minus_{self.prefix}_return_{horizon}h"] = (
                target_return - result[f"{self.prefix}_log_return_{horizon}h"]
            )
        return result.replace([np.inf, -np.inf], np.nan)


class SpxFeatureProvider(NativeMarketFeatureProvider):
    name = SPX_PROVIDER_NAME

    def __init__(self, hourly: pd.DataFrame, source_file: str = "") -> None:
        super().__init__(
            hourly,
            name=self.name,
            prefix="spx",
            source_symbol="SPX/USD",
            source_unit="index_points",
            label="S&P 500",
            source_file=source_file,
        )

    @classmethod
    def from_hourly_csv(cls, path: str | Path) -> "SpxFeatureProvider":
        frame, source_file = _load_hourly_csv(path)
        return cls(frame, source_file=source_file)


class WtiFeatureProvider(NativeMarketFeatureProvider):
    name = WTI_PROVIDER_NAME

    def __init__(self, hourly: pd.DataFrame, source_file: str = "") -> None:
        super().__init__(
            hourly,
            name=self.name,
            prefix="wti",
            source_symbol="WTI/USD",
            source_unit="usd_per_barrel",
            label="WTI Crude Oil",
            source_file=source_file,
        )

    @classmethod
    def from_hourly_csv(cls, path: str | Path) -> "WtiFeatureProvider":
        frame, source_file = _load_hourly_csv(path)
        return cls(frame, source_file=source_file)

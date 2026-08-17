from __future__ import annotations

import numpy as np
import pandas as pd


BASE_PRICE_COLUMNS = [
    "open_usd_per_kg",
    "high_usd_per_kg",
    "low_usd_per_kg",
    "close_usd_per_kg",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def load_m1_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    missing = [c for c in BASE_PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.set_index("timestamp_utc").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in BASE_PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume_source_units" in df.columns:
        df["volume_source_units"] = pd.to_numeric(
            df["volume_source_units"], errors="coerce"
        )
    else:
        df["volume_source_units"] = 0.0

    return df.dropna(subset=BASE_PRICE_COLUMNS)


def resample_to_hourly(m1: pd.DataFrame) -> pd.DataFrame:
    hourly = pd.DataFrame(
        {
            "open": m1["open_usd_per_kg"].resample("1h").first(),
            "high": m1["high_usd_per_kg"].resample("1h").max(),
            "low": m1["low_usd_per_kg"].resample("1h").min(),
            "close": m1["close_usd_per_kg"].resample("1h").last(),
            "volume": m1["volume_source_units"].resample("1h").sum(min_count=1),
            "minute_count": m1["close_usd_per_kg"].resample("1h").count(),
        }
    )
    return hourly.dropna(subset=["open", "high", "low", "close"])


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    close = df["close"]

    log_close = np.log(close)
    log_ret_1h = log_close.diff()
    df["ret_1h"] = log_ret_1h

    for lag in (2, 3, 6, 12, 18, 24, 48, 72, 168):
        df[f"log_return_{lag}h"] = log_close.diff(lag)

    for window in (6, 12, 24, 48, 72, 168, 336, 720):
        rolling = close.rolling(window, min_periods=window)
        df[f"sma_ratio_{window}h"] = close / rolling.mean() - 1.0
        df[f"volatility_{window}h"] = log_ret_1h.rolling(
            window, min_periods=window
        ).std()
        df[f"range_ratio_{window}h"] = (
            df["high"].rolling(window, min_periods=window).max()
            / df["low"].rolling(window, min_periods=window).min()
            - 1.0
        )

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["ema12_ratio"] = close / ema12 - 1.0
    df["ema26_ratio"] = close / ema26 - 1.0
    df["macd_pct"] = macd / close
    df["macd_signal_pct"] = signal / close
    df["macd_hist_pct"] = (macd - signal) / close

    df["rsi_14"] = _rsi(close, 14) / 100.0
    df["atr_14_pct"] = _atr(df, 14) / close

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df["bollinger_position_20"] = (close - lower) / (upper - lower)
    df["bollinger_width_20"] = (upper - lower) / sma20

    df["candle_body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["high_low_pct"] = (df["high"] - df["low"]) / df["open"]
    df["upper_wick_pct"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / df["open"]
    df["lower_wick_pct"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / df["open"]

    hours = pd.Series(df.index.hour, index=df.index, dtype=float)
    dow = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    return df.replace([np.inf, -np.inf], np.nan)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in excluded and not c.startswith("target_")]

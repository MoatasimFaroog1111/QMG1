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
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr = _atr(df, period)
    plus_di = (
        100.0
        * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr.replace(0.0, np.nan)
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return (series - mean) / std


def load_m1_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    missing = [column for column in BASE_PRICE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.set_index("timestamp_utc").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for column in BASE_PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

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
    """Build causal technical-analysis features from information known at each row."""

    df = hourly.copy()
    close = df["close"]
    log_close = np.log(close)
    log_ret_1h = log_close.diff()
    df["ret_1h"] = log_ret_1h

    # Multi-scale momentum aligned with the requested short/medium/long horizons.
    for lag in (2, 4, 8, 12, 24, 48, 72, 168, 360, 720):
        df[f"log_return_{lag}h"] = log_close.diff(lag)

    for window in (6, 12, 24, 48, 72, 168, 360, 720):
        close_window = close.rolling(window, min_periods=window)
        high_window = df["high"].rolling(window, min_periods=window)
        low_window = df["low"].rolling(window, min_periods=window)

        df[f"sma_ratio_{window}h"] = close / close_window.mean() - 1.0
        df[f"close_zscore_{window}h"] = _rolling_zscore(close, window)
        df[f"volatility_{window}h"] = log_ret_1h.rolling(
            window, min_periods=window
        ).std()
        rolling_high = high_window.max()
        rolling_low = low_window.min()
        df[f"range_ratio_{window}h"] = rolling_high / rolling_low - 1.0
        df[f"range_position_{window}h"] = (
            (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
        )

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    df["ema12_ratio"] = close / ema12 - 1.0
    df["ema26_ratio"] = close / ema26 - 1.0
    df["ema50_ratio"] = close / ema50 - 1.0
    df["ema200_ratio"] = close / ema200 - 1.0
    df["ema50_200_spread"] = ema50 / ema200 - 1.0
    df["macd_pct"] = macd / close
    df["macd_signal_pct"] = signal / close
    df["macd_hist_pct"] = (macd - signal) / close

    df["rsi_14"] = _rsi(close, 14) / 100.0
    df["atr_14_pct"] = _atr(df, 14) / close

    adx, plus_di, minus_di = _adx(df, 14)
    df["adx_14"] = adx / 100.0
    df["plus_di_14"] = plus_di / 100.0
    df["minus_di_14"] = minus_di / 100.0
    df["di_spread_14"] = (plus_di - minus_di) / 100.0

    low14 = df["low"].rolling(14, min_periods=14).min()
    high14 = df["high"].rolling(14, min_periods=14).max()
    stochastic_k = 100.0 * (close - low14) / (high14 - low14).replace(0.0, np.nan)
    df["stochastic_k_14"] = stochastic_k / 100.0
    df["stochastic_d_3"] = stochastic_k.rolling(3, min_periods=3).mean() / 100.0

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df["bollinger_position_20"] = (close - lower) / (upper - lower).replace(0.0, np.nan)
    df["bollinger_width_20"] = (upper - lower) / sma20

    df["candle_body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["high_low_pct"] = (df["high"] - df["low"]) / df["open"]
    df["upper_wick_pct"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / df["open"]
    df["lower_wick_pct"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / df["open"]

    volume = df["volume"].fillna(0.0).clip(lower=0.0)
    log_volume = np.log1p(volume)
    df["log_volume"] = log_volume
    df["volume_zscore_24h"] = _rolling_zscore(log_volume, 24)
    df["volume_zscore_168h"] = _rolling_zscore(log_volume, 168)
    df["minute_coverage"] = df["minute_count"] / 60.0

    hours = pd.Series(df.index.hour, index=df.index, dtype=float)
    day_of_week = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    return df.replace([np.inf, -np.inf], np.nan)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"open", "high", "low", "close", "volume", "minute_count"}
    return [
        column
        for column in df.columns
        if column not in excluded
        and not column.startswith("target_")
        and not column.startswith("future_")
    ]

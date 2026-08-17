from __future__ import annotations

import numpy as np
import pandas as pd


class CalendarHorizonTargetBuilder:
    """Build targets using real elapsed clock time, not a row-count shift.

    If the exact target timestamp falls inside a market closure, the first
    available hourly close at or after the requested timestamp is used.
    """

    def __init__(self, max_forward_tolerance_hours: int = 72) -> None:
        self.max_forward_tolerance = pd.Timedelta(hours=max_forward_tolerance_hours)

    def attach(
        self,
        feature_frame: pd.DataFrame,
        hourly_prices: pd.DataFrame,
        horizon_hours: int,
    ) -> pd.DataFrame:
        if feature_frame.empty:
            return feature_frame.copy()

        source_close = hourly_prices["close"].dropna().sort_index()
        if source_close.empty:
            raise ValueError("No hourly close prices are available")

        requested_times = feature_frame.index + pd.Timedelta(hours=horizon_hours)
        source_index = source_close.index
        positions = source_index.searchsorted(requested_times, side="left")

        future_values = np.full(len(feature_frame), np.nan, dtype=float)
        effective_times = np.full(len(feature_frame), np.datetime64("NaT"), dtype="datetime64[ns]")

        valid_position = positions < len(source_index)
        valid_rows = np.flatnonzero(valid_position)

        for row_idx in valid_rows:
            source_pos = int(positions[row_idx])
            effective_ts = source_index[source_pos]
            requested_ts = requested_times[row_idx]
            if effective_ts - requested_ts <= self.max_forward_tolerance:
                future_values[row_idx] = float(source_close.iloc[source_pos])
                effective_times[row_idx] = effective_ts.tz_convert("UTC").tz_localize(None).to_datetime64()

        result = feature_frame.copy()
        future_col = f"future_close_{horizon_hours}h"
        target_col = f"target_{horizon_hours}h"
        timestamp_col = f"target_timestamp_{horizon_hours}h"

        result[future_col] = future_values
        result[timestamp_col] = pd.to_datetime(effective_times, utc=True)
        result[target_col] = np.log(result[future_col]) - np.log(result["close"])
        return result

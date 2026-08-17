from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qmg1.data.hourly import HOURLY_COLUMNS
from qmg1.features import build_features, feature_columns, load_m1_csv, resample_to_hourly
from qmg1.ml.targets import CalendarHorizonTargetBuilder


@dataclass(frozen=True)
class FeatureBase:
    hourly: pd.DataFrame
    features: pd.DataFrame


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    horizon_hours: int


class ForecastDatasetBuilder:
    def __init__(self, target_builder: CalendarHorizonTargetBuilder | None = None) -> None:
        self.target_builder = target_builder or CalendarHorizonTargetBuilder()

    def load_feature_base(self, csv_path: str) -> FeatureBase:
        """Load normalized M1 data and build the reusable hourly feature base."""
        m1 = load_m1_csv(csv_path)
        hourly = resample_to_hourly(m1)
        features = build_features(hourly)
        return FeatureBase(hourly=hourly, features=features)

    def load_hourly_feature_base(self, csv_path: str) -> FeatureBase:
        """Load a compact H1 training dataset without reprocessing M1."""
        hourly = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
        missing = [column for column in HOURLY_COLUMNS if column not in hourly.columns]
        if missing:
            raise ValueError(f"Missing required hourly columns: {missing}")

        hourly = hourly.set_index("timestamp_utc").sort_index()
        hourly = hourly[~hourly.index.duplicated(keep="last")]
        for column in HOURLY_COLUMNS:
            hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
        hourly = hourly.dropna(subset=["open", "high", "low", "close"])
        features = build_features(hourly)
        return FeatureBase(hourly=hourly, features=features)

    def build_from_base(
        self,
        base: FeatureBase,
        horizon_hours: int,
    ) -> PreparedDataset:
        frame = self.target_builder.attach(base.features, base.hourly, horizon_hours)
        cols = feature_columns(frame)
        required = cols + [
            "close",
            f"target_{horizon_hours}h",
            f"future_close_{horizon_hours}h",
            f"target_timestamp_{horizon_hours}h",
        ]
        frame = frame.dropna(subset=required)
        return PreparedDataset(
            frame=frame,
            feature_columns=cols,
            horizon_hours=horizon_hours,
        )

    def build(self, csv_path: str, horizon_hours: int) -> PreparedDataset:
        return self.build_from_base(self.load_feature_base(csv_path), horizon_hours)

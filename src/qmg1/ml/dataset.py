from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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
        """Perform expensive M1 loading/resampling/feature work exactly once."""
        m1 = load_m1_csv(csv_path)
        hourly = resample_to_hourly(m1)
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

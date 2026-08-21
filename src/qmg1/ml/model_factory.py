from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qmg1.config import TrainingConfig
from qmg1.ml.purged_selective import PurgedCrossFittedSelectiveRegressor
from qmg1.ml.selective import (
    CrossFittedSelectiveRegressor,
    SelectiveShrinkageRegressor,
)
from qmg1.ml.targets import DEFAULT_MAX_FORWARD_TOLERANCE_HOURS


class RegressorFactory(Protocol):
    name: str
    lookback_days: int | None

    def create(self):
        ...


@dataclass(frozen=True)
class HistGradientBoostingFactory:
    config: TrainingConfig
    loss: str = "absolute_error"
    lookback_days: int | None = None

    @property
    def name(self) -> str:
        base = f"hist_gradient_boosting_{self.loss}"
        return _with_lookback_name(base, self.lookback_days)

    def create(self) -> HistGradientBoostingRegressor:
        return _base_hgb(self.config, self.loss)


@dataclass(frozen=True)
class SelectiveHistGradientBoostingFactory:
    config: TrainingConfig
    activation_quantile: float
    shrinkage: float
    lookback_days: int | None = None

    @property
    def name(self) -> str:
        quantile = int(round(self.activation_quantile * 100))
        shrink = int(round(self.shrinkage * 100))
        base = f"selective_hgb_q{quantile}_s{shrink}"
        return _with_lookback_name(base, self.lookback_days)

    def create(self) -> SelectiveShrinkageRegressor:
        return SelectiveShrinkageRegressor(
            base_estimator=_base_hgb(self.config),
            activation_quantile=self.activation_quantile,
            shrinkage=self.shrinkage,
        )


@dataclass(frozen=True)
class CrossFittedSelectiveHgbFactory:
    """Legacy unpurged calibrator retained for reproducibility only."""

    config: TrainingConfig
    lookback_days: int | None = None

    @property
    def name(self) -> str:
        return _with_lookback_name("cross_fitted_selective_hgb", self.lookback_days)

    def create(self) -> CrossFittedSelectiveRegressor:
        return CrossFittedSelectiveRegressor(
            base_estimator=_base_hgb(self.config),
            calibration_splits=3,
            activation_quantiles=(0.50, 0.70, 0.80, 0.90, 0.95),
            shrinkages=(0.10, 0.25, 0.50, 0.75, 1.00),
        )


@dataclass(frozen=True)
class PurgedCrossFittedSelectiveHgbFactory:
    config: TrainingConfig
    horizon_hours: int
    lookback_days: int | None = None
    target_forward_tolerance_hours: int = DEFAULT_MAX_FORWARD_TOLERANCE_HOURS

    @property
    def name(self) -> str:
        return _with_lookback_name(
            "purged_cross_fitted_selective_hgb",
            self.lookback_days,
        )

    @property
    def purge_hours(self) -> int:
        return self.horizon_hours + self.target_forward_tolerance_hours

    def create(self) -> PurgedCrossFittedSelectiveRegressor:
        return PurgedCrossFittedSelectiveRegressor(
            base_estimator=_base_hgb(self.config),
            calibration_splits=3,
            activation_quantiles=(0.50, 0.70, 0.80, 0.90, 0.95),
            shrinkages=(0.10, 0.25, 0.50, 0.75, 1.00),
            purge_hours=self.purge_hours,
        )


@dataclass(frozen=True)
class RidgeFactory:
    """Retained for controlled experiments; not in the production candidate set."""

    alpha: float
    lookback_days: int | None = None

    @property
    def name(self) -> str:
        base = f"ridge_alpha_{self.alpha:g}"
        return _with_lookback_name(base, self.lookback_days)

    def create(self) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )


@dataclass(frozen=True)
class MedianReturnFactory:
    lookback_days: int | None = None

    @property
    def name(self) -> str:
        return _with_lookback_name("median_return", self.lookback_days)

    def create(self) -> DummyRegressor:
        return DummyRegressor(strategy="median")


def _base_hgb(
    config: TrainingConfig,
    loss: str = "absolute_error",
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.random_state,
    )


def _with_lookback_name(base: str, lookback_days: int | None) -> str:
    if lookback_days is None:
        return base
    return f"{base}_lookback_{lookback_days}d"


def apply_training_lookback(
    factory: RegressorFactory,
    frame: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    if factory.lookback_days is None:
        return frame
    cutoff = reference_timestamp - pd.Timedelta(days=factory.lookback_days)
    return frame[frame.index >= cutoff]


def candidate_factories(
    config: TrainingConfig,
    horizon_hours: int = 0,
) -> tuple[RegressorFactory, ...]:
    """Focused candidate set with leakage-safe nested calibration.

    Ridge and recency challengers remain available for explicit experiments but
    are not part of routine training. The nested calibrator now applies a
    conservative inner purge equal to ``horizon + target-forward tolerance``.
    """
    return (
        MedianReturnFactory(),
        SelectiveHistGradientBoostingFactory(
            config=config,
            activation_quantile=0.80,
            shrinkage=0.25,
        ),
        PurgedCrossFittedSelectiveHgbFactory(
            config=config,
            horizon_hours=horizon_hours,
        ),
    )

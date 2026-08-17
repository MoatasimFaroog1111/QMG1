from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qmg1.config import TrainingConfig


class RegressorFactory(Protocol):
    name: str
    lookback_days: int | None

    def create(self) -> RegressorMixin:
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
        return HistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization,
            early_stopping=False,
            random_state=self.config.random_state,
        )


@dataclass(frozen=True)
class RidgeFactory:
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


def _with_lookback_name(base: str, lookback_days: int | None) -> str:
    if lookback_days is None:
        return base
    return f"{base}_lookback_{lookback_days}d"


def apply_training_lookback(
    factory: RegressorFactory,
    frame: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Apply a causal rolling training window for regime-aware challengers."""
    if factory.lookback_days is None:
        return frame
    cutoff = reference_timestamp - pd.Timedelta(days=factory.lookback_days)
    return frame[frame.index >= cutoff]


def candidate_factories(config: TrainingConfig) -> tuple[RegressorFactory, ...]:
    """Diverse full-history and recent-regime challengers.

    Hyperparameters and lookback windows are fixed before the untouched holdout
    is evaluated. Recency candidates test whether old regimes dilute current
    signal without using holdout performance to tune the window length.
    """
    full_history: tuple[RegressorFactory, ...] = (
        HistGradientBoostingFactory(config=config, loss="absolute_error"),
        RidgeFactory(alpha=10.0),
        RidgeFactory(alpha=100.0),
        RidgeFactory(alpha=1_000.0),
        RidgeFactory(alpha=10_000.0),
        MedianReturnFactory(),
    )
    recent_regime: tuple[RegressorFactory, ...] = tuple(
        factory
        for lookback_days in (730, 1_825)
        for factory in (
            HistGradientBoostingFactory(
                config=config,
                loss="absolute_error",
                lookback_days=lookback_days,
            ),
            RidgeFactory(alpha=10_000.0, lookback_days=lookback_days),
            MedianReturnFactory(lookback_days=lookback_days),
        )
    )
    return full_history + recent_regime

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
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


class SelectiveShrinkageRegressor(BaseEstimator, RegressorMixin):
    """Use a base regressor only for its strongest in-sample signals.

    Persistence (zero log return) is difficult to beat under MAE. This wrapper
    therefore abstains for weak signals and shrinks the remaining predictions.
    The activation threshold is learned only from the training fold, so no
    validation or holdout information is used to decide when to act.
    """

    def __init__(
        self,
        base_estimator: RegressorMixin,
        activation_quantile: float = 0.90,
        shrinkage: float = 0.50,
    ) -> None:
        self.base_estimator = base_estimator
        self.activation_quantile = activation_quantile
        self.shrinkage = shrinkage

    def fit(self, X, y):
        if not 0.0 < self.activation_quantile < 1.0:
            raise ValueError("activation_quantile must be between 0 and 1")
        if not 0.0 < self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in (0, 1]")

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)
        train_prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        finite = np.abs(train_prediction[np.isfinite(train_prediction)])
        if finite.size == 0:
            self.activation_threshold_ = float("inf")
        else:
            self.activation_threshold_ = float(
                np.quantile(finite, self.activation_quantile)
            )
        return self

    def predict(self, X):
        if not hasattr(self, "estimator_"):
            raise RuntimeError("SelectiveShrinkageRegressor must be fitted first")
        raw = np.asarray(self.estimator_.predict(X), dtype=float)
        active = np.abs(raw) >= self.activation_threshold_
        return np.where(active, raw * self.shrinkage, 0.0)


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
        base = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization,
            early_stopping=False,
            random_state=self.config.random_state,
        )
        return SelectiveShrinkageRegressor(
            base_estimator=base,
            activation_quantile=self.activation_quantile,
            shrinkage=self.shrinkage,
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
    """Diverse challengers evaluated only on development walk-forward folds."""
    full_history: tuple[RegressorFactory, ...] = (
        HistGradientBoostingFactory(config=config, loss="absolute_error"),
        RidgeFactory(alpha=10.0),
        RidgeFactory(alpha=100.0),
        RidgeFactory(alpha=1_000.0),
        RidgeFactory(alpha=10_000.0),
        MedianReturnFactory(),
    )

    selective: tuple[RegressorFactory, ...] = tuple(
        SelectiveHistGradientBoostingFactory(
            config=config,
            activation_quantile=quantile,
            shrinkage=shrinkage,
        )
        for quantile in (0.80, 0.90, 0.95)
        for shrinkage in (0.25, 0.50)
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
    return full_history + selective + recent_regime

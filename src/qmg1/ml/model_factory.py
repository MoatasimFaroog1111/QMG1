from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qmg1.config import TrainingConfig


class RegressorFactory(Protocol):
    name: str

    def create(self) -> RegressorMixin:
        ...


@dataclass(frozen=True)
class HistGradientBoostingFactory:
    config: TrainingConfig
    loss: str = "absolute_error"

    @property
    def name(self) -> str:
        return f"hist_gradient_boosting_{self.loss}"

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

    @property
    def name(self) -> str:
        return f"ridge_alpha_{self.alpha:g}"

    def create(self) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )


@dataclass(frozen=True)
class MedianReturnFactory:
    name: str = "median_return"

    def create(self) -> DummyRegressor:
        return DummyRegressor(strategy="median")


def candidate_factories(config: TrainingConfig) -> tuple[RegressorFactory, ...]:
    """Small, deliberately diverse challenger set for leakage-safe selection."""
    return (
        HistGradientBoostingFactory(config=config, loss="absolute_error"),
        RidgeFactory(alpha=10.0),
        RidgeFactory(alpha=100.0),
        RidgeFactory(alpha=1_000.0),
        RidgeFactory(alpha=10_000.0),
        MedianReturnFactory(),
    )

from __future__ import annotations

from typing import Protocol

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qmg1.config import TrainingConfig


class RegressorFactory(Protocol):
    @property
    def name(self) -> str:
        ...

    def create(self) -> RegressorMixin:
        ...


class ZeroReturnFactory:
    """Exact persistence/no-change benchmark expressed as a regressor."""

    @property
    def name(self) -> str:
        return "persistence_zero_return"

    def create(self) -> DummyRegressor:
        return DummyRegressor(strategy="constant", constant=0.0)


class RidgeFactory:
    """Strongly regularized linear return model with standardized features."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"ridge_alpha_{self.alpha:g}"

    def create(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )


class HistGradientBoostingFactory:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "hist_gradient_boosting"

    def create(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization,
            # Do not let sklearn create an internal random validation split.
            # Time-series validation is owned by WalkForwardEvaluator.
            early_stopping=False,
            random_state=self.config.random_state,
        )

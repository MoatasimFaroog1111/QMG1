from __future__ import annotations

from typing import Protocol

from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

from qmg1.config import TrainingConfig


class RegressorFactory(Protocol):
    def create(self) -> RegressorMixin:
        ...


class HistGradientBoostingFactory:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

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

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qmg1.config import TrainingConfig
from qmg1.ml.evaluation import HorizonMetrics, WalkForwardEvaluator
from qmg1.ml.model_factory import (
    HistGradientBoostingFactory,
    RegressorFactory,
    RidgeFactory,
    ZeroReturnFactory,
)


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    factory: RegressorFactory
    max_train_rows: int | None = None

    def final_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.max_train_rows is None or len(frame) <= self.max_train_rows:
            return frame
        return frame.iloc[-self.max_train_rows :]


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: ModelCandidate
    metrics: HorizonMetrics

    @property
    def robust_improvement_pct(self) -> float:
        # A current-market model must earn improvement over persistence both
        # across the long OOS history and on the most recent validation block.
        return min(
            self.metrics.improvement_vs_persistence_pct,
            self.metrics.latest_fold_improvement_vs_persistence_pct,
        )


@dataclass(frozen=True)
class ModelSelectionResult:
    selected: CandidateEvaluation
    evaluations: tuple[CandidateEvaluation, ...]


class ForecastModelSelector:
    """Benchmark several model/regime candidates and select conservatively."""

    def __init__(
        self,
        config: TrainingConfig,
        candidates: tuple[ModelCandidate, ...] | None = None,
    ) -> None:
        self.config = config
        self.candidates = candidates or self.default_candidates(config)

    @staticmethod
    def default_candidates(config: TrainingConfig) -> tuple[ModelCandidate, ...]:
        hgb = HistGradientBoostingFactory(config)
        return (
            # Including persistence as an actual candidate makes it impossible
            # for selection to silently choose an OOS-worse price forecast.
            ModelCandidate("persistence", ZeroReturnFactory()),
            ModelCandidate("ridge_a100_full", RidgeFactory(100.0)),
            ModelCandidate("ridge_a1000_full", RidgeFactory(1_000.0)),
            ModelCandidate("ridge_a10000_full", RidgeFactory(10_000.0)),
            ModelCandidate(
                "ridge_a1000_recent_30000",
                RidgeFactory(1_000.0),
                max_train_rows=30_000,
            ),
            ModelCandidate("hgb_full", hgb),
            ModelCandidate("hgb_recent_30000", hgb, max_train_rows=30_000),
            ModelCandidate("hgb_recent_15000", hgb, max_train_rows=15_000),
        )

    def select(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        horizon_hours: int,
    ) -> ModelSelectionResult:
        evaluations: list[CandidateEvaluation] = []

        for candidate in self.candidates:
            print(
                f"  [CANDIDATE] {candidate.name} "
                f"max_train_rows={candidate.max_train_rows or 'all'}"
            )
            metrics = WalkForwardEvaluator(
                self.config,
                candidate.factory,
                max_train_rows=candidate.max_train_rows,
            ).evaluate(frame, feature_columns, horizon_hours)
            evaluation = CandidateEvaluation(candidate=candidate, metrics=metrics)
            evaluations.append(evaluation)
            print(
                f"  [RESULT] {candidate.name} "
                f"overall={metrics.improvement_vs_persistence_pct:+.2f}% "
                f"recent={metrics.latest_fold_improvement_vs_persistence_pct:+.2f}% "
                f"robust={evaluation.robust_improvement_pct:+.2f}% "
                f"MAE={metrics.mae_usd_per_kg:.4f}"
            )

        if not evaluations:
            raise ValueError("No forecasting model candidates were evaluated")

        selected = max(
            evaluations,
            key=lambda item: (
                item.robust_improvement_pct,
                item.metrics.improvement_vs_persistence_pct,
                item.metrics.latest_fold_improvement_vs_persistence_pct,
                -item.metrics.mae_usd_per_kg,
            ),
        )
        print(
            f"  [SELECTED] {selected.candidate.name} "
            f"robust={selected.robust_improvement_pct:+.2f}%"
        )
        return ModelSelectionResult(
            selected=selected,
            evaluations=tuple(evaluations),
        )

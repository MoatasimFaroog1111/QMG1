from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from qmg1.config import TrainingConfig
from qmg1.ml.evaluation import HorizonMetrics, WalkForwardEvaluator, score_predictions
from qmg1.ml.model_factory import RegressorFactory, candidate_factories


@dataclass(frozen=True)
class CandidateScore:
    model_name: str
    metrics: HorizonMetrics


@dataclass(frozen=True)
class ChampionSelection:
    selected_model_name: str
    active_strategy: str
    development_candidates: tuple[CandidateScore, ...]
    challenger_holdout_metrics: HorizonMetrics
    active_holdout_metrics: HorizonMetrics
    holdout_start_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_model_name": self.selected_model_name,
            "active_strategy": self.active_strategy,
            "development_candidates": [
                {"model_name": score.model_name, "metrics": asdict(score.metrics)}
                for score in self.development_candidates
            ],
            "challenger_holdout_metrics": asdict(self.challenger_holdout_metrics),
            "active_holdout_metrics": asdict(self.active_holdout_metrics),
            "holdout_start_utc": self.holdout_start_utc,
        }


class ChampionChallengerSelector:
    """Select on development history, then verify once on an untouched holdout."""

    def __init__(
        self,
        config: TrainingConfig,
        holdout_fraction: float = 0.20,
    ) -> None:
        if not 0.10 <= holdout_fraction <= 0.40:
            raise ValueError("holdout_fraction must be between 0.10 and 0.40")
        self.config = config
        self.holdout_fraction = holdout_fraction

    def select(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        horizon_hours: int,
    ) -> tuple[RegressorFactory, ChampionSelection]:
        split_at = int(len(frame) * (1.0 - self.holdout_fraction))
        if split_at < self.config.min_rows:
            raise ValueError("Development segment is too small for model selection")

        development = frame.iloc[:split_at]
        holdout = frame.iloc[split_at:]
        if holdout.empty:
            raise ValueError("Holdout segment is empty")

        scores: list[CandidateScore] = []
        factories = candidate_factories(self.config)
        for factory in factories:
            print(f"  [CHALLENGER] {factory.name}")
            metrics = WalkForwardEvaluator(self.config, factory).evaluate(
                development,
                feature_columns,
                horizon_hours,
            )
            scores.append(CandidateScore(factory.name, metrics))
            print(
                f"    MAE={metrics.mae_usd_per_kg:.4f} "
                f"vs_persistence={metrics.improvement_vs_persistence_pct:+.3f}%"
            )

        winning_score = min(scores, key=lambda score: score.metrics.mae_usd_per_kg)
        winning_factory = next(
            factory for factory in factories if factory.name == winning_score.model_name
        )

        target_col = f"target_{horizon_hours}h"
        future_close_col = f"future_close_{horizon_hours}h"
        target_timestamp_col = f"target_timestamp_{horizon_hours}h"
        holdout_start = holdout.index[0]
        holdout_train = development[
            development[target_timestamp_col] < holdout_start
        ]
        if holdout_train.empty:
            raise ValueError("No leakage-safe development rows remain before holdout")

        challenger = winning_factory.create()
        challenger.fit(holdout_train[feature_columns], holdout_train[target_col])
        challenger_log_return = np.asarray(
            challenger.predict(holdout[feature_columns]), dtype=float
        )
        current_close = holdout["close"].to_numpy(dtype=float)
        actual_close = holdout[future_close_col].to_numpy(dtype=float)
        actual_log_return = holdout[target_col].to_numpy(dtype=float)
        challenger_close = current_close * np.exp(challenger_log_return)

        challenger_metrics = score_predictions(
            horizon_hours=horizon_hours,
            rows_total=len(frame),
            cv_splits=1,
            current_close=current_close,
            actual_close=actual_close,
            predicted_close=challenger_close,
            actual_log_return=actual_log_return,
            predicted_log_return=challenger_log_return,
        )

        persistence_log_return = np.zeros(len(holdout), dtype=float)
        persistence_metrics = score_predictions(
            horizon_hours=horizon_hours,
            rows_total=len(frame),
            cv_splits=1,
            current_close=current_close,
            actual_close=actual_close,
            predicted_close=current_close,
            actual_log_return=actual_log_return,
            predicted_log_return=persistence_log_return,
        )

        challenger_wins = (
            challenger_metrics.mae_usd_per_kg < persistence_metrics.mae_usd_per_kg
        )
        active_strategy = winning_factory.name if challenger_wins else "persistence"
        active_metrics = challenger_metrics if challenger_wins else persistence_metrics

        print(
            f"  [HOLDOUT] selected={winning_factory.name} "
            f"challenger_MAE={challenger_metrics.mae_usd_per_kg:.4f} "
            f"persistence_MAE={persistence_metrics.mae_usd_per_kg:.4f} "
            f"active={active_strategy}"
        )

        selection = ChampionSelection(
            selected_model_name=winning_factory.name,
            active_strategy=active_strategy,
            development_candidates=tuple(scores),
            challenger_holdout_metrics=challenger_metrics,
            active_holdout_metrics=active_metrics,
            holdout_start_utc=holdout_start.isoformat(),
        )
        return winning_factory, selection

from __future__ import annotations

import numpy as np
import pandas as pd

from qmg1.config import TrainingConfig
from qmg1.ml.selection import ChampionChallengerSelector


def test_persistence_remains_champion_when_challengers_do_not_improve() -> None:
    index = pd.date_range("2024-01-01", periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    current = 1000.0 + rng.normal(0.0, 1.0, len(index)).cumsum()
    frame = pd.DataFrame(
        {
            "signal_a": rng.normal(size=len(index)),
            "signal_b": rng.normal(size=len(index)),
            "close": current,
            "target_2h": 0.0,
            "future_close_2h": current,
            "target_timestamp_2h": index + pd.Timedelta(2, unit="h"),
        },
        index=index,
    )
    config = TrainingConfig(
        min_rows=100,
        cv_splits=3,
        max_iter=10,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        min_promotion_improvement_pct=0.5,
    )

    _, result = ChampionChallengerSelector(config).select(
        frame,
        ["signal_a", "signal_b"],
        horizon_hours=2,
    )

    assert result.active_strategy == "persistence"
    assert result.active_holdout_metrics.mae_usd_per_kg == 0.0
    assert len(result.development_candidates) >= 4
    names = {score.model_name for score in result.development_candidates}
    assert "cross_fitted_shrinkage_hgb" in names
    assert result.development_qualified is False
    assert result.holdout_qualified is False
    assert result.promotion_threshold_pct == 0.5

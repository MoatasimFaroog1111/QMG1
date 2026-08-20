from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.ml.artifacts import ModelArtifactRepository  # noqa: E402
from qmg1.ml.predictor import ForecastPredictor  # noqa: E402
from qmg1.ml.serving_policy import PersistedArtifactServingPolicy  # noqa: E402


def _metrics(
    *,
    directional_accuracy_pct: float,
    improvement_vs_persistence_pct: float,
) -> dict[str, float | int]:
    return {
        "horizon_hours": 2,
        "cv_splits": 1,
        "rows_total": 100,
        "rows_validation_total": 20,
        "mae_usd_per_kg": 12.5,
        "rmse_usd_per_kg": 18.0,
        "smape_pct": 1.25,
        "directional_accuracy_pct": directional_accuracy_pct,
        "persistence_mae_usd_per_kg": 12.0,
        "improvement_vs_persistence_pct": improvement_vs_persistence_pct,
        "residual_log_return_q10": -0.02,
        "residual_log_return_q90": 0.03,
    }


def _legacy_artifact() -> dict[str, object]:
    features = pd.DataFrame([{"feature_a": 0.0, "feature_b": 0.0}])
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit(features, [0.01])

    challenger_metrics = _metrics(
        directional_accuracy_pct=47.25,
        improvement_vs_persistence_pct=-4.5,
    )
    persistence_metrics = _metrics(
        directional_accuracy_pct=0.0,
        improvement_vs_persistence_pct=0.0,
    )
    return {
        "schema_version": 6,
        "metal": "silver",
        "horizon_hours": 2,
        "feature_columns": ["feature_a", "feature_b"],
        "active_strategy": "persistence",
        "selected_challenger": "median_return",
        "selection": {
            "selected_model_name": "median_return",
            "active_strategy": "persistence",
            "challenger_holdout_metrics": challenger_metrics,
        },
        "metrics": persistence_metrics,
        "model": model,
    }


def test_legacy_persistence_artifact_resolves_to_saved_model() -> None:
    decision = PersistedArtifactServingPolicy.resolve(_legacy_artifact())

    assert decision.serving_strategy == "median_return"
    assert decision.governance_strategy == "persistence"
    assert decision.feature_data_required is False
    assert decision.model_metrics["directional_accuracy_pct"] == 47.25
    assert decision.model_metrics["improvement_vs_persistence_pct"] == -4.5


def test_live_prediction_uses_estimator_and_challenger_metrics(tmp_path: Path) -> None:
    predictor = ForecastPredictor(ModelArtifactRepository(tmp_path / "models"))

    result = predictor.predict_live_from_artifact(
        artifact=_legacy_artifact(),
        timestamp_utc=datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc),
        close_usd_per_kg=1_000.0,
    )

    assert result["active_strategy"] == "median_return"
    assert result["serving_strategy"] == "median_return"
    assert result["governance_strategy"] == "persistence"
    assert math.isclose(
        float(result["predicted_usd_per_kg"]),
        1_000.0 * math.exp(0.01),
        rel_tol=1e-12,
    )
    assert float(result["predicted_usd_per_kg"]) != 1_000.0
    assert result["validation_directional_accuracy_pct"] == 47.25
    assert result["validation_improvement_vs_persistence_pct"] == -4.5


def test_feature_dependent_strategy_requires_serving_features() -> None:
    artifact = _legacy_artifact()
    artifact["selected_challenger"] = "selective_hgb_q80_s25"
    artifact["selection"]["selected_model_name"] = "selective_hgb_q80_s25"

    decision = PersistedArtifactServingPolicy.resolve(artifact)

    assert decision.feature_data_required is True

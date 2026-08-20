from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.api.schemas import PredictionRequest  # noqa: E402
from qmg1.api.service import ForecastApiService, RuntimeSettings  # noqa: E402
from qmg1.config import HORIZONS_HOURS  # noqa: E402
from qmg1.ml.artifacts import ModelArtifactRepository  # noqa: E402
from qmg1.serving.live_price import LiveQuote  # noqa: E402


class FixedQuoteProvider:
    configured = True

    def latest_quote(self, metal_key: str) -> LiveQuote:
        return LiveQuote(
            metal=metal_key,
            timestamp_utc=datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc),
            close_usd_per_kg=2050.0,
            source="test-live-quote",
        )


def _serving_settings() -> RuntimeSettings:
    return RuntimeSettings(
        project_root=ROOT,
        models_dir=ROOT / "serving_artifacts" / "models",
        target_data_dir=ROOT / "metals_m1_usd_per_kg" / "final",
        hourly_context_dir=ROOT / "training_data" / "hourly",
        required_metals=("silver",),
        required_horizons=HORIZONS_HOURS,
        production_mode=False,
    )


def _expected_live_model_price(artifact: dict[str, object], current: float) -> float:
    feature_columns = list(artifact["feature_columns"])
    features = pd.DataFrame(
        [{column: 0.0 for column in feature_columns}],
        columns=feature_columns,
    )
    predicted_log_return = float(artifact["model"].predict(features)[0])
    return current * math.exp(predicted_log_return)


def test_committed_training_bundle_contains_every_silver_horizon() -> None:
    repository = ModelArtifactRepository(ROOT / "serving_artifacts" / "models")

    assert repository.available_trained() == {"silver": list(HORIZONS_HOURS)}

    for horizon in HORIZONS_HOURS:
        artifact = repository.load_trained("silver", horizon)
        assert artifact["schema_version"] == 6
        assert artifact["metal"] == "silver"
        assert artifact["horizon_hours"] == horizon
        assert artifact["trained_at_utc"]
        assert artifact["active_strategy"] == "persistence"
        assert artifact["selected_challenger"] == "median_return"
        assert "selection" in artifact
        assert "model" in artifact
        assert "metrics" in artifact


def test_prediction_service_uses_previously_trained_artifact_without_retraining() -> None:
    service = ForecastApiService(
        _serving_settings(),
        live_price_provider=FixedQuoteProvider(),
    )
    artifact = service.repository.load_trained("silver", 2)

    result = service.predict(PredictionRequest(metal="silver", horizon_hours=2))

    assert result["metal"] == "silver"
    assert result["horizon_hours"] == 2
    assert result["active_strategy"] == "median_return"
    assert result["serving_strategy"] == "median_return"
    assert result["governance_strategy"] == "persistence"
    assert result["selected_challenger"] == "median_return"
    assert result["current_usd_per_kg"] == 2050.0
    assert math.isclose(
        float(result["predicted_usd_per_kg"]),
        _expected_live_model_price(artifact, 2050.0),
        rel_tol=1e-12,
    )
    assert (
        result["validation_mae_usd_per_kg"]
        == artifact["selection"]["challenger_holdout_metrics"]["mae_usd_per_kg"]
    )
    assert result["market_data_source"] == "test-live-quote"

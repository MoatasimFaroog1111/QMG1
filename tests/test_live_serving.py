from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.api.service import ForecastApiService, RuntimeSettings  # noqa: E402
from qmg1.serving.live_price import LiveQuote  # noqa: E402


class FixedLivePriceProvider:
    configured = True

    def latest_quote(self, metal_key: str) -> LiveQuote:
        assert metal_key == "silver"
        return LiveQuote(
            metal="silver",
            timestamp_utc=datetime(2026, 8, 17, 20, 0, tzinfo=timezone.utc),
            close_usd_per_kg=1250.0,
            source="test",
        )


def _expected_live_model_price(artifact: dict[str, object], current: float) -> float:
    feature_columns = list(artifact["feature_columns"])
    features = pd.DataFrame(
        [{column: 0.0 for column in feature_columns}],
        columns=feature_columns,
    )
    predicted_log_return = float(artifact["model"].predict(features)[0])
    return current * math.exp(predicted_log_return)


def test_packaged_silver_trained_artifact_is_available() -> None:
    settings = RuntimeSettings.from_environment()
    service = ForecastApiService(
        settings,
        live_price_provider=FixedLivePriceProvider(),
    )

    assert service.repository.has_any() is True
    artifact = service.repository.load_trained("silver", 2)
    assert artifact["active_strategy"] == "persistence"
    assert artifact["selected_challenger"] == "median_return"
    assert artifact["horizon_hours"] == 2
    assert "model" in artifact


def test_silver_predict_uses_saved_model_with_live_quote_without_training() -> None:
    settings = RuntimeSettings.from_environment()
    service = ForecastApiService(
        settings,
        live_price_provider=FixedLivePriceProvider(),
    )
    artifact = service.repository.load_trained("silver", 2)
    request = type("Request", (), {"metal": "silver", "horizon_hours": 2})()
    result = service.predict(request)

    assert result["active_strategy"] == "median_return"
    assert result["serving_strategy"] == "median_return"
    assert result["governance_strategy"] == "persistence"
    assert result["current_usd_per_kg"] == 1250.0
    assert math.isclose(
        float(result["predicted_usd_per_kg"]),
        _expected_live_model_price(artifact, 1250.0),
        rel_tol=1e-12,
    )
    assert result["market_data_source"] == "test"
    assert (
        result["validation_directional_accuracy_pct"]
        == artifact["selection"]["challenger_holdout_metrics"][
            "directional_accuracy_pct"
        ]
    )
    assert result["prediction_interval_80_low_usd_per_kg"] > 0
    assert result["prediction_interval_80_high_usd_per_kg"] > 0

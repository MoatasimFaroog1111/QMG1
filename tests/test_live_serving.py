from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

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


def test_packaged_silver_champion_manifest_is_available() -> None:
    settings = RuntimeSettings.from_environment()
    service = ForecastApiService(
        settings,
        live_price_provider=FixedLivePriceProvider(),
    )

    assert service.repository.has_any() is True
    artifact = service.repository.load("silver", 2)
    assert artifact["active_strategy"] == "persistence"
    assert artifact["horizon_hours"] == 2


def test_silver_predict_uses_live_quote_without_training() -> None:
    settings = RuntimeSettings.from_environment()
    service = ForecastApiService(
        settings,
        live_price_provider=FixedLivePriceProvider(),
    )
    request = type("Request", (), {"metal": "silver", "horizon_hours": 2})()
    result = service.predict(request)

    assert result["active_strategy"] == "persistence"
    assert result["current_usd_per_kg"] == 1250.0
    assert result["predicted_usd_per_kg"] == 1250.0
    assert result["predicted_change_pct"] == 0.0
    assert result["market_data_source"] == "test"
    assert result["prediction_interval_80_low_usd_per_kg"] > 0
    assert result["prediction_interval_80_high_usd_per_kg"] > 0


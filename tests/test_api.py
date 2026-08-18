from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.api.app import create_app  # noqa: E402
from qmg1.api.service import RuntimeSettings  # noqa: E402


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(
        project_root=tmp_path,
        models_dir=tmp_path / "models",
        target_data_dir=tmp_path / "data",
        hourly_context_dir=tmp_path / "hourly",
    )


def test_health_is_boot_safe_without_models_or_market_data(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "QMG1"
    assert payload["architecture"] == "train-once-persist-load-predict"
    assert payload["models_available"] is False
    assert payload["target_data_available"] is False
    assert payload["hourly_context_available"] is False


def test_dashboard_is_served_at_root(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "QMG1" in response.text
    assert 'id="prediction-form"' in response.text
    assert 'src="/assets/js/app.js"' in response.text


def test_api_metadata_exposes_requested_forecast_horizons(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    response = client.get("/api/meta")

    assert response.status_code == 200
    assert response.json()["forecast_horizons_hours"] == [
        2,
        4,
        8,
        12,
        24,
        72,
        168,
        360,
        720,
    ]


def test_dashboard_static_assets_are_mounted(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    css_response = client.get("/assets/css/app.css")
    js_response = client.get("/assets/js/app.js")

    assert css_response.status_code == 200
    assert "--accent" in css_response.text
    assert js_response.status_code == 200
    assert "DashboardController" in js_response.text


def test_predict_returns_service_unavailable_without_persisted_data(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    response = client.post(
        "/predict",
        json={"metal": "silver", "horizon_hours": 2},
    )

    assert response.status_code == 503
    assert "Serving data for silver is not available" in response.json()["detail"]

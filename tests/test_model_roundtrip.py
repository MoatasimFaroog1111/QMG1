from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import TrainingConfig  # noqa: E402
from qmg1.ml.artifacts import ModelArtifactRepository  # noqa: E402
from qmg1.ml.predictor import ForecastPredictor  # noqa: E402
from qmg1.ml.trainer import ForecastTrainer  # noqa: E402


def _write_synthetic_m1(path: Path) -> None:
    minutes = 60 * 1_100
    timestamps = pd.date_range("2024-01-01", periods=minutes, freq="min", tz="UTC")
    x = np.arange(minutes, dtype=float)
    center = 1_000.0 + 0.002 * x + 4.0 * np.sin(x / 2_000.0)
    open_price = center + 0.10 * np.sin(x / 17.0)
    close_price = center + 0.10 * np.cos(x / 19.0)
    high_price = np.maximum(open_price, close_price) + 0.35
    low_price = np.minimum(open_price, close_price) - 0.35

    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open_usd_per_kg": open_price,
            "high_usd_per_kg": high_price,
            "low_usd_per_kg": low_price,
            "close_usd_per_kg": close_price,
            "volume_source_units": 100.0 + 5.0 * np.sin(x / 31.0),
        }
    )
    frame.to_csv(path, index=False)


def test_train_persist_load_predict_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic_m1.csv"
    _write_synthetic_m1(csv_path)

    repository = ModelArtifactRepository(tmp_path / "models")
    trainer = ForecastTrainer(
        artifact_repository=repository,
        config=TrainingConfig(
            min_rows=100,
            cv_splits=2,
            random_state=42,
            max_iter=20,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
        ),
    )

    metrics = trainer.train_one(str(csv_path), "silver", 2)
    artifact_path = repository.path_for("silver", 2)

    assert artifact_path.exists()
    assert metrics.rows_total >= 100
    assert metrics.cv_splits == 1
    assert math.isfinite(metrics.mae_usd_per_kg)

    artifact = repository.load("silver", 2)
    assert artifact["schema_version"] == 6
    assert artifact["validation_method"].startswith("development walk-forward")
    assert artifact["exogenous_features"] == []
    assert "candidate_training_lookback_days" in artifact
    development_candidates = artifact["selection"]["development_candidates"]
    assert len(development_candidates) >= 9
    assert any("lookback_730d" in item["model_name"] for item in development_candidates)
    assert any("lookback_1825d" in item["model_name"] for item in development_candidates)
    assert all(
        candidate["metrics"]["cv_splits"] == 2
        for candidate in development_candidates
    )

    prediction = ForecastPredictor(repository).predict_latest(
        csv_path=str(csv_path),
        metal="silver",
        horizon_hours=2,
    )

    assert prediction["metal"] == "silver"
    assert prediction["horizon_hours"] == 2
    assert prediction["active_strategy"] in {
        "persistence",
        artifact["selected_challenger"],
    }
    assert float(prediction["current_usd_per_kg"]) > 0
    assert float(prediction["predicted_usd_per_kg"]) > 0
    assert math.isfinite(float(prediction["predicted_change_pct"]))

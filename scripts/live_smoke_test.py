#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import TrainingConfig
from qmg1.data.dukascopy import DukascopyDownloader
from qmg1.data.metals import METALS
from qmg1.data.normalizer import UsdPerKgNormalizer
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.predictor import ForecastPredictor
from qmg1.ml.trainer import ForecastTrainer


def main() -> None:
    """Exercise the real market-data -> model -> persisted prediction path."""

    smoke_root = ROOT / "smoke_live"
    raw_root = smoke_root / "raw"
    incoming_root = smoke_root / ".incoming"
    final_root = smoke_root / "final"
    models_root = smoke_root / "models"
    result_path = smoke_root / "smoke_result.json"

    silver = next(metal for metal in METALS if metal.key == "silver")

    # 75 calendar days provides enough history for the longest 720-hour
    # causal feature window while keeping this CI smoke test lightweight.
    now_utc = datetime.now(timezone.utc)
    end_exclusive = now_utc.date()
    start = end_exclusive - timedelta(days=75)
    end_inclusive = (end_exclusive - timedelta(days=1)).isoformat()

    downloader = DukascopyDownloader(raw_root=raw_root, incoming_root=incoming_root)
    downloader.validate_runtime()

    print(f"[SMOKE] live silver data {start} -> {end_exclusive} (exclusive)")
    raw_csv = downloader.download(silver, start, end_exclusive)

    normalizer = UsdPerKgNormalizer(
        output_root=final_root,
        price_side=downloader.config.price_type,
    )
    normalization = normalizer.normalize(silver, [raw_csv], end_inclusive)
    if normalization.rows_written < 50_000:
        raise RuntimeError(
            "Live data smoke test returned too few M1 rows: "
            f"{normalization.rows_written:,}"
        )
    if normalization.output_file is None:
        raise RuntimeError("Normalizer did not produce an output CSV")

    dataset_csv = Path(normalization.output_file)
    repository = ModelArtifactRepository(models_root)
    trainer = ForecastTrainer(
        artifact_repository=repository,
        config=TrainingConfig(
            min_rows=200,
            cv_splits=3,
            random_state=42,
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
        ),
    )

    # A 2-hour live model is enough to prove the entire operational path;
    # full training still uses all nine configured horizons.
    metrics = trainer.train_one(str(dataset_csv), "silver", 2)
    artifact_path = repository.path_for("silver", 2)
    if not artifact_path.exists():
        raise RuntimeError(f"Persisted model missing after training: {artifact_path}")

    prediction = ForecastPredictor(repository).predict_latest(
        csv_path=str(dataset_csv),
        metal="silver",
        horizon_hours=2,
    )
    predicted_price = float(prediction["predicted_usd_per_kg"])
    current_price = float(prediction["current_usd_per_kg"])
    if not (math.isfinite(predicted_price) and predicted_price > 0):
        raise RuntimeError(f"Invalid predicted USD/kg value: {predicted_price}")
    if not (math.isfinite(current_price) and current_price > 0):
        raise RuntimeError(f"Invalid current USD/kg value: {current_price}")

    result = {
        "status": "success",
        "generated_at_utc": now_utc.isoformat(),
        "metal": "silver",
        "source": "Dukascopy live M1 bid data",
        "source_range": {
            "start": start.isoformat(),
            "end_exclusive": end_exclusive.isoformat(),
        },
        "normalization": asdict(normalization),
        "training": {
            "horizon_hours": 2,
            "metrics": asdict(metrics),
            "artifact": str(artifact_path),
        },
        "prediction": prediction,
    }
    smoke_root.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[SMOKE] SUCCESS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

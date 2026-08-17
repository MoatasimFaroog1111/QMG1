#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import TrainingConfig  # noqa: E402
from qmg1.data.dukascopy_direct import (  # noqa: E402
    DirectDukascopyConfig,
    DirectDukascopyM1Downloader,
)
from qmg1.data.metals import METALS  # noqa: E402
from qmg1.data.normalizer import UsdPerKgNormalizer  # noqa: E402
from qmg1.ml.artifacts import ModelArtifactRepository  # noqa: E402
from qmg1.ml.predictor import ForecastPredictor  # noqa: E402
from qmg1.ml.trainer import ForecastTrainer  # noqa: E402


def main() -> None:
    """Exercise real BI5 data -> USD/kg -> train -> persist -> predict."""

    smoke_root = ROOT / "smoke_live"
    raw_root = smoke_root / "raw"
    final_root = smoke_root / "final"
    models_root = smoke_root / "models"
    result_path = smoke_root / "smoke_result.json"

    silver = next(metal for metal in METALS if metal.key == "silver")

    # Stable historical dates make CI reproducible while still proving real
    # connectivity to Dukascopy's production datafeed. The range provides
    # enough trading hours to populate the 720-row longest causal feature.
    start = date(2025, 1, 2)
    end_exclusive = date(2025, 3, 15)
    end_inclusive = "2025-03-14"
    now_utc = datetime.now(timezone.utc)

    downloader = DirectDukascopyM1Downloader(
        raw_root=raw_root,
        config=DirectDukascopyConfig(
            timeout_seconds=60,
            max_attempts=4,
            base_backoff_seconds=2.0,
            request_pause_seconds=0.05,
        ),
    )
    downloader.validate_runtime()

    print(
        f"[SMOKE] silver {start} -> {end_exclusive} (exclusive) "
        f"provider={downloader.provider_description}"
    )
    raw_csv = downloader.download(silver, start, end_exclusive)

    normalizer = UsdPerKgNormalizer(
        output_root=final_root,
        price_side=downloader.price_side,
    )
    normalization = normalizer.normalize(silver, [raw_csv], end_inclusive)
    if normalization.rows_written < 40_000:
        raise RuntimeError(
            "Real-data smoke test returned too few M1 rows: "
            f"{normalization.rows_written:,}"
        )
    if normalization.output_file is None:
        raise RuntimeError("Normalizer did not produce an output CSV")

    dataset_csv = Path(normalization.output_file)
    repository = ModelArtifactRepository(models_root)
    trainer = ForecastTrainer(
        artifact_repository=repository,
        config=TrainingConfig(
            min_rows=150,
            cv_splits=3,
            random_state=42,
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
        ),
    )

    # One horizon proves the complete operational path. Full training uses all
    # nine configured horizons and remains resume-safe through persisted files.
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
        "source": "Dukascopy M1 bid data",
        "source_engine": downloader.provider_description,
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

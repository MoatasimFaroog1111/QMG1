#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import HORIZONS_HOURS, TrainingConfig
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.trainer import ForecastTrainer


PATTERNS = {
    "gold": "XAUUSD_H1_USD_PER_KG_*.csv",
    "silver": "XAGUSD_H1_USD_PER_KG_*.csv",
    "palladium": "XPDCMDUSD_H1_USD_PER_KG_*.csv",
    "platinum": "XPTCMDUSD_H1_USD_PER_KG_*.csv",
}


def newest_matching(data_dir: Path, pattern: str) -> Path:
    matches = sorted(
        data_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No hourly dataset matching {pattern} in {data_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and persist QMG1 models from compact H1 datasets"
    )
    parser.add_argument("--metal", choices=PATTERNS, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "training_data" / "hourly")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=list(HORIZONS_HOURS),
    )
    args = parser.parse_args()

    csv_path = newest_matching(args.data_dir, PATTERNS[args.metal])
    trainer = ForecastTrainer(
        artifact_repository=ModelArtifactRepository(args.models_dir),
        config=TrainingConfig(),
    )
    metrics = trainer.train_all_hourly(
        str(csv_path),
        args.metal,
        horizons=args.horizons,
    )

    for item in metrics:
        print(
            f"[OK] {args.metal:10s} {item.horizon_hours:4d}h "
            f"MAE={item.mae_usd_per_kg:.2f} USD/kg "
            f"direction={item.directional_accuracy_pct:.2f}% "
            f"vs_persistence={item.improvement_vs_persistence_pct:+.2f}%"
        )


if __name__ == "__main__":
    main()

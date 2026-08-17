#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import HORIZONS_HOURS, ProjectPaths, TrainingConfig  # noqa: E402
from qmg1.ml.artifacts import ModelArtifactRepository  # noqa: E402
from qmg1.ml.trainer import ForecastTrainer  # noqa: E402


METAL_PATTERNS = {
    "gold": "XAUUSD_M1_USD_PER_KG_*.csv",
    "silver": "XAGUSD_M1_USD_PER_KG_*.csv",
    "palladium": "XPDCMDUSD_M1_USD_PER_KG_*.csv",
    "platinum": "XPTCMDUSD_M1_USD_PER_KG_*.csv",
}


def newest_matching(data_dir: Path, pattern: str) -> Path:
    matches = sorted(
        data_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No dataset matching {pattern} in {data_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train once, validate candidates, and persist QMG1 models"
    )
    parser.add_argument("--metal", choices=[*METAL_PATTERNS, "all"], default="all")
    parser.add_argument(
        "--horizon",
        type=int,
        choices=HORIZONS_HOURS,
        nargs="+",
        default=None,
        help="Optional subset of forecast horizons in hours",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when a persisted artifact already exists",
    )
    args = parser.parse_args()

    paths = ProjectPaths(ROOT)
    data_dir = args.data_dir or paths.data_dir
    models_dir = args.models_dir or paths.models_dir
    requested_horizons = tuple(args.horizon or HORIZONS_HOURS)
    selected = (
        METAL_PATTERNS
        if args.metal == "all"
        else {args.metal: METAL_PATTERNS[args.metal]}
    )

    repository = ModelArtifactRepository(models_dir)
    trainer = ForecastTrainer(
        artifact_repository=repository,
        config=TrainingConfig(),
    )

    for metal, pattern in selected.items():
        csv_path = newest_matching(data_dir, pattern)
        print(f"[DATA] {metal}: {csv_path}")

        pending: list[int] = []
        for horizon in requested_horizons:
            artifact_path = repository.path_for(metal, horizon)
            if artifact_path.exists() and not args.force:
                print(f"[SKIP] {metal:10s} {horizon:4d}h artifact={artifact_path}")
            else:
                pending.append(horizon)

        if not pending:
            print(f"[DONE] {metal}: all requested persisted models already exist")
            continue

        metrics = trainer.train_all(str(csv_path), metal, horizons=pending)
        for item in metrics:
            artifact = repository.load(metal, item.horizon_hours)
            status = artifact["operational_status"]
            gate_label = "ACCEPTED" if status["accepted"] else "REJECTED"
            print(
                f"[EVAL] {metal:10s} {item.horizon_hours:4d}h "
                f"model={artifact['selected_model']} gate={gate_label} "
                f"MAE={item.mae_usd_per_kg:.2f} USD/kg "
                f"overall_vs_persistence={item.improvement_vs_persistence_pct:+.2f}% "
                f"recent_vs_persistence="
                f"{item.latest_fold_improvement_vs_persistence_pct:+.2f}%"
            )


if __name__ == "__main__":
    main()

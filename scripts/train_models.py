#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import ProjectPaths, TrainingConfig  # noqa: E402
from qmg1.modeling import train_all_horizons  # noqa: E402


METAL_PATTERNS = {
    "gold": "XAUUSD_M1_USD_PER_KG_*.csv",
    "silver": "XAGUSD_M1_USD_PER_KG_*.csv",
    "palladium": "XPDCMDUSD_M1_USD_PER_KG_*.csv",
    "platinum": "XPTCMDUSD_M1_USD_PER_KG_*.csv",
}


def newest_matching(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No dataset matching {pattern} in {data_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist QMG1 metal forecasting models")
    parser.add_argument("--metal", choices=[*METAL_PATTERNS, "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = ProjectPaths(ROOT)
    data_dir = args.data_dir or paths.data_dir
    models_dir = args.models_dir or paths.models_dir
    selected = METAL_PATTERNS if args.metal == "all" else {args.metal: METAL_PATTERNS[args.metal]}

    cfg = TrainingConfig()
    for metal, pattern in selected.items():
        csv_path = newest_matching(data_dir, pattern)
        output_dir = models_dir / metal
        metrics = train_all_horizons(str(csv_path), output_dir, metal, cfg)
        for m in metrics:
            print(
                f"[OK] {metal:10s} {m.horizon_hours:4d}h "
                f"MAE={m.mae_usd_per_kg:.2f} USD/kg "
                f"direction={m.directional_accuracy_pct:.2f}% "
                f"vs_persistence={m.improvement_vs_persistence_pct:+.2f}%"
            )


if __name__ == "__main__":
    main()

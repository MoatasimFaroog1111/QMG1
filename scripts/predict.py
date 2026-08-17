#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import HORIZONS_HOURS, ProjectPaths  # noqa: E402
from qmg1.modeling import predict_latest  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Load persisted QMG1 models and predict")
    parser.add_argument("--metal", choices=METAL_PATTERNS, required=True)
    parser.add_argument("--horizon", type=int, choices=HORIZONS_HOURS, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = ProjectPaths(ROOT)
    data_dir = args.data_dir or paths.data_dir
    models_dir = args.models_dir or paths.models_dir

    csv_path = newest_matching(data_dir, METAL_PATTERNS[args.metal])
    artifact = models_dir / args.metal / f"{args.metal}_{args.horizon}h.joblib"
    if not artifact.exists():
        raise FileNotFoundError(
            f"Persisted model not found: {artifact}. Run scripts/train_models.py first."
        )

    result = predict_latest(str(csv_path), artifact)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

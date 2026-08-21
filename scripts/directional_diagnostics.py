#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.config import HORIZONS_HOURS, TrainingConfig  # noqa: E402
from qmg1.ml.dataset import ForecastDatasetBuilder  # noqa: E402
from qmg1.ml.directional import DirectionDiagnostics  # noqa: E402
from qmg1.ml.exogenous import (  # noqa: E402
    GoldSilverFeatureProvider,
    SpxFeatureProvider,
    UsdIndexFeatureProvider,
)


PATTERNS = {
    "gold": "XAUUSD_H1_USD_PER_KG_*.csv",
    "silver": "XAGUSD_H1_USD_PER_KG_*.csv",
}
UDX_PATTERN = "UDXUSD_H1_NATIVE_*.csv"
SPX_PATTERN = "SPXUSD_H1_NATIVE_*.csv"


def newest_matching(data_dir: Path, pattern: str) -> Path:
    matches = sorted(
        data_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No hourly dataset matching {pattern} in {data_dir}")
    return matches[0]


def silver_dataset_builder(data_dir: Path) -> ForecastDatasetBuilder:
    gold_csv = newest_matching(data_dir, PATTERNS["gold"])
    udx_csv = newest_matching(data_dir, UDX_PATTERN)
    spx_csv = newest_matching(data_dir, SPX_PATTERN)
    print(f"[EXOG] silver <- gold {gold_csv.name}")
    print(f"[EXOG] silver <- UDX  {udx_csv.name}")
    print(f"[EXOG] silver <- SPX  {spx_csv.name}")
    return ForecastDatasetBuilder(
        exogenous_providers=[
            GoldSilverFeatureProvider.from_hourly_csv(gold_csv),
            UsdIndexFeatureProvider.from_hourly_csv(udx_csv),
            SpxFeatureProvider.from_hourly_csv(spx_csv),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe Silver direction diagnostics on Development and an "
            "untouched Holdout without changing production models."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "training_data" / "hourly",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "directional_diagnostics.json",
    )
    args = parser.parse_args()

    silver_csv = newest_matching(args.data_dir, PATTERNS["silver"])
    builder = silver_dataset_builder(args.data_dir)
    base = builder.load_hourly_feature_base(silver_csv)
    config = TrainingConfig()

    report_rows: list[dict[str, object]] = []
    for horizon in HORIZONS_HOURS:
        prepared = builder.build_from_base(base, horizon)
        report = DirectionDiagnostics.evaluate_with_holdout(
            frame=prepared.frame,
            feature_columns=prepared.feature_columns,
            horizon_hours=horizon,
            cv_splits=config.cv_splits,
        )
        report_rows.append(asdict(report))
        dev = report.development
        holdout = report.holdout
        print(
            f"{horizon:>4}h  "
            f"DEV acc={dev.accuracy_pct:6.2f}% bal={dev.balanced_accuracy_pct:6.2f}% "
            f"base={dev.majority_baseline_accuracy_pct:6.2f}% "
            f"delta={dev.improvement_vs_majority_pp:+6.2f}pp | "
            f"HOLDOUT acc={holdout.accuracy_pct:6.2f}% "
            f"bal={holdout.balanced_accuracy_pct:6.2f}% "
            f"base={holdout.majority_baseline_accuracy_pct:6.2f}% "
            f"delta={holdout.improvement_vs_majority_pp:+6.2f}pp"
        )

    payload = {
        "metal": "silver",
        "horizons_hours": list(HORIZONS_HOURS),
        "method": (
            "balanced logistic classifier; development walk-forward + target-time "
            "purge; untouched 20% holdout; diagnostic only; no production effect"
        ),
        "source_csv": str(silver_csv),
        "results": report_rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from qmg1.config import HORIZONS_HOURS, TrainingConfig
from qmg1.ml.dataset import ForecastDatasetBuilder
from qmg1.ml.directional import DirectionDiagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe directional diagnostics without changing production models."
    )
    parser.add_argument("csv_path", help="Hourly or M1 source dataset path")
    parser.add_argument("--hourly", action="store_true", help="Treat source as H1 data")
    parser.add_argument("--output", default="directional_diagnostics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = ForecastDatasetBuilder()
    base = (
        builder.load_hourly_feature_base(args.csv_path)
        if args.hourly
        else builder.load_feature_base(args.csv_path)
    )
    config = TrainingConfig()
    rows: list[dict[str, object]] = []

    for horizon in HORIZONS_HOURS:
        prepared = builder.build_from_base(base, horizon)
        split_at = int(len(prepared.frame) * 0.80)
        development = prepared.frame.iloc[:split_at]
        metrics = DirectionDiagnostics.walk_forward_classifier(
            frame=development,
            feature_columns=prepared.feature_columns,
            horizon_hours=horizon,
            cv_splits=config.cv_splits,
        )
        row = {"horizon_hours": horizon, **asdict(metrics)}
        rows.append(row)
        print(
            f"{horizon:>4}h  accuracy={metrics.accuracy_pct:.2f}%  "
            f"balanced={metrics.balanced_accuracy_pct:.2f}%  "
            f"majority={metrics.majority_baseline_accuracy_pct:.2f}%  "
            f"delta={metrics.improvement_vs_majority_pp:+.2f}pp"
        )

    output = Path(args.output)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

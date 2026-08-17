#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.data.hourly import HourlyTrainingDataPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download M1 chunks and compact them to H1 USD/kg training data"
    )
    parser.add_argument(
        "--metal",
        choices=["gold", "silver", "palladium", "platinum"],
        required=True,
    )
    parser.add_argument("--root", type=Path, default=ROOT / "training_data")
    parser.add_argument("--cleanup-raw", action="store_true")
    args = parser.parse_args()

    result = HourlyTrainingDataPipeline(args.root).run(
        args.metal,
        cleanup_raw=args.cleanup_raw,
    )
    payload = asdict(result)
    payload["output_file"] = str(result.output_file)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

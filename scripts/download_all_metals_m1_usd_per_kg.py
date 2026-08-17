#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.data.metals import METALS  # noqa: E402
from qmg1.data.pipeline import MetalsDataPipeline  # noqa: E402


METAL_BY_KEY = {metal.key: metal for metal in METALS}


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected YYYY-MM-DD") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize precious-metals M1 history to USD/kg"
    )
    parser.add_argument(
        "--metal",
        choices=[*METAL_BY_KEY, "all"],
        default="all",
        help="Download one metal or all configured metals",
    )
    parser.add_argument(
        "--start",
        type=_iso_date,
        default=None,
        help="Optional inclusive UTC start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=_iso_date,
        default=None,
        help="Optional exclusive UTC end date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    selected = METALS if args.metal == "all" else (METAL_BY_KEY[args.metal],)
    pipeline = MetalsDataPipeline(ROOT / "metals_m1_usd_per_kg")
    result = pipeline.run(
        metals=selected,
        start=args.start,
        end_exclusive=args.end,
    )

    print("\nFINAL SUMMARY")
    for report in result["reports"]:
        print(
            f"{report['metal']:10s} source={report['source']} "
            f"rows={report['rows_written']:,} "
            f"first={report['first_timestamp_utc']} "
            f"last={report['last_timestamp_utc']}"
        )
        print(f"  -> {report['output_file']}")

    failures = result["failures"]
    if failures:
        print(f"WARNING: {len(failures)} chunk(s) failed; re-run to retry them.")
        for failure in failures[:10]:
            print(f"  {failure}")

    print(f"Report: {ROOT / 'metals_m1_usd_per_kg' / 'download_report.json'}")


if __name__ == "__main__":
    main()

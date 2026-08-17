#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.data.pipeline import MetalsDataPipeline  # noqa: E402


def main() -> None:
    pipeline = MetalsDataPipeline(ROOT / "metals_m1_usd_per_kg")
    result = pipeline.run()

    print("\nFINAL SUMMARY")
    for report in result["reports"]:
        print(
            f"{report['metal']:10s} rows={report['rows_written']:,} "
            f"first={report['first_timestamp_utc']} "
            f"last={report['last_timestamp_utc']}"
        )
        print(f"  -> {report['output_file']}")

    failures = result["failures"]
    if failures:
        print(f"WARNING: {len(failures)} chunk(s) failed; re-run to retry them.")

    print(f"Report: {ROOT / 'metals_m1_usd_per_kg' / 'download_report.json'}")


if __name__ == "__main__":
    main()

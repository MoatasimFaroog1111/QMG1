#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.data.histdata import HistDataConfig, HistDataM1Downloader  # noqa: E402
from qmg1.data.metals import METALS  # noqa: E402
from qmg1.data.normalizer import UsdPerKgNormalizer  # noqa: E402


def main() -> None:
    """Exercise HistData bulk ZIP -> provider CSV -> USD/kg normalization."""

    smoke_root = ROOT / "smoke_histdata"
    raw_root = smoke_root / "raw"
    final_root = smoke_root / "final"
    result_path = smoke_root / "smoke_result.json"

    silver = next(metal for metal in METALS if metal.key == "silver")
    start = date(2026, 7, 1)
    end_exclusive = date(2026, 8, 1)

    provider = HistDataM1Downloader(
        raw_root=raw_root,
        config=HistDataConfig(
            timeout_seconds=120,
            max_attempts=5,
            download_pause_seconds=0.0,
        ),
    )
    provider.validate_runtime()

    print(
        f"[SMOKE] HistData silver {start} -> {end_exclusive} (exclusive) "
        f"provider={provider.provider_description}"
    )
    raw_csv = provider.download(silver, start, end_exclusive)

    normalizer = UsdPerKgNormalizer(
        output_root=final_root,
        price_side=provider.price_side,
    )
    normalization = normalizer.normalize(
        silver,
        [raw_csv],
        "2026-07-31",
        source_name=provider.source_name_for(silver),
        start_inclusive=start.isoformat(),
    )
    if normalization.rows_written < 20_000:
        raise RuntimeError(
            "HistData smoke returned too few July M1 rows: "
            f"{normalization.rows_written:,}"
        )
    if normalization.output_file is None:
        raise RuntimeError("HistData normalization did not produce an output CSV")

    final_csv = Path(normalization.output_file)
    first: dict[str, str] | None = None
    last: dict[str, str] | None = None
    with final_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if first is None:
                first = row
            last = row

    if first is None or last is None:
        raise RuntimeError("HistData normalized CSV is empty")

    first_close = float(first["close_usd_per_kg"])
    last_close = float(last["close_usd_per_kg"])
    if not all(
        math.isfinite(value) and value > 0 for value in (first_close, last_close)
    ):
        raise RuntimeError("HistData normalized USD/kg prices are invalid")

    result = {
        "status": "success",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metal": "silver",
        "source": provider.source_name_for(silver),
        "source_engine": provider.provider_description,
        "source_range": {
            "start": start.isoformat(),
            "end_exclusive": end_exclusive.isoformat(),
        },
        "normalization": asdict(normalization),
        "first_close_usd_per_kg": first_close,
        "last_close_usd_per_kg": last_close,
    }
    smoke_root.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[SMOKE] HISTDATA SUCCESS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

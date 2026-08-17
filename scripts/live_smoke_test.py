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

from qmg1.data.dukascopy_direct import (  # noqa: E402
    DirectDukascopyConfig,
    DirectDukascopyM1Downloader,
)
from qmg1.data.metals import METALS  # noqa: E402
from qmg1.data.normalizer import UsdPerKgNormalizer  # noqa: E402


def main() -> None:
    """Exercise the real Dukascopy BI5 -> USD/kg source path."""

    smoke_root = ROOT / "smoke_live"
    raw_root = smoke_root / "raw"
    final_root = smoke_root / "final"
    result_path = smoke_root / "smoke_result.json"

    silver = next(metal for metal in METALS if metal.key == "silver")
    start = date(2025, 1, 3)
    end_exclusive = date(2025, 1, 4)
    now_utc = datetime.now(timezone.utc)

    downloader = DirectDukascopyM1Downloader(
        raw_root=raw_root,
        config=DirectDukascopyConfig(
            timeout_seconds=60,
            max_attempts=6,
            base_backoff_seconds=2.0,
            request_pause_seconds=0.0,
            download_passes=2,
            pass_backoff_seconds=5.0,
        ),
    )
    downloader.validate_runtime()

    print(
        f"[SMOKE] silver {start} -> {end_exclusive} (exclusive) "
        f"provider={downloader.provider_description}"
    )
    raw_csv = downloader.download(silver, start, end_exclusive)

    normalizer = UsdPerKgNormalizer(
        output_root=final_root,
        price_side=downloader.price_side,
    )
    normalization = normalizer.normalize(silver, [raw_csv], "2025-01-03")
    if normalization.rows_written < 1_000:
        raise RuntimeError(
            "Real-data smoke test returned too few M1 rows: "
            f"{normalization.rows_written:,}"
        )
    if normalization.output_file is None:
        raise RuntimeError("Normalizer did not produce an output CSV")

    final_csv = Path(normalization.output_file)
    with final_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    first = rows[0]
    last = rows[-1]
    first_close = float(first["close_usd_per_kg"])
    last_close = float(last["close_usd_per_kg"])
    if not all(
        math.isfinite(value) and value > 0 for value in (first_close, last_close)
    ):
        raise RuntimeError("Normalized live-market USD/kg prices are invalid")

    result = {
        "status": "success",
        "generated_at_utc": now_utc.isoformat(),
        "metal": "silver",
        "source": "Dukascopy real M1 bid data",
        "source_engine": downloader.provider_description,
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

    print("[SMOKE] SUCCESS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

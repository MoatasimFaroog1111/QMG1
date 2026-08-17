from __future__ import annotations

import lzma
import struct
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.data.dukascopy_direct import DirectDukascopyM1Downloader  # noqa: E402
from qmg1.data.metals import METALS  # noqa: E402


def test_dukas_m1_url_uses_zero_based_month(tmp_path: Path) -> None:
    silver = next(metal for metal in METALS if metal.key == "silver")
    downloader = DirectDukascopyM1Downloader(tmp_path)

    assert downloader.url_for_day(silver, date(2025, 1, 3)) == (
        "https://datafeed.dukascopy.com/datafeed/"
        "XAGUSD/2025/00/03/BID_candles_min_1.bi5"
    )
    assert downloader.url_for_day(silver, date(2025, 12, 31)) == (
        "https://datafeed.dukascopy.com/datafeed/"
        "XAGUSD/2025/11/31/BID_candles_min_1.bi5"
    )


def test_decode_lzma_big_endian_m1_payload(tmp_path: Path) -> None:
    silver = next(metal for metal in METALS if metal.key == "silver")
    downloader = DirectDukascopyM1Downloader(tmp_path)

    raw = b"".join(
        [
            struct.pack(">5if", 0, 31_000, 31_100, 30_900, 31_200, 1.25),
            struct.pack(">5if", 60, 31_100, 31_050, 31_000, 31_250, 2.50),
        ]
    )
    payload = lzma.compress(raw, format=lzma.FORMAT_ALONE)
    day = date(2025, 1, 3)

    rows = downloader.decode_day(payload, day, silver.dukascopy_decimal_factor)

    day_start_ms = int(
        datetime(2025, 1, 3, tzinfo=timezone.utc).timestamp() * 1000
    )
    assert rows == [
        (day_start_ms, 31.0, 31.2, 30.9, 31.1, 1_250_000.0),
        (day_start_ms + 60_000, 31.1, 31.25, 31.0, 31.05, 2_500_000.0),
    ]

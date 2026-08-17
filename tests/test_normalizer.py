from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.data.metals import METALS  # noqa: E402
from qmg1.data.normalizer import UsdPerKgNormalizer  # noqa: E402


def test_troy_ounce_to_kg_conversion(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1735689600000,31.1034768,31.1034768,31.1034768,31.1034768,1\n",
        encoding="utf-8",
    )

    gold = next(metal for metal in METALS if metal.key == "gold")
    report = UsdPerKgNormalizer(tmp_path / "final").normalize(
        gold,
        [raw],
        "2025-01-01",
    )

    assert report.rows_written == 1
    assert report.output_file is not None

    with Path(report.output_file).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["open_usd_per_kg"] == "1000.000000"
    assert row["close_usd_per_kg"] == "1000.000000"
    assert row["price_unit"] == "USD/kg"

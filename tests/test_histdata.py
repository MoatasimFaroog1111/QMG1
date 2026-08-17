from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.data.histdata import HistDataM1Downloader, HistDataPeriod, _TokenParser  # noqa: E402
from qmg1.data.hybrid import HybridPreciousMetalsM1Provider  # noqa: E402
from qmg1.data.metals import METALS  # noqa: E402


def test_histdata_fixed_est_timestamp_is_converted_to_utc(tmp_path: Path) -> None:
    provider = HistDataM1Downloader(tmp_path)
    timestamp_ms = provider._timestamp_utc_ms("20250103 000000")
    expected_ms = int(
        datetime(2025, 1, 3, 5, 0, tzinfo=timezone.utc).timestamp() * 1000
    )
    assert timestamp_ms == expected_ms


def test_histdata_uses_annual_archives_for_past_year(tmp_path: Path) -> None:
    provider = HistDataM1Downloader(tmp_path)
    periods = provider._periods("XAGUSD", date(2025, 5, 1), date(2026, 1, 1))
    assert periods == [HistDataPeriod(pair="XAGUSD", year=2025)]


def test_histdata_uses_monthly_archives_for_current_2026(tmp_path: Path) -> None:
    provider = HistDataM1Downloader(tmp_path)
    periods = provider._periods("XAGUSD", date(2026, 7, 1), date(2026, 9, 1))
    assert periods == [
        HistDataPeriod(pair="XAGUSD", year=2026, month=7),
        HistDataPeriod(pair="XAGUSD", year=2026, month=8),
    ]


def test_histdata_token_parser_reads_hidden_token() -> None:
    parser = _TokenParser()
    parser.feed('<html><input type="hidden" id="tk" value="abc123"></html>')
    assert parser.token == "abc123"


def test_hybrid_routes_gold_silver_to_histdata_and_pgms_to_dukascopy(
    tmp_path: Path,
) -> None:
    provider = HybridPreciousMetalsM1Provider(tmp_path)
    by_key = {metal.key: metal for metal in METALS}

    assert provider.source_name_for(by_key["gold"]) == "HistData"
    assert provider.source_name_for(by_key["silver"]) == "HistData"
    assert provider.source_name_for(by_key["palladium"]) == "Dukascopy"
    assert provider.source_name_for(by_key["platinum"]) == "Dukascopy"

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.ml.artifacts import ModelArtifactRepository  # noqa: E402
from qmg1.serving.live_price import DukascopyLivePriceProvider  # noqa: E402


class CountingDownloader:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.calls = 0

    def download(self, *_args) -> Path:
        self.calls += 1
        return self.csv_path


def test_live_quote_requests_are_single_flight_and_cached(tmp_path: Path) -> None:
    csv_path = tmp_path / "quote.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1787097600000,40,41,39,40,1\n",
        encoding="utf-8",
    )
    downloader = CountingDownloader(csv_path)
    provider = DukascopyLivePriceProvider(tmp_path, cache_ttl_seconds=60)
    provider._downloader = lambda _path: downloader  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=8) as executor:
        quotes = list(executor.map(lambda _index: provider.latest_quote("silver"), range(8)))

    assert downloader.calls == 1
    assert {quote.close_usd_per_kg for quote in quotes} == {quotes[0].close_usd_per_kg}


def test_model_artifact_checksum_blocks_tampering(tmp_path: Path) -> None:
    repository = ModelArtifactRepository(tmp_path)
    path = repository.save("silver", 2, {"active_strategy": "persistence"})

    assert repository.load("silver", 2)["active_strategy"] == "persistence"
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        repository.load("silver", 2)

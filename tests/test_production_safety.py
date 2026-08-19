from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.ml.artifacts import ModelArtifactRepository  # noqa: E402
from qmg1.serving import live_price  # noqa: E402
from qmg1.serving.live_price import (  # noqa: E402
    BullionVaultLivePriceProvider,
    DukascopyLivePriceProvider,
    LivePriceUnavailableError,
    LiveQuote,
    ResilientLivePriceProvider,
)


class CountingDownloader:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.calls = 0

    def download(self, *_args) -> Path:
        self.calls += 1
        return self.csv_path


class QuoteProvider:
    configured = True

    def __init__(self, quote: LiveQuote | None = None) -> None:
        self.quote = quote
        self.calls = 0

    def latest_quote(self, _metal_key: str) -> LiveQuote:
        self.calls += 1
        if self.quote is None:
            raise LivePriceUnavailableError("provider unavailable")
        return self.quote


class XmlResponse:
    headers = {"Date": "Wed, 19 Aug 2026 07:00:00 GMT"}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return b"""<envelope><message><market><pitches>
        <pitch securityId="AGXLN" considerationCurrency="USD">
        <buyPrices><price limit="2027"/></buyPrices>
        <sellPrices><price limit="2042"/></sellPrices>
        </pitch></pitches></market></message></envelope>"""


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


def test_bullionvault_quote_uses_usd_per_kg_midpoint(monkeypatch) -> None:
    monkeypatch.setattr(live_price, "urlopen", lambda *_args, **_kwargs: XmlResponse())

    quote = BullionVaultLivePriceProvider().latest_quote("silver")

    assert quote.close_usd_per_kg == 2034.5
    assert quote.timestamp_utc == datetime(2026, 8, 19, 7, tzinfo=timezone.utc)
    assert "BullionVault" in quote.source


def test_model_artifact_checksum_blocks_tampering(tmp_path: Path) -> None:
    repository = ModelArtifactRepository(tmp_path)
    path = repository.save("silver", 2, {"active_strategy": "persistence"})

    assert repository.load("silver", 2)["active_strategy"] == "persistence"
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        repository.load("silver", 2)


def test_resilient_provider_fails_over_and_persists_quote(tmp_path: Path) -> None:
    quote = LiveQuote(
        metal="silver",
        timestamp_utc=datetime.now(timezone.utc),
        close_usd_per_kg=2028.5,
        source="BullionVault test midpoint",
    )
    failing = QuoteProvider()
    working = QuoteProvider(quote)
    provider = ResilientLivePriceProvider([failing, working], tmp_path)

    assert provider.latest_quote("silver").close_usd_per_kg == 2028.5
    assert failing.calls == 1
    assert working.calls == 1
    assert (tmp_path / "silver.json").is_file()


def test_resilient_provider_uses_persistent_quote_during_outage(tmp_path: Path) -> None:
    quote = LiveQuote(
        metal="silver",
        timestamp_utc=datetime.now(timezone.utc),
        close_usd_per_kg=2028.5,
        source="BullionVault test midpoint",
    )
    warm = ResilientLivePriceProvider([QuoteProvider(quote)], tmp_path)
    warm.latest_quote("silver")

    cold = ResilientLivePriceProvider([QuoteProvider()], tmp_path)
    cached = cold.latest_quote("silver")

    assert cached.close_usd_per_kg == 2028.5
    assert "persistent cache" in cached.source


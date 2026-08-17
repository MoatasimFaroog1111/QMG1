#!/usr/bin/env python3
"""Download M1 precious-metals data and normalize OHLC to USD/kg.

Source transport: dukascopy-node (invoked with npx).
Requested window: 2009-05-01 through August 2026.
Gold/silver start at the requested date; platinum/palladium start at the first
configured M1 date available from the source.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path

DUKASCOPY_NODE_VERSION = "1.50.0"
TIMEFRAME = "m1"
PRICE_TYPE = "bid"
REQUESTED_START = date(2009, 5, 1)
REQUESTED_END_EXCLUSIVE = date(2026, 9, 1)
ROOT = Path("metals_m1_usd_per_kg")
RAW_ROOT = ROOT / "raw"
FINAL_ROOT = ROOT / "final"
REPORT_FILE = ROOT / "download_report.json"

getcontext().prec = 40
TROY_OUNCE_GRAMS = Decimal("31.1034768")
TROY_OUNCES_PER_KG = Decimal("1000") / TROY_OUNCE_GRAMS
PRICE_QUANT = Decimal("0.000001")


@dataclass(frozen=True)
class Metal:
    key: str
    name: str
    symbol: str
    instrument: str
    first_m1: date

    @property
    def start(self) -> date:
        return max(REQUESTED_START, self.first_m1)


METALS = (
    Metal("gold", "Gold", "XAU/USD", "xauusd", date(2003, 5, 5)),
    Metal("silver", "Silver", "XAG/USD", "xagusd", date(2003, 5, 4)),
    Metal("palladium", "Palladium", "XPD.CMD/USD", "xpdcmdusd", date(2021, 7, 4)),
    Metal("platinum", "Platinum", "XPT.CMD/USD", "xptcmdusd", date(2021, 11, 1)),
)


@dataclass
class Report:
    metal: str
    source_symbol: str
    start: str
    end_exclusive: str
    rows: int = 0
    duplicates: int = 0
    malformed: int = 0
    first_timestamp_utc: str | None = None
    last_timestamp_utc: str | None = None
    output_file: str | None = None


def require_runtime() -> None:
    for binary in ("node", "npx"):
        if shutil.which(binary) is None:
            raise SystemExit(f"ERROR: {binary} is required")
    version = subprocess.run(
        ["node", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip().lstrip("v")
    if int(version.split(".", 1)[0]) < 18:
        raise SystemExit(f"ERROR: Node.js 18+ required; found {version}")


def end_exclusive() -> date:
    return min(REQUESTED_END_EXCLUSIVE, datetime.now(timezone.utc).date())


def chunks(start: date, end: date):
    cursor = start
    while cursor < end:
        next_year = date(cursor.year + 1, 1, 1)
        chunk_end = min(next_year, end)
        yield cursor, chunk_end
        cursor = chunk_end


def raw_path(metal: Metal, start: date, end: date) -> Path:
    return RAW_ROOT / metal.key / f"{metal.instrument}_{start}_{end}_{TIMEFRAME}.csv"


def download_chunk(metal: Metal, start: date, end: date) -> Path:
    destination = raw_path(metal, start, end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100:
        print(f"[SKIP] {metal.name:10s} {start} -> {end}")
        return destination

    incoming = ROOT / ".incoming" / metal.key / f"{start}_{end}"
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir(parents=True, exist_ok=True)

    command = [
        "npx", "--yes", f"dukascopy-node@{DUKASCOPY_NODE_VERSION}",
        "-i", metal.instrument,
        "-from", start.isoformat(),
        "-to", end.isoformat(),
        "-t", TIMEFRAME,
        "-p", PRICE_TYPE,
        "-v",
        "-vu", "units",
        "-f", "csv",
        "-dir", str(incoming.resolve()),
        "-r", "5",
        "-rp", "1000",
        "-s",
    ]
    print(f"[GET ] {metal.name:10s} {start} -> {end}")
    subprocess.run(command, check=True)

    candidates = [p for p in incoming.rglob("*.csv") if p.is_file()]
    if not candidates:
        raise RuntimeError(f"No CSV produced for {metal.name} {start} -> {end}")
    source = max(candidates, key=lambda p: p.stat().st_size)
    shutil.move(str(source), destination)
    shutil.rmtree(incoming, ignore_errors=True)
    return destination


def decimal_price(value: str) -> Decimal:
    try:
        price = Decimal(value.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(value) from exc
    if not price.is_finite() or price <= 0:
        raise ValueError(value)
    return price


def usd_kg(value: str) -> str:
    return format((decimal_price(value) * TROY_OUNCES_PER_KG).quantize(PRICE_QUANT), "f")


def timestamp_ms(value: str) -> int:
    ts = int(Decimal(value.strip()))
    if 0 < ts < 10_000_000_000:
        ts *= 1000
    return ts


def normalize_headers(row: dict[str, str]) -> dict[str, str]:
    return {str(k).strip().lower(): v for k, v in row.items() if k is not None}


def convert_metal(metal: Metal, files: list[Path], end: date) -> Report:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    output = FINAL_ROOT / (
        f"{metal.symbol.replace('/', '').replace('.', '')}_M1_USD_PER_KG_"
        f"{metal.start}_to_{end - timedelta(days=1)}.csv"
    )
    report = Report(metal.name, metal.symbol, metal.start.isoformat(), end.isoformat())
    report.output_file = str(output)
    last_ts: int | None = None

    with output.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "timestamp_utc", "timestamp_ms",
                "open_usd_per_kg", "high_usd_per_kg",
                "low_usd_per_kg", "close_usd_per_kg",
                "volume_source_units", "metal", "source_symbol",
                "source", "price_side", "price_unit",
            ],
        )
        writer.writeheader()

        for path in files:
            print(f"[CONV] {metal.name:10s} {path.name}")
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as src:
                for original in csv.DictReader(src):
                    row = normalize_headers(original)
                    try:
                        ts = timestamp_ms(row["timestamp"])
                        if last_ts is not None and ts <= last_ts:
                            report.duplicates += 1
                            continue
                        o = decimal_price(row["open"])
                        h = decimal_price(row["high"])
                        l = decimal_price(row["low"])
                        c = decimal_price(row["close"])
                        if l > h or not (l <= o <= h and l <= c <= h):
                            raise ValueError("invalid OHLC")
                        iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(
                            timespec="seconds"
                        ).replace("+00:00", "Z")
                        writer.writerow(
                            {
                                "timestamp_utc": iso,
                                "timestamp_ms": ts,
                                "open_usd_per_kg": usd_kg(row["open"]),
                                "high_usd_per_kg": usd_kg(row["high"]),
                                "low_usd_per_kg": usd_kg(row["low"]),
                                "close_usd_per_kg": usd_kg(row["close"]),
                                "volume_source_units": row.get("volume", ""),
                                "metal": metal.name,
                                "source_symbol": metal.symbol,
                                "source": "Dukascopy",
                                "price_side": PRICE_TYPE,
                                "price_unit": "USD/kg",
                            }
                        )
                        report.rows += 1
                        report.first_timestamp_utc = report.first_timestamp_utc or iso
                        report.last_timestamp_utc = iso
                        last_ts = ts
                    except (KeyError, ValueError, InvalidOperation, OverflowError):
                        report.malformed += 1

    return report


def main() -> None:
    require_runtime()
    ROOT.mkdir(parents=True, exist_ok=True)
    end = end_exclusive()
    if end <= REQUESTED_START:
        raise SystemExit("ERROR: invalid date range")

    reports: list[Report] = []
    failures: list[dict[str, str]] = []
    print(f"Troy ounces per kg: {TROY_OUNCES_PER_KG}")
    print(f"Completed UTC data requested through: {end - timedelta(days=1)}")

    for metal in METALS:
        files: list[Path] = []
        for start, stop in chunks(metal.start, end):
            try:
                files.append(download_chunk(metal, start, stop))
            except Exception as exc:
                failures.append(
                    {"metal": metal.name, "from": str(start), "to": str(stop), "error": str(exc)}
                )
                print(f"[FAIL] {metal.name} {start}->{stop}: {exc}", file=sys.stderr)
        if files:
            reports.append(convert_metal(metal, files, end))

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": REQUESTED_START.isoformat(),
        "requested_end_inclusive": "2026-08-31",
        "actual_end_exclusive": end.isoformat(),
        "source": "Dukascopy",
        "downloader": f"dukascopy-node {DUKASCOPY_NODE_VERSION}",
        "timeframe": TIMEFRAME,
        "source_price_unit": "USD/troy_ounce",
        "final_price_unit": "USD/kg",
        "troy_ounce_grams": str(TROY_OUNCE_GRAMS),
        "troy_ounces_per_kg": str(TROY_OUNCES_PER_KG),
        "reports": [asdict(r) for r in reports],
        "failures": failures,
    }
    REPORT_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFINAL SUMMARY")
    for report in reports:
        print(
            f"{report.metal:10s} rows={report.rows:,} "
            f"first={report.first_timestamp_utc} last={report.last_timestamp_utc}"
        )
        print(f"  -> {report.output_file}")
    if failures:
        print(f"WARNING: {len(failures)} chunk(s) failed; re-run to retry them.")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()

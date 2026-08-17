from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path

from .metals import MetalSpec


getcontext().prec = 40
TROY_OUNCE_GRAMS = Decimal("31.1034768")
TROY_OUNCES_PER_KG = Decimal("1000") / TROY_OUNCE_GRAMS
PRICE_QUANT = Decimal("0.000001")


@dataclass
class NormalizationReport:
    metal: str
    rows_written: int = 0
    duplicates_skipped: int = 0
    malformed_skipped: int = 0
    first_timestamp_utc: str | None = None
    last_timestamp_utc: str | None = None
    output_file: str | None = None


def _decimal_price(value: str) -> Decimal:
    try:
        price = Decimal(value.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"Invalid price: {value!r}") from exc
    if not price.is_finite() or price <= 0:
        raise ValueError(f"Invalid price: {value!r}")
    return price


def _to_usd_per_kg(value: str) -> str:
    converted = _decimal_price(value) * TROY_OUNCES_PER_KG
    return format(converted.quantize(PRICE_QUANT), "f")


def _timestamp_ms(value: str) -> int:
    try:
        timestamp = int(Decimal(value.strip()))
    except (InvalidOperation, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if 0 < timestamp < 10_000_000_000:
        timestamp *= 1000
    return timestamp


class UsdPerKgNormalizer:
    """Normalize source OHLC values from USD/troy ounce to USD/kg."""

    def __init__(self, output_root: Path, price_side: str = "bid") -> None:
        self.output_root = output_root
        self.price_side = price_side

    def normalize(
        self,
        metal: MetalSpec,
        files: list[Path],
        end_inclusive: str,
    ) -> NormalizationReport:
        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / (
            f"{metal.output_symbol}_M1_USD_PER_KG_"
            f"{metal.effective_start.isoformat()}_to_{end_inclusive}.csv"
        )

        report = NormalizationReport(metal=metal.name, output_file=str(output))
        last_timestamp: int | None = None

        with output.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(
                out_handle,
                fieldnames=[
                    "timestamp_utc",
                    "timestamp_ms",
                    "open_usd_per_kg",
                    "high_usd_per_kg",
                    "low_usd_per_kg",
                    "close_usd_per_kg",
                    "volume_source_units",
                    "metal",
                    "source_symbol",
                    "source",
                    "price_side",
                    "price_unit",
                ],
            )
            writer.writeheader()

            for path in files:
                print(f"[CONV] {metal.name:10s} {path.name}")
                with path.open(
                    "r",
                    encoding="utf-8-sig",
                    errors="replace",
                    newline="",
                ) as source_handle:
                    reader = csv.DictReader(source_handle)
                    for original in reader:
                        row = {
                            str(key).strip().lower(): value
                            for key, value in original.items()
                            if key is not None
                        }
                        try:
                            timestamp = _timestamp_ms(row["timestamp"])
                            if (
                                last_timestamp is not None
                                and timestamp <= last_timestamp
                            ):
                                report.duplicates_skipped += 1
                                continue

                            open_price = _decimal_price(row["open"])
                            high_price = _decimal_price(row["high"])
                            low_price = _decimal_price(row["low"])
                            close_price = _decimal_price(row["close"])
                            if low_price > high_price or not (
                                low_price <= open_price <= high_price
                                and low_price <= close_price <= high_price
                            ):
                                raise ValueError("Invalid OHLC relationship")

                            iso_timestamp = datetime.fromtimestamp(
                                timestamp / 1000,
                                tz=timezone.utc,
                            ).isoformat(timespec="seconds").replace("+00:00", "Z")

                            writer.writerow(
                                {
                                    "timestamp_utc": iso_timestamp,
                                    "timestamp_ms": timestamp,
                                    "open_usd_per_kg": _to_usd_per_kg(row["open"]),
                                    "high_usd_per_kg": _to_usd_per_kg(row["high"]),
                                    "low_usd_per_kg": _to_usd_per_kg(row["low"]),
                                    "close_usd_per_kg": _to_usd_per_kg(row["close"]),
                                    "volume_source_units": row.get("volume", ""),
                                    "metal": metal.name,
                                    "source_symbol": metal.source_symbol,
                                    "source": "Dukascopy",
                                    "price_side": self.price_side,
                                    "price_unit": "USD/kg",
                                }
                            )

                            report.rows_written += 1
                            report.first_timestamp_utc = (
                                report.first_timestamp_utc or iso_timestamp
                            )
                            report.last_timestamp_utc = iso_timestamp
                            last_timestamp = timestamp
                        except (KeyError, ValueError, InvalidOperation, OverflowError):
                            report.malformed_skipped += 1

        return report

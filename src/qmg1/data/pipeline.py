from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .dukascopy import DukascopyConfig, DukascopyDownloader
from .metals import METALS, REQUESTED_END_EXCLUSIVE, REQUESTED_START, MetalSpec
from .normalizer import TROY_OUNCE_GRAMS, TROY_OUNCES_PER_KG, NormalizationReport, UsdPerKgNormalizer


class MetalsDataPipeline:
    """Orchestrates acquisition and normalization without owning their implementation details."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw_root = root / "raw"
        self.final_root = root / "final"
        self.report_file = root / "download_report.json"
        self.downloader = DukascopyDownloader(
            raw_root=self.raw_root,
            incoming_root=root / ".incoming",
        )
        self.normalizer = UsdPerKgNormalizer(
            output_root=self.final_root,
            price_side=self.downloader.config.price_type,
        )

    @staticmethod
    def _actual_end_exclusive() -> date:
        # During August 2026, stop at the latest completed UTC day.
        return min(REQUESTED_END_EXCLUSIVE, datetime.now(timezone.utc).date())

    @staticmethod
    def _yearly_chunks(start: date, end_exclusive: date):
        cursor = start
        while cursor < end_exclusive:
            next_year = date(cursor.year + 1, 1, 1)
            chunk_end = min(next_year, end_exclusive)
            yield cursor, chunk_end
            cursor = chunk_end

    def _download_metal(self, metal: MetalSpec, end_exclusive: date) -> tuple[list[Path], list[dict[str, str]]]:
        files: list[Path] = []
        failures: list[dict[str, str]] = []

        for start, stop in self._yearly_chunks(metal.effective_start, end_exclusive):
            try:
                files.append(self.downloader.download(metal, start, stop))
            except Exception as exc:
                failures.append(
                    {
                        "metal": metal.name,
                        "from": start.isoformat(),
                        "to": stop.isoformat(),
                        "error": str(exc),
                    }
                )
                print(f"[FAIL] {metal.name} {start} -> {stop}: {exc}")

        return files, failures

    def run(self) -> dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.downloader.validate_runtime()
        end_exclusive = self._actual_end_exclusive()
        if end_exclusive <= REQUESTED_START:
            raise RuntimeError("Invalid requested data range")

        end_inclusive = (end_exclusive - timedelta(days=1)).isoformat()
        reports: list[NormalizationReport] = []
        failures: list[dict[str, str]] = []

        print(f"Troy ounces per kg: {TROY_OUNCES_PER_KG}")
        print(f"Completed UTC data requested through: {end_inclusive}")

        for metal in METALS:
            files, metal_failures = self._download_metal(metal, end_exclusive)
            failures.extend(metal_failures)
            if files:
                reports.append(self.normalizer.normalize(metal, files, end_inclusive))

        metadata: dict[str, object] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_start": REQUESTED_START.isoformat(),
            "requested_end_inclusive": "2026-08-31",
            "actual_end_exclusive": end_exclusive.isoformat(),
            "source": "Dukascopy",
            "downloader": f"dukascopy-node {DukascopyConfig().package_version}",
            "timeframe": DukascopyConfig().timeframe,
            "price_side": DukascopyConfig().price_type,
            "source_price_unit": "USD/troy_ounce",
            "final_price_unit": "USD/kg",
            "troy_ounce_grams": str(TROY_OUNCE_GRAMS),
            "troy_ounces_per_kg": str(TROY_OUNCES_PER_KG),
            "reports": [asdict(report) for report in reports],
            "failures": failures,
        }
        self.report_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

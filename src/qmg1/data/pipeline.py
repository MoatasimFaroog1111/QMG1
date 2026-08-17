from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from .hybrid import HybridPreciousMetalsM1Provider
from .metals import METALS, REQUESTED_END_EXCLUSIVE, REQUESTED_START, MetalSpec
from .normalizer import (
    TROY_OUNCE_GRAMS,
    TROY_OUNCES_PER_KG,
    NormalizationReport,
    UsdPerKgNormalizer,
)
from .provider import HistoricalM1Provider


class MetalsDataPipeline:
    """Orchestrate acquisition and normalization through replaceable providers."""

    def __init__(
        self,
        root: Path,
        provider: HistoricalM1Provider | None = None,
    ) -> None:
        self.root = root
        self.raw_root = root / "raw"
        self.final_root = root / "final"
        self.report_file = root / "download_report.json"
        self.provider = provider or HybridPreciousMetalsM1Provider(raw_root=self.raw_root)
        self.normalizer = UsdPerKgNormalizer(
            output_root=self.final_root,
            price_side=self.provider.price_side,
        )

    @staticmethod
    def _actual_end_exclusive() -> date:
        # During August 2026, stop at the latest completed UTC day.
        return min(REQUESTED_END_EXCLUSIVE, datetime.now(timezone.utc).date())

    def _download_metal(
        self,
        metal: MetalSpec,
        start: date,
        end_exclusive: date,
    ) -> tuple[list[Path], list[dict[str, str]]]:
        files: list[Path] = []
        failures: list[dict[str, str]] = []

        for chunk_start, chunk_stop in self.provider.chunk_ranges(
            metal,
            start,
            end_exclusive,
        ):
            try:
                files.append(self.provider.download(metal, chunk_start, chunk_stop))
            except Exception as exc:
                failures.append(
                    {
                        "metal": metal.name,
                        "from": chunk_start.isoformat(),
                        "to": chunk_stop.isoformat(),
                        "error": str(exc),
                    }
                )
                print(f"[FAIL] {metal.name} {chunk_start} -> {chunk_stop}: {exc}")

        return files, failures

    def run(
        self,
        metals: Sequence[MetalSpec] = METALS,
        start: date | None = None,
        end_exclusive: date | None = None,
    ) -> dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.provider.validate_runtime()

        requested_start = start or REQUESTED_START
        actual_end = min(end_exclusive or self._actual_end_exclusive(), REQUESTED_END_EXCLUSIVE)
        if actual_end <= requested_start:
            raise RuntimeError("Invalid requested data range")

        end_inclusive = (actual_end - timedelta(days=1)).isoformat()
        reports: list[NormalizationReport] = []
        failures: list[dict[str, str]] = []

        print(f"Troy ounces per kg: {TROY_OUNCES_PER_KG}")
        print(f"Completed UTC data requested through: {end_inclusive}")
        print(f"Historical provider: {self.provider.provider_description}")

        for metal in metals:
            metal_start = max(requested_start, metal.effective_start)
            if metal_start >= actual_end:
                continue

            files, metal_failures = self._download_metal(
                metal,
                metal_start,
                actual_end,
            )
            failures.extend(metal_failures)
            if files:
                reports.append(
                    self.normalizer.normalize(
                        metal,
                        files,
                        end_inclusive,
                        source_name=self.provider.source_name_for(metal),
                        start_inclusive=metal_start.isoformat(),
                    )
                )

        metadata: dict[str, object] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_start": requested_start.isoformat(),
            "requested_end_inclusive": "2026-08-31",
            "actual_end_exclusive": actual_end.isoformat(),
            "source": self.provider.source_name,
            "provider": self.provider.provider_description,
            "timeframe": self.provider.timeframe,
            "price_side": self.provider.price_side,
            "source_price_unit": "USD/troy_ounce",
            "final_price_unit": "USD/kg",
            "troy_ounce_grams": str(TROY_OUNCE_GRAMS),
            "troy_ounces_per_kg": str(TROY_OUNCES_PER_KG),
            "reports": [asdict(report) for report in reports],
            "failures": failures,
        }
        self.report_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

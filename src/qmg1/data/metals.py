from __future__ import annotations

from dataclasses import dataclass
from datetime import date


REQUESTED_START = date(2009, 5, 1)
REQUESTED_END_EXCLUSIVE = date(2026, 9, 1)


@dataclass(frozen=True)
class MetalSpec:
    key: str
    name: str
    source_symbol: str
    downloader_instrument: str
    first_m1_available: date
    dukascopy_decimal_factor: int = 1000

    @property
    def effective_start(self) -> date:
        return max(REQUESTED_START, self.first_m1_available)

    @property
    def output_symbol(self) -> str:
        return self.source_symbol.replace("/", "").replace(".", "")


METALS: tuple[MetalSpec, ...] = (
    MetalSpec("gold", "Gold", "XAU/USD", "xauusd", date(2003, 5, 5)),
    MetalSpec("silver", "Silver", "XAG/USD", "xagusd", date(2003, 5, 4)),
    MetalSpec("palladium", "Palladium", "XPD.CMD/USD", "xpdcmdusd", date(2021, 7, 4)),
    MetalSpec("platinum", "Platinum", "XPT.CMD/USD", "xptcmdusd", date(2021, 11, 1)),
)

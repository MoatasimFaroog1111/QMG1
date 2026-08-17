from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Forecast horizons requested by the project owner.
# 168h = 7 days, 360h = 15 days, 720h = 30 days.
HORIZONS_HOURS: tuple[int, ...] = (2, 4, 8, 12, 24, 72, 168, 360, 720)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(".")

    @property
    def data_dir(self) -> Path:
        return self.root / "metals_m1_usd_per_kg" / "final"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"


@dataclass(frozen=True)
class TrainingConfig:
    min_rows: int = 5_000
    cv_splits: int = 5
    random_state: int = 42
    max_iter: int = 350
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    # Operational forecasts must beat the no-change/persistence benchmark both
    # across all OOS folds and on the most recent fold. Strictly greater than
    # these thresholds is required; equality is still considered unproven.
    min_operational_improvement_pct: float = 0.0
    min_recent_improvement_pct: float = 0.0

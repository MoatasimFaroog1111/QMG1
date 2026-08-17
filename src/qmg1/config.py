from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HORIZONS_HOURS: tuple[int, ...] = (6, 12, 18, 24, 48, 168, 720)


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

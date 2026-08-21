from qmg1.config import HORIZONS_HOURS


def test_directional_diagnostics_covers_all_requested_horizons() -> None:
    assert HORIZONS_HOURS == (2, 4, 8, 12, 24, 72, 168, 360, 720)

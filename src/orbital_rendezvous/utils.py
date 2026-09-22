"""Small shared helpers: configuration loading and reproducible seeding."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file into a plain dictionary."""
    raise NotImplementedError


def set_global_seed(seed: int) -> None:
    """Seed ``random``, NumPy and PyTorch so a run can be reproduced."""
    raise NotImplementedError


def delta_v(thrust_history: Any, mass: float, dt: float) -> float:
    """Total velocity increment spent over a trajectory, in m/s."""
    raise NotImplementedError

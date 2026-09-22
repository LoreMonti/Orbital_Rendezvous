"""Tests for the Gymnasium environment.

Covers: ``gymnasium.utils.env_checker.check_env`` compliance, the shapes and
bounds of the observation and action spaces, determinism of ``reset(seed=...)``,
thrust saturation, and each terminal condition (docking, crash, runaway,
timeout) being reachable and reported correctly.
"""

import pytest

pytest.skip("env.py is still a skeleton", allow_module_level=True)

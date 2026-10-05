"""Tests for the optional brakes on PPO's updates.

Off by default, so every earlier run keeps its settings; on, the learning
rate must fall from its start to its end, not the other way round (a schedule
read backwards would freeze the first stages and shake the last ones).
"""

from pathlib import Path

import pytest

from orbital_rendezvous.training import build_model, ppo_stability
from orbital_rendezvous.utils import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_off_by_default():
    assert ppo_stability({"learning_rate": 3e-4}) == {}


def test_the_learning_rate_falls_linearly_to_its_final_value():
    schedule = ppo_stability({"learning_rate": 3e-4, "learning_rate_final": 3e-5})["learning_rate"]
    # Stable-Baselines3 passes the progress remaining: 1 at the start, 0 at the end.
    assert schedule(1.0) == pytest.approx(3e-4)
    assert schedule(0.5) == pytest.approx(1.65e-4)
    assert schedule(0.0) == pytest.approx(3e-5)


def test_the_stable_configuration_builds_a_braked_ppo(tmp_path):
    config = load_config(CONFIGS / "ppo_corridor_stable.yaml")
    config["training"].update(n_envs=2, n_steps=64, batch_size=64)
    model = build_model(config, 0, tmp_path / "run")
    assert model.target_kl == pytest.approx(0.02)
    assert model.lr_schedule(1.0) == pytest.approx(3e-4)
    assert model.lr_schedule(0.0) == pytest.approx(3e-5)

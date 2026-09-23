"""Tests for loading the YAML configuration."""

from pathlib import Path

import pytest

from orbital_rendezvous import EnvConfig, RewardConfig
from orbital_rendezvous.utils import build_configs, load_config

DEFAULT = Path(__file__).parents[1] / "configs" / "ppo_default.yaml"


def test_default_yaml_matches_the_dataclass_defaults():
    # The YAML is what a user reads; the dataclass defaults are what the tests
    # and the library snippet use. If they drift apart, one of them is lying.
    env_config, reward_config = build_configs(load_config(DEFAULT))
    assert env_config == EnvConfig()
    assert reward_config == RewardConfig()


def test_training_section_is_complete():
    training = load_config(DEFAULT)["training"]
    assert training["total_timesteps"] == 2_000_000
    assert training["n_steps"] * training["n_envs"] % training["batch_size"] == 0


def test_unknown_key_is_rejected():
    config = load_config(DEFAULT)
    config["environment"]["docking_raduis"] = 2.0
    with pytest.raises(ValueError, match="docking_raduis"):
        build_configs(config)


def test_gamma_mismatch_is_rejected():
    config = load_config(DEFAULT)
    config["training"]["gamma"] = 0.995
    with pytest.raises(ValueError, match="gamma"):
        build_configs(config)

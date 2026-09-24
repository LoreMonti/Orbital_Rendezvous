"""Tests for the fuel curriculum and the fuel study plumbing.

The study itself takes half an hour of training; what is tested here is that
each run is set up as intended: the discount agrees between PPO and the
shaping, the fuel weight follows its schedule inside every environment, and
the results are aggregated per configuration.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.callbacks import FuelCurriculum
from orbital_rendezvous.study import aggregate, make_jobs
from orbital_rendezvous.training import train, with_overrides
from orbital_rendezvous.utils import build_configs, load_config

DEFAULT = Path(__file__).parents[1] / "configs" / "ppo_default.yaml"


def test_set_fuel_weight_changes_only_the_fuel_term():
    env = RendezvousEnv()
    env.reset(seed=0)
    env.state = np.array([100.0, 0.0, 0.0, 0.0])
    _, _, _, _, light = env.step(np.array([1.0, 0.0]))
    env.set_fuel_weight(4 * env.reward_config.fuel_weight)
    env.state = np.array([100.0, 0.0, 0.0, 0.0])
    env.steps = 0
    _, _, _, _, heavy = env.step(np.array([1.0, 0.0]))
    assert heavy["reward_terms"]["fuel"] == pytest.approx(4 * light["reward_terms"]["fuel"])
    assert heavy["reward_terms"]["shaping"] == pytest.approx(light["reward_terms"]["shaping"])


def test_curriculum_schedule_is_linear_then_flat():
    curriculum = FuelCurriculum(start=2.0, end=10.0, ramp=0.5)
    assert curriculum.weight_at(0.0) == 2.0
    assert curriculum.weight_at(0.25) == pytest.approx(6.0)
    assert curriculum.weight_at(0.5) == pytest.approx(10.0)
    assert curriculum.weight_at(0.9) == pytest.approx(10.0)


def test_curriculum_rejects_a_ramp_outside_the_training():
    with pytest.raises(ValueError):
        FuelCurriculum(2.0, 10.0, ramp=0.0)


def test_curriculum_sets_the_weight_in_every_environment():
    calls = []
    curriculum = FuelCurriculum(start=2.0, end=10.0, ramp=0.5)
    envs = SimpleNamespace(env_method=lambda name, value: calls.append((name, value)))
    logger = SimpleNamespace(record=lambda key, value: None)
    # training_env and logger are both read from the model in Stable-Baselines3.
    curriculum.model = SimpleNamespace(_total_timesteps=1000, get_env=lambda: envs, logger=logger)
    curriculum.num_timesteps = 250
    curriculum._on_rollout_start()
    assert calls == [("set_fuel_weight", pytest.approx(6.0))]


def test_overrides_keep_the_two_discounts_equal():
    config = with_overrides(load_config(DEFAULT), gamma=0.999, fuel_weight=10.0,
                            curriculum={"start": 2.0, "ramp": 0.5})
    assert config["training"]["gamma"] == config["rewards"]["gamma"] == 0.999
    assert config["rewards"]["fuel_weight"] == 10.0
    # The check in build_configs, that would reject a mismatch, passes.
    build_configs(config)
    # And the original is left alone.
    assert load_config(DEFAULT)["training"]["fuel_curriculum"] is None


def test_training_with_a_curriculum_ends_at_the_target_weight(tmp_path):
    config = with_overrides(load_config(DEFAULT), fuel_weight=10.0,
                            curriculum={"start": 2.0, "ramp": 0.5})
    config["training"]["n_envs"] = 2
    config["training"]["n_steps"] = 64
    config["training"]["batch_size"] = 64
    model = train(config, 0, 1024, tmp_path / "run", tmp_path / "model.zip", checkpoints=False)
    weights = model.get_env().get_attr("reward_config")
    assert all(w.fuel_weight == pytest.approx(10.0) for w in weights)


def test_jobs_cover_the_grid_slowest_first():
    jobs = make_jobs([0.99, 0.999], [2.0, 10.0], [0, 1], 1000, 5, "runs/x")
    assert len(jobs) == 8
    assert len({job.label for job in jobs}) == 8
    assert jobs[0].gamma == 0.999


def test_aggregate_takes_medians_over_seeds():
    def result(gamma, seed, dv, rate=1.0):
        summary = {"success_rate": rate, "delta_v_median": dv, "time_median": 100.0 * seed,
                   "docking_speed_median": 0.03}
        return {"gamma": gamma, "fuel_weight": 2.0, "seed": seed, "summary": summary}

    rows = aggregate([result(0.99, 0, 1.0), result(0.99, 1, 3.0), result(0.99, 2, 2.0),
                      result(0.999, 0, float("nan"), rate=0.0),
                      # Docks rarely, from the easy starts only: cheap, and not a solution.
                      result(0.999, 1, 0.5, rate=0.05), result(0.999, 2, 0.9)])
    by_gamma = {row["gamma"]: row for row in rows}
    assert by_gamma[0.99]["delta_v_median"] == 2.0
    assert by_gamma[0.99]["time_median"] == 100.0
    assert by_gamma[0.99]["reliable_seeds"] == 3
    assert by_gamma[0.999]["success_rates"] == [0.0, 0.05, 1.0]
    assert by_gamma[0.999]["reliable_seeds"] == 1
    # Only the reliable seed counts, not the flattering 0.5 of the unreliable one.
    assert by_gamma[0.999]["delta_v_median"] == 0.9

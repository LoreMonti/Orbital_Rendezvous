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


def make_budget(**kwargs):
    from orbital_rendezvous.callbacks import FuelBudget

    return FuelBudget(budget=0.6, make_env=RendezvousEnv, **kwargs)


def test_dual_ascent_follows_the_worked_example():
    # The example of the README: budget 0.6 m/s, step size 5, starting from zero.
    budget = make_budget(step_size=5.0)
    weight = 0.0
    for spent, expected in [(1.20, 5.0), (0.90, 7.5), (0.70, 8.333), (0.58, 8.167)]:
        weight = budget.dual_step(weight, spent)
        assert weight == pytest.approx(expected, abs=1e-3)
    # On budget, the price stops moving.
    assert budget.dual_step(weight, 0.6) == pytest.approx(weight)


def test_multiplier_stays_within_its_bounds():
    budget = make_budget(step_size=5.0, max_weight=20.0)
    assert budget.dual_step(0.5, 0.0) == 0.0     # spending nothing never makes it negative
    assert budget.dual_step(19.0, 6.0) == 20.0   # a runaway is capped


def fake_training(budget, measurements):
    """Drive the callback's rollout hook with scripted (success, delta_v) measurements."""
    applied = []
    envs = SimpleNamespace(env_method=lambda name, value: applied.append(value))
    logger = SimpleNamespace(record=lambda key, value: None)
    budget.model = SimpleNamespace(get_env=lambda: envs, logger=logger)
    budget.num_timesteps = 0
    readings = iter(measurements)
    budget.measure = lambda: next(readings)
    budget._on_training_start()
    for _ in measurements:
        budget._on_rollout_end()
    return applied


def test_price_stays_at_zero_until_the_agent_docks():
    # Fuel must not get expensive before docking is learned: the stay-put trap.
    budget = make_budget(step_size=1.0, evaluate_every=1, warmup_success=0.5)
    fake_training(budget, [(0.0, 3.0), (0.2, 2.5), (0.6, 1.2), (0.9, 0.9)])
    weights = [h["fuel_weight"] for h in budget.history]
    assert weights[:2] == [0.0, 0.0]
    assert weights[2] == pytest.approx(1.0)          # (1.2 - 0.6) / 0.6
    assert weights[3] == pytest.approx(1.5)          # + (0.9 - 0.6) / 0.6
    assert [h["active"] for h in budget.history] == [False, False, True, True]


def test_price_is_measured_only_every_few_rollouts():
    budget = make_budget(evaluate_every=3)
    fake_training(budget, [(1.0, 1.2)] * 7)
    assert len(budget.history) == 7 // 3


def test_measurement_uses_the_deterministic_policy():
    calls = []

    class Model:
        def predict(self, obs, deterministic=False):
            calls.append(deterministic)
            return np.zeros(2), None

    budget = make_budget(episodes=2)
    budget.model = Model()
    success, delta_v = budget.measure()
    assert calls and all(calls)
    assert success == 0.0 and delta_v == 0.0


def test_budget_jobs_and_their_aggregation():
    jobs = make_jobs([0.999], [2.0, 10.0], [0, 1], 1000, 5, "runs/x", budgets=[0.6, 0.3])
    assert len(jobs) == 4
    assert {job.fuel_budget for job in jobs} == {0.6, 0.3}
    assert all("budget" in job.label for job in jobs)

    def result(budget, seed, dv):
        summary = {"success_rate": 1.0, "delta_v_median": dv, "time_median": 700.0,
                   "docking_speed_median": 0.03}
        return {"gamma": 0.999, "fuel_weight": 0.0, "fuel_budget": budget, "seed": seed,
                "summary": summary, "final_fuel_weight": 4.0 + seed}

    rows = aggregate([result(0.6, 0, 0.62), result(0.6, 1, 0.58), result(0.3, 0, 0.45)])
    assert [row["fuel_budget"] for row in rows] == [0.3, 0.6]
    assert rows[1]["delta_v_median"] == pytest.approx(0.60)
    assert rows[1]["final_fuel_weights"] == [4.0, 5.0]

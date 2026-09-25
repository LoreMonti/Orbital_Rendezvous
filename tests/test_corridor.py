"""Tests for the oriented target: keep-out sphere, approach cone and the V-bar baseline.

The chaser is placed by hand where a single step must, or must not, break the
rule, so a mislabelled violation shows up directly. The rule: inside the
keep-out sphere, outside the docking sphere, the chaser must be within the
approach cone around +y.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, Outcome, RendezvousEnv
from orbital_rendezvous.baselines import VbarApproach
from orbital_rendezvous.callbacks import KeepOutBudget
from orbital_rendezvous.evaluation import HELD_OUT_SEED, evaluate
from orbital_rendezvous.rewards import in_approach_cone

STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)
PENALTY = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0, keep_out_mode="penalty",
                    keep_out_weight=7.0)


def step_from(config, state, action=(0.0, 0.0)):
    env = RendezvousEnv(config)
    env.reset(seed=0)
    env.state = np.asarray(state, dtype=float)
    return env.step(np.asarray(action))


def at_angle(degrees, distance=10.0):
    """A point ``distance`` from the port, ``degrees`` away from the docking axis +y."""
    return distance * np.array([np.sin(np.radians(degrees)), np.cos(np.radians(degrees))])


def test_cone_geometry():
    assert in_approach_cone(at_angle(0), 15.0)
    assert in_approach_cone(at_angle(14), 15.0)
    assert not in_approach_cone(at_angle(16), 15.0)
    assert not in_approach_cone(at_angle(180), 15.0)


def test_entering_the_sphere_outside_the_cone_is_a_violation():
    # 15 m radial, at rest: inside the sphere and far outside the cone.
    _, reward, terminated, _, info = step_from(STRICT, [15.0, 0.0, 0.0, 0.0])
    assert info["outcome"] is Outcome.KEEP_OUT and terminated
    assert info["reward_terms"]["terminal"] < 0
    assert info["keep_out_violated"]


def test_inside_the_cone_is_allowed():
    _, _, terminated, _, info = step_from(STRICT, [0.0, 15.0, 0.0, -0.02])
    assert not terminated and not info["keep_out_violated"]


def test_crossing_the_sphere_in_one_step_is_caught():
    # 5 m/s across the sphere, radially: both ends outside, the middle inside.
    _, _, _, _, info = step_from(STRICT, [30.0, 5.0, -5.0, 0.0])
    assert np.hypot(*info["position"]) > 20.0
    assert info["outcome"] is Outcome.KEEP_OUT


def test_the_docking_sphere_is_exempt():
    # The cone's apex is the port: docking from slightly off-axis still counts.
    _, _, _, _, info = step_from(STRICT, [0.5, 0.3, 0.0, 0.0])
    assert info["outcome"] is Outcome.DOCKED


def test_penalty_mode_charges_and_continues():
    _, _, terminated, _, info = step_from(PENALTY, [15.0, 0.0, 0.0, 0.0])
    assert not terminated and info["outcome"] is None
    assert info["keep_out_violated"]
    assert info["reward_terms"]["keep_out"] == -7.0


def test_keep_out_weight_can_be_changed():
    env = RendezvousEnv(PENALTY)
    env.reset(seed=0)
    env.set_keep_out_weight(30.0)
    env.state = np.array([15.0, 0.0, 0.0, 0.0])
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["reward_terms"]["keep_out"] == -30.0


def test_default_environment_has_no_keep_out_term():
    _, _, _, _, info = step_from(EnvConfig(), [15.0, 0.0, 0.0, 0.0])
    assert "keep_out" not in info["reward_terms"]
    assert not info["keep_out_violated"]


def test_vbar_approach_docks_without_violations():
    env = RendezvousEnv(STRICT)
    runs = evaluate(env, VbarApproach(env), range(HELD_OUT_SEED, HELD_OUT_SEED + 8))
    assert all(r.outcome is Outcome.DOCKED for r in runs)
    assert not any(r.violated for r in runs)


def test_keep_out_price_waits_for_docking_then_follows_violations():
    budget = KeepOutBudget(lambda: RendezvousEnv(PENALTY), step_size=20.0, evaluate_every=1)
    applied = []
    envs = SimpleNamespace(env_method=lambda name, value: applied.append((name, value)))
    logger = SimpleNamespace(record=lambda key, value: None)
    budget.model = SimpleNamespace(get_env=lambda: envs, logger=logger)
    budget.num_timesteps = 0
    readings = iter([(0.1, 1.0), (0.6, 1.0), (0.9, 0.5), (1.0, 0.02)])
    budget.measure = lambda: next(readings)
    budget._on_training_start()
    for _ in range(4):
        budget._on_rollout_end()
    weights = [h["weight"] for h in budget.history]
    # Held at zero before docking; then 20 * (1 - 0.02), + 20 * (0.5 - 0.02), steady at 2 %.
    assert weights == pytest.approx([0.0, 19.6, 29.2, 29.2])
    assert applied[-1] == ("set_keep_out_weight", pytest.approx(29.2))


def test_keep_out_price_relaxes_when_docking_is_lost():
    budget = KeepOutBudget(lambda: RendezvousEnv(PENALTY), step_size=20.0, relax=0.5,
                           evaluate_every=1)
    envs = SimpleNamespace(env_method=lambda name, value: None)
    logger = SimpleNamespace(record=lambda key, value: None)
    budget.model = SimpleNamespace(get_env=lambda: envs, logger=logger)
    budget.num_timesteps = 0
    # Docks and violates, then stops docking: the price must fall, not rise.
    readings = iter([(0.8, 1.0), (0.1, 0.0), (0.1, 0.0)])
    budget.measure = lambda: next(readings)
    budget._on_training_start()
    for _ in range(3):
        budget._on_rollout_end()
    assert [h["weight"] for h in budget.history] == pytest.approx([19.6, 9.8, 4.9])


def test_a_cone_of_180_degrees_is_no_constraint():
    env = RendezvousEnv(STRICT)
    env.reset(seed=0)
    env.set_approach_cone(180.0)
    env.state = np.array([0.0, -15.0, 0.0, 0.0])   # straight behind the station
    _, _, _, _, info = env.step(np.zeros(2))
    assert not info["keep_out_violated"]


def test_cone_narrows_only_once_mastered_and_stops_at_the_final_angle():
    from orbital_rendezvous.callbacks import ConeCurriculum

    cone = ConeCurriculum(lambda: RendezvousEnv(STRICT), final_deg=15.0, step_deg=10.0,
                          success_threshold=0.9)
    assert cone.narrowed(180.0, 0.95) == 170.0
    assert cone.narrowed(180.0, 0.80) == 180.0   # not mastered: stays
    assert cone.narrowed(20.0, 1.00) == 15.0     # never below the final cone
    assert cone.narrowed(15.0, 1.00) == 15.0


def scales(cone):
    from orbital_rendezvous.rewards import Scales

    return Scales(500.0, 0.5, 0.05, 500.0, 10.0, keep_out_radius=20.0, approach_cone_deg=cone)


def test_path_goes_around_the_sphere_from_behind():
    from orbital_rendezvous.rewards import path_length

    # Tangent sqrt(100^2 - 20^2), arc to the 15 degree edge, then 20 m along the cone.
    half = np.arccos(20 / 100)
    arc = abs(np.pi - half - np.radians(15))
    expected = np.sqrt(100**2 - 20**2) + 20 * arc + 20
    assert path_length(np.array([0.0, -100.0]), scales(15.0)) == pytest.approx(expected)


def test_path_is_straight_inside_the_cone_and_with_no_constraint():
    from orbital_rendezvous.rewards import path_length

    assert path_length(at_angle(5, 80.0), scales(15.0)) == pytest.approx(80.0)
    # At 180 degrees the curriculum has not started: the plain straight distance.
    assert path_length(np.array([0.0, -100.0]), scales(180.0)) == pytest.approx(100.0)


def test_path_is_continuous_across_the_edge_of_the_cone():
    from orbital_rendezvous.rewards import path_length

    inside = path_length(at_angle(14.999, 100.0), scales(15.0))
    outside = path_length(at_angle(15.001, 100.0), scales(15.0))
    assert inside == pytest.approx(outside, abs=1e-3)

"""Tests for the two learned pilots flying the V-bar procedure's plan (Step 16).

The go-to task must be the default task with the target moved, and nothing
else: with the goal at the origin it pays exactly the default rewards. Its
observation must carry the absolute position too, since holding still off the
V-bar needs a thrust that depends on where the chaser is. The pilot must hand
over at exactly the thresholds the go-to agent was trained to reach, and give
each agent the clock of its own leg.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, GoToConfig, GoToEnv, Outcome, RendezvousEnv
from orbital_rendezvous.baselines import side_waypoint
from orbital_rendezvous.callbacks import BestModel
from orbital_rendezvous.env import goto_observation
from orbital_rendezvous.hierarchy import HierarchicalPilot
from orbital_rendezvous.training import train
from orbital_rendezvous.utils import load_config, make_env

CONFIGS = Path(__file__).parents[1] / "configs"
GOTO = EnvConfig(docking_radius=2.0, docking_speed=0.04)
HOLD = np.array([0.0, 30.0])


def goto_env(goal=(0.0, 30.0), **kwargs):
    env = GoToEnv(GOTO, goal_config=GoToConfig(goals=(goal,), goal_jitter=0.0,
                                               moving_start_fraction=0.0, **kwargs))
    env.reset(seed=0)
    return env


def test_goal_at_the_origin_pays_the_default_rewards():
    state = np.array([60.0, -40.0, 0.1, 0.2])
    action = np.array([0.3, -0.7])
    default = RendezvousEnv(GOTO)
    default.reset(seed=0)
    default.state = state.copy()
    _, _, _, _, expected = default.step(action)
    env = goto_env(goal=(0.0, 0.0))
    env.state = state.copy()
    _, _, _, _, info = env.step(action)
    for term in ("shaping", "fuel"):
        assert info["reward_terms"][term] == expected["reward_terms"][term]


def test_shaping_measures_distance_from_the_goal():
    # The same displacement relative to the goal earns the same shaping.
    near_goal = goto_env(goal=(50.0, 0.0))
    near_goal.state = np.array([90.0, 0.0, -0.2, 0.0])
    _, _, _, _, moved = near_goal.step(np.zeros(2))
    at_origin = goto_env(goal=(0.0, 0.0))
    at_origin.state = np.array([40.0, 0.0, -0.2, 0.0])
    _, _, _, _, plain = at_origin.step(np.zeros(2))
    # Not identical: the dynamics depend on the absolute position.
    assert moved["reward_terms"]["shaping"] == pytest.approx(
        plain["reward_terms"]["shaping"], rel=0.05)


def test_the_goal_is_reached_only_close_and_slow():
    env = goto_env()
    env.state = np.array([0.5, 30.5, 0.0, 0.01])
    _, reward, terminated, _, info = env.step(np.zeros(2))
    assert info["outcome"] is Outcome.DOCKED and terminated
    assert info["reward_terms"]["terminal"] == 100.0
    # Through the goal too fast: not a success, and not a failure either.
    env = goto_env()
    env.state = np.array([0.0, 31.0, 0.0, -0.15])
    _, _, terminated, _, info = env.step(np.zeros(2))
    assert info["goal_distance"] < 2.0
    assert info["outcome"] is None and not terminated


def test_holding_still_off_the_vbar_needs_minus_three_n_squared_x_m():
    # Why the go-to agent sees its absolute position: at rest at x, the
    # Clohessy-Wiltshire x equation needs a thrust -3 n^2 x m to stay there.
    env = goto_env(goal=(50.0, 0.0))
    env.state = np.array([50.0, 0.0, 0.0, 0.0])
    thrust = -3 * env.n**2 * 50.0 * env.config.mass
    assert thrust == pytest.approx(-0.096, abs=1e-3)
    for _ in range(30):
        env.step(np.array([thrust / env.config.max_thrust, 0.0]))
    np.testing.assert_allclose(env.state, [50.0, 0.0, 0.0, 0.0], atol=1e-9)
    # Coasting from the same point, it drifts away.
    env.state = np.array([50.0, 0.0, 0.0, 0.0])
    for _ in range(30):
        env.step(np.zeros(2))
    assert np.hypot(*(env.state[:2] - [50.0, 0.0])) > 5.0


def test_the_observation_carries_the_goal_and_the_absolute_position():
    obs = goto_observation(np.array([100.0, -50.0, 0.25, -0.5]), np.array([50.0, 0.0]), 0.3,
                           GOTO)
    np.testing.assert_allclose(obs, [0.1, -0.1, 0.5, -1.0, 0.1, 0.0, 0.3], rtol=1e-6)
    env = goto_env()
    assert env.observation_space.contains(env._observation())


def test_goals_and_moving_starts_follow_the_configuration():
    cfg = GoToConfig()
    env = GoToEnv(GOTO, goal_config=cfg)
    points = np.array(cfg.goals)
    moving = 0
    for seed in range(1000):
        env.reset(seed=seed)
        assert np.min(np.max(np.abs(points - env.goal), axis=1)) <= cfg.goal_jitter
        near = np.min(np.max(np.abs(points - env.state[:2]), axis=1)) <= cfg.moving_start_offset
        if near:
            moving += 1
            assert np.hypot(*env.state[2:]) <= cfg.moving_start_speed
        else:
            assert 80.0 <= np.hypot(*env.state[:2]) <= 200.0
    assert 250 < moving < 350


def test_the_planner_goes_around_on_the_start_side_with_clearance():
    assert side_waypoint(np.array([0.0, 150.0]), HOLD, 20.0) is None      # in front
    np.testing.assert_array_equal(side_waypoint(np.array([-80.0, -100.0]), HOLD, 20.0),
                                  [-50.0, 0.0])
    np.testing.assert_array_equal(side_waypoint(np.array([0.0, -150.0]), HOLD, 20.0),
                                  [50.0, 0.0])                            # exactly behind
    for angle in np.radians(np.linspace(100, 260, 50)):
        start = 150.0 * np.array([np.sin(angle), np.cos(angle)])
        waypoint = side_waypoint(start, HOLD, 20.0)
        legs = [(start, HOLD)] if waypoint is None else [(start, waypoint), (waypoint, HOLD)]
        for a, b in legs:
            d = b - a
            t = np.clip(-(a @ d) / (d @ d), 0.0, 1.0)
            assert np.hypot(*(a + t * d)) > 20.0


def fake_pilot():
    calls = []
    pilot = HierarchicalPilot(
        lambda obs: calls.append(("go_to", obs)) or np.zeros(2),
        lambda obs: calls.append(("final", obs)) or np.zeros(2),
        GOTO, EnvConfig(max_episode_steps=150),
    )
    return pilot, calls


def test_the_pilot_passes_the_waypoint_and_hands_over_at_the_hold_point():
    pilot, calls = fake_pilot()
    env = SimpleNamespace(state=np.array([0.0, -150.0, 0.0, 0.0]), steps=0)
    pilot(env, None)
    np.testing.assert_array_equal(pilot.plan[0], [50.0, 0.0])
    # Within 5 m of the waypoint, still moving fast: on to the hold point.
    env.state, env.steps = np.array([53.0, 2.0, -0.2, 0.2]), 40
    pilot(env, None)
    assert pilot.leg == 1 and pilot.phase == "fly"
    obs = calls[-1][1]
    np.testing.assert_allclose(obs[4:6], HOLD / 500.0)     # the goal is now the hold point
    assert obs[6] == 0.0                                    # and its clock starts afresh
    # At the hold point but too fast: not yet.
    env.state, env.steps = np.array([0.5, 30.5, 0.0, -0.05]), 90
    pilot(env, None)
    assert pilot.phase == "fly"
    # Close and slow: the final-approach agent takes over, with its own clock.
    env.state, env.steps = np.array([0.5, 30.5, 0.0, -0.03]), 100
    pilot(env, None)
    assert pilot.phase == "final" and calls[-1][0] == "final"
    np.testing.assert_allclose(calls[-1][1], [0.001, 0.061, 0.0, -0.06, 0.0], atol=1e-6)
    env.steps = 115
    pilot(env, None)
    assert calls[-1][1][4] == pytest.approx(15 / 150)


def test_the_pilot_goes_straight_to_the_hold_point_from_the_front():
    pilot, _ = fake_pilot()
    pilot(SimpleNamespace(state=np.array([20.0, 150.0, 0.0, 0.0]), steps=0), None)
    assert len(pilot.plan) == 1


def test_best_model_keeps_the_best_not_the_last(tmp_path):
    best = BestModel(lambda: None, tmp_path / "best.zip", evaluate_every=1)
    saved = []
    best.model = SimpleNamespace(save=lambda path: saved.append(best.num_timesteps),
                                 logger=SimpleNamespace(record=lambda k, v: None))
    readings = iter([(0.6, 1.0), (0.9, 1.2), (0.9, 1.1), (0.4, 0.8), (0.9, 1.3)])
    best.measure = lambda: next(readings)
    for step in range(5):
        best.num_timesteps = step
        best._on_rollout_end()
    # Saved when docking improved, and on a tie only with less fuel.
    assert saved == [0, 1, 2]
    assert best.best == (0.9, 1.1)


@pytest.mark.parametrize("name, kind", [("ppo_goto", GoToEnv), ("ppo_final_approach",
                                                                  RendezvousEnv)])
def test_pilot_configs_build_their_environments(name, kind):
    env = make_env(load_config(CONFIGS / f"{name}.yaml"))
    assert type(env) is kind
    assert env.observation_space.contains(env.reset(seed=0)[0])


def test_the_final_approach_trains_near_the_hold_point_with_its_best_model(tmp_path):
    config = load_config(CONFIGS / "ppo_final_approach.yaml")
    config["training"]["n_envs"] = 2
    config["training"]["n_steps"] = 64
    config["training"]["batch_size"] = 64
    config["training"]["best_model"] = {"evaluate_every": 1, "episodes": 2}
    model = train(config, 0, 256, tmp_path / "run", tmp_path / "final.zip", checkpoints=False)
    configs = model.get_env().get_attr("config")
    assert [c.start_angle_range_deg for c in configs] == [(0.0, 5.0), (0.0, 5.0)]
    assert [c.approach_cone_deg for c in configs] == [180.0, 180.0]   # phase 1 under way
    assert (tmp_path / "final_best.zip").exists()
    assert len(json.loads((tmp_path / "run" / "best_model.json").read_text())) == 2
    stages = json.loads((tmp_path / "run" / "curriculum.json").read_text())
    assert stages["starts"] == []   # start_deg = final_deg: nothing to widen

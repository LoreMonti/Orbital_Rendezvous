"""Tests for the learned planner (Step 17).

The menu must be the points it claims to be, and adding it must leave the
go-to pilot of Step 16 untouched. The planner's environment must fly exactly
the plan it was given, one waypoint or none, and pay +100 only for a docking,
-100 for anything else, less the fuel.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from orbital_rendezvous import EnvConfig, Outcome, RendezvousEnv
from orbital_rendezvous.core.utils import load_config
from orbital_rendezvous.rl.training import train
from teacher_student.config import make_env
from teacher_student.goto import GoToConfig, GoToEnv, waypoint_menu
from teacher_student.hierarchy import HierarchicalPilot, PlannerEnv

CONFIGS = Path(__file__).parents[1] / "configs"
STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)
GOTO = EnvConfig(docking_radius=2.0, docking_speed=0.04)


def test_the_menu_rings_the_station_evenly():
    menu = waypoint_menu((40.0, 60.0), 16)
    assert menu.shape == (32, 2)
    np.testing.assert_allclose(menu[0], [0.0, 40.0], atol=1e-12)       # first on the axis
    np.testing.assert_allclose(menu[4], [40.0, 0.0], atol=1e-12)       # a quarter turn on
    np.testing.assert_allclose(np.hypot(*menu[:16].T), 40.0)
    np.testing.assert_allclose(np.hypot(*menu[16:].T), 60.0)
    angles = np.degrees(np.arctan2(menu[:16, 0], menu[:16, 1])) % 360
    np.testing.assert_allclose(np.diff(angles), 22.5)


def test_the_menu_is_added_to_the_goals_and_the_step_16_pilot_is_unchanged():
    assert len(GoToConfig().all_goals()) == 3
    np.testing.assert_array_equal(GoToConfig().all_goals(), GoToConfig().goals)
    with_menu = GoToConfig(menu_radii=(40.0, 60.0)).all_goals()
    assert len(with_menu) == 35
    np.testing.assert_array_equal(with_menu[:3], GoToConfig().goals)
    config = load_config(CONFIGS / "ppo_goto_menu.yaml")
    assert len(make_env(config).goal_config.all_goals()) == 35


def still_pilot():
    return HierarchicalPilot(lambda obs: np.zeros(2), lambda obs: np.zeros(2),
                             GOTO, EnvConfig(max_episode_steps=150))


def planner_env(max_steps=3):
    corridor = RendezvousEnv(EnvConfig(keep_out_radius=20.0, max_episode_steps=max_steps))
    return PlannerEnv(corridor, still_pilot(), waypoint_menu((40.0, 60.0), 16), 10.0)


def test_a_custom_planner_replaces_the_rule():
    pilot = still_pilot()
    pilot.planner = lambda state: np.array([-60.0, 0.0])
    pilot(type("Env", (), {"state": np.array([0.0, 150.0, 0.0, 0.0]), "steps": 0})(), None)
    assert len(pilot.plan) == 2
    np.testing.assert_array_equal(pilot.plan[0], [-60.0, 0.0])


def test_the_planner_flies_the_waypoint_it_chose():
    env = planner_env()
    assert env.action_space.n == 33
    env.reset(seed=0)
    info = env.fly(5)
    np.testing.assert_array_equal(env.pilot.plan[0], env.menu[4])
    assert info["choice"] == 5 and info["outcome"] is Outcome.TIMEOUT
    env.reset(seed=0)
    env.fly(0)
    assert len(env.pilot.plan) == 1                     # straight to the hold point


def test_the_reward_is_docking_or_not_less_the_fuel():
    env = planner_env()
    env.reset(seed=0)
    for outcome, expected in ((Outcome.DOCKED, 100.0 - 10 * 0.8),
                              (Outcome.KEEP_OUT, -100.0 - 10 * 0.8),
                              (Outcome.TIMEOUT, -100.0 - 10 * 0.8)):
        env.fly = lambda action, o=outcome: {"outcome": o, "delta_v": 0.8, "time": 1500.0}
        _, reward, terminated, _, info = env.step(3)
        assert reward == pytest.approx(expected) and terminated
        assert info["is_success"] is (outcome is Outcome.DOCKED)


def test_time_counts_in_the_cost_when_weighted():
    env = planner_env()
    env.time_weight = 0.01
    env.reset(seed=0)
    env.fly = lambda action: {"outcome": Outcome.DOCKED, "delta_v": 0.8, "time": 1500.0}
    _, reward, _, _, _ = env.step(3)
    assert reward == pytest.approx(100.0 - 10 * 0.8 - 0.01 * 1500.0)


def test_the_relative_reward_is_measured_against_the_rule_from_the_same_start():
    env = planner_env(max_steps=5)
    env.relative, env.time_weight = True, 0.01
    env.reset(seed=3)
    start = env.state.copy()
    seen = []
    env.fly = lambda action: {"outcome": Outcome.DOCKED, "delta_v": 0.8, "time": 1500.0}

    def rule_flight(state):
        seen.append(state.copy())
        return {"outcome": Outcome.DOCKED, "delta_v": 1.0, "time": 1200.0}

    env.fly_rule = rule_flight
    _, reward, _, _, info = env.step(3)
    np.testing.assert_array_equal(seen[0], start)          # the rule flies the same start
    # Docked, plus J_rule - J = (10 + 12) - (8 + 15) = -1: worse than the rule by 1.
    assert reward == pytest.approx(100.0 - 1.0)
    assert info["rule_cost"] == pytest.approx(22.0)


def test_the_rule_replay_starts_where_the_choice_did():
    env = planner_env(max_steps=5)
    env.reset(seed=3)
    start = env.state.copy()
    env.fly(7)
    assert not np.array_equal(env.state, start)
    env.fly_rule(start)
    # The pilot's plan was rebuilt from that start with the rule, not the choice.
    assert env.corridor.steps == 5
    assert env.pilot.planner is env.rule


def test_the_observation_is_the_start_state():
    env = planner_env()
    obs, _ = env.reset(seed=4)
    np.testing.assert_allclose(obs, env.state / np.array([500.0, 500.0, 0.5, 0.5]), rtol=1e-6)
    assert env.observation_space.contains(obs)


@pytest.fixture
def planner_config(tmp_path):
    """The planner configuration, with untrained pilots saved to disk."""
    config = load_config(CONFIGS / "ppo_planner.yaml")
    go_to = PPO("MlpPolicy", GoToEnv(GOTO), seed=0)
    final = PPO("MlpPolicy", RendezvousEnv(STRICT), seed=0)
    go_to.save(tmp_path / "go_to.zip")
    final.save(tmp_path / "final.zip")
    config["planner"].update(go_to=str(tmp_path / "go_to.zip"),
                             final=str(tmp_path / "final.zip"))
    config["planner"]["go_to_config"] = str(CONFIGS / "ppo_goto_menu.yaml")
    config["planner"]["final_config"] = str(CONFIGS / "ppo_final_approach.yaml")
    return config


def test_the_planner_config_builds_its_environment(planner_config):
    env = make_env(planner_config)
    assert type(env) is PlannerEnv and env.action_space.n == 33
    assert env.pilot.keep_out == 20.0 and env.fuel_weight == 50.0


def test_the_planner_trains_and_keeps_its_best_model(planner_config, tmp_path):
    planner_config["environment"]["max_episode_steps"] = 20   # short approaches, for speed
    planner_config["training"].update(n_envs=2, n_steps=4, batch_size=8,
                                      best_model={"evaluate_every": 1, "episodes": 2})
    train(planner_config, 0, 16, tmp_path / "run", tmp_path / "planner.zip", checkpoints=False,
          env_factory=make_env)
    assert (tmp_path / "planner_best.zip").exists()
    assert len(json.loads((tmp_path / "run" / "best_model.json").read_text())) == 2


def test_a_planner_can_return_several_waypoints():
    pilot = still_pilot()
    pilot.planner = lambda state: [np.array([-60.0, 0.0]), np.array([-40.0, 20.0])]
    pilot(type("Env", (), {"state": np.array([0.0, -150.0, 0.0, 0.0]), "steps": 0})(), None)
    assert len(pilot.plan) == 3
    np.testing.assert_array_equal(pilot.plan[1], [-40.0, 20.0])
    np.testing.assert_array_equal(pilot.plan[2], [0.0, 30.0])      # always ending at the hold


def test_the_beam_search_extends_only_the_cheapest_plans_and_finds_the_best():
    from teacher_student.hierarchy import beam_search, best_plan

    env = planner_env()
    menu = [tuple(w) for w in env.menu]
    good = (menu[1], menu[20])          # the one two-waypoint plan that is cheaper
    hidden = (menu[3], menu[20])        # as cheap, but after a first waypoint outside the beam

    def fake_fly(planner):
        plan = tuple(tuple(w) for w in planner(None))
        cost = 60.0 + 2.0 * len(plan) + 0.1 * (menu.index(plan[0]) if plan else 0)
        if plan in (good, hidden):
            cost = 55.0
        return {"outcome": Outcome.DOCKED, "delta_v": 0.0, "time": cost / 0.01}

    env._fly = fake_fly
    env.time_weight, env.fuel_weight = 0.01, 0.0
    flown = beam_search(env, seed=0, width=3, depth=2)
    # None, 32 single waypoints, then 3 of them each extended by the 31 others.
    assert len(flown) == 1 + 32 + 3 * 31
    plan, value = best_plan(flown, 2)
    assert plan == good and value["cost"] == pytest.approx(55.0)
    assert best_plan(flown, 1)[0] == ()                            # without it: straight ahead
    assert hidden not in flown       # the price of the beam: only the 3 cheapest are extended

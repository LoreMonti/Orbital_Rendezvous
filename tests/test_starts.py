"""Tests for the curriculum on starting points (Step 15).

The option must leave the default environment untouched, seed for seed, since
every earlier model and result depends on which start a seed selects. With a
narrower range, every start must lie inside it, on both sides of the docking
axis; the curriculum must widen the range only once the agent has mastered
the outer edge of the current one, and only after the cone curriculum of
phase 1 has finished.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.utils import seeding

from orbital_rendezvous import EnvConfig, Outcome, RendezvousEnv
from orbital_rendezvous.callbacks import ConeCurriculum, StartCurriculum
from orbital_rendezvous.evaluation import (
    HELD_OUT_SEED,
    docked_by_sector,
    evaluate,
    start_angle,
)
from orbital_rendezvous.training import train
from orbital_rendezvous.utils import build_configs, load_config

CORRIDOR = Path(__file__).parents[1] / "configs" / "ppo_corridor.yaml"
STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)


def angle_and_side(position):
    """Angle between a position and the docking axis +y, in degrees, and its side."""
    return np.degrees(np.arctan2(abs(position[0]), position[1])), np.sign(position[0])


def test_every_direction_draws_the_same_starts_as_before():
    # The draw before the option existed, redone with gymnasium's generator.
    cfg = EnvConfig()
    env = RendezvousEnv(cfg)
    for seed in range(50):
        env.reset(seed=seed)
        rng, _ = seeding.np_random(seed)
        radius = rng.uniform(*cfg.initial_radius_range)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        velocity = rng.normal(0.0, cfg.initial_velocity_scale, size=2)
        expected = [radius * np.cos(angle), radius * np.sin(angle), *velocity]
        np.testing.assert_array_equal(env.state, expected)


def test_starts_stay_inside_a_narrower_range_on_both_sides():
    env = RendezvousEnv(EnvConfig(start_angle_range_deg=(0.0, 30.0)))
    low, high = env.config.initial_radius_range
    angles, sides = [], []
    for seed in range(1000):
        env.reset(seed=seed)
        angle, side = angle_and_side(env.state[:2])
        angles.append(angle)
        sides.append(side)
        assert low <= np.hypot(*env.state[:2]) <= high
    assert max(angles) <= 30.0 + 1e-9
    assert max(angles) > 29.0            # the whole range is used, not just its middle
    assert 400 < sides.count(1.0) < 600  # both sides of the axis, about equally


def test_set_start_angles_moves_the_range():
    env = RendezvousEnv(STRICT)
    env.set_start_angles(150.0, 180.0)
    for seed in range(200):
        env.reset(seed=seed)
        angle, _ = angle_and_side(env.state[:2])
        assert 150.0 - 1e-9 <= angle <= 180.0 + 1e-9


def test_starts_widen_only_once_mastered_and_stop_behind_the_station():
    starts = StartCurriculum(lambda: RendezvousEnv(STRICT), start_deg=15.0, step_deg=10.0,
                             success_threshold=0.9)
    assert starts.widened(15.0, 0.95) == 25.0
    assert starts.widened(15.0, 0.80) == 15.0    # not mastered: stays
    assert starts.widened(175.0, 1.00) == 180.0  # never beyond every direction
    assert starts.widened(180.0, 1.00) == 180.0


def test_mastery_is_measured_on_the_outer_band_only():
    starts = StartCurriculum(lambda: RendezvousEnv(STRICT), start_deg=100.0, band_deg=30.0)
    env = RendezvousEnv(STRICT)
    starts._configure(env)
    for seed in starts.seeds:
        env.reset(seed=seed)
        angle, _ = angle_and_side(env.state[:2])
        assert 70.0 - 1e-9 <= angle <= 100.0 + 1e-9
    # Early on the band would reach behind the axis: it stops at zero.
    starts.value = 15.0
    starts._configure(env)
    assert env.config.start_angle_range_deg == (0.0, 15.0)


def test_curriculum_sets_the_range_in_every_training_environment():
    starts = StartCurriculum(lambda: RendezvousEnv(STRICT), start_deg=15.0, evaluate_every=2)
    applied = []
    envs = SimpleNamespace(env_method=lambda name, *args: applied.append((name, *args)))
    logger = SimpleNamespace(record=lambda key, value: None)
    starts.model = SimpleNamespace(get_env=lambda: envs, logger=logger)
    starts.num_timesteps = 0
    readings = iter([0.95, 0.5, 0.92])
    starts.measure = lambda: next(readings)
    starts._on_training_start()
    for _ in range(6):   # measured on every second rollout: 3 readings
        starts._on_rollout_end()
    assert [h["start_deg"] for h in starts.history] == [25.0, 25.0, 35.0]
    assert applied[0] == ("set_start_angles", 0.0, 15.0)
    assert applied[-1] == ("set_start_angles", 0.0, 35.0)


def test_start_curriculum_waits_for_the_cone_to_finish():
    cone = ConeCurriculum(lambda: RendezvousEnv(STRICT), final_deg=15.0, start_deg=35.0,
                          evaluate_every=1)
    starts = StartCurriculum(lambda: RendezvousEnv(STRICT), start_deg=15.0, evaluate_every=1,
                             after=cone)
    envs = SimpleNamespace(env_method=lambda name, *args: None)
    logger = SimpleNamespace(record=lambda key, value: None)
    for c in (cone, starts):
        c.model = SimpleNamespace(get_env=lambda: envs, logger=logger)
        c.num_timesteps = 0
        c.measure = lambda: 1.0   # every stage mastered at once
    progress = []
    for _ in range(3):
        for c in (cone, starts):
            c._on_rollout_end()
        progress.append((cone.value, starts.value, starts.active))
    # The starts wait while the cone narrows, take over in the rollout it
    # reaches its final angle, and the cone is no longer measured after that.
    assert progress == [(25.0, 15.0, False), (15.0, 25.0, True), (15.0, 35.0, True)]
    assert [h["cone_deg"] for h in cone.history] == [25.0, 15.0]


def test_coasting_down_the_vbar_leaves_the_cone_the_vbar_thrust_does_not():
    # Why the cone must be learned first: sliding down the V-bar at 4 cm/s, the
    # Coriolis term 2 n |y'| pushes the chaser sideways out of a 15 degree cone;
    # the steady radial thrust 2 n v m of the V-bar procedure cancels it.
    def slide(thrust_x):
        env = RendezvousEnv(STRICT)
        env.reset(seed=0)
        env.state = np.array([0.0, 19.0, 0.0, -0.04])
        info = {"outcome": None}
        while info["outcome"] is None:
            _, _, _, _, info = env.step(np.array([thrust_x / env.config.max_thrust, 0.0]))
        return info["outcome"]

    env = RendezvousEnv(STRICT)
    assert slide(0.0) is Outcome.KEEP_OUT
    assert slide(2 * env.n * 0.04 * env.config.mass) is Outcome.DOCKED


def test_corridor_config_trains_with_both_curricula(tmp_path):
    config = load_config(CORRIDOR)
    env_config, _ = build_configs(config)
    # The task stays the full one: the curricula only change training.
    assert env_config.start_angle_range_deg == (0.0, 180.0)
    assert env_config.approach_cone_deg == 15.0
    config["training"]["n_envs"] = 2
    config["training"]["n_steps"] = 64
    config["training"]["batch_size"] = 64
    model = train(config, 3, 256, tmp_path / "run", tmp_path / "model.zip", checkpoints=False)
    configs = model.get_env().get_attr("config")
    # Phase 1 under way: starts in front of the port, no constraint yet.
    assert [c.start_angle_range_deg for c in configs] == [(0.0, 15.0), (0.0, 15.0)]
    assert [c.approach_cone_deg for c in configs] == [180.0, 180.0]
    saved = json.loads((tmp_path / "run" / "curriculum.json").read_text())
    assert saved == {"cone": [], "starts": []}
    saved = load_config(tmp_path / "run" / "config.yaml")
    assert saved["environment"]["start_angle_range_deg"] == [0.0, 180.0]
    assert saved["training"]["seed"] == 3   # the seed used, not the one in the YAML


def test_dockings_are_counted_by_the_direction_of_the_start():
    def attempt(angle, outcome):
        position = 100.0 * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
        return SimpleNamespace(start=np.append(position, [0.0, 0.0]), outcome=outcome)

    runs = [attempt(10, Outcome.DOCKED), attempt(-30, Outcome.KEEP_OUT),   # both sides count
            attempt(50, Outcome.DOCKED),
            attempt(120, Outcome.DOCKED), attempt(180, Outcome.KEEP_OUT)]   # 180 is included
    assert start_angle(runs[1].start) == pytest.approx(30.0)
    assert docked_by_sector(runs) == [(1, 2), (1, 1), (1, 1), (0, 1)]


def test_a_rollout_records_where_it_started():
    env = RendezvousEnv(STRICT)
    run = evaluate(env, lambda e, obs: np.zeros(2), [HELD_OUT_SEED])[0]
    env.reset(seed=HELD_OUT_SEED)
    np.testing.assert_array_equal(run.start, env.state)


@pytest.mark.parametrize("name", ["ppo_default", "ppo_corridor"])
def test_both_configs_draw_from_every_direction(name):
    env_config, _ = build_configs(load_config(CORRIDOR.parent / f"{name}.yaml"))
    assert env_config.start_angle_range_deg == (0.0, 180.0)

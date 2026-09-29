"""Tests for the reverse curriculum (Step 21): starts next to the port first, then further out.

The option must leave every earlier start unchanged, seed for seed. Inside the
keep-out sphere a start must lie in the approach cone, or it would fail before
the first thrust. Each stage must respect its distances and its angle, advance
only once mastered, and be tested on its outer band; the training must start
at the first stage from the first episode.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, RendezvousEnv
from orbital_rendezvous.callbacks import ReverseCurriculum
from orbital_rendezvous.training import train
from orbital_rendezvous.utils import build_configs, load_config

CONFIGS = Path(__file__).parents[1] / "configs"
STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)
STAGES = [(5.0, 10.0), (10.0, 10.0), (20.0, 10.0), (40.0, 45.0), (200.0, 180.0)]


def angle_and_distance(state):
    return (np.degrees(np.arctan2(abs(state[0]), state[1])), float(np.hypot(*state[:2])))


def test_the_keep_out_sphere_does_not_change_the_usual_starts():
    # Starts from 80 m out never fall inside the sphere, so they draw nothing more.
    plain, oriented = RendezvousEnv(EnvConfig()), RendezvousEnv(STRICT)
    for seed in range(50):
        plain.reset(seed=seed)
        oriented.reset(seed=seed)
        np.testing.assert_array_equal(plain.state, oriented.state)


def test_starts_inside_the_sphere_lie_in_the_approach_cone():
    env = RendezvousEnv(STRICT)
    env.set_start_region(2.0, 40.0, 180.0)
    inside = 0
    for seed in range(1000):
        env.reset(seed=seed)
        angle, distance = angle_and_distance(env.state)
        assert 2.0 <= distance <= 40.0
        if distance < 20.0:
            inside += 1
            assert angle <= 10.0 + 1e-9
    assert inside > 300


def test_each_stage_keeps_to_its_distances_and_its_angle():
    reverse = ReverseCurriculum(lambda: RendezvousEnv(STRICT), STAGES, min_radius=2.0)
    env = RendezvousEnv(STRICT)
    for stage, (radius, angle) in enumerate(STAGES):
        env.set_start_region(*reverse.region(stage))
        for seed in range(200):
            env.reset(seed=seed)
            a, d = angle_and_distance(env.state)
            assert 2.0 - 1e-9 <= d <= radius + 1e-9
            assert a <= max(angle, 10.0) + 1e-9


def test_the_stage_advances_only_once_mastered_and_stops_at_the_task():
    reverse = ReverseCurriculum(lambda: RendezvousEnv(STRICT), STAGES, success_threshold=0.9)
    assert reverse.advanced(0.0, 0.95) == 1.0
    assert reverse.advanced(0.0, 0.85) == 0.0
    assert reverse.advanced(4.0, 1.0) == 4.0          # the last stage is the task
    assert reverse.final_deg == 4.0


def test_mastery_is_tested_on_the_outer_band_of_distances():
    reverse = ReverseCurriculum(lambda: RendezvousEnv(STRICT), STAGES, band=0.3)
    reverse.value = 3.0                                # up to 40 m and 45 degrees
    env = RendezvousEnv(STRICT)
    reverse._configure(env)
    for seed in reverse.seeds:
        env.reset(seed=seed)
        assert 28.0 - 1e-9 <= angle_and_distance(env.state)[1] <= 40.0 + 1e-9
    reverse.value = 0.0                                # up to 5 m: the band stays above 2 m
    reverse._configure(env)
    assert env.config.initial_radius_range == (3.5, 5.0)


def test_the_curriculum_moves_the_starts_of_every_training_environment():
    reverse = ReverseCurriculum(lambda: RendezvousEnv(STRICT), STAGES, evaluate_every=1)
    applied = []
    envs = SimpleNamespace(env_method=lambda name, *args: applied.append((name, *args)))
    reverse.model = SimpleNamespace(get_env=lambda: envs,
                                    logger=SimpleNamespace(record=lambda k, v: None))
    reverse.num_timesteps = 0
    readings = iter([0.95, 0.5, 0.92])
    reverse.measure = lambda: next(readings)
    reverse._on_training_start()
    for _ in range(3):
        reverse._on_rollout_end()
    assert [h["stage"] for h in reverse.history] == [1.0, 1.0, 2.0]
    assert applied[0] == ("set_start_region", 2.0, 5.0, 10.0, None)
    assert applied[-1] == ("set_start_region", 2.0, 20.0, 10.0, None)


def test_the_reverse_configuration_trains_from_the_first_stage(tmp_path):
    config = load_config(CONFIGS / "ppo_corridor_reverse.yaml")
    env_config, _ = build_configs(config)
    # The task itself is unchanged: the curriculum only moves the training starts.
    assert env_config.initial_radius_range == (80.0, 200.0)
    assert env_config.start_angle_range_deg == (0.0, 180.0)
    config["training"].update(n_envs=2, n_steps=64, batch_size=64,
                              best_model={"evaluate_every": 1, "episodes": 2})
    model = train(config, 0, 256, tmp_path / "run", tmp_path / "reverse.zip", checkpoints=False)
    configs = model.get_env().get_attr("config")
    assert [c.initial_radius_range for c in configs] == [(2.0, 5.0), (2.0, 5.0)]
    assert [c.start_angle_range_deg for c in configs] == [(0.0, 10.0), (0.0, 10.0)]
    # The cone opened first, then narrowed on the first stage (variant B).
    assert [c.approach_cone_deg for c in configs] == [180.0, 180.0]
    assert set(json.loads((tmp_path / "run" / "curriculum.json").read_text())) == {"cone",
                                                                                   "reverse"}
    assert (tmp_path / "reverse_best.zip").exists()


def test_start_velocities_shrink_close_to_the_port_and_stay_the_same_far_away():
    near = RendezvousEnv(STRICT)
    near.set_start_region(2.0, 5.0, 10.0)
    speeds = []
    for seed in range(500):
        near.reset(seed=seed)
        distance = float(np.hypot(*near.state[:2]))
        speeds.append(np.hypot(*near.state[2:]) * 15.0 / distance)
    # Rescaled by 15 m / r, the speeds are the usual ones: median about 0.05 * 1.18.
    assert 0.05 < np.median(speeds) < 0.07
    # A radius so small that no start is closer: the draws before the option existed.
    far_default = RendezvousEnv(EnvConfig())
    far_old = RendezvousEnv(EnvConfig(start_speed_radius=1e-9))
    for seed in range(20):
        far_default.reset(seed=seed)
        far_old.reset(seed=seed)
        np.testing.assert_array_equal(far_default.state, far_old.state)


def test_a_timeout_can_be_made_a_failure():
    from orbital_rendezvous import Outcome, RewardConfig
    from orbital_rendezvous.rewards import terminal_reward

    assert terminal_reward(Outcome.TIMEOUT, RewardConfig()) == 0.0        # the default
    assert terminal_reward(Outcome.TIMEOUT, RewardConfig(timeout_reward=-100.0)) == -100.0
    assert terminal_reward(Outcome.DOCKED, RewardConfig(timeout_reward=-100.0)) == 100.0
    env = RendezvousEnv(STRICT, RewardConfig(timeout_reward=-100.0))
    env.reset(seed=0)
    env.steps = env.config.max_episode_steps - 1
    env.state = np.array([0.0, 150.0, 0.0, 0.0])
    _, _, terminated, _, info = env.step(np.zeros(2))
    assert info["outcome"] is Outcome.TIMEOUT and terminated
    assert info["reward_terms"]["terminal"] == -100.0


def test_the_cone_can_stay_at_the_rule_from_the_start(tmp_path):
    config = load_config(CONFIGS / "ppo_corridor_reverse.yaml")
    config["training"]["reverse_curriculum"]["cone_start_deg"] = None
    config["training"].update(n_envs=2, n_steps=64, batch_size=64, best_model=None)
    model = train(config, 0, 256, tmp_path / "run", tmp_path / "reverse.zip", checkpoints=False)
    configs = model.get_env().get_attr("config")
    assert [c.approach_cone_deg for c in configs] == [15.0, 15.0]    # the rule from the start
    assert [c.initial_radius_range for c in configs] == [(2.0, 5.0), (2.0, 5.0)]
    stages = json.loads((tmp_path / "run" / "curriculum.json").read_text())
    assert set(stages) == {"reverse"}


def margin_scales(radius_margin=5.0, cone_margin=8.0):
    from orbital_rendezvous.rewards import Scales

    return Scales(500.0, 0.5, 0.05, 500.0, 10.0, keep_out_radius=20.0, approach_cone_deg=15.0,
                  shaping_radius_margin=radius_margin, shaping_cone_margin_deg=cone_margin)


def at(degrees, distance):
    return distance * np.array([np.sin(np.radians(degrees)), np.cos(np.radians(degrees))])


def test_without_margins_the_shaping_path_is_the_old_one():
    from orbital_rendezvous.rewards import path_length

    plain = margin_scales(0.0, 0.0)
    half = np.arccos(20 / 100)
    expected = np.sqrt(100**2 - 20**2) + 20 * abs(np.pi - half - np.radians(15)) + 20
    assert path_length(np.array([0.0, -100.0]), plain) == pytest.approx(expected)


def test_with_margins_the_path_goes_around_a_wider_sphere_to_a_narrower_mouth():
    from orbital_rendezvous.rewards import path_length

    half = np.arccos(25 / 100)
    expected = np.sqrt(100**2 - 25**2) + 25 * abs(np.pi - half - np.radians(7)) + 25
    assert path_length(np.array([0.0, -100.0]), margin_scales()) == pytest.approx(expected)


def test_inside_the_true_cone_the_way_stays_straight():
    from orbital_rendezvous.rewards import path_length

    # 5 m out and 10 degrees off the axis: legal, and no detour is asked for.
    assert path_length(at(10, 5.0), margin_scales()) == pytest.approx(5.0)
    assert path_length(at(14, 60.0), margin_scales()) == pytest.approx(60.0)


def test_cutting_the_corner_on_the_rim_looks_further_with_margins():
    from orbital_rendezvous.rewards import path_length

    # Where the agent's violations were: on the rim, about 22 degrees off the axis.
    corner = at(22, 20.5)
    assert path_length(corner, margin_scales()) > path_length(corner, margin_scales(0.0, 0.0)) + 5


def test_the_frontier_is_the_newest_part_of_a_stage():
    reverse = ReverseCurriculum(lambda: RendezvousEnv(STRICT), STAGES, band=0.3,
                                frontier_fraction=0.5)
    # Up to 40 m and 45 degrees, after 20 m and 10: the outer 30 % of distances,
    # and the angles added since the last stage.
    assert reverse.frontier(3.0) == (0.5, (28.0, 40.0), (10.0, 45.0))
    # A stage that only moves further out keeps every angle.
    assert reverse.frontier(1.0) == (0.5, (7.0, 10.0), (0.0, 10.0))


def test_a_share_of_the_starts_comes_from_the_frontier():
    env = RendezvousEnv(STRICT)
    env.set_start_region(2.0, 40.0, 45.0, frontier=(0.5, (28.0, 40.0), (35.0, 45.0)))
    hard = 0
    for seed in range(1000):
        env.reset(seed=seed)
        angle, distance = angle_and_distance(env.state)
        assert 2.0 <= distance <= 40.0 and angle <= 45.0 + 1e-9
        hard += distance >= 28.0 and angle >= 35.0 - 1e-9
    # Half from the frontier, plus the few whole-stage draws that fall there by chance.
    assert 480 < hard < 600
    env.set_start_region(2.0, 40.0, 45.0)            # and without it, as before
    assert env.frontier is None


def test_sac_trains_on_the_same_task_and_is_loaded_back_as_sac(tmp_path):
    from stable_baselines3 import SAC

    from orbital_rendezvous.training import load_model

    config = load_config(CONFIGS / "sac_corridor_reverse.yaml")
    config["training"].update(n_envs=2, learning_starts=0, buffer_size=1000, batch_size=32,
                              train_freq=[16, "step"], gradient_steps=1, best_model=None)
    config["training"]["policy_kwargs"] = {"net_arch": [16, 16]}
    model = train(config, 0, 64, tmp_path / "run", tmp_path / "sac.zip", checkpoints=False)
    assert isinstance(model, SAC)
    assert [c.approach_cone_deg for c in model.get_env().get_attr("config")] == [180.0, 180.0]
    assert isinstance(load_model(tmp_path / "sac.zip"), SAC)


def test_an_unknown_algorithm_is_refused(tmp_path):
    import pytest as _pytest

    config = load_config(CONFIGS / "ppo_corridor_reverse.yaml")
    config["training"]["algorithm"] = "DQN"
    with _pytest.raises(ValueError, match="DQN"):
        train(config, 0, 64, tmp_path / "run", tmp_path / "x.zip", checkpoints=False)

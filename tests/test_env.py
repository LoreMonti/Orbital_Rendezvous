"""Tests for the Gymnasium environment.

Each terminal condition is reached by placing the chaser by hand in a state
that must lead to it in one step, so a mislabelled outcome or a wrong
terminated/truncated flag shows up directly.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from orbital_rendezvous import EnvConfig, Outcome, RendezvousEnv, RewardConfig


@pytest.fixture
def env():
    env = RendezvousEnv()
    env.reset(seed=0)
    return env


def place(env, state):
    env.state = np.asarray(state, dtype=float)


@pytest.mark.filterwarnings("error")
def test_passes_gymnasium_checker():
    check_env(RendezvousEnv(), skip_render_check=True)


def test_spaces():
    env = RendezvousEnv()
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (4,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


def test_reset_is_deterministic_under_a_seed():
    obs_a, _ = RendezvousEnv().reset(seed=42)
    obs_b, _ = RendezvousEnv().reset(seed=42)
    obs_c, _ = RendezvousEnv().reset(seed=43)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert not np.array_equal(obs_a, obs_c)


def test_initial_distance_within_range():
    env = RendezvousEnv()
    low, high = env.config.initial_radius_range
    for seed in range(200):
        _, info = env.reset(seed=seed)
        assert low <= info["distance"] <= high


def test_observation_is_normalised(env):
    place(env, [250.0, -100.0, 0.25, -0.5])
    obs = env._observation()
    np.testing.assert_allclose(obs, [0.5, -0.2, 0.5, -1.0])
    assert obs.dtype == np.float32


def test_thrust_saturates(env):
    start = [100.0, 0.0, 0.0, 0.0]
    place(env, start)
    _, _, _, _, info_clipped = env.step(np.array([5.0, -5.0]))
    clipped = env.state.copy()

    place(env, start)
    _, _, _, _, info_max = env.step(np.array([1.0, -1.0]))
    np.testing.assert_array_equal(clipped, env.state)
    np.testing.assert_array_equal(info_clipped["thrust"], info_max["thrust"])


def test_running_step_carries_no_terminal_reward(env):
    place(env, [100.0, 0.0, 0.0, 0.0])
    _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert info["reward_terms"]["terminal"] == 0.0
    assert not terminated and not truncated
    assert info["outcome"] is None
    assert "is_success" not in info


def test_reward_is_the_sum_of_its_terms(env):
    place(env, [100.0, 20.0, -0.1, 0.05])
    _, reward, _, _, info = env.step(np.array([0.3, -0.9]))
    assert set(info["reward_terms"]) == {"shaping", "fuel", "terminal"}
    assert reward == pytest.approx(sum(info["reward_terms"].values()))


def test_docking(env):
    place(env, [0.5, 0.0, 0.0, 0.0])
    _, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert info["outcome"] is Outcome.DOCKED
    assert info["reward_terms"]["terminal"] == RewardConfig().success_bonus
    assert reward > 0.9 * RewardConfig().success_bonus
    assert terminated and not truncated
    assert info["is_success"]


def test_crash_inside_the_sphere(env):
    # 2 m/s towards the target: ends the step inside the sphere, far too fast.
    place(env, [0.0, 1.5, 0.0, -2.0])
    _, reward, terminated, _, info = env.step(np.zeros(2))
    assert info["outcome"] is Outcome.CRASHED
    assert info["reward_terms"]["terminal"] == RewardConfig().failure_penalty
    assert reward < 0.0
    assert terminated
    assert not info["is_success"]


def test_crash_through_the_sphere(env):
    # 5 m/s: starts 3 m away and ends 2 m past the target. Neither endpoint
    # is inside the sphere, but the segment between them crosses it.
    place(env, [0.0, 3.0, 0.0, -5.0])
    _, _, terminated, _, info = env.step(np.zeros(2))
    assert info["distance"] > env.config.docking_radius
    assert info["outcome"] is Outcome.CRASHED
    assert terminated


def test_escape(env):
    place(env, [499.9, 0.0, 1.0, 0.0])
    _, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert info["outcome"] is Outcome.ESCAPED
    assert info["reward_terms"]["terminal"] == RewardConfig().failure_penalty
    assert reward < 0.0
    assert terminated and not truncated


def test_timeout_is_a_truncation_not_a_failure():
    env = RendezvousEnv(EnvConfig(max_episode_steps=5))
    env.reset(seed=0)
    place(env, [100.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        assert not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert info["outcome"] is Outcome.TIMEOUT
    assert truncated and not terminated
    assert info["reward_terms"]["terminal"] == 0.0
    assert not info["is_success"]


def test_step_follows_the_dynamics(env):
    # The environment must not add anything of its own to the physics.
    from orbital_rendezvous.dynamics import propagate

    start = np.array([120.0, -30.0, 0.02, 0.01])
    action = np.array([0.4, -0.8])
    place(env, start)
    env.step(action)
    expected = propagate(start, env.config.max_thrust * action, env.phi, env.gamma)
    np.testing.assert_array_equal(env.state, expected)

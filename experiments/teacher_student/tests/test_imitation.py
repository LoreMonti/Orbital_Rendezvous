"""Tests for learning the planner's choice from the oracle (Step 17c).

The targets must never favour a choice that fails, and must share weight
between near-ties. The fit must be able to learn a choice that jumps, as the
best side does behind the station: a categorical policy fitted to such labels
has to switch sharply where they switch.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from orbital_rendezvous import EnvConfig, RendezvousEnv
from teacher_student.goto import waypoint_menu
from teacher_student.hierarchy import HierarchicalPilot, PlannerEnv, oracle_costs
from teacher_student.imitation import expected_costs, imitate, soft_targets


def test_targets_never_favour_a_choice_that_fails():
    costs = np.array([[10.0, 20.0, 5.0]])
    docked = np.array([[True, True, False]])        # the cheapest one did not dock
    p = soft_targets(costs, docked, temperature=1.0)
    assert p[0, 2] == 0.0
    assert p[0, 0] > p[0, 1] and p.sum() == pytest.approx(1.0)


def test_near_ties_share_the_target_and_the_temperature_sets_how_near():
    costs = np.array([[10.0, 10.2, 30.0]])
    docked = np.ones((1, 3), dtype=bool)
    p = soft_targets(costs, docked, temperature=1.0)
    assert p[0, 1] / p[0, 0] == pytest.approx(np.exp(-0.2))
    assert p[0, 2] < 1e-8
    cold = soft_targets(costs, docked, temperature=0.01)
    assert cold[0, 0] > 0.99                          # nearly the argmin


def test_starts_where_nothing_docks_are_rejected():
    with pytest.raises(ValueError):
        soft_targets(np.zeros((2, 3)), np.array([[True, False, False], [False] * 3]), 1.0)


class Bandit(gym.Env):
    """Spaces only: a four-number observation and three choices."""

    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {}


def test_the_fit_learns_a_choice_that_jumps():
    # Choice 0 always fails; the best of the other two jumps at x = 0, like
    # the side to go around behind the station.
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1.0, 1.0, size=(2000, 4)).astype(np.float32)
    left = obs[:, 0] < 0
    costs = np.stack([np.zeros(2000), np.where(left, 1.0, 5.0), np.where(left, 5.0, 1.0)], 1)
    docked = np.stack([np.zeros(2000, bool), np.ones(2000, bool), np.ones(2000, bool)], 1)
    model = PPO("MlpPolicy", Bandit(), seed=0, device="cpu")
    history = imitate(model, obs, soft_targets(costs, docked, 1.0), epochs=40, seed=0)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    probe = np.zeros((2, 4), dtype=np.float32)
    probe[:, 0] = [-0.1, 0.1]
    np.testing.assert_array_equal(model.predict(probe, deterministic=True)[0], [1, 2])


def test_the_oracle_flies_every_choice_from_the_same_start():
    pilot = HierarchicalPilot(lambda obs: np.zeros(2), lambda obs: np.zeros(2),
                              EnvConfig(docking_radius=2.0, docking_speed=0.04),
                              EnvConfig(max_episode_steps=150))
    corridor = RendezvousEnv(EnvConfig(keep_out_radius=20.0, max_episode_steps=3))
    env = PlannerEnv(corridor, pilot, waypoint_menu((40.0, 60.0), 16), 50.0, 0.01)
    obs, costs, docked = oracle_costs(env, seed=7)
    expected, _ = env.reset(seed=7)
    np.testing.assert_array_equal(obs, expected)
    assert costs.shape == (33,) and not docked.any()     # three steps: all time out
    # Coasting from the same start costs the same whatever the waypoint: 3 steps, no fuel.
    np.testing.assert_allclose(costs, 0.01 * 30.0)


def test_a_failure_is_priced_far_above_the_cost_of_fuel():
    costs = np.array([[60.0, 65.0]])
    docked = np.array([[False, True]])
    np.testing.assert_allclose(expected_costs(costs, docked, 200.0), [[1.3, 0.325]])


def test_the_expected_cost_moves_probability_away_from_failing_choices():
    # Two choices the targets rate equally; the second fails. Only the
    # expected-cost term can tell them apart.
    obs = np.random.default_rng(0).uniform(-1.0, 1.0, size=(500, 4)).astype(np.float32)
    targets = np.full((500, 3), [0.5, 0.5, 0.0])
    penalties = np.tile([0.3, 1.3, 1.3], (500, 1))

    def probabilities(risk_weight):
        model = PPO("MlpPolicy", Bandit(), seed=0, device="cpu")
        imitate(model, obs, targets, penalties, risk_weight, epochs=30, seed=0)
        dist = model.policy.get_distribution(torch.as_tensor(obs[:50]))
        return dist.distribution.probs.detach().numpy().mean(axis=0)

    plain, careful = probabilities(0.0), probabilities(5.0)
    assert abs(plain[0] - plain[1]) < 0.05
    # The optimum of -ln(pi_0)/2 - ln(pi_1)/2 + 5 (0.3 pi_0 + 1.3 pi_1) is
    # pi_0 = 0.91; thirty epochs get well on the way.
    assert careful[0] - careful[1] > 0.3

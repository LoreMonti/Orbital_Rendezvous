"""Tests for the reward function.

The property that makes potential-based shaping safe is that it telescopes:
over any trajectory the discounted sum of the shaping terms depends only on the
two ends. That is tested directly, together with the signs of every term.
"""

import numpy as np
import pytest

from orbital_rendezvous.rewards import (
    Outcome,
    RewardConfig,
    Scales,
    potential,
    speed_limit,
    step_reward,
    terminal_reward,
)

SCALES = Scales(
    max_distance=500.0, velocity_scale=0.5, docking_speed=0.05, mass=500.0, time_step=1.0
)
CONFIG = RewardConfig()
UNDISCOUNTED = RewardConfig(gamma=1.0)
NO_THRUST = np.zeros(2)


def shaping(previous, state, config=UNDISCOUNTED, terminated=False):
    terms = step_reward(
        np.asarray(previous, float), np.asarray(state, float), NO_THRUST, terminated,
        config, SCALES,
    )
    return terms["shaping"]


def test_potential_is_zero_at_the_target_at_rest():
    assert potential(np.zeros(4), CONFIG, SCALES) == 0.0


def test_potential_is_negative_elsewhere():
    assert potential(np.array([10.0, 0.0, 0.0, 0.0]), CONFIG, SCALES) < 0.0


def test_glide_slope_ends_at_the_docking_speed():
    assert speed_limit(0.0, CONFIG, SCALES) == SCALES.docking_speed
    # tau = 200 s: 0.55 m/s allowed at 100 m.
    assert speed_limit(100.0, CONFIG, SCALES) == pytest.approx(0.55)


def test_closing_in_is_rewarded_and_receding_is_penalised():
    assert shaping([100.0, 0, 0, 0], [90.0, 0, 0, 0]) > 0.0
    assert shaping([90.0, 0, 0, 0], [100.0, 0, 0, 0]) < 0.0


def test_speeding_near_the_target_is_penalised():
    # Same position; 0.3 m/s is above the 0.1 m/s allowed at 10 m.
    slow = potential(np.array([10.0, 0.0, 0.05, 0.0]), CONFIG, SCALES)
    fast = potential(np.array([10.0, 0.0, 0.3, 0.0]), CONFIG, SCALES)
    assert slow == potential(np.array([10.0, 0.0, 0.0, 0.0]), CONFIG, SCALES)
    assert fast < slow


def test_shaping_telescopes_so_it_cannot_be_farmed():
    # Over any trajectory, sum_k gamma^k F_k = gamma^K Phi(s_K) - Phi(s_0).
    # A closed loop therefore earns nothing, however it is flown.
    rng = np.random.default_rng(0)
    states = rng.normal(0.0, [100.0, 100.0, 0.3, 0.3], size=(50, 4))
    states[-1] = states[0]

    gamma = CONFIG.gamma
    total = sum(
        gamma**k * shaping(states[k], states[k + 1], CONFIG) for k in range(len(states) - 1)
    )
    expected = gamma ** (len(states) - 1) * potential(states[-1], CONFIG, SCALES) - potential(
        states[0], CONFIG, SCALES
    )
    assert total == pytest.approx(expected, rel=1e-12)

    loop = sum(shaping(states[k], states[k + 1]) for k in range(len(states) - 1))
    assert loop == pytest.approx(0.0, abs=1e-9)


def test_terminal_state_has_zero_potential():
    previous = np.array([0.8, 0.0, -0.01, 0.0])
    after = np.array([0.5, 0.0, -0.01, 0.0])
    assert shaping(previous, after, CONFIG, terminated=True) == pytest.approx(
        -potential(previous, CONFIG, SCALES)
    )


def test_thrusting_is_never_free():
    state = np.array([100.0, 0.0, 0.0, 0.0])
    idle = step_reward(state, state, NO_THRUST, False, CONFIG, SCALES)["fuel"]
    burn = step_reward(state, state, np.array([0.6, -0.8]), False, CONFIG, SCALES)["fuel"]
    assert idle == 0.0
    # |u| = 1 N for 1 s on 500 kg: 2 mm/s of delta-v, at 10 per m/s.
    assert burn == pytest.approx(-CONFIG.fuel_weight * 1.0 * 1.0 / 500.0)


def test_terminal_rewards():
    assert terminal_reward(Outcome.DOCKED, CONFIG) == CONFIG.success_bonus
    assert terminal_reward(Outcome.CRASHED, CONFIG) == CONFIG.failure_penalty
    assert terminal_reward(Outcome.ESCAPED, CONFIG) == CONFIG.failure_penalty
    assert terminal_reward(Outcome.TIMEOUT, CONFIG) == 0.0
    assert terminal_reward(None, CONFIG) == 0.0

"""Tests for the reward function.

The property that makes potential-based shaping safe is that it telescopes:
over any trajectory the discounted sum of the shaping terms depends only on the
two ends. That is tested directly, together with the signs of every term.
"""

import numpy as np
import pytest

from orbital_rendezvous.core.rewards import (
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


GRADED = RewardConfig(graded_failure=True)


def test_graded_failure_is_off_by_default():
    for miss in (0.0, 0.5, 1.0):
        assert terminal_reward(Outcome.KEEP_OUT, CONFIG, miss) == CONFIG.failure_penalty
        assert terminal_reward(Outcome.CRASHED, CONFIG, miss) == CONFIG.failure_penalty


def test_graded_failure_runs_from_the_floor_to_the_full_penalty():
    floor = GRADED.failure_floor * GRADED.failure_penalty
    assert terminal_reward(Outcome.KEEP_OUT, GRADED, 0.0) == pytest.approx(floor)
    assert terminal_reward(Outcome.KEEP_OUT, GRADED, 1.0) == pytest.approx(GRADED.failure_penalty)
    # A runaway keeps the full penalty, however small the miss: fleeing must
    # never cost less than a near miss. Docking and timeout are untouched.
    assert terminal_reward(Outcome.ESCAPED, GRADED, 0.0) == GRADED.failure_penalty
    assert terminal_reward(Outcome.DOCKED, GRADED, 0.0) == GRADED.success_bonus
    assert terminal_reward(Outcome.TIMEOUT, GRADED, 0.0) == GRADED.timeout_reward
    # Every failure stays worse than a timeout.
    assert terminal_reward(Outcome.KEEP_OUT, GRADED, 0.0) < GRADED.timeout_reward


def test_miss_size_grows_with_angle_and_speed_and_saturates():
    from orbital_rendezvous.core.rewards import miss_size

    assert miss_size(15.0, 0.10, 0.10, 15.0, GRADED) == 0.0          # on the rim, at the limit
    assert miss_size(10.0, 0.05, 0.10, 15.0, GRADED) == 0.0          # inside: no error
    angles = [miss_size(a, 0.0, 0.10, 15.0, GRADED) for a in (16, 20, 30, 45)]
    speeds = [miss_size(15.0, v, 0.10, 15.0, GRADED) for v in (0.11, 0.14, 0.18, 0.20)]
    assert angles == sorted(angles) and angles[-1] == pytest.approx(1.0)
    assert speeds == sorted(speeds) and speeds[-1] == pytest.approx(1.0)
    # The worked example of the README: 17 deg at 0.07 m/s costs about 53.
    miss = miss_size(17.0, 0.07, 0.10, 15.0, GRADED)
    assert terminal_reward(Outcome.KEEP_OUT, GRADED, miss) == pytest.approx(-53.3, abs=0.1)
    # A fast arrival saturates even when well centred.
    assert miss_size(16.0, 0.25, 0.10, 15.0, GRADED) == 1.0

"""Tests for the classical references and the evaluation helpers.

The LQR is checked against the Riccati equation it is supposed to solve, and
the two-impulse transfer by flying it: after the first impulse, propagating
with the closed-form state-transition matrix must land exactly on the origin,
with exactly the arrival velocity the second impulse cancels.
"""

import numpy as np
import pytest

from orbital_rendezvous import Outcome, RendezvousEnv
from orbital_rendezvous.baselines import LQRController, best_two_impulse, two_impulse
from orbital_rendezvous.dynamics import mean_motion, orbital_period, state_transition
from orbital_rendezvous.evaluation import evaluate, pareto_front, summarise

N = mean_motion(6.778e6)


@pytest.fixture(scope="module")
def env():
    return RendezvousEnv()


def test_lqr_solves_the_riccati_equation(env):
    lqr = LQRController.from_env(env, approach_time=200.0, fuel_weight=1e-2)
    a, b, p, q, r = env.phi, env.gamma, lqr.p, lqr.q, lqr.r
    residual = q + a.T @ p @ a - a.T @ p @ b @ np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a) - p
    np.testing.assert_allclose(residual, 0.0, atol=1e-9 * np.abs(p).max())


def test_lqr_closed_loop_is_stable(env):
    for tau in (50.0, 200.0):
        for weight in (1e-4, 1.0):
            lqr = LQRController.from_env(env, tau, weight)
            assert np.abs(np.linalg.eigvals(lqr.closed_loop)).max() < 1.0


def test_free_fuel_lqr_approaches_at_its_time_constant(env):
    # With fuel almost free, the velocity weight is an implicit glide slope
    # dr/dt = -r / tau: the slowest closed-loop mode is exp(-dt / tau).
    tau = 300.0
    lqr = LQRController.from_env(env, tau, 1e-9)
    slowest = np.abs(np.linalg.eigvals(lqr.closed_loop)).max()
    assert slowest == pytest.approx(np.exp(-env.config.time_step / tau), rel=1e-3)


def test_lqr_actions_are_saturated(env):
    lqr = LQRController.from_env(env, 50.0, 1e-6)
    action = lqr.act(np.array([400.0, -400.0, 1.0, 1.0]))
    assert np.all(np.abs(action) <= 1.0)
    assert np.abs(action).max() == 1.0


def test_lqr_docks(env):
    lqr = LQRController.from_env(env, 200.0, 1e-3)
    runs = evaluate(env, lambda e, obs: lqr.act(e.state), range(10_000, 10_010))
    assert all(run.outcome is Outcome.DOCKED for run in runs)


def test_two_impulse_transfer_lands_on_the_origin():
    state = np.array([120.0, -80.0, 0.03, -0.01])
    duration = 1500.0
    phi = state_transition(N, duration)
    v0_plus = -np.linalg.solve(phi[:2, 2:], phi[:2, :2] @ state[:2])
    arrival = phi @ np.concatenate([state[:2], v0_plus])
    np.testing.assert_allclose(arrival[:2], 0.0, atol=1e-9)

    first, second = two_impulse(state, N, duration)
    assert first == pytest.approx(np.linalg.norm(v0_plus - state[2:]))
    assert second == pytest.approx(np.linalg.norm(arrival[2:]))


def test_two_impulse_from_rest_at_the_origin_costs_nothing():
    assert sum(two_impulse(np.zeros(4), N, 1000.0)) == pytest.approx(0.0, abs=1e-12)


def test_best_two_impulse_skips_singular_durations():
    # At a full orbital period Phi_rv is singular; the search must step over it.
    period = orbital_period(N)
    delta_v, duration = best_two_impulse(
        np.array([100.0, 0.0, 0.0, 0.0]), N, np.array([period, 0.5 * period])
    )
    assert np.isfinite(delta_v)
    assert duration == pytest.approx(0.5 * period)


def test_pareto_front():
    points = [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0), (2.5, 2.5), (4.0, 1.5)]
    assert pareto_front(points) == [1, 2, 0]


def test_summary_counts_costs_on_docked_attempts_only(env):
    runs = evaluate(env, lambda e, obs: np.zeros(2), range(10_000, 10_004))
    summary = summarise(runs)
    assert summary.success_rate == 0.0
    assert np.isnan(summary.delta_v_median)
    assert sum(summary.outcomes.values()) == 4


def test_lqr_keeps_the_engine_switch_on():
    from orbital_rendezvous import EnvConfig

    env = RendezvousEnv(EnvConfig(engine_switch=True))
    lqr = LQRController.from_env(env, 200.0, 1e-3)
    action = lqr.act(np.array([100.0, 0.0, 0.0, 0.0]))
    assert action.shape == (3,) and action[2] == 1.0
    runs = evaluate(env, lambda e, obs: lqr.act(e.state), range(10_000, 10_005))
    assert all(run.outcome is Outcome.DOCKED for run in runs)

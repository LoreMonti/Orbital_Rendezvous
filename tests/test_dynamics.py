"""Tests for the Clohessy-Wiltshire propagation.

The dynamics are linear, so every check here is exact rather than statistical.
A sign flip or a missing factor of n would still produce plausible-looking
trajectories; these invariants are the ones that would catch it.
"""

import numpy as np
import pytest

from orbital_rendezvous.dynamics import (
    discretize,
    mean_motion,
    orbital_period,
    propagate,
    state_transition,
)

# Target at 400 km altitude, as in configs/ppo_default.yaml.
N = mean_motion(6778.0e3)
MASS = 500.0
T = orbital_period(N)


def test_mean_motion_matches_leo_period():
    # A 400 km orbit takes about 92.6 minutes.
    assert abs(T / 60.0 - 92.56) < 0.05


@pytest.mark.parametrize("dt", [1.0, 60.0, 1000.0, T])
def test_expm_matches_analytical_transition(dt):
    phi_expm, _ = discretize(N, dt, MASS)
    np.testing.assert_allclose(phi_expm, state_transition(N, dt), rtol=1e-9, atol=1e-9)


def test_rest_at_origin_stays_at_rest():
    phi, gamma = discretize(N, 10.0, MASS)
    state = np.zeros(4)
    for _ in range(1000):
        state = propagate(state, np.zeros(2), phi, gamma)
    np.testing.assert_array_equal(state, np.zeros(4))


def test_radial_offset_drifts_along_track():
    # After exactly one period the radial motion has returned to x0, and the
    # along-track position has drifted by -6 n x0 T: the secular term.
    x0 = 10.0
    state = state_transition(N, T) @ np.array([x0, 0.0, 0.0, 0.0])
    assert state[0] == pytest.approx(x0, rel=1e-9)
    assert state[1] == pytest.approx(-6.0 * N * x0 * T, rel=1e-9)


def test_matched_velocity_gives_closed_orbit():
    # With vy0 = -2 n x0 the secular terms cancel: a periodic 2:1 ellipse.
    x0 = 50.0
    initial = np.array([x0, 0.0, 0.0, -2.0 * N * x0])
    final = state_transition(N, T) @ initial
    np.testing.assert_allclose(final, initial, atol=1e-8)


def test_two_steps_equal_one_double_step():
    # Covers Gamma as well as Phi: under a constant thrust,
    # Phi(dt) Phi(dt) s + (Phi(dt) Gamma(dt) + Gamma(dt)) u = Phi(2dt) s + Gamma(2dt) u.
    dt = 30.0
    state = np.array([20.0, -40.0, 0.01, -0.02])
    thrust = np.array([0.3, -0.7])

    phi, gamma = discretize(N, dt, MASS)
    twice = propagate(propagate(state, thrust, phi, gamma), thrust, phi, gamma)

    phi2, gamma2 = discretize(N, 2.0 * dt, MASS)
    once = propagate(state, thrust, phi2, gamma2)

    np.testing.assert_allclose(twice, once, rtol=1e-10, atol=1e-12)


def test_small_n_reduces_to_double_integrator_plus_coriolis():
    # For n -> 0 a constant thrust over dt must give dx = u dt^2 / (2m) and
    # dv = u dt / m, which pins the 1/m in B. The first-order corrections are
    # the Coriolis coupling, antisymmetric between the axes: n dt^3 / (3m) on
    # position and n dt^2 / m on velocity. Checking them pins its sign.
    n, dt = 1e-12, 5.0
    _, gamma = discretize(n, dt, MASS)
    p = dt**2 / (2 * MASS)
    v = dt / MASS
    cp = n * dt**3 / (3 * MASS)
    cv = n * dt**2 / MASS
    expected = np.array(
        [
            [p, cp],
            [-cp, p],
            [v, cv],
            [-cv, v],
        ]
    )
    np.testing.assert_allclose(gamma, expected, rtol=1e-9, atol=1e-20)


def test_propagate_accepts_batches():
    phi, gamma = discretize(N, 1.0, MASS)
    states = np.random.default_rng(0).normal(size=(8, 4))
    thrusts = np.random.default_rng(1).normal(size=(8, 2))
    batch = propagate(states, thrusts, phi, gamma)
    single = np.stack([propagate(s, u, phi, gamma) for s, u in zip(states, thrusts, strict=True)])
    np.testing.assert_allclose(batch, single)

"""Clohessy-Wiltshire relative motion, with no reinforcement learning in sight.

The target flies a circular orbit of semi-major axis :math:`a`, with mean
motion :math:`n = \\sqrt{\\mu / a^3}`. In the LVLH frame centred on the target
(:math:`x` radial, :math:`y` along-track), the chaser obeys

.. math::

    \\ddot{x} - 3 n^2 x - 2 n \\dot{y} = u_x / m,
    \\qquad
    \\ddot{y} + 2 n \\dot{x} = u_y / m,

a linear time-invariant system :math:`\\dot{s} = A s + B u` on the state
:math:`s = [x, y, \\dot{x}, \\dot{y}]`.

Because the system is linear, the propagation over one step is exact. Two
independent routes to it are kept on purpose:

- `state_transition` writes the classical closed-form matrix :math:`\\Phi(t)`;
- `discretize` obtains :math:`\\Phi` and the zero-order-hold input matrix
  :math:`\\Gamma` together from a single matrix exponential (Van Loan, 1978).

The tests require the two :math:`\\Phi` to agree to machine precision, which is
the strongest check available on either of them.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

EARTH_MU = 3.986004418e14
"""Standard gravitational parameter of the Earth, in m^3/s^2."""


def mean_motion(semi_major_axis: float, mu: float = EARTH_MU) -> float:
    """Return the mean motion ``n = sqrt(mu / a**3)`` in rad/s."""
    if semi_major_axis <= 0.0:
        raise ValueError("semi_major_axis must be positive")
    return float(np.sqrt(mu / semi_major_axis**3))


def orbital_period(n: float) -> float:
    """Return the period ``T = 2 pi / n`` of the target orbit, in seconds."""
    return 2.0 * np.pi / n


def system_matrices(n: float, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time matrices ``A`` (4x4) and ``B`` (4x2) of the CW system."""
    a = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [3.0 * n**2, 0.0, 0.0, 2.0 * n],
            [0.0, 0.0, -2.0 * n, 0.0],
        ]
    )
    b = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ) / mass
    return a, b


def state_transition(n: float, dt: float) -> np.ndarray:
    """Closed-form CW state-transition matrix over ``dt`` for the free motion.

    With ``tau = n dt``, ``s = sin(tau)`` and ``c = cos(tau)``, this is the
    classical matrix acting on ``[x, y, vx, vy]``. The entry ``6 (s - tau)`` is
    the secular along-track drift caused by a radial offset.
    """
    tau = n * dt
    s, c = np.sin(tau), np.cos(tau)
    return np.array(
        [
            [4.0 - 3.0 * c, 0.0, s / n, 2.0 * (1.0 - c) / n],
            [6.0 * (s - tau), 1.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * tau) / n],
            [3.0 * n * s, 0.0, c, 2.0 * s],
            [-6.0 * n * (1.0 - c), 0.0, -2.0 * s, 4.0 * c - 3.0],
        ]
    )


def discretize(n: float, dt: float, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discrete matrices ``(Phi, Gamma)`` for a thrust held over ``dt``.

    Van Loan's method: the exponential of the block matrix
    ``[[A, B], [0, 0]] dt`` equals ``[[Phi, Gamma], [0, I]]``, so both matrices
    are read off one call to ``expm``. The result is exact, not an
    approximation, and is meant to be computed once and cached.
    """
    a, b = system_matrices(n, mass)
    block = np.zeros((6, 6))
    block[:4, :4] = a
    block[:4, 4:] = b
    exp_block = expm(block * dt)
    return exp_block[:4, :4], exp_block[:4, 4:]


def control_matrix(n: float, dt: float, mass: float) -> np.ndarray:
    """Discrete input matrix ``Gamma`` (4x2) for a thrust held constant over ``dt``."""
    return discretize(n, dt, mass)[1]


def propagate(
    state: np.ndarray,
    thrust: np.ndarray,
    phi: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Advance ``[x, y, vx, vy]`` by one step: ``s' = Phi s + Gamma u``.

    Takes the precomputed matrices from `discretize`, so a step costs two small
    matrix products. Also works on batches of shape ``(..., 4)`` and ``(..., 2)``.
    """
    return np.asarray(state) @ phi.T + np.asarray(thrust) @ gamma.T

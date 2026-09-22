"""Clohessy-Wiltshire relative motion, with no reinforcement learning in sight.

The target flies a circular orbit of semi-major axis :math:`a`, with mean
motion :math:`n = \\sqrt{\\mu / a^3}`. In the LVLH frame centred on the target
(:math:`x` radial, :math:`y` along-track), the chaser obeys

.. math::

    \\ddot{x} - 3 n^2 x - 2 n \\dot{y} = u_x / m,
    \\qquad
    \\ddot{y} + 2 n \\dot{x} = u_y / m.

The system is linear, so the propagation over one step is exact: this module
exposes the closed-form state-transition matrices rather than a numerical
integrator, which is both faster and testable against an analytical solution.
"""

from __future__ import annotations

import numpy as np

EARTH_MU = 3.986004418e14
"""Standard gravitational parameter of the Earth, in m^3/s^2."""


def mean_motion(semi_major_axis: float, mu: float = EARTH_MU) -> float:
    """Return the mean motion ``n = sqrt(mu / a**3)`` in rad/s."""
    raise NotImplementedError


def state_transition(n: float, dt: float) -> np.ndarray:
    """Closed-form CW state-transition matrix over ``dt`` for the free motion.

    Returns the 4x4 matrix acting on the state ``[x, y, vx, vy]``.
    """
    raise NotImplementedError


def control_matrix(n: float, dt: float, mass: float) -> np.ndarray:
    """Discrete input matrix for a thrust held constant over ``dt``.

    Returns the 4x2 matrix acting on the commanded thrust ``[ux, uy]``.
    """
    raise NotImplementedError


def propagate(
    state: np.ndarray,
    thrust: np.ndarray,
    n: float,
    dt: float,
    mass: float,
) -> np.ndarray:
    """Advance ``[x, y, vx, vy]`` by one step under a constant thrust."""
    raise NotImplementedError

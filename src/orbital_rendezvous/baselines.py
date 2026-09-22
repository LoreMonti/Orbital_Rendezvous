"""Classical controllers, so the learned policy has something honest to beat.

Because the Clohessy-Wiltshire dynamics are linear, an infinite-horizon LQR is
available in closed form and is very hard to beat on fuel. Reporting that
comparison is the point: the interesting result is how close the agent gets,
not a claim that it wins.
"""

from __future__ import annotations

import numpy as np


class LQRController:
    """Infinite-horizon LQR for the discrete CW dynamics."""

    def __init__(
        self,
        n: float,
        dt: float,
        mass: float,
        max_thrust: float,
        q_position: float = 1.0,
        q_velocity: float = 1.0,
        r_thrust: float = 1.0,
    ) -> None:
        raise NotImplementedError

    def act(self, state: np.ndarray) -> np.ndarray:
        """Return the thrust command, saturated at ``max_thrust``."""
        raise NotImplementedError

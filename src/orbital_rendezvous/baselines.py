"""Classical references for the learned policy: an LQR controller and the two-impulse transfer.

**LQR.** On the exact discrete dynamics ``s' = Phi s + Gamma u`` it minimises
``sum s^T Q s + u^T R u``, solving the discrete algebraic Riccati equation for
``P`` and applying ``u = -K s`` with ``K = (R + Gamma^T P Gamma)^-1 Gamma^T P Phi``,
saturated at the same thrust limit as the agent. Its fuel term is quadratic,
``|u|^2``, while the delta-v the agent pays is ``|u|``: LQR prefers to thrust a
little all the time, whereas delta-v is minimised by a few decisive burns. It is
therefore a solid classical controller, not a fuel-optimal bound. The ratio of
``R`` to ``Q`` trades time for fuel; sweeping it traces a curve rather than
picking one point by hand.

**Two-impulse transfer.** The textbook reference for delta-v. A first impulse
sets the velocity that reaches the origin after a time ``T``,
``v0+ = -Phi_rv^-1 Phi_rr r0``, and a second one cancels the arrival velocity.
Impulses are instantaneous and ignore the thrust limit, so this is not a
controller that can fly in the environment: it is the delta-v an ideal
manoeuvre would need, minimised over ``T``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from .dynamics import state_transition


class LQRController:
    """Infinite-horizon discrete LQR, returning actions in the environment's ``[-1, 1]^2``.

    ``Q = diag(1, 1, tau^2, tau^2) / r_ref^2`` and ``R = fuel_weight / u_max^2``.
    With free fuel, the balance between a position error ``r^2`` and a speed
    ``tau^2 v^2`` is an exponential approach ``dr/dt = -r / tau``: the velocity
    weight acts as an implicit glide slope, and ``tau`` is its time constant,
    the same quantity as the glide slope of the reward. ``fuel_weight`` then
    trades time for fuel on top of it.
    """

    def __init__(
        self,
        phi: np.ndarray,
        gamma: np.ndarray,
        max_thrust: float,
        position_scale: float,
        approach_time: float = 200.0,
        fuel_weight: float = 1e-3,
    ) -> None:
        self.max_thrust = max_thrust
        # With an engine switch in the environment the LQR keeps it always on.
        self.engine_switch = False
        self.approach_time = approach_time
        self.fuel_weight = fuel_weight
        self.q = np.diag([1.0, 1.0, approach_time**2, approach_time**2]) / position_scale**2
        self.r = np.eye(2) * fuel_weight / max_thrust**2
        self.p = solve_discrete_are(phi, gamma, self.q, self.r)
        self.k = np.linalg.solve(self.r + gamma.T @ self.p @ gamma, gamma.T @ self.p @ phi)
        self.closed_loop = phi - gamma @ self.k

    @classmethod
    def from_env(
        cls, env, approach_time: float = 200.0, fuel_weight: float = 1e-3
    ) -> LQRController:
        cfg = env.config
        controller = cls(
            env.phi, env.gamma, cfg.max_thrust, cfg.max_distance, approach_time, fuel_weight
        )
        controller.engine_switch = cfg.engine_switch
        return controller

    def act(self, state: np.ndarray) -> np.ndarray:
        """Action for the physical state ``[x, y, vx, vy]``, saturated to ``[-1, 1]``."""
        action = np.clip(-self.k @ state / self.max_thrust, -1.0, 1.0)
        return np.append(action, 1.0) if self.engine_switch else action


def two_impulse(state: np.ndarray, n: float, duration: float) -> tuple[float, float]:
    """Delta-v of the two impulses that bring ``state`` to rest at the origin in ``duration``.

    Returns ``(first, second)``, in m/s.
    """
    phi = state_transition(n, duration)
    r0, v0 = state[:2], state[2:]
    v0_plus = -np.linalg.solve(phi[:2, 2:], phi[:2, :2] @ r0)
    v_arrival = phi[2:, :2] @ r0 + phi[2:, 2:] @ v0_plus
    return float(np.linalg.norm(v0_plus - v0)), float(np.linalg.norm(v_arrival))


def best_two_impulse(
    state: np.ndarray, n: float, durations: np.ndarray
) -> tuple[float, float]:
    """The cheapest two-impulse transfer over ``durations``: ``(delta_v, duration)``.

    Durations where ``Phi_rv`` is nearly singular, at multiples of the orbital
    period, are skipped: there the required impulse diverges.
    """
    best = (np.inf, np.nan)
    for duration in durations:
        phi_rv = state_transition(n, duration)[:2, 2:]
        if np.linalg.cond(phi_rv) > 1e8:
            continue
        total = sum(two_impulse(state, n, duration))
        if total < best[0]:
            best = (total, float(duration))
    return best

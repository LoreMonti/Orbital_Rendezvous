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


def side_waypoint(
    start: np.ndarray,
    hold: np.ndarray,
    keep_out: float,
    distance: float = 50.0,
    margin: float = 5.0,
) -> np.ndarray | None:
    """A point beside the keep-out sphere, if the straight way to the hold point crosses it.

    ``None`` when the segment from ``start`` to ``hold`` stays more than
    ``margin`` outside the sphere; otherwise a point ``distance`` out on the
    radial axis, on the side the chaser starts from. Shared by the V-bar
    procedure and the learned pilots, so that both fly the same plan.
    """
    d = hold - start
    t = np.clip(-(start @ d) / max(d @ d, 1e-12), 0.0, 1.0)
    if np.hypot(*(start + t * d)) > keep_out + margin:
        return None
    side = np.sign(start[0]) or 1.0
    return np.array([side * distance, 0.0])


class VbarApproach:
    """The classical procedure for an oriented target: a hold point, then along the V-bar.

    Phase 1 flies an LQR to a hold point on the docking axis, outside the
    keep-out sphere; if the way there would cut through the sphere, it goes
    first to a waypoint beside it. Points on the V-bar (``x = 0``, at rest) are
    equilibria of the Clohessy-Wiltshire equations, which is why real missions
    hold there. Phase 2 tracks a reference sliding down the axis at a constant
    ``closing_speed`` to the port. Staying on the axis while moving along it
    needs a steady radial thrust against the Coriolis term: with ``x = 0`` and
    ``y' = -v_c``, the x equation gives ``u_x = 2 n v_c m``.

    Callable as a controller, ``(env, obs) -> action``; it resets itself when the
    environment starts a new episode.
    """

    def __init__(
        self,
        env,
        approach_time: float = 100.0,
        fuel_weight: float = 1e-3,
        hold_distance: float = 30.0,
        closing_speed: float = 0.04,
        hold_tolerance: float = 2.0,
        waypoint_distance: float = 50.0,
    ) -> None:
        cfg = env.config
        self.lqr = LQRController.from_env(env, approach_time, fuel_weight)
        self.n, self.mass, self.dt = env.n, cfg.mass, cfg.time_step
        self.keep_out = cfg.keep_out_radius
        self.hold = np.array([0.0, hold_distance])
        self.closing_speed = closing_speed
        self.hold_tolerance = hold_tolerance
        self.waypoint_distance = waypoint_distance
        self.reset(np.zeros(4))

    def reset(self, state: np.ndarray) -> None:
        self.phase = "waypoint"
        self.slide_steps = 0
        self.waypoint = self._waypoint(state[:2])
        if self.waypoint is None:
            self.phase = "hold"

    def _waypoint(self, start: np.ndarray) -> np.ndarray | None:
        return side_waypoint(start, self.hold, self.keep_out, self.waypoint_distance)

    def _track(self, state: np.ndarray, reference: np.ndarray, feedforward: np.ndarray):
        thrust = feedforward - self.lqr.k @ (state - reference)
        action = np.clip(thrust / self.lqr.max_thrust, -1.0, 1.0)
        return np.append(action, 1.0) if self.lqr.engine_switch else action

    def __call__(self, env, obs) -> np.ndarray:
        state = env.state
        if env.steps == 0:
            self.reset(state)
        position, speed = state[:2], float(np.hypot(*state[2:]))
        if self.phase == "waypoint":
            if np.hypot(*(position - self.waypoint)) < 3 * self.hold_tolerance:
                self.phase = "hold"
            else:
                return self._track(state, np.append(self.waypoint, [0.0, 0.0]), np.zeros(2))
        if self.phase == "hold":
            at_hold = np.hypot(*(position - self.hold)) < self.hold_tolerance
            if at_hold and speed < self.closing_speed:
                self.phase = "slide"
            else:
                return self._track(state, np.append(self.hold, [0.0, 0.0]), np.zeros(2))
        self.slide_steps += 1
        along = max(0.0, self.hold[1] - self.closing_speed * self.dt * self.slide_steps)
        reference = np.array([0.0, along, 0.0, -self.closing_speed if along > 0 else 0.0])
        coriolis = 2 * self.n * self.closing_speed * self.mass if along > 0 else 0.0
        return self._track(state, reference, np.array([coriolis, 0.0]))

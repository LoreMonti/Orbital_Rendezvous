"""A graph of exact manoeuvres round the station, and the value it gives (Step 28).

The planner of Step 26 docks through the port 195 times in 200 with a value
beyond its horizon written by hand from the geometry, and 100 times with one
learned from its own flights. This module computes the value instead, from the
structure of the problem: the Clohessy-Wiltshire dynamics are linear and
solved in closed form, so the exact manoeuvre between two points is known.

**Nodes.** Points round the station: rings outside the keep-out sphere, and
inside it only the approach cone, where a chaser may be; the port is the
origin.

**Edges.** From rest at ``p_i`` to rest at ``p_j`` in a time ``T``, the
two-impulse transfer, with ``Phi(T)`` split into position and velocity blocks:

    v0+ = Phi_rv(T)^-1 (p_j - Phi_rr(T) p_i),
    dv1 = |v0+|,   dv2 = |Phi_vr(T) p_i + Phi_vv(T) v0+|.

An edge is kept only if its arc, sampled along the way, never enters the
keep-out sphere outside the cone, and if neither impulse exceeds ``dv_max``: the
thruster gives 2 mm/s^2, so an impulse of 0.3 m/s already takes two and a half
minutes of full thrust, and larger ones would not be impulses at all. Nor may
an impulse need more than a quarter of the time of flight to give
(`impulse_limit`).

**Value.** Dijkstra from the port on the additive cost ``dv1 + dv2 + lambda T``
gives each node its cheapest way in. Along that way the value is written in the
units of the environment's reward, so that the planner can add it to the
rewards it simulates:

    V(p) = gamma^(T_tot / dt) * bonus - w_f * dv_tot.

The way round the sphere, and the side, come out of the shortest path; nobody
writes them. Blocks 1-3 follow the sampling-based planners of Starek, Pavone
and co-authors (J. Guid. Control Dyn. 40, 2017); the use of the value at the end
of a sampling MPC's horizon is `GraphValue`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from orbital_rendezvous.core.dynamics import state_transition
from orbital_rendezvous.core.env import RendezvousEnv


@dataclass(frozen=True)
class GraphConfig:
    """Settings of the graph. See the module docstring."""

    rings: tuple[float, ...] = (25.0, 30.0, 40.0, 60.0, 80.0, 110.0, 150.0, 200.0)  # m
    ring_step_deg: float = 15.0
    axis: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0)         # m, nodes on the axis inside
    cone_edge_deg: float = 7.0                               # nodes either side of the axis
    times: tuple[float, ...] = (200.0, 400.0, 800.0)         # s, times of flight of an edge
    # Times of flight from the planner's state to a node: shorter ones too, or
    # every state near the port would be worth the same.
    # Nearly continuous up to 300 s: with a few times only, states at different
    # distances from the port shared the same shortest transfer, and so the same
    # value, and the planner hovered on that plateau.
    entry_times: tuple[float, ...] = tuple(float(s) for s in range(30, 310, 10)) + (
        400.0, 600.0, 800.0)
    entry_samples: int = 15                                  # points checked on an entry
    reach: float = 60.0                                      # m, longest edge
    dv_max: float = 0.3                                      # m/s, largest impulse
    # An impulse must be one the thruster can give in this share of the time of
    # flight, at its full acceleration u_max / m.
    burn_share: float = 0.25
    nearest: int = 12                                        # nodes tried from a state
    time_weight: float = 0.001                               # lambda, m/s per s
    samples: int = 30                                        # points checked along an arc
    # Arcs inside the keep-out sphere must stay this far within the cone. On
    # the rim itself, the way in suited the graph pilot, but its distilled
    # student, off by a couple of degrees, entered the sphere outside the cone
    # 62 times in 200, at a median of 17 degrees. The rule is unchanged.
    cone_margin_deg: float = 5.0

    # Inside the sphere arcs keep under the glide slope v_dock + r / glide_time;
    # None turns it off. Without it the way in from the front was one long arc
    # down the axis, entered at up to 0.24 m/s: the pilot held it, its
    # distilled student, a little off, did not (15 of 41 from the front).
    glide_time: float | None = 400.0                         # s

    def glide(self, env_config) -> tuple[float, float] | None:
        """``(v_dock, tau)`` for `legal`, or None."""
        if self.glide_time is None:
            return None
        return env_config.docking_speed, self.glide_time

    def cone(self, cone_deg: float) -> float:
        """The cone the graph's arcs keep to: the true one less the margin."""
        return max(1.0, cone_deg - self.cone_margin_deg)


def nodes(config: GraphConfig, keep_out: float, cone_deg: float) -> np.ndarray:
    """Node positions; the port, the origin, is node 0."""
    points = [np.zeros(2)]
    for r in config.rings:
        if r <= keep_out:
            continue
        for a in np.arange(0.0, 360.0, config.ring_step_deg):
            points.append(r * np.array([np.sin(np.radians(a)), np.cos(np.radians(a))]))
    edge = min(config.cone_edge_deg, cone_deg * 0.5)
    for r in config.axis:
        if r >= keep_out:
            continue
        for a in (0.0, -edge, edge) if r > 5.0 else (0.0,):
            points.append(r * np.array([np.sin(np.radians(a)), np.cos(np.radians(a))]))
    return np.array(points)


def blocks(n: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The position and velocity blocks of ``Phi(t)``: rr, rv, vr, vv."""
    phi = state_transition(n, t)
    return phi[:2, :2], phi[:2, 2:], phi[2:, :2], phi[2:, 2:]


def transfer(n: float, t: float, p0: np.ndarray, v0: np.ndarray, p1: np.ndarray):
    """Two-impulse transfer in time ``t`` from ``(p0, v0)`` to rest at ``p1``, batched.

    Returns the departure velocity, the first impulse and the arrival velocity
    to cancel, each of shape ``(..., 2)``.
    """
    rr, rv, vr, vv = blocks(n, t)
    # Accelerate's matmul on macOS raises spurious flags on finite products.
    with np.errstate(all="ignore"):
        depart = (np.asarray(p1) - np.asarray(p0) @ rr.T) @ np.linalg.inv(rv).T
        arrive = np.asarray(p0) @ vr.T + depart @ vv.T
    return depart, depart - np.asarray(v0), arrive


def impulse_limit(config: GraphConfig, env: RendezvousEnv, t: float) -> float:
    """The largest impulse of a transfer lasting ``t``: what the thruster gives in a share of it.

    Impulses are an idealisation of a thruster of 2 mm/s^2, which needs 100 s for
    0.2 m/s. Without this limit a state 10 m from the port was worth a transfer
    of 50 s with two impulses of 0.2 m/s, gamma^5 * 100 = 94, more than the
    planner's own slow docking, 82, and the planner hovered there instead of
    docking. With it, the shortest transfer that can be flown grows with the
    distance, and so does the value as the chaser gets closer.
    """
    acceleration = env.config.max_thrust / env.config.mass
    return min(config.dv_max, config.burn_share * acceleration * t)


def legal(n: float, t: float, p0: np.ndarray, depart: np.ndarray, samples: int,
          keep_out: float, cone_deg: float, docking: float,
          glide: tuple[float, float] | None = None) -> np.ndarray:
    """Whether each arc stays out of the keep-out sphere, or within the cone inside it.

    With ``glide = (v_dock, tau)``, inside the sphere the arc must also keep under
    the glide slope, ``|v| <= v_dock + r / tau``, as a real final approach does.
    """
    ok = np.ones(len(p0), dtype=bool)
    cos_cone = np.cos(np.radians(cone_deg))
    for s in np.linspace(0.0, t, samples)[1:]:
        rr, rv, vr, vv = blocks(n, s)
        with np.errstate(all="ignore"):
            point = p0 @ rr.T + depart @ rv.T
        r = np.hypot(point[:, 0], point[:, 1])
        inside = (r >= docking) & (r < keep_out)
        ok &= ~(inside & (point[:, 1] < r * cos_cone))
        if glide is not None:
            with np.errstate(all="ignore"):
                velocity = p0 @ vr.T + depart @ vv.T
            speed = np.hypot(velocity[:, 0], velocity[:, 1])
            ok &= ~(inside & (speed > glide[0] + r / glide[1]))
    return ok


@dataclass
class CWGraph:
    """Nodes, legal edges and the value of every node, for one environment."""

    env: RendezvousEnv
    config: GraphConfig = field(default_factory=GraphConfig)

    def __post_init__(self) -> None:
        ec, rc = self.env.config, self.env.reward_config
        cfg = self.config
        self.points = nodes(cfg, ec.keep_out_radius, ec.approach_cone_deg)
        count = len(self.points)
        cost = np.full((count, count), np.inf)
        dv = np.zeros((count, count))
        time = np.zeros((count, count))
        i, j = np.nonzero((np.linalg.norm(self.points[:, None] - self.points[None], axis=2)
                           <= cfg.reach) & ~np.eye(count, dtype=bool))
        for t in cfg.times:
            p0, p1 = self.points[i], self.points[j]
            depart, dv1, arrive = transfer(self.env.n, t, p0, np.zeros_like(p0), p1)
            d1, d2 = np.linalg.norm(dv1, axis=1), np.linalg.norm(arrive, axis=1)
            limit = impulse_limit(cfg, self.env, t)
            good = (d1 <= limit) & (d2 <= limit) & legal(
                self.env.n, t, p0, depart, cfg.samples, ec.keep_out_radius,
                cfg.cone(ec.approach_cone_deg), ec.docking_radius, cfg.glide(ec))
            c = d1 + d2 + cfg.time_weight * t
            better = good & (c < cost[i, j])
            cost[i[better], j[better]] = c[better]
            dv[i[better], j[better]] = (d1 + d2)[better]
            time[i[better], j[better]] = t
        self.edges = int(np.isfinite(cost).sum())
        # Cost to go: Dijkstra from the port on the reversed graph.
        weights = np.where(np.isfinite(cost), cost, 0.0)
        reverse = csr_matrix(weights.T)
        self.cost, predecessors = dijkstra(reverse, indices=0, return_predecessors=True)
        # Along each node's way in, its fuel and time, then its value in reward units.
        self.dv_to_go = np.full(count, np.inf)
        self.time_to_go = np.full(count, np.inf)
        self.dv_to_go[0] = self.time_to_go[0] = 0.0
        for k in np.argsort(self.cost):
            nxt = predecessors[k]
            if k == 0 or nxt < 0:
                continue
            self.dv_to_go[k] = dv[k, nxt] + self.dv_to_go[nxt]
            self.time_to_go[k] = time[k, nxt] + self.time_to_go[nxt]
        self.next = predecessors
        reachable = np.isfinite(self.dv_to_go)
        self.value = np.full(count, -rc.success_bonus)
        self.value[reachable] = (rc.gamma ** (self.time_to_go[reachable] / ec.time_step)
                                 * rc.success_bonus - rc.fuel_weight * self.dv_to_go[reachable])

    def path(self, k: int) -> list[int]:
        """Node indices of the cheapest way in from node ``k`` to the port."""
        way = [k]
        while way[-1] != 0 and self.next[way[-1]] >= 0:
            way.append(int(self.next[way[-1]]))
        return way


class GraphValue:
    """``V(s)``: the best legal transfer from ``s`` to a node nearby, plus that node's value.

    For a batch of states at the end of the planner's horizon,

        V(s) = max over j, T of gamma^(T/dt) V(p_j) - w_f (dv1 + dv2),

    over the ``nearest`` nodes within ``reach`` and the times of flight ``entry_times``. A
    state with no legal transfer gets ``fallback``, a failure.

    The entry times include short ones, 50 and 100 s. With the graph's 200 s as
    the shortest, every state within reach of the port was worth the same,
    gamma^20 * 100 = 82: the planner, guided round the station into the cone,
    then hovered 8 m from the port until the timeout, since getting closer
    paid nothing and trying to dock risked a crash.
    """

    def __init__(self, graph: CWGraph, fallback: float | None = None):
        self.graph = graph
        rc = graph.env.reward_config
        self.fallback = -rc.success_bonus if fallback is None else fallback

    def entries(self, states: np.ndarray, steps: int):
        """For every state and nearby node, the best legal transfer: (state, node, value).

        Value ``gamma^(T/dt) V(p_j) - w_f (dv1 + dv2)``, maximised over the entry
        times; ``-inf`` where no transfer is legal.
        """
        g, ec, rc, cfg = self.graph, self.graph.env.config, self.graph.env.reward_config, \
            self.graph.config
        states = np.atleast_2d(states)
        distance = np.linalg.norm(states[:, None, :2] - g.points[None], axis=2)
        distance[:, ~np.isfinite(g.dv_to_go)] = np.inf
        # Not the port itself: the docking is the planner's to fly, within its
        # horizon, with real thrust. Valued as an impulsive stop at the port, every
        # state near it promised an easy docking later, and the planner, which
        # replans each step, put it off for good.
        distance[:, 0] = np.inf
        k = min(cfg.nearest, len(g.points))
        nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
        si = np.repeat(np.arange(len(states)), k)
        nj = nearest.ravel()
        keep = distance[si, nj] <= cfg.reach
        si, nj = si[keep], nj[keep]
        values = np.full(len(si), -np.inf)
        if len(si) == 0:
            return si, nj, values
        p0, v0, p1 = states[si, :2], states[si, 2:], g.points[nj]
        left = ec.max_episode_steps - steps
        for t in cfg.entry_times:
            if t / ec.time_step > left:
                continue
            depart, dv1, arrive = transfer(g.env.n, t, p0, v0, p1)
            d1, d2 = np.linalg.norm(dv1, axis=1), np.linalg.norm(arrive, axis=1)
            limit = impulse_limit(cfg, g.env, t)
            ok = (d1 <= limit) & (d2 <= limit) & legal(
                g.env.n, t, p0, depart, cfg.entry_samples, ec.keep_out_radius,
                cfg.cone(ec.approach_cone_deg), ec.docking_radius, cfg.glide(ec))
            v = rc.gamma ** (t / ec.time_step) * g.value[nj] - rc.fuel_weight * (d1 + d2)
            values = np.maximum(values, np.where(ok, v, -np.inf))
        return si, nj, values

    def __call__(self, states: np.ndarray, steps: int) -> np.ndarray:
        states = np.atleast_2d(states)
        best = np.full(len(states), self.fallback, dtype=float)
        si, _, values = self.entries(states, steps)
        if len(si):
            np.maximum.at(best, si, np.where(np.isfinite(values), values, self.fallback))
        return np.maximum(best, self.final_approach(states))

    def final_approach(self, states: np.ndarray) -> np.ndarray:
        """The value of the slow descent down the cone, for states inside it.

        Impulsive transfers describe the way round the station, not the last
        metres: down the cone the chaser descends slowly under the glide slope
        of the task, ``v(r) = v_dock + r / tau``, which takes

            t(r) = tau ln(1 + r / (v_dock tau)),

        worth ``gamma^(t/dt) * bonus``: 95 at 3 m, 85 at 10 m, 76 at 20 m. Without
        it, and with the port excluded from the transfers, the value near the
        port did not grow towards it, and the planner flew from 2-10 m out to
        10-14 m. Outside the cone, or beyond the keep-out sphere, the graph
        alone values the state.
        """
        g = self.graph
        ec, rc = g.env.config, g.env.reward_config
        r = np.hypot(states[:, 0], states[:, 1])
        inside = (states[:, 1] >= r * np.cos(np.radians(ec.approach_cone_deg))) & (
            r <= ec.keep_out_radius)
        tau, v_dock = rc.approach_time, ec.docking_speed
        t = tau * np.log1p(r / (v_dock * tau))
        return np.where(inside, rc.gamma ** (t / ec.time_step) * rc.success_bonus, -np.inf)


class GraphPilot:
    """The graph as a map: fly its cheapest way in, node by node, then down the cone.

    A `Controller`. At the start it picks the node the chaser can best reach,
    the one maximising `GraphValue.entries`, and takes the graph's cheapest way
    from there to the port. A sampling MPC with a short horizon flies towards
    the current node, its value beyond the horizon the distance to that node;
    within ``switch`` metres of it the next node becomes the target. Once the
    chaser is inside the approach cone, the MPC flies the last metres with the
    value of the slow descent (`GraphValue.final_approach`). If the chaser
    makes no progress for ``patience`` steps, the way is planned again from
    where it is.

    The graph as the planner's value, instead, did not dock: impulsive
    transfers promised what a thruster of 2 mm/s^2 cannot do, and the planner
    hovered short of the port or parked on a node. Followed as a map, the
    graph decides only where to go, and the planner only how to get to the
    next point, which it does well.
    """

    def __init__(self, env: RendezvousEnv, graph: CWGraph, planner_config=None,
                 seed: int = 0, switch: float = 5.0, distance_weight: float = 0.2,
                 patience: int = 60, docking_margin: float = 0.8):
        from dataclasses import replace

        from orbital_rendezvous.planning.mpc import PlannerConfig, SamplingPlanner

        self.graph, self.value = graph, GraphValue(graph)
        self.switch, self.distance_weight, self.patience = switch, distance_weight, patience
        settings = planner_config or PlannerConfig(horizon=10, block=2, init_std=0.3)
        # The planner's model docks only below a share of the true docking speed:
        # planned at the limit itself, the chaser arrived at 0.049-0.050 m/s, and a
        # thrust off by 10 % turned 69 dockings in 200 into crashes.
        model = RendezvousEnv(replace(env.config,
                                      docking_speed=docking_margin * env.config.docking_speed),
                              env.reward_config)
        self.planner = SamplingPlanner(model, PlannerConfig(**{**settings.__dict__,
                                                               "value": "graph"}),
                                       seed=seed, value_fn=self._value)
        self.way: list[int] = []
        self.final = False
        self.since = 0

    def _inside_cone(self, state: np.ndarray) -> bool:
        return bool(np.isfinite(self.value.final_approach(np.atleast_2d(state))[0]))

    def plan(self, state: np.ndarray, steps: int) -> None:
        """The cheapest way in from ``state``: the best node to reach, then the graph's path."""
        self.final = self._inside_cone(state)
        self.since = 0
        if self.final:
            self.way = [0]
            return
        si, nj, values = self.value.entries(state[None], steps)
        if len(nj) == 0 or not np.isfinite(values).any():
            # Nothing reachable: head for the nearest node and plan again there.
            distances = np.linalg.norm(self.graph.points[1:] - state[:2], axis=1)
            self.way = self.graph.path(1 + int(np.argmin(distances)))
            return
        self.way = self.graph.path(int(nj[int(np.argmax(values))]))

    def _value(self, states: np.ndarray, steps: int) -> np.ndarray:
        if self.final:
            v = self.value.final_approach(states)
            return np.where(np.isfinite(v), v, -self.graph.env.reward_config.success_bonus)
        target = self.graph.points[self.way[0]]
        return -self.distance_weight * np.linalg.norm(states[:, :2] - target, axis=1)

    def __call__(self, env: RendezvousEnv, obs: np.ndarray) -> np.ndarray:
        state = env.state
        if env.steps == 0:
            self.plan(state, 0)
        # The descent starts only once the chaser is inside the cone: its value
        # is defined there alone, and switching on the last node, at the mouth
        # of the cone but outside the sphere, left the planner with a value of
        # -100 everywhere and a chaser flying off at random.
        self.final = self._inside_cone(state)
        if self.final:
            self.way = [0]
        else:
            target = self.graph.points[self.way[0]]
            gap = np.linalg.norm(state[:2] - target)
            if self.way[0] != 0 and gap < self.switch:
                self.way = self.way[1:] or [0]
                self.since = 0
            elif self.since > self.patience or gap > self.graph.config.reach:
                self.plan(state, env.steps)
        self.since += 1
        return self.planner(env, obs)

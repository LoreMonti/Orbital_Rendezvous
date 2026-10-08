"""Go-Explore, phase 1: find docking trajectories by exploring from an archive.

Trial and error from the start (Steps 14-25) and a planner with a learned
value (Step 26) did not find the way through the port from behind the
station. Go-Explore [Ecoffet et al., 2021] finds it in legs. It keeps an
archive of *cells*, the states reached so far grouped on a coarse grid; each
round it returns to one of them, restoring the simulator's state exactly, and
explores from there with a few random thrusts; a cell reached for the first
time, or reached with a better score, enters the archive. The way round the
sphere is then found one leg at a time, and the last metre is retried from
close to the port as often as needed.

A cell is the position on a grid and a class of speed; the grid is ``grid``
metres far out and shrinks to ``fine`` next to the port, where the last metres
have to be told apart. The
cell to return to is drawn with a weight

    w = exp(-r / length) / sqrt(n + 1),

``n`` the times it was chosen and ``r`` its distance to the port: the count
spreads the search, the distance is the generic prior of Step 26, blind to
the sphere and the cone. Chosen by count alone, the archive reached the mouth
of the cone from behind the station but never the port, 0 dockings in 10
starts after 100 000 rounds each; with the distance, 8 of 10 in 200 000.
Random thrusts have a random magnitude too, since the corridor asks for
small, steady ones.

Only the environment's own outcomes and fuel score a trajectory; the archive
keeps, for each cell, the actions that reached it, so a docking found is a
sequence of thrusts from the start that `replay` flies again from a fresh
reset. Phase 2, learning a policy from the trajectories found, is the
distillation of Step 20.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .env import RendezvousEnv
from .rewards import Outcome


@dataclass(frozen=True)
class GoExploreConfig:
    """Exploration settings. See the module docstring."""

    grid: float = 5.0                                   # m, largest cell size in position
    fine: float = 0.5                                   # m, smallest, next to the port
    speed_edges: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)   # m/s, speed classes
    length: float = 30.0                                # m, preference for cells near the port
    max_len: int = 10                                   # most steps explored per round
    hold_max: int = 5                                   # most steps a random thrust is held
    budget: int = 200_000                               # rounds per start, at most
    after_docking: int = 20_000                         # rounds kept after the first docking


@dataclass
class Exploration:
    """What the exploration of one start found."""

    seed: int
    start: np.ndarray
    docked: bool
    actions: np.ndarray = field(repr=False)     # (T, 2), the best docking, or empty
    score: float = float("-inf")               # its fuel and outcome rewards
    rounds: int = 0
    first_docking: int | None = None
    cells: int = 0
    closest: float = float("inf")              # m, nearest the archive came to the port
    toward: np.ndarray = field(default=None, repr=False)   # actions to that nearest cell


def cell_size(distance: float, config: GoExploreConfig) -> float:
    """Side of a cell at ``distance`` from the port: a quarter of it, within limits."""
    return float(np.clip(distance / 4.0, config.fine, config.grid))


def cell(state: np.ndarray, config: GoExploreConfig) -> tuple[int, int, int, int]:
    """The cell of a state: its size class, its square on that grid, its class of speed.

    Cells shrink near the port. On a grid of 5 m everywhere the last 5 m were
    one cell, in which a state 1 m from the port and one 4 m away looked the
    same, and the archive could not get closer: from 4 m it never docked.
    """
    size = cell_size(float(np.hypot(state[0], state[1])), config)
    level = int(np.round(np.log2(size / config.fine)))
    side = config.fine * 2.0**level
    return (level, int(np.floor(state[0] / side)), int(np.floor(state[1] / side)),
            int(np.searchsorted(config.speed_edges, np.hypot(state[2], state[3]))))


def weights(distances: np.ndarray, visits: np.ndarray, config: GoExploreConfig) -> np.ndarray:
    """Probability of returning to each cell: near the port and seldom chosen."""
    w = np.exp(-distances / config.length) / np.sqrt(visits + 1.0)
    return w / w.sum()


def explore(
    env: RendezvousEnv,
    seed: int,
    config: GoExploreConfig | None = None,
    state: np.ndarray | None = None,
    steps_done: int = 0,
) -> Exploration:
    """Explore from the start that ``seed`` selects; return the best docking found.

    With ``state`` the exploration starts there instead, ``steps_done`` steps
    into the episode: a planner flying a chaser explores again from where it is.
    """
    config = config or GoExploreConfig()
    if env.config.mirror_symmetry or env.config.engine_switch:
        raise ValueError("the exploration commands the true thrust on two axes")
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    if state is not None:
        env.state, env.steps = np.asarray(state, dtype=float).copy(), steps_done
    start = env.state.copy()
    index = {cell(start, config): 0}
    states, steps, scores, sequences = [start.copy()], [env.steps], [0.0], [np.zeros((0, 2))]
    capacity = 1024
    visits, distances = np.zeros(capacity), np.zeros(capacity)
    distances[0] = np.hypot(*start[:2])
    found = Exploration(seed=seed, start=start, docked=False, actions=np.zeros((0, 2)))
    rounds = 0
    while rounds < config.budget:
        if found.first_docking is not None and rounds - found.first_docking >= config.after_docking:
            break
        n = len(states)
        i = rng.choice(n, p=weights(distances[:n], visits[:n], config))
        visits[i] += 1
        rounds += 1
        env.reset(seed=seed)
        env.state, env.steps = states[i].copy(), steps[i]
        score, taken = scores[i], [sequences[i]]
        action, hold = np.zeros(2), 0
        for _ in range(rng.integers(1, config.max_len + 1)):
            if hold == 0:
                direction = rng.normal(size=2)
                action = direction / np.linalg.norm(direction) * rng.uniform(0.0, 1.0)
                hold = rng.integers(1, config.hold_max + 1)
            hold -= 1
            _, _, done, _, info = env.step(action)
            score += info["reward_terms"]["fuel"] + info["reward_terms"]["terminal"]
            taken.append(action[None, :])
            if done:
                if info["outcome"] is Outcome.DOCKED and score > found.score:
                    found.docked, found.score = True, score
                    found.actions = np.concatenate(taken)
                    if found.first_docking is None:
                        found.first_docking = rounds
                break
            c = cell(env.state, config)
            j = index.get(c)
            if j is None:
                j = index[c] = len(states)
                states.append(env.state.copy())
                steps.append(env.steps)
                scores.append(float("-inf"))
                sequences.append(np.zeros((0, 2)))
                if j >= capacity:
                    capacity *= 2
                    visits = np.resize(visits, capacity)
                    distances = np.resize(distances, capacity)
                visits[j] = 0.0
            if score > scores[j]:
                states[j], steps[j], scores[j] = env.state.copy(), env.steps, score
                sequences[j] = np.concatenate(taken)
                distances[j] = np.hypot(*env.state[:2])
    found.rounds, found.cells = rounds, len(states)
    nearest = int(np.argmin(distances[: len(states)]))
    found.closest, found.toward = float(distances[nearest]), sequences[nearest]
    return found


def replay(env: RendezvousEnv, seed: int, actions: np.ndarray) -> dict:
    """Fly ``actions`` again from a fresh reset: how it ends, and what it costs."""
    env.reset(seed=seed)
    delta_v, violated, info = 0.0, False, {"outcome": None}
    for action in actions:
        _, _, done, _, info = env.step(np.asarray(action))
        delta_v += info["delta_v"]
        violated = violated or info.get("keep_out_violated", False)
        if done:
            break
    return {"outcome": info["outcome"], "delta_v": delta_v, "violated": violated,
            "time": env.steps * env.config.time_step}


class GoExplorePlanner:
    """Go-Explore as a planner: explore from the chaser's state, fly, explore again.

    A `Controller`. At the start, and then every ``replan_every`` steps, it runs
    `explore` from the chaser's current state on a private copy of the
    environment, and flies the thrusts of the best docking found; if none was
    found, the thrusts toward the cell nearest the port. Without errors the
    first plan would dock alone, as its replay shows; replanning is what lets
    the chaser recover when the thrust is not the one commanded. There is no
    learning: every plan is a new search, guided only by the reward and by the
    preference for cells near the port.
    """

    def __init__(self, env: RendezvousEnv, config: GoExploreConfig | None = None,
                 replan_every: int = 30, seed: int = 0, replan_after_docking: int = 2_000):
        self.model = RendezvousEnv(env.config, env.reward_config)
        self.config = config or GoExploreConfig()
        # The first plan is searched in full; a replan keeps looking for a
        # cheaper docking for fewer rounds once it has one, since a flight
        # replans about eight times and the margin of phase 1, 20 000 rounds,
        # made a flight cost five minutes.
        self.replan_config = replace(self.config, after_docking=replan_after_docking)
        self.replan_every, self.seed = replan_every, seed
        self.plan, self.position, self.searches = np.zeros((0, 2)), 0, 0

    def __call__(self, env: RendezvousEnv, obs: np.ndarray) -> np.ndarray:
        if env.steps == 0:
            self.plan, self.position, self.searches = np.zeros((0, 2)), 0, 0
        if self.position >= len(self.plan) or self.position >= self.replan_every:
            settings = self.config if self.searches == 0 else self.replan_config
            found = explore(self.model, self.seed + 1000 * self.searches, settings,
                            state=env.state, steps_done=env.steps)
            self.searches += 1
            self.plan = found.actions if found.docked else found.toward
            self.position = 0
            if len(self.plan) == 0:
                return np.zeros(2)
        action = self.plan[self.position]
        self.position += 1
        return np.asarray(action, dtype=float)

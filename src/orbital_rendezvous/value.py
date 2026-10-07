"""A value learned from the planner's own flights, to replace the one written by hand.

The planner of `planning` needs ``V(s)``, the return still to come from a state
beyond its horizon. Step 26 found that blind (``V = 0`` or the straight
distance) it docks 0 of 200, and with the shaping potential, written by hand
from the geometry of the corridor, 195. Here ``V`` is learned instead:

    V(s) = prior(s) + 100 dV_theta(o(s)),    prior(s) = -w r / r_max,

a generic prior, the straight distance to the port, blind to the sphere and
the cone, plus a correction a small network learns from experience; ``o(s)``
is the observation, the state normalised and the elapsed fraction of the
episode, and the factor 100 keeps the network's output of order one.

The network is fitted to the return each visited state actually got, with the
environment's own rewards and no shaping:

    G_t = r_t + gamma G_{t+1},   L(theta) = mean (V(s_i) - G_i)^2.

Without the prior the loop cannot start: with ``V = 0`` the planner docked 4
times in 30 even from 2 to 10 m, since no sampled sequence reaches the port
slowly enough to be rewarded. With it, it docks 30 in 30 there; the starts
then move out, as in a reverse curriculum, and the network carries what the
search found near the port further out (`scripts/value_loop.py`).

The network is trained with PyTorch and evaluated inside the planner with
numpy, on hundreds of states at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .env import RendezvousEnv

SCALE = 100.0   # returns are of order the +-100 of the outcomes


def observation(env: RendezvousEnv, states: np.ndarray, steps: int | np.ndarray) -> np.ndarray:
    """The agent's observation of a batch of physical states, at a given step count."""
    states = np.atleast_2d(states)
    elapsed = np.broadcast_to(np.asarray(steps, dtype=float) / env.config.max_episode_steps,
                              (len(states),))
    return np.column_stack([states / env._obs_scale, elapsed])


def prior(env: RendezvousEnv, states: np.ndarray) -> np.ndarray:
    """The straight distance to the port, as a value: ``-w r / r_max``."""
    states = np.atleast_2d(states)
    return (-env.reward_config.distance_weight * np.hypot(states[:, 0], states[:, 1])
            / env.config.max_distance)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = r_t + gamma G_{t+1}``, with nothing after the last reward."""
    returns = np.zeros(len(rewards))
    following = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        following = rewards[t] + gamma * following
        returns[t] = following
    return returns


@dataclass
class ValueNet:
    """``dV_theta``: an MLP from the observation to a correction of the prior."""

    hidden: tuple[int, ...] = (64, 64)
    inputs: int = 5
    seed: int = 0
    model: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        width = self.inputs
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        layers.append(nn.Linear(width, 1))
        self.model = nn.Sequential(*layers)
        # Start at zero: before any data the value is the prior alone.
        nn.init.zeros_(self.model[-1].weight)
        nn.init.zeros_(self.model[-1].bias)
        self._cache()

    def _cache(self) -> None:
        """The weights as numpy arrays, for fast evaluation inside the planner."""
        self.weights = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                        for m in self.model if isinstance(m, nn.Linear)]

    def correction(self, obs: np.ndarray) -> np.ndarray:
        """``SCALE * dV_theta(o)`` for a batch of observations, in numpy."""
        h = obs
        # Accelerate's matmul on macOS raises spurious floating-point flags on
        # small, finite products; the results are exact.
        with np.errstate(all="ignore"):
            for i, (w, b) in enumerate(self.weights):
                h = h @ w + b
                if i < len(self.weights) - 1:
                    h = np.tanh(h)
        return SCALE * h[:, 0]

    def fit(self, obs: np.ndarray, targets: np.ndarray, steps: int = 2000,
            batch: int = 512, lr: float = 1e-3, seed: int = 0) -> float:
        """Fit ``SCALE * dV`` to ``targets`` (returns minus prior); the final mean squared error."""
        rng = np.random.default_rng(seed)
        x = torch.as_tensor(obs, dtype=torch.float32)
        y = torch.as_tensor(targets / SCALE, dtype=torch.float32)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(steps):
            idx = torch.as_tensor(rng.integers(0, len(x), size=min(batch, len(x))))
            loss = ((self.model(x[idx])[:, 0] - y[idx]) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        self._cache()
        with torch.no_grad():
            return float(((self.model(x)[:, 0] - y) ** 2).mean()) * SCALE**2

    def state_dict(self) -> dict:
        return {k: v.numpy() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        self._cache()

    def save(self, path) -> None:
        np.savez(path, **self.state_dict())

    @classmethod
    def load(cls, path) -> ValueNet:
        net = cls()
        with np.load(path) as data:
            net.load_state_dict({k: data[k] for k in data.files})
        return net


class LearnedValue:
    """``V(s) = prior(s) + w(r) C(s)``, callable on a batch of states.

    ``C`` is the learned correction: with several networks fitted on different
    resamplings of the data, their mean less ``beta`` times their spread,
    ``C = mean - beta std``. Where the data are, the networks agree; beyond
    them each invents its own value and the spread lowers the correction, so
    the search is not drawn to places nobody has flown. With a single network
    and no spread the planner flew away from the port, 31 escapes in 64.
    With ``penalty_only`` the correction can only lower the prior.

    The correction counts only where the search sees the port anyway: ``w`` is
    0 within ``near`` metres, 1 beyond ``far``, linear between. Used everywhere,
    the learned value rated the states just short of the port above the +100
    of docking itself (+112 at 2 m on the axis), and the planner, which rarely
    samples the exact slow arrival a docking needs, hovered there until the
    timeout: 92 % of dockings from 2-10 m became 6-15 %. A ``near`` of 25 m was
    too wide: the chaser then stopped against the side of the keep-out sphere,
    22 m out and 72 degrees off the axis, where the prior pushes it into the
    sphere and only the learned value can say that the mouth of the cone is
    worth sliding round to. With ``cone_zone`` the correction is off only
    within the cone, near the port, where the port is in sight: there it
    stays off, and the side of the sphere gets the learned value.
    """

    def __init__(self, env: RendezvousEnv, nets: ValueNet | list[ValueNet],
                 near: float = 25.0, far: float = 40.0, beta: float = 1.0,
                 penalty_only: bool = True, cone_zone: bool = True, ramp_deg: float = 15.0):
        self.env = env
        self.nets = nets if isinstance(nets, list) else [nets]
        self.near, self.far, self.beta, self.penalty_only = near, far, beta, penalty_only
        self.cone_zone, self.ramp_deg = cone_zone, ramp_deg

    def weight(self, states: np.ndarray) -> np.ndarray:
        """``w``: 0 where the search sees the port, 1 where it needs the learned value.

        By distance alone, or, with ``cone_zone``, only within the approach cone
        (widened by ``ramp_deg``): the port is in sight down the cone, not from
        the side of the keep-out sphere at the same distance.
        """
        r = np.hypot(states[:, 0], states[:, 1])
        radial = np.clip((r - self.near) / max(self.far - self.near, 1e-9), 0.0, 1.0)
        if not self.cone_zone:
            return radial
        angle = np.degrees(np.arctan2(np.abs(states[:, 0]), states[:, 1]))
        cone = self.env.config.approach_cone_deg
        sideways = np.clip((angle - cone) / self.ramp_deg, 0.0, 1.0)
        return 1.0 - (1.0 - radial) * (1.0 - sideways)

    def correction(self, states: np.ndarray, steps: int) -> np.ndarray:
        obs = observation(self.env, states, steps)
        values = np.array([net.correction(obs) for net in self.nets])
        spread = values.std(axis=0) if len(self.nets) > 1 else 0.0
        correction = values.mean(axis=0) - self.beta * spread
        return np.minimum(correction, 0.0) if self.penalty_only else correction

    def __call__(self, states: np.ndarray, steps: int) -> np.ndarray:
        states = np.atleast_2d(states)
        weight = self.weight(states)
        correction = np.zeros(len(states))
        far = weight > 0.0
        if far.any():
            correction[far] = self.correction(states[far], steps)
        return prior(self.env, states) + weight * correction

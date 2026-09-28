"""Reinforcement learning for a planar orbital rendezvous.

The chaser spacecraft is described in the LVLH frame attached to a target on a
circular orbit, where the relative motion obeys the Clohessy-Wiltshire
equations. See `dynamics` for the physics and `env` for the Gymnasium wrapper.
"""

from .env import EnvConfig, RendezvousEnv
from .rewards import Outcome, RewardConfig

__all__ = ["EnvConfig", "Outcome", "RendezvousEnv", "RewardConfig"]
__version__ = "0.1.0"

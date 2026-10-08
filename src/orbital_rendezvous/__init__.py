"""Reinforcement learning for a planar orbital rendezvous.

The chaser spacecraft is described in the LVLH frame attached to a target on a
circular orbit, where the relative motion obeys the Clohessy-Wiltshire
equations. The library is split by function: `core` (dynamics, environment,
reward, baselines, evaluation), `rl` (model-free training), `planning` (the
sampling MPC, learned values, Go-Explore) and `viz` (drawing). The studies that
use it live in ``studies/``.
"""

from orbital_rendezvous.core.env import EnvConfig, RendezvousEnv
from orbital_rendezvous.core.rewards import Outcome, RewardConfig

__all__ = ["EnvConfig", "Outcome", "RendezvousEnv", "RewardConfig"]
__version__ = "0.1.0"

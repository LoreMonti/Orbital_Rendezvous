"""Building the study's environments from a configuration.

The package's `orbital_rendezvous.core.utils.make_env` builds the rendezvous
environment; this one also builds `goto.GoToEnv` for a configuration with a
``goal`` section, and `hierarchy.PlannerEnv` for one with a ``planner``
section. Training goes through `orbital_rendezvous.rl.training.train` with this
function as its environment factory.
"""

from __future__ import annotations

from typing import Any

from orbital_rendezvous.core.env import RendezvousEnv
from orbital_rendezvous.core.utils import _build, build_configs, load_config

from .goto import GoToConfig, GoToEnv, waypoint_menu


def build_goal_config(config: dict[str, Any]) -> GoToConfig | None:
    """The ``goal`` section, present only for the pilot that flies to a point."""
    if "goal" not in config:
        return None
    values = dict(config["goal"])
    if "goals" in values:
        values["goals"] = tuple(tuple(float(c) for c in g) for g in values["goals"])
    if "menu_radii" in values:
        values["menu_radii"] = tuple(float(r) for r in values["menu_radii"])
    return _build(GoToConfig, values, "goal")


def load_pilot(planner: dict[str, Any]):
    """The two frozen pilots of a ``planner`` section, as a `hierarchy.HierarchicalPilot`."""
    from stable_baselines3 import PPO

    from .hierarchy import HierarchicalPilot

    configs = [build_configs(load_config(planner[key]))[0]
               for key in ("go_to_config", "final_config")]
    return HierarchicalPilot.from_models(
        PPO.load(planner["go_to"], device="cpu"), PPO.load(planner["final"], device="cpu"),
        *configs,
    )


def make_env(config: dict[str, Any]):
    """The environment a configuration describes.

    `GoToEnv` with a ``goal`` section; `hierarchy.PlannerEnv`, choosing the
    waypoint that two frozen pilots then fly, with a ``planner`` section; the
    plain `RendezvousEnv` otherwise.
    """
    env_config, reward_config = build_configs(config)
    if "planner" in config:
        from .hierarchy import PlannerEnv

        planner = config["planner"]
        pilot = load_pilot(planner)
        pilot.keep_out = env_config.keep_out_radius
        menu = waypoint_menu(tuple(planner["menu_radii"]), planner["menu_directions"])
        return PlannerEnv(RendezvousEnv(env_config, reward_config), pilot, menu,
                          planner["fuel_weight"], planner.get("time_weight", 0.0),
                          planner.get("relative", False))
    goal_config = build_goal_config(config)
    if goal_config is not None:
        return GoToEnv(env_config, reward_config, goal_config)
    return RendezvousEnv(env_config, reward_config)

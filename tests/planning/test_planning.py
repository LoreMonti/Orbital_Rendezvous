"""Tests for the sampling planner.

A planner is only as good as its model of the task: if its score disagreed
with what the environment pays, it would optimise the wrong thing and still
look busy. So the score is checked against the environment step by step, with
every outcome the rule can end in.
"""

from dataclasses import replace

import numpy as np
import pytest

from orbital_rendezvous import EnvConfig, Outcome, RendezvousEnv
from orbital_rendezvous.core.evaluation import HELD_OUT_SEED, rollout
from orbital_rendezvous.planning.mpc import PlannerConfig, SamplingPlanner

STRICT = EnvConfig(keep_out_radius=20.0, approach_cone_deg=15.0)


def environment_return(env, state, actions):
    """What the environment pays for ``actions`` from ``state``, without the shaping."""
    env.reset(seed=0)
    env.state = np.asarray(state, dtype=float).copy()
    total, discount = 0.0, 1.0
    for action in actions:
        _, _, done, _, info = env.step(action)
        terms = info["reward_terms"]
        total += discount * (terms["fuel"] + terms["terminal"])
        discount *= env.reward_config.gamma
        if done:
            return total, info["outcome"]
    return total, None


@pytest.mark.parametrize("state", [
    [60.0, 60.0, -0.1, -0.1],     # heads into the sphere from the side: a violation
    [0.0, 30.0, 0.0, -0.08],      # down the corridor
    [0.0, 3.0, 0.0, -0.3],        # into the port too fast: a crash
    [100.0, 400.0, 0.5, 0.5],     # away: an escape
])
def test_the_score_is_what_the_environment_pays(state):
    env = RendezvousEnv(STRICT)
    planner = SamplingPlanner(env, PlannerConfig(horizon=40, value="none"))
    actions = np.random.default_rng(1).uniform(-0.3, 0.3, size=(5, 40, 2))
    scores = planner.score(np.asarray(state), actions, steps_done=0)
    for score, sequence in zip(scores, actions, strict=True):
        expected, _ = environment_return(RendezvousEnv(STRICT), state, sequence)
        assert score == pytest.approx(expected, abs=1e-9)


def test_the_value_counts_only_beyond_the_horizon_and_only_if_still_flying():
    env = RendezvousEnv(STRICT)
    far = np.array([[0.0, 100.0, 0.0, 0.0]])
    none = SamplingPlanner(env, PlannerConfig(horizon=1, value="none"))
    distance = SamplingPlanner(env, PlannerConfig(horizon=1, value="distance"))
    hold = np.zeros((1, 1, 2))
    gap = distance.score(far[0], hold, 0) - none.score(far[0], hold, 0)
    moved = np.hypot(*(far[0] @ env.phi.T)[:2])
    # gamma^H V(s_H), with H = 1.
    rc = env.reward_config
    assert gap[0] == pytest.approx(rc.gamma * -rc.distance_weight * moved / 500.0)
    # On the last step of the episode there is no future to value.
    assert distance.score(far[0], hold, STRICT.max_episode_steps - 1) == pytest.approx(
        none.score(far[0], hold, STRICT.max_episode_steps - 1))


def test_without_the_port_the_planner_docks():
    env = RendezvousEnv(EnvConfig())
    planner = SamplingPlanner(env, PlannerConfig(init_std=0.2), seed=0)
    with np.errstate(all="ignore"):
        run = rollout(env, planner, HELD_OUT_SEED)
    assert run.outcome is Outcome.DOCKED
    assert np.all(np.abs(run.thrusts) <= EnvConfig().max_thrust + 1e-12)


def test_the_planner_refuses_a_mirrored_environment_and_unknown_values():
    with pytest.raises(ValueError):
        SamplingPlanner(RendezvousEnv(replace(STRICT, mirror_symmetry=True)))
    with pytest.raises(ValueError):
        SamplingPlanner(RendezvousEnv(STRICT), PlannerConfig(value="oracle"))

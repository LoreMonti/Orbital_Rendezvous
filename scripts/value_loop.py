"""Learn the planner's value from its own flights, from the port outwards (Step 26).

Each iteration the sampling planner, with the current value
``V = prior + dV_theta``, flies a batch of episodes from the starts of the
current stage, in parallel; every visited state is labelled with the return it
actually got, ``G_t = r_t + gamma G_{t+1}``, from the environment's own fuel
and outcomes; the network is refitted on all the states collected so far; and
once the planner docks ``--threshold`` of the episodes from the outer band of
the stage, the starts move out to the next one.

The stages grow by less than the planner sees ahead, about 200 s, or 20-40 m
at the speeds of an approach, so that every new start can reach, within its
horizon, states where the value has already been learned: the value is built
from the port outwards, layer on layer, as in dynamic programming. No stage
says how to go around the station; the search finds it, the network keeps it.

Resumable: the run directory holds the network, the collected states and the
stage after every iteration, and a new call with the same ``--run`` continues.

Usage:
    python scripts/value_loop.py --run runs/value_loop/seed0 --seed 0
    python scripts/value_loop.py --run runs/value_loop/seed1 --seed 1 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.planning import PlannerConfig, SamplingPlanner
from orbital_rendezvous.rewards import Outcome
from orbital_rendezvous.utils import build_configs, load_config
from orbital_rendezvous.value import (
    LearnedValue,
    ValueNet,
    discounted_returns,
    observation,
    prior,
)

# (largest distance [m], largest angle from the docking axis [deg]); the
# smallest distance stays at 2 m. Inside the keep-out sphere starts lie in the cone.
STAGES = [(10, 10), (20, 10), (40, 15), (40, 30), (40, 45), (60, 65), (80, 90),
          (100, 110), (120, 130), (150, 150), (200, 165), (200, 180)]
MIN_RADIUS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_corridor.yaml")
    parser.add_argument("--run", default="runs/value_loop/seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=64, help="per iteration")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=60, help="largest number in all")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--band", type=float, default=0.3)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--init-std", type=float, default=0.3)
    parser.add_argument("--fit-steps", type=int, default=2000)
    parser.add_argument("--replay", type=int, default=400_000, help="states kept, newest")
    parser.add_argument("--near", type=float, default=25.0,
                        help="within this distance [m] the learned correction is off")
    parser.add_argument("--far", type=float, default=40.0,
                        help="beyond this distance [m] it is fully on")
    parser.add_argument("--ensemble", type=int, default=1, help="networks, on resampled data")
    parser.add_argument("--beta", type=float, default=1.0, help="weight of their spread")
    parser.add_argument("--both-signs", action="store_true",
                        help="let the correction raise the prior too")
    parser.add_argument("--radial-zone", action="store_true",
                        help="switch the correction off by distance alone, not within the cone")
    return parser.parse_args()


def make_env(config_path: str) -> RendezvousEnv:
    env_config, reward_config = build_configs(load_config(config_path))
    return RendezvousEnv(env_config, reward_config)


def fly(job):
    """One episode from the stage's starts: observations, rewards, outcome, start."""
    config_path, planner, weights, seed, (radius, angle), value_settings = job
    env = make_env(config_path)
    env.set_start_region(MIN_RADIUS, float(radius), float(angle))
    nets = []
    for state in weights:
        net = ValueNet()
        net.load_state_dict(state)
        nets.append(net)
    value = LearnedValue(env, nets, **value_settings)
    controller = SamplingPlanner(env, planner, seed=seed, value_fn=value)
    env.reset(seed=seed)
    start = env.state.copy()
    # rollout() resets the environment itself; fly by hand to record each state.
    obs_list, rewards, done = [], [], False
    with np.errstate(all="ignore"):   # spurious matmul warnings of Accelerate on macOS
        obs = env._observation()
        while not done:
            obs_list.append(observation(env, env.state, env.steps)[0])
            obs, _, done, _, info = env.step(controller(env, obs))
            terms = info["reward_terms"]
            rewards.append(terms["fuel"] + terms["terminal"])
    return np.array(obs_list), np.array(rewards), info["outcome"].value, start


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    env = make_env(args.config)
    planner = PlannerConfig(value="learned", horizon=args.horizon, block=args.block,
                            init_std=args.init_std)
    nets = [ValueNet(seed=args.seed * 100 + k) for k in range(args.ensemble)]
    value_settings = {"near": args.near, "far": args.far, "beta": args.beta,
                      "penalty_only": not args.both_signs, "cone_zone": not args.radial_zone}
    state = {"iteration": 0, "stage": 0}
    obs_bank = np.zeros((0, 5))
    target_bank = np.zeros(0)
    if (run / "state.json").exists():
        state = json.loads((run / "state.json").read_text())
        nets = [ValueNet.load(run / f"value_{k}.npz") for k in range(args.ensemble)]
        with np.load(run / "replay.npz") as data:
            obs_bank, target_bank = data["obs"], data["targets"]
        print(f"Resumed at iteration {state['iteration']}, stage {state['stage']}")
    log_path = run / "progress.csv"
    new_log = not log_path.exists()
    began = time.time()

    while state["iteration"] < args.iterations and state["stage"] < len(STAGES):
        radius, angle = STAGES[state["stage"]]
        first = args.seed * 1_000_000 + state["iteration"] * args.episodes
        weights = [net.state_dict() for net in nets]
        jobs = [(args.config, planner, weights, first + i, (radius, angle), value_settings)
                for i in range(args.episodes)]
        with ProcessPoolExecutor(args.workers) as pool:
            flights = list(pool.map(fly, jobs))

        gamma = env.reward_config.gamma
        for obs, rewards, _, _ in flights:
            states = obs[:, :4] * env._obs_scale
            targets = discounted_returns(rewards, gamma) - prior(env, states)
            obs_bank = np.vstack([obs_bank, obs])[-args.replay:]
            target_bank = np.concatenate([target_bank, targets])[-args.replay:]
        # Each network on its own resampling of the data, so that they agree
        # where the data are and disagree beyond them.
        rng = np.random.default_rng(state["iteration"])
        errors = []
        for k, net in enumerate(nets):
            idx = (rng.integers(0, len(target_bank), len(target_bank)) if len(nets) > 1
                   else np.arange(len(target_bank)))
            errors.append(net.fit(obs_bank[idx], target_bank[idx], steps=args.fit_steps,
                                  seed=state["iteration"] * 100 + k))
        mse = float(np.mean(errors))

        docked = np.array([f[2] == Outcome.DOCKED.value for f in flights])
        starts = np.array([f[3] for f in flights])
        distances = np.hypot(starts[:, 0], starts[:, 1])
        # Mastery is tested on the outer band of distances, as in Step 21.
        test = distances >= (1.0 - args.band) * radius
        success = float(docked[test].mean()) if test.any() else 0.0
        outcomes = {o: sum(f[2] == o for f in flights) for o in {f[2] for f in flights}}
        row = {"iteration": state["iteration"], "stage": state["stage"], "radius": radius,
               "angle": angle, "docked": float(docked.mean()), "test_docked": success,
               "test_episodes": int(test.sum()), "outcomes": json.dumps(outcomes),
               "mse": round(mse, 2), "states": len(target_bank),
               "minutes": round((time.time() - began) / 60, 1)}
        print(row, flush=True)
        with open(log_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new_log:
                writer.writeheader()
                new_log = False
            writer.writerow(row)

        state["iteration"] += 1
        if success >= args.threshold and test.sum() >= 10:
            state["stage"] += 1
        for k, net in enumerate(nets):
            net.save(run / f"value_{k}.npz")
        np.savez(run / "replay.npz", obs=obs_bank, targets=target_bank)
        (run / "state.json").write_text(json.dumps(state))

    print("Finished:", state, "stages", len(STAGES))


if __name__ == "__main__":
    main()

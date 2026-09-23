"""Train a PPO agent on the rendezvous task.

Reads every parameter from the YAML configuration, trains on parallel
environments, and writes:

- the final policy to ``--output`` (default ``models/ppo_rendezvous.zip``);
- a run directory ``runs/<timestamp>/`` with periodic checkpoints, the
  Stable-Baselines3 log as CSV (losses included), one row per episode from the
  Monitor wrapper, a copy of the configuration, and the final training window;
- with ``--record``, a GIF of the replays at the milestones set in
  ``live_view.record_at``, plus the last one: the agent learning, in a few seconds.

Usage:
    python scripts/train.py
    python scripts/train.py --no-render
    python scripts/train.py --timesteps 200000 --seed 1
    python scripts/train.py --record assets/training.gif
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from orbital_rendezvous import RendezvousEnv
from orbital_rendezvous.utils import build_configs, load_config

PPO_KEYS = (
    "learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda", "clip_range", "ent_coef",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--output", default="models/ppo_rendezvous.zip")
    parser.add_argument("--runs", default="runs", help="Parent directory of the run directories.")
    parser.add_argument(
        "--no-render", action="store_true", help="Train without the live window, at full speed."
    )
    parser.add_argument(
        "--record", default=None,
        help="Save a GIF of the milestone replays here. Works with --no-render too.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Overrides training.seed.")
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Overrides training.total_timesteps."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_config, reward_config = build_configs(config)
    training = config["training"]
    if training.get("algorithm", "PPO") != "PPO":
        raise ValueError("only PPO is supported")

    seed = training["seed"] if args.seed is None else args.seed
    total_timesteps = int(args.timesteps or training["total_timesteps"])
    render = config["live_view"]["enabled"] and not args.no_render
    if args.record and not render:
        # Recording needs the window drawn, but not shown.
        import matplotlib

        matplotlib.use("Agg")

    run_dir = Path(args.runs) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    set_random_seed(seed)
    vec_env = make_vec_env(
        lambda: RendezvousEnv(env_config, reward_config),
        n_envs=training["n_envs"],
        seed=seed,
        monitor_dir=str(run_dir / "monitor"),
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    model = PPO(
        training["policy"],
        vec_env,
        policy_kwargs=training.get("policy_kwargs"),
        seed=seed,
        verbose=0,
        **{key: training[key] for key in PPO_KEYS},
    )
    model.set_logger(configure(str(run_dir), ["csv"]))

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, 200_000 // training["n_envs"]),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        )
    ]
    view = live = None
    if render or args.record:
        from orbital_rendezvous.callbacks import LiveViewCallback
        from orbital_rendezvous.live_view import LiveView

        view = LiveView.from_env(
            RendezvousEnv(env_config, reward_config),
            trail_length=config["live_view"]["trail_length"],
        )
        live = LiveViewCallback(
            view,
            config["live_view"]["episode_stride"],
            record_at=config["live_view"]["record_at"] if args.record else (),
        )
        callbacks.append(live)

    print(f"Training PPO for {total_timesteps:,} steps on {training['n_envs']} environments")
    print(f"Run directory: {run_dir}")
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    elapsed = time.perf_counter() - start

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)

    successes = list(model.ep_success_buffer)
    rate = float(np.mean(successes)) if successes else float("nan")
    print(f"Done in {elapsed / 60:.1f} min. Model saved to {output}")
    print(f"Docking rate over the last {len(successes)} episodes: {100 * rate:.0f} %")

    if args.record:
        from orbital_rendezvous.live_view import save_gif

        clips = live.recorded_clips()
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_gif(clips, args.record)
        size = Path(args.record).stat().st_size / 1e6
        print(f"Training GIF of {len(clips)} replays saved to {args.record} ({size:.1f} MB)")

    if view is not None:
        view.save(str(run_dir / "training_window.png"))
        if render:
            print("Close the window to exit.")
            view.plt.show(block=True)


if __name__ == "__main__":
    main()

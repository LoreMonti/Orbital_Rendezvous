# Orbital Rendezvous with Reinforcement Learning

A PPO agent learns to fly a chaser spacecraft onto a target in low Earth orbit,
by trial and error, in the linearised relative dynamics of the Clohessy-Wiltshire
equations. A live window shows the trajectory and the training progress side by
side, and the learned policy is measured against a classical LQR controller on
fuel and on success rate.

> Status: early work in progress. The repository layout is in place; the
> dynamics, the environment and the training loop are being implemented.

## The physics

The target flies a circular orbit of semi-major axis $a$, with mean motion
$n = \sqrt{\mu / a^3}$. In the LVLH frame centred on the target, with $x$ radial
and $y$ along-track, the chaser of mass $m$ under a thrust $\mathbf{u}$ obeys the
planar Clohessy-Wiltshire equations

$$\ddot{x} - 3n^2 x - 2n\dot{y} = \frac{u_x}{m}, \qquad \ddot{y} + 2n\dot{x} = \frac{u_y}{m}.$$

The out-of-plane motion, $\ddot{z} + n^2 z = u_z/m$, is a harmonic oscillator
decoupled from the other two axes: it is left out, since it would double the
training cost without adding any coupling to learn.

These equations are linear, so the propagation over one time step is exact and
the implementation can be tested against the closed-form solution.

## The learning problem

| | |
|---|---|
| Observation | the relative state $[x, y, \dot{x}, \dot{y}]$, normalised |
| Action | continuous thrust $[u_x, u_y]$, saturated at $u_{\max}$ |
| Reward | closing on the target, minus fuel spent, plus a docking bonus and a failure penalty |
| Success | inside the docking radius and below the docking speed |
| Algorithm | PPO (Stable-Baselines3) |

## The live training window

```
┌─────────────────────────┬──────────────────────────┐
│  LVLH plane             │  mean episode reward     │
│   target, chaser, trail │  success rate            │
│   thrust vector         │  delta-v per episode     │
│                         │  PPO losses (secondary)  │
└─────────────────────────┴──────────────────────────┘
```

The trajectory is redrawn once every `episode_stride` episodes, so the training
runs at full speed; the curves are refreshed after every policy update.

A note on reading those curves: in reinforcement learning the losses do **not**
decrease the way they do in supervised learning, because the policy changes the
data it collects. The PPO policy loss oscillates around zero, and the value loss
often grows once the agent starts reaching the docking bonus. The curves that
show real learning are the mean episode reward and the success rate, which is
why they are the prominent ones.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,play]"
```

## Usage

```bash
# Train, with the live window
python scripts/train.py --config configs/ppo_default.yaml

# Train at full speed, no window
python scripts/train.py --config configs/ppo_default.yaml --no-render

# Evaluate the policy and the LQR baseline
python scripts/evaluate.py --model models/ppo_rendezvous.zip --baseline

# Watch the agent, or fly the chaser yourself
python scripts/play.py --model models/ppo_rendezvous.zip
python scripts/play.py --human
```

## Roadmap

The project is built one step at a time; see [ROADMAP.md](ROADMAP.md) for the
design decisions and the physics behind each step.

## Repository layout

```
Orbital_Rendezvous/
├── pyproject.toml          # metadata, dependencies, ruff and pytest config
├── README.md               # physics, agent GIF, results, how to reproduce them
├── ROADMAP.md              # the steps, and the reasoning behind each one
├── LICENSE                 # MIT
├── .gitignore
├── configs/
│   └── ppo_default.yaml    # environment parameters and PPO hyperparameters
├── src/
│   └── orbital_rendezvous/
│       ├── __init__.py
│       ├── dynamics.py     # Clohessy-Wiltshire propagation (pure physics, no RL)
│       ├── env.py          # RendezvousEnv (Gymnasium API)
│       ├── rewards.py      # reward function, kept apart so it can be tuned alone
│       ├── baselines.py    # LQR controller, the honest yardstick for the agent
│       ├── live_view.py    # the training window: trajectory + progress curves
│       ├── callbacks.py    # SB3 callback feeding the window during training
│       └── utils.py        # config loading, seeding, delta-v accounting
├── scripts/
│   ├── train.py            # train PPO and save the model
│   ├── evaluate.py         # metrics: success rate, total delta-v, time to dock
│   └── play.py             # game mode: watch the agent, or fly it yourself
├── notebooks/              # exploration and figures only, no logic
├── tests/
│   ├── test_dynamics.py    # analytical solution, closed orbits, limit cases
│   ├── test_env.py         # Gymnasium check_env, spaces, reset/step
│   └── test_rewards.py     # term signs, breakdown consistency
├── models/                 # checkpoints (git-ignored, except the final one)
└── assets/                 # GIFs and plots for the README
```

Two departures from the plan sketched at the start: the renderer is split into
`live_view.py` (the figure) and `callbacks.py` (the Stable-Baselines3 hook that
drives it during training), because drawing and training have to be decoupled;
and `ROADMAP.md` records the design decisions so each step stays reviewable.

## Tests

```bash
pytest
ruff check .
```

## License

MIT. Author: Lorenzo Monti.

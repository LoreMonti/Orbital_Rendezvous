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
configs/    training and environment parameters
src/        the package: dynamics, environment, rewards, baselines, live view
scripts/    train, evaluate, play
tests/      unit tests, including the dynamics against the analytical solution
notebooks/  exploration and figures only, no logic
```

## Tests

```bash
pytest
ruff check .
```

## License

MIT. Author: Lorenzo Monti.

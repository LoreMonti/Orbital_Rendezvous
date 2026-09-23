# Orbital Rendezvous

A **PPO agent** learning to fly a chaser spacecraft onto a target in low Earth
orbit, by trial and error, in the linearised relative dynamics of the
**Clohessy-Wiltshire equations** — the same problem a Dragon capsule solves on
its final approach to the ISS, reduced to its essential geometry.

The target flies a circular orbit of semi-major axis $a$, with mean motion
$n = \sqrt{\mu/a^3}$. In the LVLH frame centred on the target, with $x$ radial
and $y$ along-track, a chaser of mass $m$ under a thrust $\mathbf{u}$ obeys

```math
\ddot{x} - 3n^2 x - 2n\dot{y} = \frac{u_x}{m}, \qquad \ddot{y} + 2n\dot{x} = \frac{u_y}{m}
```

The coupling terms are what make a rendezvous counter-intuitive. Thrusting
straight at a target ahead of you raises your orbit, slows you down, and leaves
you further behind: the secular drift in the closed-form solution is
$-6 n x_0 t$, proportional to the *radial* offset. The agent has to discover
that it must go down to catch up.

The out-of-plane motion, $\ddot{z} + n^2 z = u_z/m$, is a harmonic oscillator
fully decoupled from the two in-plane axes. It is left out: it would double the
training cost without adding any coupling to learn.

> **Status: early work in progress.** The Clohessy-Wiltshire dynamics, the
> Gymnasium environment, the reward and the live training window are
> implemented and tested; the training script comes next, one reviewable step
> at a time. See [ROADMAP.md](ROADMAP.md).

## Install

```bash
git clone https://github.com/LoreMonti/Orbital_Rendezvous.git
cd Orbital_Rendezvous
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,play]"
```

That one command reads `pyproject.toml` and pulls in `numpy`, `scipy`,
`matplotlib`, `gymnasium`, `stable-baselines3` and `torch`, plus `pytest` and
`ruff` from the `dev` extra and `pygame` from `play`. Python 3.10 or newer.
scipy is pinned below 1.15, whose macOS arm64 wheels fail to load on macOS 27.
The environment used here is Python 3.10 with numpy 2.2.6, scipy 1.14.1,
gymnasium 1.3.0, stable-baselines3 2.9.0 and torch 2.14.0.

## Usage

Three drivers cover the whole workflow. Each takes its parameters from
`configs/ppo_default.yaml`, needs no arguments for the default run, and lists
its options with `--help`.

| script | what it does |
| --- | --- |
| `train.py` | trains PPO and saves the policy to `models/`, with the live window open |
| `evaluate.py` | success rate, total $\Delta v$ and time to dock over a fixed set of seeded initial conditions, against the LQR baseline |
| `play.py` | watch the trained agent, or fly the chaser yourself and compare |

```bash
python scripts/train.py --config configs/ppo_default.yaml
python scripts/train.py --no-render          # full speed, no window
python scripts/evaluate.py --model models/ppo_rendezvous.zip --baseline
python scripts/play.py --human
```

As a library:

```python
from orbital_rendezvous import RendezvousEnv, EnvConfig

env = RendezvousEnv(EnvConfig(max_thrust=1.0, docking_radius=1.0))
obs, info = env.reset(seed=0)
```

## Layout

```
Orbital_Rendezvous/
├── README.md
├── ROADMAP.md
├── LICENSE
├── pyproject.toml          # metadata, dependencies, ruff and pytest config
├── .gitignore
├── configs/
│   └── ppo_default.yaml    # environment parameters and PPO hyperparameters
├── src/orbital_rendezvous/
│   ├── dynamics.py         # Clohessy-Wiltshire propagation: pure physics, no RL
│   ├── env.py              # RendezvousEnv, the Gymnasium API
│   ├── rewards.py          # reward function, kept apart so it can be tuned alone
│   ├── baselines.py        # LQR controller: the honest yardstick for the agent
│   ├── live_view.py        # the training window: trajectory and progress curves
│   ├── callbacks.py        # SB3 callback feeding that window during training
│   └── utils.py            # config loading, seeding, delta-v accounting
├── scripts/
│   ├── train.py            # trains PPO and saves the model
│   ├── evaluate.py         # metrics: success rate, total delta-v, time to dock
│   └── play.py             # game mode: watch the agent, or fly it yourself
├── tests/
│   ├── test_dynamics.py    # analytical solution, closed orbits, limit cases
│   ├── test_env.py         # Gymnasium check_env, spaces, reset and step
│   ├── test_rewards.py     # term signs, consistency of the breakdown
│   ├── test_live_view.py   # callback bookkeeping, headless drawing, short PPO run
│   └── conftest.py         # draws off-screen, so tests never open a window
├── notebooks/              # exploration and figures only, no logic
├── models/                 # checkpoints, git-ignored except the final one
└── assets/                 # GIFs and plots used by this README
```

The physics lives in `dynamics.py` and knows nothing about agents, so it can be
tested against the closed-form solution on its own; the renderer is split into
the figure (`live_view.py`) and the training hook that drives it
(`callbacks.py`), which is what lets the window draw one rollout every $N$
episodes while training keeps running at full speed.

## Tests

```bash
pytest
ruff check .
```

Errors in orbital mechanics rarely crash: a sign flip or a missing factor of
$n$ produces plausible-looking trajectories. The suite pins the invariants that
would catch that — the state-transition matrix from `expm` against the
analytical one, a chaser at rest staying at rest, the secular drift matching
$-6 n x_0 t$, the initial condition $\dot{y}_0 = -2 n x_0$ closing into a
periodic $2\!:\!1$ ellipse, two steps of $\Delta t$ agreeing with one of
$2\Delta t$ under a constant thrust, and, for $n \to 0$, the input matrix
reducing to a double integrator plus the first-order Coriolis coupling
$\pm n\,\Delta t^3/3m$, which pins both the $1/m$ and the sign of the coupling.

The environment is tested by placing the chaser by hand in a state that must
lead to a given outcome in one step, so a mislabelled ending or a wrong
`terminated`/`truncated` flag shows up directly. That includes the tunnelling
case, a chaser fast enough to cross the docking sphere between two steps with
neither endpoint inside it, and `check_env` with warnings treated as errors.

For the reward, besides the sign of every term, the suite checks the property
that makes the shaping safe: over a random trajectory the discounted sum of the
shaping terms equals $\gamma^K\Phi(\mathbf{s}_K) - \Phi(\mathbf{s}_0)$ to
$10^{-12}$, so a closed loop earns nothing and the shaping cannot be farmed.

The training window is tested off-screen: its callback is fed episodes whose
outcome, return and $\Delta v$ are known in advance, so the bookkeeping behind
every curve is checked exactly, and a short real PPO run checks that the pieces
fit together inside Stable-Baselines3.

## The learning problem

| | |
| --- | --- |
| observation | the relative state $[x, y, \dot{x}, \dot{y}]$, normalised |
| action | continuous thrust $[u_x, u_y]$, saturated at $u_\mathrm{max}$ |
| reward | potential-based shaping towards the target under a glide slope, minus the $\Delta v$ spent, plus $\pm 100$ at the end |
| success | inside the docking radius **and** below the docking speed |
| algorithm | PPO, on vectorised environments |

The thrust is continuous rather than a set of discrete on/off burns: it is
closer to a throttleable engine, and easier to learn, because the agent can
correct gently instead of choosing between a few abrupt impulses.

Because the dynamics are linear, the propagation over one step is exact. Rather
than deriving the zero-order-hold input matrix by hand, both blocks come from a
single matrix exponential (Van Loan, 1978):

```math
\exp\left( \begin{pmatrix} A & B \\ 0 & 0 \end{pmatrix} \Delta t \right) = \begin{pmatrix} \Phi & \Gamma \\ 0 & I \end{pmatrix}
```

evaluated once when the environment is built, so it costs nothing during
training. The $\Phi$ it returns must agree with the analytical matrix to machine
precision, which is the strongest available check on the implementation.

The reward is shaped with a potential (Ng, Harada & Russell, 1999),

```math
F = \gamma\,\Phi(\mathbf{s}') - \Phi(\mathbf{s}), \qquad
\Phi(\mathbf{s}) = -\,w_r\,\frac{r}{r_\mathrm{max}} - w_v\,\frac{\max\left(0,\ |\mathbf{v}| - v_\mathrm{dock} - r/\tau\right)}{v_\mathrm{ref}}
```

which pulls the chaser in under a speed limit that tightens to the docking
speed at the target. Because $F$ telescopes, it speeds learning up without
changing which policy is optimal. Fuel, instead, is a real cost:
$-w_f\,|\mathbf{u}|\,\Delta t/m$ per step.

## The live training window

```
┌──────────────────────────────────────┬─────────────────────────────┐
│          ATTEMPT 140 · docked!       │  how often it docks         │
│ TIME  DISTANCE  SPEED  LIMIT    FUEL │                             │
│ 11:07 34.7 m   0.16   0.22 ✓  ███░   ├─────────────────────────────┤
├──────────────────────────────────────┤  score: fuel + docking      │
│                    direction ──▶     │                             │
│         ▭■▭ TARGET                   ├─────────────────────────────┤
│              ╲ 50 m                  │  fuel spent per attempt     │
│        ▲ chaser                      ├─────────────────────────────┤
│  ▼ EARTH, 400 km below               │  legend                     │
└──────────────────────────────────────┴─────────────────────────────┘
```

The left panel is meant to read like a video game for someone with no
background in orbits: the station at the centre, the Earth below, the orbit
running to the right, and the chaser as an arrow with its engine flame. A
status bar above the scene shows the distance, the speed against the speed
limit at that distance, and a fuel gauge; it and the legend are kept off the
scene, so nothing ever hides the chaser. One real training attempt,
exploration noise included, is replayed sped up once every $N$ attempts and
ends on a banner (DOCKED!, CRASHED, LOST IN SPACE, OUT OF TIME); the curves are
refreshed after every rollout. Training pauses only during the replays, about
two minutes over a full run.

A note on reading those curves. The network losses are deliberately not
plotted: in reinforcement learning they do **not** fall the way they do in
supervised learning, because the policy changes the very data it collects, so
the PPO policy loss oscillates around zero and the value loss often *grows*
once the agent starts reaching the docking bonus. The curves that show learning
are the success rate and the mean episode return *without* the shaping term: with $\gamma < 1$ and a negative potential, a step spent standing
still earns $(\gamma - 1)\,\Phi > 0$, so the shaped undiscounted return
rewards wandering for a long time and would make a poor agent look good.

## The honest part

A reinforcement learning agent on Clohessy-Wiltshire dynamics is not a new
result, and it is not expected to win. The problem is linear with a quadratic
cost, which is exactly the setting where an infinite-horizon LQR is optimal, so
the classical controller in `baselines.py` should be very hard to beat on fuel.
That comparison is the point of including it: the interesting number is how
close a policy that was given no model of the dynamics gets to one that was
handed the equations.

What the learned policy can do that LQR cannot is absorb the parts of the
problem that break the linear-quadratic assumptions — thrust saturation, a
docking cone, a hard failure at a fixed distance — without any of them having to
be linearised away.

## Roadmap

The steps, and the reasoning and formulas behind each one, live in
**[ROADMAP.md](ROADMAP.md)**.

## References

- G. W. Hill, *Researches in the Lunar Theory*, Am. J. Math. **1**, 5 (1878)
- W. H. Clohessy & R. S. Wiltshire, *Terminal Guidance System for Satellite
  Rendezvous*, J. Aerospace Sci. **27**, 653 (1960)
- C. F. Van Loan, *Computing integrals involving the matrix exponential*,
  IEEE Trans. Automat. Contr. **23**, 395 (1978)
- H. Schaub & J. L. Junkins, *Analytical Mechanics of Space Systems*,
  AIAA Education Series, 4th ed. (2018)
- A. Y. Ng, D. Harada & S. Russell, *Policy invariance under reward
  transformations: theory and application to reward shaping*, Proc. ICML (1999)
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford & O. Klimov,
  *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017)
- A. Raffin et al., *Stable-Baselines3: Reliable Reinforcement Learning
  Implementations*, J. Mach. Learn. Res. **22**, 268 (2021)
- M. Towers et al., *Gymnasium: A Standard Interface for Reinforcement Learning
  Environments*, arXiv:2407.17032 (2024)

## License

MIT. Author: Lorenzo Monti.

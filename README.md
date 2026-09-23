# Orbital Rendezvous

A **PPO agent** learning to fly a chaser spacecraft onto a target in low Earth
orbit, by trial and error, in the linearised relative dynamics of the
**Clohessy-Wiltshire equations** — the same problem a Dragon capsule solves on
its final approach to the ISS, reduced to its essential geometry.

The target flies a circular orbit of semi-major axis $`a`$, with mean motion
$`n = \sqrt{\mu/a^3}`$. In the LVLH frame centred on the target, with $`x`$ radial
and $`y`$ along-track, a chaser of mass $`m`$ under a thrust $`\mathbf{u}`$ obeys

```math
\ddot{x} - 3n^2 x - 2n\dot{y} = \frac{u_x}{m}, \qquad \ddot{y} + 2n\dot{x} = \frac{u_y}{m}
```

The coupling terms are what make a rendezvous counter-intuitive. Thrusting
straight at a target ahead of you raises your orbit, slows you down, and leaves
you further behind: the secular drift in the closed-form solution is
$`-6 n x_0 t`$, proportional to the *radial* offset. The agent has to discover
that it must go down to catch up.

![The agent and an LQR controller flying the same approach](assets/side_by_side.gif)

*The trained agent (left) and the fastest LQR controller that never crashes
(right), from the same starting point never seen in training. The status bars
compare them directly: time, distance, speed against the speed limit, and fuel
used.*

The out-of-plane motion, $`\ddot{z} + n^2 z = u_z/m`$, is a harmonic oscillator
fully decoupled from the two in-plane axes. It is left out: it would double the
training cost without adding any coupling to learn.

> **Status: complete.** The agent learns to dock: after 2 million
> steps, about two minutes on a laptop, it docks from 200 out of 200 unseen
> starting points in a median $`600\,\text{s}`$, spending $`0.98\,\text{m/s}`$ of
> $`\Delta v`$. That is faster *and* cheaper than the fastest LQR controller that
> never crashes; a patient LQR spends half as much in four times the time. See
> [ROADMAP.md](ROADMAP.md) for how each step was reached.

## Install

```bash
git clone https://github.com/LoreMonti/Orbital_Rendezvous.git
cd Orbital_Rendezvous
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

That one command reads `pyproject.toml` and pulls in `numpy`, `scipy`,
`matplotlib`, `gymnasium`, `stable-baselines3` and `torch`, plus `pytest` and
`ruff` from the `dev` extra. Python 3.10 or newer.
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
| `evaluate.py` | the agent against a sweep of LQR controllers and the ideal two-impulse transfer, on 200 unseen starts: a table, a plot and a JSON file |
| `play.py` | the agent and an LQR controller flying the same approach side by side, in a window or as a GIF |

```bash
python scripts/train.py --config configs/ppo_default.yaml
python scripts/train.py --no-render          # full speed, no window
python scripts/evaluate.py --watch 5        # and replay 5 attempts
python scripts/play.py                       # agent against the fastest LQR
python scripts/play.py --lqr cheapest        # against the patient one
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
│   ├── baselines.py        # LQR controller and two-impulse transfer
│   ├── evaluation.py       # flies any controller on fixed starts, summarises
│   ├── game_view.py        # one attempt drawn like a video game, reusable
│   ├── live_view.py        # the training window: a game view and the curves
│   ├── callbacks.py        # SB3 callback feeding that window during training
│   └── utils.py            # YAML config into the dataclasses, with checks
├── scripts/
│   ├── train.py            # trains PPO and saves the model
│   ├── evaluate.py         # agent against LQR and two impulses: table and plot
│   └── play.py             # agent against LQR, side by side, same start
├── tests/
│   ├── test_dynamics.py    # analytical solution, closed orbits, limit cases
│   ├── test_env.py         # Gymnasium check_env, spaces, reset and step
│   ├── test_rewards.py     # term signs, consistency of the breakdown
│   ├── test_live_view.py   # callback bookkeeping, headless drawing, short PPO run
│   ├── test_config.py      # YAML and code defaults agree; typos rejected
│   ├── test_baselines.py   # Riccati residual, stability, two impulses land exactly
│   └── conftest.py         # draws off-screen, so tests never open a window
├── notebooks/              # exploration and figures only, no logic
├── models/                 # checkpoints, git-ignored except the final one
└── assets/                 # GIFs and plots used by this README
```

The physics lives in `dynamics.py` and knows nothing about agents, so it can be
tested against the closed-form solution on its own; the renderer is split into
the figure (`live_view.py`) and the training hook that drives it
(`callbacks.py`), which is what lets the window draw one rollout every $`N`$
episodes while training keeps running at full speed.

## Tests

```bash
pytest
ruff check .
```

Errors in orbital mechanics rarely crash: a sign flip or a missing factor of
$`n`$ produces plausible-looking trajectories. The suite pins the invariants that
would catch that — the state-transition matrix from `expm` against the
analytical one, a chaser at rest staying at rest, the secular drift matching
$`-6 n x_0 t`$, the initial condition $`\dot{y}_0 = -2 n x_0`$ closing into a
periodic $`2\!:\!1`$ ellipse, two steps of $`\Delta t`$ agreeing with one of
$`2\Delta t`$ under a constant thrust, and, for $`n \to 0`$, the input matrix
reducing to a double integrator plus the first-order Coriolis coupling
$`\pm n\,\Delta t^3/3m`$, which pins both the $`1/m`$ and the sign of the coupling.

The environment is tested by placing the chaser by hand in a state that must
lead to a given outcome in one step, so a mislabelled ending or a wrong
`terminated`/`truncated` flag shows up directly. That includes the tunnelling
case, a chaser fast enough to cross the docking sphere between two steps with
neither endpoint inside it, and `check_env` with warnings treated as errors.

For the reward, besides the sign of every term, the suite checks the property
that makes the shaping safe: over a random trajectory the discounted sum of the
shaping terms equals $`\gamma^K\Phi(\mathbf{s}_K) - \Phi(\mathbf{s}_0)`$ to
$`10^{-12}`$, so a closed loop earns nothing and the shaping cannot be farmed.

The training window is tested off-screen: its callback is fed episodes whose
outcome, return and $`\Delta v`$ are known in advance, so the bookkeeping behind
every curve is checked exactly, and a short real PPO run checks that the pieces
fit together inside Stable-Baselines3.

The baselines are checked against what they claim to be: the LQR gain against
the residual of the Riccati equation, its closed loop for stability, and, with
free fuel, its slowest mode against $`e^{-\Delta t/\tau}`$. The two-impulse
transfer is flown with the closed-form $`\Phi`$ and must land on the origin to
$`10^{-9}\,\text{m}`$.

## The learning problem

| | |
| --- | --- |
| observation | the relative state $`[x, y, \dot{x}, \dot{y}]`$, normalised |
| action | continuous thrust $`[u_x, u_y]`$, saturated at $`u_\mathrm{max}`$ |
| reward | potential-based shaping towards the target under a glide slope, minus the $`\Delta v`$ spent, plus $`\pm 100`$ at the end |
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
training. The $`\Phi`$ it returns must agree with the analytical matrix to machine
precision, which is the strongest available check on the implementation.

The reward is shaped with a potential (Ng, Harada & Russell, 1999),

```math
F = \gamma\,\Phi(\mathbf{s}') - \Phi(\mathbf{s}), \qquad
\Phi(\mathbf{s}) = -\,w_r\,\frac{r}{r_\mathrm{max}} - w_v\,\frac{\max\left(0,\ |\mathbf{v}| - v_\mathrm{dock} - r/\tau\right)}{v_\mathrm{ref}}
```

which pulls the chaser in under a speed limit that tightens to the docking
speed at the target. Because $`F`$ telescopes, it speeds learning up without
changing which policy is optimal. Fuel, instead, is a real cost:
$`-w_f\,|\mathbf{u}|\,\Delta t/m`$ per step.

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
exploration noise included, is replayed sped up once every $`N`$ attempts and
ends on a banner (DOCKED!, CRASHED, LOST IN SPACE, OUT OF TIME); the curves are
refreshed after every rollout. Training pauses only during the replays, about
two minutes over a full run.

A note on reading those curves. The network losses are deliberately not
plotted: in reinforcement learning they do **not** fall the way they do in
supervised learning, because the policy changes the very data it collects, so
the PPO policy loss oscillates around zero and the value loss often *grows*
once the agent starts reaching the docking bonus. The curves that show learning
are the success rate and the mean episode return *without* the shaping term: with $`\gamma \lt  1`$ and a negative potential, a step spent standing
still earns $`(\gamma - 1)\,\Phi \gt  0`$, so the shaped undiscounted return
rewards wandering for a long time and would make a poor agent look good.

## Results

![Fuel against time to dock: the agent and a sweep of LQR controllers](assets/delta_v_vs_time.png)

On 200 starting points never seen in training, each controller flown on the
same ones:

| | docked | $`\Delta v`$, median (5–95 %) | time, median | speed at docking |
| --- | --- | --- | --- | --- |
| PPO agent, deterministic | 200 / 200 | $`0.98\,\text{m/s}`$ ($`0.68`$–$`1.37`$) | $`600\,\text{s}`$ | $`2.9\,\text{cm/s}`$ |
| LQR, fastest that always docks | 200 / 200 | $`1.05\,\text{m/s}`$ ($`0.71`$–$`1.48`$) | $`650\,\text{s}`$ | $`1.3\,\text{cm/s}`$ |
| LQR, cheapest that always docks | 200 / 200 | $`0.45\,\text{m/s}`$ ($`0.25`$–$`0.80`$) | $`2620\,\text{s}`$ | $`0.2\,\text{cm/s}`$ |
| ideal two-impulse transfer | not flyable | $`0.26\,\text{m/s}`$ ($`0.08`$–$`0.51`$) | $`2880\,\text{s}`$ | |

The LQR minimises $`\sum \mathbf{s}^T Q\,\mathbf{s} + \mathbf{u}^T R\,\mathbf{u}`$
on the exact discrete dynamics, with $`Q = \mathrm{diag}(1, 1, \tau^2, \tau^2)/r_\mathrm{ref}^2`$:
with free fuel the velocity weight acts as a glide slope $`\dot r = -r/\tau`$.
Rather than picking one tuning, 54 of them are swept over $`\tau`$ and the fuel
weight, and only those that dock every time compete. Their best trade-offs form
the blue curve. The two-impulse transfer, $`\mathbf{v}_0^+ = -\Phi_{rv}^{-1}\Phi_{rr}\,\mathbf{r}_0`$
followed by a braking impulse, is minimised over its duration; its impulses
ignore the thrust limit, so it is a reference number and not a controller.

## The honest part

A reinforcement learning agent on Clohessy-Wiltshire dynamics is not a new
result. What this project adds is a careful comparison, and it cuts both ways.

The agent sits outside the LQR front at the fast end: it docks sooner *and*
spends less than the fastest LQR that never crashes. The reason is the docking
speed limit, a hard constraint that a quadratic cost cannot express. Tuned for
speed, the LQR arrives too fast and crashes, in up to 68 % of the attempts;
the agent learned the constraint from the crashes themselves. But nothing in
the plot shows the agent is fuel-efficient in absolute terms: an LQR allowed
four times longer spends half as much, and an ideal two-impulse transfer less
than a third. The agent was trained with a time limit and a glide slope that
reward a quick approach, and it found one.

An earlier version of this README claimed the LQR would be very hard to beat on
fuel because the problem is "linear with a quadratic cost". That was wrong: the
fuel paid here is $`\sum|\mathbf{u}|`$, not $`\sum|\mathbf{u}|^2`$. A quadratic
cost prefers to thrust a little all the time, while delta-v is minimised by a
few decisive burns, which is why the two-impulse transfer undercuts both.

The first training run learned nothing: with a decision every second and
$`\gamma = 0.99`$, the agent looked 100 seconds ahead, while an approach takes
about 1000, so the docking bonus was discounted to
$`100 \cdot 0.99^{1000} \approx 0.004`$ and never seen. Deciding every 10 seconds instead, with the
physics unchanged since the propagation is exact, took the docking rate from
0 % to 200 out of 200. The timescale of the decisions mattered more than any
hyperparameter; the full account is in the roadmap, Step 5.

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

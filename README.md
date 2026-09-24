# Orbital Rendezvous

A reinforcement learning agent that learns, by trial and error, to fly a chaser
spacecraft onto a target in low Earth orbit. The relative motion follows the
Clohessy-Wiltshire equations, propagated exactly; the agent is trained with PPO
and compared, on identical starting conditions, with a sweep of LQR controllers
and with the ideal two-impulse transfer.

![The agent and an LQR controller flying the same approach](assets/side_by_side.gif)

*The trained agent (left) and the fastest LQR controller that never crashes
(right), from the same starting point never seen in training. The status bars
compare them directly: time, distance, speed against the speed limit, and fuel
used.*

**Key results**, on 200 starting points never seen in training:

- the agent docks **199 times out of 200**, after 2 million training steps,
  about two minutes on a laptop, and between 197 and 200 out of 200 over three
  training seeds;
- it docks in a median $`600\ \text{s}`$ for $`0.98\ \text{m/s}`$ of
  $`\Delta v`$: sooner **and** cheaper than the fastest LQR controller that
  never crashes ($`650\ \text{s}`$, $`1.05\ \text{m/s}`$), whose more aggressive
  tunings crash in up to 68 % of the attempts;
- a patient LQR spends less than half the fuel in four times the time, and the
  ideal two-impulse transfer less than a third: the agent learned a *quick*
  approach, not a fuel-optimal one.

## Contents

1. [Physics](#physics)
2. [The control problem](#the-control-problem)
3. [Classical baselines](#classical-baselines)
4. [Results](#results)
5. [Discussion and limitations](#discussion-and-limitations)
6. [Getting started](#getting-started)
7. [Repository layout](#repository-layout)
8. [Tests](#tests)
9. [Roadmap](#roadmap)
10. [References](#references)

## Physics

### Relative motion near a circular orbit

The target flies a circular orbit of semi-major axis $`a`$, with mean motion
$`n = \sqrt{\mu/a^3}`$. The chaser is described in the LVLH frame centred on the
target, with $`x`$ radial (pointing away from the Earth) and $`y`$ along-track
(in the direction of motion). For separations much smaller than $`a`$, the
linearised relative motion of a chaser of mass $`m`$ under a thrust
$`\mathbf{u}`$ obeys the Clohessy-Wiltshire (Hill) equations [1, 2]:

```math
\ddot{x} - 3n^2 x - 2n\dot{y} = \frac{u_x}{m}, \qquad
\ddot{y} + 2n\dot{x} = \frac{u_y}{m}, \qquad
\ddot{z} + n^2 z = \frac{u_z}{m}
```

The out-of-plane motion $`z`$ is a harmonic oscillator fully decoupled from the
in-plane axes, so it is left out: it would double the training cost without
adding any coupling to learn. The target orbits at $`400\ \text{km}`$ of
altitude, $`a = 6778\ \text{km}`$, which gives $`n = 1.131 \times 10^{-3}\ \text{rad/s}`$
and an orbital period of $`92.6`$ minutes.

### A linear system, propagated exactly

On the state $`\mathbf{s} = [x, y, \dot{x}, \dot{y}]^T`$ the in-plane equations
form a linear time-invariant system $`\dot{\mathbf{s}} = A\mathbf{s} + B\mathbf{u}`$:

```math
A = \begin{pmatrix} 0&0&1&0 \\ 0&0&0&1 \\ 3n^2&0&0&2n \\ 0&0&-2n&0 \end{pmatrix},
\qquad
B = \frac{1}{m}\begin{pmatrix} 0&0 \\ 0&0 \\ 1&0 \\ 0&1 \end{pmatrix}
```

A system with constant coefficients has an exact solution,
$`\mathbf{s}(t) = \Phi(t)\,\mathbf{s}(0)`$ with $`\Phi(t) = e^{At}`$. A numerical
integrator such as Runge-Kutta would give the same answer with a truncation
error, run slower, and leave nothing exact to test against. With $`\tau = nt`$,
$`s = \sin\tau`$ and $`c = \cos\tau`$, the state-transition matrix is the
classical closed form [3]:

```math
\Phi(t) = \begin{pmatrix}
4-3c & 0 & \dfrac{s}{n} & \dfrac{2(1-c)}{n} \\[2mm]
6(s-\tau) & 1 & -\dfrac{2(1-c)}{n} & \dfrac{4s-3\tau}{n} \\[2mm]
3ns & 0 & c & 2s \\[2mm]
-6n(1-c) & 0 & -2s & 4c-3
\end{pmatrix}
```

Two properties can be read directly off this matrix.

**The secular drift.** The entry $`6(s-\tau)`$ grows linearly in time: a radial
offset $`x_0`$ produces an along-track drift of $`-6 n x_0 t`$ on average,
exactly $`-12\pi x_0`$ per orbit. This is what makes a rendezvous
counter-intuitive. A chaser below the target ($`x_0 \lt 0`$) moves faster and
drifts ahead; thrusting straight at a target in front raises the orbit, slows
the chaser down and leaves it further behind. To catch up, one has to go down.

**Closed relative orbits.** The column of $`\dot{y}`$ contains $`4s - 3\tau`$.
With $`\dot{y}_0 = -2 n x_0`$ the secular terms cancel, and the relative orbit
closes into a periodic ellipse twice as long along-track as it is radially.

### Thrust held over a step

The agent chooses a thrust every $`\Delta t`$ and holds it constant over the step
(a zero-order hold). The state then advances as
$`\mathbf{s}_{k+1} = \Phi\,\mathbf{s}_k + \Gamma\,\mathbf{u}_k`$, with

```math
\Gamma(\Delta t) = \int_0^{\Delta t} \Phi(\Delta t - \sigma)\, B \,\mathrm{d}\sigma
```

This integral has a closed form, but a long one that is easy to get wrong by
hand. It is computed instead with Van Loan's method [4]: a single exponential of
a block matrix contains both matrices,

```math
\exp\left( \begin{pmatrix} A & B \\ 0 & 0 \end{pmatrix} \Delta t \right)
= \begin{pmatrix} \Phi & \Gamma \\ 0 & I \end{pmatrix}
```

The result is exact, not an approximation, for any $`\Delta t`$, and it is
evaluated once when the environment is built, so a simulation step costs two
small matrix products. It also gives the strongest available check on the
physics: the $`\Phi`$ obtained this way must agree with the closed form above to
machine precision, and it does.

## The control problem

### Environment

The environment follows the Gymnasium interface [5].

| | |
| --- | --- |
| **action** | $`a \in [-1, 1]^2`$, thrust $`\mathbf{u} = u_\mathrm{max}\,a`$ saturated per axis, $`u_\mathrm{max} = 1\ \text{N}`$ |
| **observation** | $`[x/r_\mathrm{max},\ y/r_\mathrm{max},\ \dot{x}/v_\mathrm{ref},\ \dot{y}/v_\mathrm{ref},\ t/T_\mathrm{max}]`$, with $`r_\mathrm{max} = 500\ \text{m}`$, $`v_\mathrm{ref} = 0.5\ \text{m/s}`$ and $`T_\mathrm{max} = 3000\ \text{s}`$ |
| **start** | a random distance $`r_0 \in [80, 200]\ \text{m}`$ in a random direction, with a small random velocity |
| **step** | $`\Delta t = 10\ \text{s}`$, at most 300 steps ($`3000\ \text{s}`$) |
| **docked** | inside $`1\ \text{m}`$ of the target and slower than $`0.05\ \text{m/s}`$ |
| **crashed** | inside $`1\ \text{m}`$, too fast |
| **escaped** | further than $`500\ \text{m}`$ |
| **timeout** | the step limit: a true end of the episode, not penalised |

The chaser has a mass of $`500\ \text{kg}`$, so its acceleration is at most
$`2\ \text{mm/s}^2`$ per axis. The thrust is continuous rather than a set of
on/off burns: closer to a throttleable engine, and easier to learn, because the
agent can correct gently. The observation is normalised because positions are
hundreds of metres and velocities centimetres per second; fed raw, the network
would barely see the velocities. The start is random so that the agent learns a
strategy rather than memorising a trajectory.

The last component of the observation is the clock. Without it, the same
position and velocity early and late in an episode would look identical to the
agent although their futures differ, and the problem would not be Markovian.
With the clock observed, the time limit is part of the task, so the timeout is
treated as a true end of the episode rather than an interruption to bootstrap
from [9]. It carries no penalty: the agent should not learn to fear the clock
itself.

At a few metres per second the chaser can cover more than the docking diameter
in one step, and fly through the target with neither end of the step inside the
docking sphere. The outcome is therefore decided on the closest approach of the
whole segment travelled during the step.

### Reward

A naive reward for getting closer can be farmed, by oscillating back and forth,
and in general changes which policy is optimal. The reward is instead shaped
with a potential [6]: with a potential $`\Phi(\mathbf{s})`$, high where the agent
should be, each step adds

```math
F = \gamma\,\Phi(\mathbf{s}') - \Phi(\mathbf{s})
```

Over any trajectory the discounted sum of these terms telescopes,

```math
\sum_{k=0}^{K-1} \gamma^k F_k = \gamma^K \Phi(\mathbf{s}_K) - \Phi(\mathbf{s}_0)
```

so it depends only on the two ends: nothing can be farmed, and the optimal
policy is provably the same as without shaping. The shaping speeds learning up
without moving the target. The potential of a terminated state is taken as
zero, which keeps the equivalence exact on episodic tasks. The potential used
here is

```math
\Phi(\mathbf{s}) = -\,w_r\,\frac{r}{r_\mathrm{max}} \;-\; w_v\,\frac{\max\left(0,\ |\mathbf{v}| - v_\mathrm{max}(r)\right)}{v_\mathrm{ref}},
\qquad
v_\mathrm{max}(r) = v_\mathrm{dock} + \frac{r}{\tau}
```

The first term pulls the chaser in. The second is a speed limit that tightens on
approach, a *glide slope*: with $`\tau = 200\ \text{s}`$ it allows
$`0.55\ \text{m/s}`$ at $`100\ \text{m}`$, $`0.10\ \text{m/s}`$ at $`10\ \text{m}`$,
and exactly the docking speed at the target. Without it, the first term alone
would teach the agent to rush in and crash.

Two further terms are real costs, meant to change the optimal policy: the fuel
spent in each step, $`-w_f\,|\mathbf{u}|\,\Delta t/m`$, and a terminal reward of
$`+100`$ on docking and $`-100`$ on a crash or an escape. The weights are
$`w_r = 20`$, $`w_v = 10`$ and $`w_f = 2`$ per $`\text{m/s}`$: flying in from
$`200\ \text{m}`$ earns $`20 \cdot 200/500 = +8`$ of shaping, an efficient
approach costs about $`-2`$ of fuel.

### Choosing the timescale of the decisions

With a discount $`\gamma`$ the agent effectively looks
$`1/(1-\gamma)`$ steps ahead, 100 steps for $`\gamma = 0.99`$. An approach
along the glide slope, $`\dot r = -r/\tau`$, takes

```math
t \approx \tau \ln\frac{r_0}{r_\mathrm{dock}} = 200 \ln 200 \approx 1060\ \text{s}
```

so the time step fixes whether the agent can see the end of its own manoeuvre.
A first training run used $`\Delta t = 1\ \text{s}`$: the horizon was 100
seconds, the docking bonus arrived about 1000 steps later and was worth
$`100 \cdot 0.99^{1000} \approx 0.004`$, and the agent never learned to approach
(0 % docking, median closest approach $`133\ \text{m}`$). With
$`\Delta t = 10\ \text{s}`$ the same 100 steps span $`1000\ \text{s}`$; since the
propagation is exact for any step, the physics is unchanged, and only the rate
of the decisions is.

### Training

PPO [7] from Stable-Baselines3 [8], on 8 parallel environments:

| | |
| --- | --- |
| network | two hidden layers of 64 units, for the policy and the value function |
| rollout | 512 steps per environment, a bit under two episodes |
| minibatch | 256 |
| learning rate | $`3 \times 10^{-4}`$ |
| discount, GAE | $`\gamma = 0.99`$, $`\lambda = 0.95`$ |
| clip range | 0.2 |
| total | 2 million steps, about 20 000 episodes, about 2 minutes on a laptop |

The shaping discount must equal the training discount, or the shaping is no
longer guaranteed to leave the optimal policy unchanged; loading a configuration
where the two differ is an error.

## Classical baselines

### LQR

On the exact discrete dynamics, the infinite-horizon LQR minimises

```math
J = \sum_k \mathbf{s}_k^T Q\,\mathbf{s}_k + \mathbf{u}_k^T R\,\mathbf{u}_k
```

through the discrete algebraic Riccati equation,

```math
P = Q + \Phi^T P \Phi - \Phi^T P \Gamma \left(R + \Gamma^T P \Gamma\right)^{-1} \Gamma^T P \Phi,
\qquad
\mathbf{u}_k = -K\mathbf{s}_k, \quad K = \left(R + \Gamma^T P \Gamma\right)^{-1}\Gamma^T P \Phi
```

with the thrust saturated at the same limit as the agent. When fuel is free,
the balance between a position error $`q_r r^2`$ and a speed $`q_v v^2`$ is an
exponential approach $`\dot r = -r/\tau`$ with $`\tau = \sqrt{q_v/q_r}`$: the
velocity weight acts as an implicit glide slope. The controller is therefore
parametrised physically,

```math
Q = \frac{1}{r_\mathrm{ref}^2}\,\mathrm{diag}\left(1, 1, \tau^2, \tau^2\right),
\qquad
R = \frac{w}{u_\mathrm{max}^2}\, I
```

by the time constant $`\tau`$ of that glide slope and by a fuel weight $`w`$ that
trades time for fuel on top of it. Note that the LQR penalises
$`|\mathbf{u}|^2`$, while the fuel actually paid is $`\sum|\mathbf{u}|`$: a
quadratic cost prefers to thrust a little all the time, whereas $`\Delta v`$ is
minimised by a few decisive burns. The LQR is a solid classical controller, not
a fuel-optimal bound.

### Two-impulse transfer

The textbook reference for $`\Delta v`$ [3]. A first impulse sets the velocity
that reaches the origin after a time $`T`$; a second one cancels the arrival
velocity. From the position and velocity blocks of $`\Phi(T)`$,

```math
\mathbf{v}_0^+ = -\Phi_{rv}^{-1}\Phi_{rr}\,\mathbf{r}_0,
\qquad
\Delta v = \left|\mathbf{v}_0^+ - \mathbf{v}_0\right| + \left|\Phi_{vr}\mathbf{r}_0 + \Phi_{vv}\mathbf{v}_0^+\right|
```

minimised over $`T`$ up to the length of an episode, skipping the durations where
$`\Phi_{rv}`$ is singular, at multiples of the orbital period. Impulses are
instantaneous and ignore the thrust limit, so this is a reference number, not a
controller that can fly in the environment.

## Results

### Learning

![Four moments of one training run](assets/training.gif)

*Four replays from one training run, with the curves as they stood at that
moment: after about 500 attempts the agent drifts off and is lost in space;
around 1500 it has learned to close in but stops short of the target; around
2500 it docks about half the time; by the end it docks cleanly almost every
time, on a fraction of the fuel it burnt while learning.*

The docking rate is 5 % over the first 2000 episodes, 85 % over the next 2000,
and 97 to 98 % from there on. The fuel curve rises first and falls later: the
agent first discovers that moving pays, and burns up to $`5.5\ \text{m/s}`$ per
attempt; once it docks reliably, it learns to do so on about $`1\ \text{m/s}`$.
Two runs with the same seed give identical numbers.

The window above is also what one watches during training. The curves plotted
are the docking rate and the *true* score, fuel plus terminal reward, without
the shaping term. With $`\gamma \lt 1`$ and a negative potential, a step spent
standing still earns $`(\gamma - 1)\,\Phi \gt 0`$, so the shaped undiscounted
return rewards wandering and would make a poor agent look good. The network
losses are not plotted either: in reinforcement learning they do not fall as
they do in supervised learning, because the policy changes the data it
collects, and they do not say whether the agent is improving.

### Comparison with the classical baselines

![Fuel against time to dock: the agent and a sweep of LQR controllers](assets/delta_v_vs_time.png)

Every controller is flown on the same 200 starting points, none of which was
seen in training. Rather than picking one LQR tuning by hand, which could favour
the agent without meaning to, 54 of them are swept: $`\tau`$ from 50 to
$`300\ \text{s}`$ and $`w`$ from $`10^{-4}`$ to 1. Only those that dock every time
compete; their best trade-offs form the blue curve.

| | docked | $`\Delta v`$, median (5–95 %) | time, median | speed at docking |
| --- | --- | --- | --- | --- |
| PPO agent, deterministic | 199 / 200 | 0.98 m/s (0.64–1.37) | 600 s | 2.9 cm/s |
| LQR, fastest that always docks | 200 / 200 | 1.05 m/s (0.71–1.48) | 650 s | 1.3 cm/s |
| LQR, cheapest that always docks | 200 / 200 | 0.45 m/s (0.25–0.80) | 2620 s | 0.2 cm/s |
| ideal two-impulse transfer | not flyable | 0.26 m/s (0.08–0.51) | 2880 s | |

The fastest reliable LQR has $`\tau = 100\ \text{s}`$ and $`w = 10^{-3}`$, the
cheapest $`\tau = 200\ \text{s}`$ and $`w = 0.3`$. More aggressive tunings do not
dock faster: they arrive too fast and crash, in up to 68 % of the attempts. The
agent's one failure is also a crash at the limit, arriving at
$`7.2\ \text{cm/s}`$ against the $`5\ \text{cm/s}`$ allowed.

### Robustness across training seeds

A single training run can be lucky or unlucky, so the effect of the clock in the
observation was measured on three seeds, each trained and evaluated on the same
200 unseen starts:

| seed | without the clock | with the clock |
| --- | --- | --- |
| 0 | 200 / 200 | 199 / 200 |
| 1 | 200 / 200 | 197 / 200 |
| 2 | **0 / 200** | 200 / 200 |

Without the clock, one run in three never learned to dock: it closed to about
$`70\ \text{m}`$, parked there at $`1\ \text{cm/s}`$ and waited for the episode
to end, a local optimum in which waiting has no visible end. With the clock, and
the timeout treated as the true end of the task, every seed docks; the cost is
one to three crashes in 200, all at the edge of the docking speed. Three seeds
are too few for a statistic, but a run that fails outright is hard to dismiss.
Docking rate, $`\Delta v`$ and time to dock are otherwise the same within a few
percent.

## Discussion and limitations

A reinforcement learning agent on Clohessy-Wiltshire dynamics is not a new
result. What this project adds is a careful comparison, and it cuts both ways.

**Where the agent wins.** At the fast end it sits outside the LQR front, docking
sooner and on less fuel than the fastest LQR that never crashes. The reason is
the docking speed limit, a hard constraint that a quadratic cost cannot express.
Tuned for speed, the LQR arrives too fast; the agent learned the constraint from
its own crashes. Without being given the dynamics, it also found much the same
path as the controller that was: in the side-by-side replays the two
trajectories are strikingly similar.

**Where it does not.** It is not perfectly reliable: one to three attempts in
200 still crash, just over the docking speed limit, where the LQR never does.
Nothing shows the agent is fuel-efficient in absolute
terms: an LQR allowed four times longer spends less than half as much, and the
ideal two-impulse transfer less than a third. The agent was trained with a time
limit and a glide slope that reward a quick approach, and it found one.

Start by start, against the fastest LQR that never crashes, the agent is never
more expensive: on the 199 starts where it docks, the LQR docks sooner on 3 and
spends less fuel on none.

**What mattered most.** The timescale of the decisions mattered more than any
hyperparameter. The first run, with a decision every second, learned nothing;
the same physics with a decision every ten seconds docked every time.

**What is left out.** The model is linear, planar and deterministic: no
out-of-plane motion, no perturbations (drag, $`J_2`$), no navigation noise, no
approach corridor or keep-out zone, and a single target on a circular orbit.
These are where a learned policy could matter more than here, since they break
the assumptions that make the LQR a natural fit.

## Getting started

### Installation

```bash
git clone https://github.com/LoreMonti/Orbital_Rendezvous.git
cd Orbital_Rendezvous
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs `numpy`, `scipy`, `matplotlib`, `gymnasium`, `stable-baselines3`,
`torch` and `pyyaml`, plus `pytest` and `ruff` from the `dev` extra. Python 3.10
or newer is required. scipy is pinned below 1.15, whose macOS arm64 wheels fail
to load on macOS 27. The results above were produced with Python 3.10, numpy
2.2.6, scipy 1.14.1, gymnasium 1.3.0, stable-baselines3 2.9.0 and torch 2.14.0.

### Usage

Every parameter lives in `configs/ppo_default.yaml`. Each script needs no
arguments for the default run and lists its options with `--help`.

| script | what it does |
| --- | --- |
| `train.py` | trains PPO with the live window open, and saves the model to `models/` and a run directory to `runs/` |
| `evaluate.py` | the agent against the LQR sweep and the two-impulse transfer on 200 unseen starts: a table, the plot and a JSON file |
| `play.py` | the agent and an LQR flying the same approach side by side, in a window or as a GIF |

```bash
python scripts/train.py                                # train, with the window
python scripts/train.py --no-render                    # full speed, no window
python scripts/train.py --record assets/training.gif   # and save the training GIF
python scripts/evaluate.py                             # table and plot
python scripts/evaluate.py --watch 5                   # and replay 5 attempts
python scripts/play.py                                 # against the fastest LQR
python scripts/play.py --lqr cheapest                  # against the patient one
python scripts/play.py --gif assets/side_by_side.gif   # save the comparison GIF
```

As a library:

```python
from orbital_rendezvous import EnvConfig, RendezvousEnv
from orbital_rendezvous.baselines import LQRController

env = RendezvousEnv(EnvConfig(docking_radius=1.0))
lqr = LQRController.from_env(env, approach_time=100.0, fuel_weight=1e-3)

obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(lqr.act(env.state))
    done = terminated or truncated
print(info["outcome"], info["distance"])
```

## Repository layout

```
Orbital_Rendezvous/
├── README.md
├── ROADMAP.md
├── LICENSE
├── pyproject.toml          # metadata, dependencies, ruff and pytest config
├── configs/
│   └── ppo_default.yaml    # environment, reward, PPO and window parameters
├── src/orbital_rendezvous/
│   ├── dynamics.py         # Clohessy-Wiltshire propagation: pure physics, no RL
│   ├── env.py              # RendezvousEnv, the Gymnasium API
│   ├── rewards.py          # potential-based shaping, fuel and terminal terms
│   ├── baselines.py        # LQR controller and two-impulse transfer
│   ├── evaluation.py       # flies any controller on fixed starts, summarises
│   ├── game_view.py        # one attempt drawn like a video game, reusable
│   ├── live_view.py        # the training window: a game view and the curves
│   ├── callbacks.py        # SB3 callback feeding the window, recording GIFs
│   └── utils.py            # YAML config into the dataclasses, with checks
├── scripts/
│   ├── train.py            # trains PPO and saves the model
│   ├── evaluate.py         # agent against LQR and two impulses: table and plot
│   └── play.py             # agent against LQR, side by side, same start
├── tests/                  # 64 tests, one file per module
├── models/                 # trained models, git-ignored
└── assets/                 # the GIFs, the plot and the evaluation numbers
```

The physics in `dynamics.py` knows nothing about agents, so it can be tested
against the closed-form solution on its own. The reward is kept apart from the
environment so that it can be tuned alone. The game view is its own module, so
that the training window holds one and the side-by-side comparison two, drawn
identically by construction.

## Tests

```bash
pytest
ruff check .
```

Errors in orbital mechanics rarely crash: a sign flip or a missing factor of
$`n`$ produces plausible-looking trajectories. The suite pins the invariants that
would catch them.

- **Dynamics.** $`\Phi`$ from Van Loan's method agrees with the closed form to
  machine precision at every step size up to a full orbit; a chaser at rest at
  the origin stays there; a radial offset drifts by exactly $`-12\pi x_0`$ per
  orbit; $`\dot{y}_0 = -2 n x_0`$ gives a closed orbit; two steps of $`\Delta t`$
  equal one of $`2\Delta t`$ under a constant thrust; and for $`n \to 0`$,
  $`\Gamma`$ reduces to a double integrator plus the first-order Coriolis
  coupling $`\pm n\,\Delta t^3/3m`$, which pins both the $`1/m`$ and the sign of
  the coupling.
- **Environment.** Each outcome is reached by placing the chaser by hand in a
  state that must lead to it in one step, including a chaser fast enough to
  cross the docking sphere between two steps; the clock starts at zero and ticks
  by $`1/T_\mathrm{max}`$ per step, and at the timeout the shaping pays back
  exactly $`-\Phi(\mathbf{s})`$; `check_env` passes with warnings treated as
  errors.
- **Reward.** The sign of every term, and the property that makes the shaping
  safe: over a random trajectory the discounted sum equals
  $`\gamma^K\Phi(\mathbf{s}_K) - \Phi(\mathbf{s}_0)`$ to $`10^{-12}`$.
- **Baselines.** The LQR gain against the residual of the Riccati equation, its
  closed loop for stability, and with free fuel its slowest mode against
  $`e^{-\Delta t/\tau}`$; the two-impulse transfer, flown with the closed-form
  $`\Phi`$, lands on the origin to $`10^{-9}\ \text{m}`$.
- **Configuration.** The YAML file and the defaults in the code agree, and
  unknown keys are rejected. This test caught PyYAML reading `6778.0e3` as a
  string: YAML 1.1 needs an explicit exponent sign.
- **Visualisation.** The callback behind the curves is fed episodes whose
  outcome, return and $`\Delta v`$ are known; the status bar is checked against
  the true final state; everything runs off-screen.

## Roadmap

How the project was built, step by step, including the runs that failed, is in
**[ROADMAP.md](ROADMAP.md)**.

## References

1. G. W. Hill, *Researches in the Lunar Theory*, Am. J. Math. **1**, 5 (1878)
2. W. H. Clohessy & R. S. Wiltshire, *Terminal Guidance System for Satellite
   Rendezvous*, J. Aerospace Sci. **27**, 653 (1960)
3. H. Schaub & J. L. Junkins, *Analytical Mechanics of Space Systems*,
   AIAA Education Series, 4th ed. (2018)
4. C. F. Van Loan, *Computing integrals involving the matrix exponential*,
   IEEE Trans. Automat. Contr. **23**, 395 (1978)
5. M. Towers et al., *Gymnasium: A Standard Interface for Reinforcement Learning
   Environments*, arXiv:2407.17032 (2024)
6. A. Y. Ng, D. Harada & S. Russell, *Policy invariance under reward
   transformations: theory and application to reward shaping*, Proc. ICML (1999)
7. J. Schulman, F. Wolski, P. Dhariwal, A. Radford & O. Klimov,
   *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017)
8. A. Raffin et al., *Stable-Baselines3: Reliable Reinforcement Learning
   Implementations*, J. Mach. Learn. Res. **22**, 268 (2021)
9. F. Pardo, A. Tavakoli, V. Levdik & P. Kormushev, *Time Limits in
   Reinforcement Learning*, Proc. ICML (2018)

## License

MIT. Author: Lorenzo Monti.

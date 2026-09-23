# Roadmap

The project is built one reviewable step at a time. Each step ends with tests
that pass, an updated README, and a commit.

---

## Step 0 — Repository skeleton *(done)*

Package layout with `src/`, editable install, `pyproject.toml` with ruff and
pytest configured, all parameters collected in `configs/ppo_default.yaml`, and
module stubs with their docstrings and signatures in place.

Design decisions taken here:

- **Planar motion only.** The out-of-plane equation $\ddot{z} + n^2 z = u_z/m$
  is a harmonic oscillator fully decoupled from the in-plane axes, so it would
  double the training cost without adding any coupling to learn.
- **Continuous thrust** $[u_x, u_y]$ saturated at $u_{\max}$, rather than
  discrete on/off burns: more realistic, and easier to learn because the agent
  can correct gently.
- **PPO** as the algorithm: robust, tolerant of imperfect hyperparameters, and
  well suited to continuous actions. SAC is kept as a possible later comparison.
- **Physics separated from the RL code**: `dynamics.py` knows nothing about
  agents, so it can be tested against the analytical solution on its own.

The README follows a fixed section order, the same one used across the other
repositories, so that it grows by filling sections in rather than by being
rearranged: title and physics, Install, Usage, Layout, Tests, what the code
computes, the honest part, Roadmap, References, License. Display formulas go in
` ```math ` blocks and inline ones in `$...$`.

---

## Step 1 — Clohessy-Wiltshire dynamics *(done)*

The target flies a circular orbit of semi-major axis $a$ with mean motion
$n = \sqrt{\mu/a^3}$. In the LVLH frame centred on the target, with $x$ radial
and $y$ along-track, the chaser of mass $m$ obeys

$$\ddot{x} - 3n^2 x - 2n\dot{y} = \frac{u_x}{m}, \qquad \ddot{y} + 2n\dot{x} = \frac{u_y}{m},$$

that is, a linear time-invariant system $\dot{\mathbf{s}} = A\mathbf{s} + B\mathbf{u}$
on the state $\mathbf{s} = [x, y, \dot{x}, \dot{y}]^T$, with

$$A = \begin{pmatrix} 0&0&1&0 \\ 0&0&0&1 \\ 3n^2&0&0&2n \\ 0&0&-2n&0 \end{pmatrix},
\qquad
B = \frac{1}{m}\begin{pmatrix} 0&0 \\ 0&0 \\ 1&0 \\ 0&1 \end{pmatrix}.$$

### Why a closed form instead of a numerical integrator

Because the system is linear with constant coefficients, the propagation is
available exactly as $\mathbf{s}(t) = \Phi(t)\,\mathbf{s}(0)$ with
$\Phi(t) = e^{At}$. A Runge-Kutta scheme would give the same answer with a
truncation error, run slower, and leave nothing exact to test against.

### The state-transition matrix

With $\tau = nt$, $s = \sin\tau$ and $c = \cos\tau$:

$$\Phi(t) = \begin{pmatrix}
4-3c & 0 & \dfrac{s}{n} & \dfrac{2(1-c)}{n} \\[2mm]
6(s-\tau) & 1 & -\dfrac{2(1-c)}{n} & \dfrac{4s-3\tau}{n} \\[2mm]
3ns & 0 & c & 2s \\[2mm]
-6n(1-c) & 0 & -2s & 4c-3
\end{pmatrix}$$

Two properties read directly off this matrix, and both become tests:

- The entry $6(s-\tau)$ is **secular**: a radial offset $x_0$ drifts along-track
  by roughly $-6 n x_0 t$. This is the physical reason a rendezvous is not
  trivial, and the drift the agent has to exploit.
- If $\dot{y}_0 = -2 n x_0$ the secular terms cancel and the relative orbit
  closes into a periodic $2\!:\!1$ ellipse.

### The control matrix

For a thrust held constant over a step (zero-order hold),

$$\Gamma(\Delta t) = \int_0^{\Delta t} \Phi(\Delta t - \sigma)\, B \,\mathrm{d}\sigma .$$

This integral has a closed form, but a long one that is easy to get wrong by
hand. It is computed instead with **Van Loan's method**, which is exact rather
than an approximation:

$$\exp\left( \begin{pmatrix} A & B \\ 0 & 0 \end{pmatrix} \Delta t \right)
= \begin{pmatrix} \Phi & \Gamma \\ 0 & I \end{pmatrix},$$

so $\Phi$ and $\Gamma$ are read off the blocks of a single matrix exponential,
evaluated once when the environment is built. As a by-product, the $\Phi$
obtained this way must agree with the analytical matrix above to machine
precision, which is the strongest available check on the implementation.

### Tests

- $\Phi$ from `expm` equals the analytical $\Phi$;
- a chaser at rest at the origin with no thrust stays there;
- the secular along-track drift matches $-6 n x_0 t$ at first order;
- the initial condition $\dot{y}_0 = -2 n x_0$ gives a closed orbit: the state
  returns to itself after one period $T = 2\pi/n$;
- propagating twice over $\Delta t$ equals propagating once over $2\Delta t$,
  under a constant thrust, which checks $\Gamma$ as well as $\Phi$;
- for $n \to 0$, $\Gamma$ reduces to the double integrator,
  $\Delta x = u\,\Delta t^2/2m$ and $\Delta v = u\,\Delta t/m$, plus the
  first-order Coriolis coupling, antisymmetric between the axes:
  $\pm n\,\Delta t^3/3m$ on position and $\pm n\,\Delta t^2/m$ on velocity.
  This pins both the $1/m$ in $B$ and the sign of the coupling.

### Outcome

`propagate` takes the matrices precomputed by `discretize`, so a step costs two
small matrix products, and it accepts batches of states. The $\Phi$ from
`expm` agrees with the analytical one to $10^{-9}$ relative at every step size
tested, up to a full orbital period.

One environment issue surfaced here: the scipy 1.15.x wheels for macOS arm64
ship a Fortran extension that the macOS 27 loader refuses to open, so
`import scipy.linalg` fails. scipy is pinned below 1.15 in `pyproject.toml`
until a fixed wheel exists for Python 3.10.

---

## Step 2 — Gymnasium environment *(done)*

`RendezvousEnv` on top of the dynamics, with the terminal reward only.

- **Action** $a \in [-1, 1]^2$, thrust $\mathbf{u} = u_\mathrm{max}\,a$ saturated
  per axis. With $u_\mathrm{max} = 1\,\text{N}$ and $m = 500\,\text{kg}$ the
  acceleration is at most $2\,\text{mm/s}^2$, so an episode of $2000\,\text{s}$
  can spend at most $\Delta v = 4\,\text{m/s}$. *(Step 5 later changed the
  step to $10\,\text{s}$ and the episode to $3000\,\text{s}$; see there.)*
- **Observation** normalised so that every component is of order one:
  $\mathbf{o} = [x/r_\mathrm{max},\ y/r_\mathrm{max},\ \dot{x}/v_\mathrm{ref},\ \dot{y}/v_\mathrm{ref}]$
  with $r_\mathrm{max} = 500\,\text{m}$ and $v_\mathrm{ref} = 0.5\,\text{m/s}$.
  Positions are hundreds of metres and velocities centimetres per second: fed
  raw, the network would barely see the velocities. The bounds are finite,
  twice the escape radius and $10\,\text{m/s}$, as `check_env` requires.
- **Initial condition** at a random distance $r_0 \in [80, 200]\,\text{m}$ in a
  random direction, with a small random velocity, so the agent has to learn a
  strategy rather than memorise one trajectory.
- **Four outcomes**: docked ($r < 1\,\text{m}$ and $|\mathbf{v}| < 0.05\,\text{m/s}$),
  crashed (same sphere, too fast), escaped ($r > 500\,\text{m}$), and timeout.
  The timeout is reported as `truncated`, not `terminated`, so PPO treats it as
  an interrupted episode rather than a failure.
- **Tunnelling.** At a few metres per second the chaser covers more than the
  docking diameter in one step, so it could fly through the target with
  neither endpoint inside the sphere. The outcome is decided on the closest
  approach of the whole segment travelled during the step.

### Outcome

The environment passes `check_env` with warnings treated as errors, and runs
at about $3 \times 10^4$ steps per second on one core. A random policy over
200 episodes escapes 157 times, times out 43 times, and **never docks**. With
only a terminal reward, PPO would almost never see a positive signal: this is
the concrete reason the shaping terms of Step 3 are needed.

## Step 3 — Reward function *(done)*

The reward of a step is the sum of three terms, each returned separately in
`info["reward_terms"]` for logging and tuning.

**Why not a naive "reward for getting closer".** Paying $+w$ whenever the
distance shrinks can be farmed, by oscillating back and forth, and in general
changes which policy is optimal. Potential-based shaping (Ng, Harada & Russell,
1999) avoids both: with a potential $\Phi(\mathbf{s})$, each step adds

$$F = \gamma\,\Phi(\mathbf{s}') - \Phi(\mathbf{s}),$$

whose discounted sum over any trajectory telescopes to
$\gamma^K\Phi(\mathbf{s}_K) - \Phi(\mathbf{s}_0)$. It depends only on the two
ends, so nothing can be farmed, and the optimal policy is provably unchanged.
$\gamma$ must be the discount factor of PPO, $0.99$. The potential of a
terminated state is taken as zero, which is what keeps the equivalence exact
on episodic tasks; a truncated state keeps its potential.

**The potential** has two terms:

$$\Phi(\mathbf{s}) = -\,w_r\,\frac{r}{r_\mathrm{max}} \;-\; w_v\,\frac{\max\!\left(0,\ |\mathbf{v}| - v_\mathrm{max}(r)\right)}{v_\mathrm{ref}}, \qquad v_\mathrm{max}(r) = v_\mathrm{dock} + \frac{r}{\tau}.$$

The first pulls towards the target. The second is a speed limit that tightens
on approach, a glide slope: with $\tau = 200\,\text{s}$ it allows
$0.55\,\text{m/s}$ at $100\,\text{m}$, $0.10\,\text{m/s}$ at $10\,\text{m}$, and
exactly the docking speed at the target. Without it the first term alone would
teach the agent to rush in and crash.

**Fuel** is a real cost, not shaping, and is meant to change the optimal policy:
$r_\mathrm{fuel} = -\,w_f\,|\mathbf{u}|\,\Delta t/m$, the $\Delta v$ spent in the step.

**Terminal**: $+100$ on docking, $-100$ on a crash or an escape, $0$ on a timeout.

**Weights**: $w_r = w_v = w_f = 10$. Flying in from $200\,\text{m}$ is worth
about $+4$ of shaping; burning the whole $4\,\text{m/s}$ budget costs $-40$; an
efficient approach of about $0.5\,\text{m/s}$ costs $-5$. The docking bonus
stays dominant, and wasting fuel hurts.

*This reasoning was wrong, and Step 5 shows why.* The docking bonus is only
dominant if the agent can see it, and with these settings it could not; before
the bonus is ever reached, $+4$ for approaching against $-5$ of fuel makes
staying put the better choice. The weights were changed to $w_r = 20$ and
$w_f = 2$ in Step 5.

### Tests

Every sign: closing in rewarded, receding penalised, speeding near the target
penalised, thrust never free. The glide slope ending at the docking speed.
Zero potential at the target at rest and on terminated states. And the property
that makes the shaping safe: over a random trajectory the discounted sum equals
$\gamma^K\Phi(\mathbf{s}_K) - \Phi(\mathbf{s}_0)$ to $10^{-12}$, and a closed
loop earns nothing. In the environment, the reward equals the sum of its terms.

### Outcome: the shaped return is not the score to plot

Over 200 episodes of a random policy the shaping term sums to $+65$ on average,
and an agent that escapes can end with a total of $+59$. For a step with
$\mathbf{s}' = \mathbf{s}$,

$$F = (\gamma - 1)\,\Phi(\mathbf{s}) = -0.01\,\Phi(\mathbf{s}) > 0,$$

because the potential is negative everywhere but at the target: at
$300\,\text{m}$, $\Phi \approx -6$, so about $+0.06$ per step and $+84$ over 1400
steps, which matches. This does not affect learning, since PPO maximises the
*discounted* return, on which the telescoping is exact. It does make the
*undiscounted* episode return meaningless as a progress measure, because a poor
agent that wanders for a long time would look good. The live window therefore
plots the unshaped return, fuel plus terminal, alongside the success rate:
train on the shaped reward, measure on the true one.

## Step 4 — Live training window *(done)*

A single matplotlib figure. On the left the LVLH plane, with the target, the
chaser, its trail and its thrust arrow, and reference circles labelled with the
glide-slope speed limit $v_\mathrm{max}(r)$ at that distance, so it is visible
whether the agent respects it. The trail is coloured by how the episode ended:
green docked, red crashed or escaped, grey timeout. On the right the success
rate and the true return, fuel plus terminal (see Step 3), as the prominent
curves, then the $\Delta v$ per episode.

**How it fits in the training.** A Stable-Baselines3 callback is called at
every step, at the end of each rollout, and before each update. At every step
it accumulates the true return and the $\Delta v$ of each parallel environment
and records the trajectory of environment 0. When an episode ends its numbers
join the curves; once every $N = 50$ episodes the last finished episode of
environment 0 is replayed. The curves are redrawn at the end of each rollout.

**Which episode is shown.** A real training episode, exploration noise
included: it shows what the agent is doing while it learns. A separate
deterministic rollout, showing what it has already learned, belongs to
`evaluate.py`.

**Replay sped up, not in real time.** An episode can last 2000 steps, which at
60 frames per second would stop training for over half a minute. The replay
takes about 130 frames whatever the length, 3 to 4 seconds. Training pauses
meanwhile, since matplotlib on macOS must run on the main thread: about 28
replays over 2 million steps cost about 2 minutes in total, and `--no-render`
removes them.

### A game view, readable by anyone

The first version of the left panel was correct but read like a plot: no
legend, no sense of where the Earth is, and cryptic speed labels on the
rings. It was redesigned to read like a video game, for someone with no
background in orbits or in machine learning:

- a dark space scene with stars, the Earth drawn below, and an arrow for the
  direction of the orbit, which is what makes a rendezvous counter-intuitive;
- the target as a station with solar panels, the chaser as an arrow pointing
  where it is going, and the thrust as an engine flame out of the back;
- rings labelled with distances in metres, down a diagonal so that the labels
  never overlap;
- a status bar above the scene with the time, the distance, the speed against the glide-slope limit
  $v_\mathrm{max}(r)$ at the current distance, marked ✓ or ✗, and a fuel gauge
  counting down from the most $\Delta v$ an attempt could spend,
  $\sqrt{2}\,u_\mathrm{max}\,T/m$;
- a closing banner: DOCKED!, CRASHED, LOST IN SPACE or OUT OF TIME;
- plain-language titles on the right: "attempt" rather than "episode", and
  "score" rather than "return".

The status bar and the legend both started inside the scene, and both ended
up covering the chaser whenever it flew into their corner. They were moved out:
the status bar above the scene, the legend under the curves on the right, laid
out wide on three columns. To make room, the PPO losses were dropped from the
window: they say nothing to a newcomer, and even to an expert they do not say
whether a PPO agent is improving. Stable-Baselines3 still logs them.

The velocity is now passed through `info` as well as the position, so the status
bar shows the exact speed rather than a finite difference.

### Tests

Run off-screen with the `Agg` backend, so they never open a window. The
callback is fed episodes whose outcome, return and $\Delta v$ are known, and
its bookkeeping is checked exactly: the true return excludes the shaping, the
accumulators reset between episodes, and the replay happens once every $N$
episodes and only for environment 0. The status-bar values are checked against the
true final state of an attempt: time, distance, speed, the glide-slope limit
and the fuel left. The figure is drawn and saved headless,
and a short real PPO run checks that the pieces fit inside Stable-Baselines3.

### Outcome

A 100 000-step run with 8 environments takes about 7 seconds, around
$1.4 \times 10^4$ steps per second including PPO, so the full 2-million-step
training should take a few minutes. After 70 episodes the agent has learned
nothing yet, as expected: no docking, a true return around $-110$, and nearly
the whole $4\,\text{m/s}$ budget burnt in every episode.

## Step 5 — Training *(done)*

`scripts/train.py` reads every parameter from the YAML file, trains PPO on 8
parallel environments, and writes the model to `models/` and a run directory
under `runs/` with checkpoints, the Stable-Baselines3 log as CSV (losses
included), one row per episode, and a copy of the configuration. Loading the
configuration rejects unknown keys, since a typo would otherwise be silently
replaced by a default, and a shaping discount different from the training one.

A test that the YAML and the dataclass defaults agree caught a real bug on its
first run: PyYAML follows YAML 1.1, where `6778.0e3` is a *string*, because a
float needs an explicit exponent sign, `6.778e+6`.

### First run: nothing learned

With the settings of Steps 2 and 3 ($\Delta t = 1\,\text{s}$, 2000 steps,
$w_r = w_f = 10$), 2 million steps gave **0 % docking**. The agent learned
something, the reward rose and it stopped escaping as often, but the median
closest approach over 100 attempts was $133\,\text{m}$: it never really moved
towards the target. Two causes, both visible in the numbers.

**The discount horizon did not cover the manoeuvre.** With $\gamma = 0.99$ the
agent looks about $1/(1-\gamma) = 100$ steps ahead, which at
$\Delta t = 1\,\text{s}$ is 100 seconds. An approach along the glide slope,
$\dot r = -r/\tau$, takes

$$t \approx \tau \ln\frac{r_0}{r_\mathrm{dock}} = 200 \ln 200 \approx 1060\,\text{s},$$

so the docking bonus arrived about 1000 steps later, worth
$100 \cdot 0.99^{1000} \approx 0.004$: invisible.

**Fuel cost more than approaching earned.** Without the bonus in sight, $+4$ of
shaping for flying in from $200\,\text{m}$ against about $-5$ of fuel made
staying put the better choice. The agent did exactly what it was paid to do.

### The fix

- $\Delta t = 10\,\text{s}$, episodes of 300 steps ($3000\,\text{s}$). The
  propagation is exact for any step, so the physics is unchanged; only the
  decision rate is. The horizon of 100 steps now spans $1000\,\text{s}$.
- $w_r = 20$, $w_f = 2$: approaching from $200\,\text{m}$ earns
  $20 \cdot 200/500 = +8$, an efficient approach costs about $-2$.
- PPO rollouts of 512 steps per environment, a bit under two episodes.
- One replay every 500 episodes instead of 50: with shorter episodes a full run
  has about 20 000 of them, and 400 replays would have paused training for
  almost half an hour.

### Outcome

The same 2 million steps, the same 2 minutes:

| | first run | after the fix |
| --- | --- | --- |
| docking, end of training | 0 % | about 98 % |
| docking, 200 unseen starts, deterministic policy | 0 / 100 | **200 / 200** |
| $\Delta v$ spent, median (5th–95th percentile) | $1.6\,\text{m/s}$, for nothing | $0.98\,\text{m/s}$ ($0.68$–$1.37$) |
| time to dock, median | | $600\,\text{s}$ |
| speed at docking, median | | $2.9\,\text{cm/s}$, limit $5$ |

The learning curve: 5 % docking over the first 2000 episodes, 75 % over the
next 2000, and 97–99 % from there on, so most of the learning happens in the
first third of the run. Two runs with the same seed give identical numbers.

## Step 6 — LQR baseline and evaluation *(done)*

### A correction first

Step 0 and the README expected the LQR to be "very hard to beat on fuel",
because the problem is linear with a quadratic cost. That was wrong. The LQR
minimises

$$J = \sum_k \mathbf{s}_k^T Q\,\mathbf{s}_k + \mathbf{u}_k^T R\,\mathbf{u}_k,$$

a cost on $|\mathbf{u}|^2$, while the fuel paid here is
$\Delta v = \sum_k |\mathbf{u}_k|\,\Delta t/m$. A quadratic cost prefers to
thrust a little all the time; $\Delta v$ is minimised by a few decisive burns
and coasting in between. The LQR is a solid classical controller, not a bound.

### Two references

**LQR** on the exact discrete dynamics $\mathbf{s}_{k+1} = \Phi\,\mathbf{s}_k + \Gamma\,\mathbf{u}_k$,
from the discrete algebraic Riccati equation,

$$P = Q + \Phi^T P \Phi - \Phi^T P \Gamma \left(R + \Gamma^T P \Gamma\right)^{-1} \Gamma^T P \Phi, \qquad \mathbf{u}_k = -K\mathbf{s}_k, \quad K = \left(R + \Gamma^T P \Gamma\right)^{-1}\Gamma^T P \Phi,$$

saturated at the same thrust limit as the agent.

**Two-impulse transfer**, the textbook reference for $\Delta v$. From the blocks
of $\Phi(T)$, the first impulse sets the velocity that reaches the origin after
$T$, and the second cancels the arrival velocity:

$$\mathbf{v}_0^+ = -\Phi_{rv}^{-1}\Phi_{rr}\,\mathbf{r}_0, \qquad \Delta v = \left|\mathbf{v}_0^+ - \mathbf{v}_0\right| + \left|\Phi_{vr}\mathbf{r}_0 + \Phi_{vv}\mathbf{v}_0^+\right|,$$

minimised over $T$ up to the episode length, skipping the durations where
$\Phi_{rv}$ is singular. Impulses ignore the thrust limit, so this is a
reference number, not a controller that can fly in the environment.

### Tuning the LQR: the velocity weight is a glide slope

The first attempt weighted position and velocity by the scales the agent's
observations are normalised with, $500\,\text{m}$ and $0.5\,\text{m/s}$. It
never docked, and its slowest closed-loop eigenvalue stayed at $0.990$ whatever
the fuel weight. With fuel free, the LQR balances a position error $q_r r^2$
against a speed $q_v v^2$, and the balance is an exponential approach

$$\dot r = -\frac{r}{\tau}, \qquad \tau = \sqrt{q_v/q_r} = \frac{500}{0.5} = 1000\,\text{s}, \qquad e^{-\Delta t/\tau} = e^{-0.01} = 0.990,$$

too slow to cover $200\,\text{m}$ in an episode. The velocity weight is an
implicit glide slope. So the LQR is parametrised physically, by that time
constant, $Q = \mathrm{diag}(1, 1, \tau^2, \tau^2)/r_\mathrm{ref}^2$, and by a
fuel weight $R = w/u_\mathrm{max}^2$ that trades time for fuel on top of it.

### A fair comparison: a sweep, not a hand-picked tuning

Picking one $(\tau, w)$ by hand would be arbitrary, and could favour the agent
without meaning to. `evaluate.py` sweeps 54 tunings, $\tau$ from 50 to
$300\,\text{s}$ and $w$ from $10^{-4}$ to $1$, flies each on the same 200
unseen starts as the agent, and keeps those that dock every time. Their Pareto
front in ($\Delta v$, time to dock) is the curve the agent is compared with.

### Tests

The LQR gain against the residual of the Riccati equation; the closed loop
stable for every tuning; with free fuel, the slowest mode equal to
$e^{-\Delta t/\tau}$; saturation; and docking from ten starts. The two-impulse
transfer flown with the closed-form $\Phi$ lands on the origin to
$10^{-9}\,\text{m}$, costs nothing from rest at the origin, and the search
steps over the singular duration of one orbital period.

### Outcome

| 200 unseen starts | docked | $\Delta v$, median (5–95 %) | time, median | docking speed |
| --- | --- | --- | --- | --- |
| PPO agent, deterministic | 100 % | $0.98\,\text{m/s}$ ($0.68$–$1.37$) | $600\,\text{s}$ | $2.9\,\text{cm/s}$ |
| LQR, fastest that always docks ($\tau = 100$, $w = 10^{-3}$) | 100 % | $1.05\,\text{m/s}$ ($0.71$–$1.48$) | $650\,\text{s}$ | $1.3\,\text{cm/s}$ |
| LQR, cheapest that always docks ($\tau = 200$, $w = 0.3$) | 100 % | $0.45\,\text{m/s}$ ($0.25$–$0.80$) | $2620\,\text{s}$ | $0.2\,\text{cm/s}$ |
| ideal two-impulse | | $0.26\,\text{m/s}$ ($0.08$–$0.51$) | $2880\,\text{s}$ | |

The agent sits outside the LQR front at its fast end, sooner and cheaper than
the fastest LQR that never crashes. More aggressive LQR tunings do not dock
faster: they arrive too fast and crash, in up to 68 % of the attempts, because
the docking speed limit is a hard constraint a quadratic cost cannot express.
At the slow end the LQR wins clearly, and the ideal two-impulse transfer shows
how much further there is to go on fuel. The whole evaluation runs in about
40 seconds.

## Step 7 — Side-by-side comparison *(done)*

### A change of plan

The original Step 7 was a game mode: fly the chaser with the keyboard and
compare your $\Delta v$ with the agent's. It was dropped. The content of the
project is the comparison between the agent and the LQR, and a human pilot adds
nothing measurable to it: one person's $\Delta v$ on one start is not a
reproducible number. It was also the most fragile piece to build, real-time
keyboard input in a second game loop, for a feature few readers of a repository
would install and try. `pygame` left the dependencies with it.

What the project still lacked was a way to *show* the comparison. The plot of
Step 6 speaks to a physicist; a replay speaks to anyone.

### What it does

`play.py` flies the agent and an LQR from the same held-out start and draws
them side by side, in two identical game views at the same scale and on the
same clock, so the status bars compare them directly. The LQR tuning is read
from the results of `evaluate.py`: by default the fastest one that always
docks, the agent's closest rival, or with `--lqr cheapest` the patient one. It
opens a window, or with `--gif` writes the animation shown at the top of the
README.

To make two views possible, the scene and its status bar were extracted from
the training window into a reusable `GameView`; the training window now holds
one and the comparison two, drawn identically by construction. Two details
noticed during training were fixed on the way: the TARGET label now sits on a
dark tab above the trail, so the final approach cannot hide it, and distance
rings that reach the edge of the view go unlabelled instead of being cut. The
status bar now shows the fuel used in m/s, the number the comparison is about.

### Outcome

On the first two held-out starts, the ones in the GIF:

| start | agent | LQR, fastest that always docks |
| --- | --- | --- |
| 1 | $700\,\text{s}$, $1.18\,\text{m/s}$ | $780\,\text{s}$, $1.32\,\text{m/s}$ |
| 2 | $820\,\text{s}$, $1.21\,\text{m/s}$ | $690\,\text{s}$, $1.21\,\text{m/s}$ |

The second start is worth keeping in view: there the LQR docks first on the
same fuel. The agent's advantage measured in Step 6 is a statement about the
median over 200 starts, not about every single one, and the side-by-side
replay shows it as it is. The paths are also strikingly similar: without being
given the dynamics, the agent found much the same way in as the controller that
was.

A 2-start GIF at 72 dpi is 2.4 MB, small enough for the README.

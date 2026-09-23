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
  can spend at most $\Delta v = 4\,\text{m/s}$.
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

## Step 4 — Live training window *(next)*

A single matplotlib figure: the LVLH plane with the chaser and its trail on the
left, and the mean unshaped episode return (fuel plus terminal, see Step 3),
success rate and delta-v on the right, with the PPO losses as secondary curves. The trajectory is redrawn once every
`episode_stride` episodes so training keeps running at full speed.

Note on reading those curves: in reinforcement learning the losses do not fall
the way they do in supervised learning, because the policy changes the data it
collects. The PPO policy loss oscillates around zero and the value loss often
grows once the agent starts reaching the docking bonus. Mean episode reward and
success rate are the curves that show learning.

## Step 5 — Training

PPO on vectorised environments, with a seeded and reproducible run, and
checkpoints saved to `models/`.

## Step 6 — LQR baseline and evaluation

An infinite-horizon LQR on the discrete CW dynamics, and a comparison on
success rate, total delta-v and time to dock over a fixed set of seeded initial
conditions. The honest expectation is that LQR is very hard to beat on fuel for
this linear problem; the result of interest is how close the learned policy
gets.

## Step 7 — The game

`play.py`: watch the trained agent, or fly the chaser with the keyboard and
compare your delta-v with the agent's and the LQR's on the same initial
condition. GIF recording for the README.

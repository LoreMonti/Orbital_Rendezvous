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

## Step 1 — Clohessy-Wiltshire dynamics *(next)*

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
- propagating twice over $\Delta t$ equals propagating once over $2\Delta t$.

---

## Step 2 — Gymnasium environment

`RendezvousEnv` on top of the dynamics: normalised observations, thrust
saturation, episode termination on docking, crash, runaway or timeout, and
compliance with `gymnasium.utils.env_checker.check_env`.

## Step 3 — Reward function

Distance shaping, fuel penalty, docking bonus and failure penalty, with each
term returned separately for logging and tuning.

## Step 4 — Live training window

A single matplotlib figure: the LVLH plane with the chaser and its trail on the
left, and the mean episode reward, success rate and delta-v on the right, with
the PPO losses as secondary curves. The trajectory is redrawn once every
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

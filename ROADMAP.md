# Roadmap

The project was built one reviewable step at a time: each step ended with tests
passing, the README updated, and a commit. The physics and the method behind
every item are in the [README](README.md); this file records what was done, in
what order, and what each step taught.

## Step 0 — Repository skeleton

- [x] `src/` package layout, editable install, `pyproject.toml` with ruff and pytest
- [x] Every parameter in `configs/ppo_default.yaml`, none hard-coded
- [x] Design choices: planar motion only, continuous thrust, PPO, physics kept apart from the RL code
- [x] README and this roadmap

## Step 1 — Clohessy-Wiltshire dynamics

- [x] Closed-form state-transition matrix $`\Phi(t)`$
- [x] Zero-order-hold input matrix $`\Gamma`$ by Van Loan's method, evaluated once
- [x] One step as two small matrix products, on single states or batches
- [x] Tests: $`\Phi`$ from Van Loan against the closed form, secular drift, closed $`2\!:\!1`$ orbit, step composition, the Coriolis limit for $`n \to 0`$

*Lesson.* A test expecting a pure double integrator for $`n \to 0`$ failed on
terms of $`10^{-13}`$: they were the first-order Coriolis coupling, now checked
explicitly, sign included. Also: scipy 1.15 wheels do not load on macOS 27, so
scipy is pinned below 1.15.

## Step 2 — Gymnasium environment

- [x] Normalised observations with finite bounds, thrust saturated per axis
- [x] Random seeded starts between 80 and 200 m
- [x] Four outcomes: docked, crashed, escaped, and timeout as a truncation
- [x] Docking decided on the closest approach of the whole step, so a fast chaser cannot tunnel through the target
- [x] Tests: each outcome forced in one step; `check_env` with warnings as errors

*Lesson.* A random policy never docks in 200 episodes: a terminal reward alone
gives PPO nothing to learn from, which is why Step 3 was needed.

## Step 3 — Reward

- [x] Potential-based shaping towards the target, under a glide slope ending at the docking speed
- [x] Fuel cost per m/s of $`\Delta v`$, and a $`\pm 100`$ terminal reward
- [x] Zero potential on terminated states; every term returned separately in `info`
- [x] Tests: the sign of every term, and the telescoping sum to $`10^{-12}`$

*Lesson.* With $`\gamma \lt 1`$ and a negative potential, standing still earns
$`(\gamma - 1)\,\Phi \gt 0`$. It does not affect learning, but it makes the
shaped return useless as a progress measure: plot the unshaped one.

## Step 4 — Live training window

- [x] Stable-Baselines3 callback: true return and $`\Delta v`$ per episode, trajectory of one environment
- [x] One real training episode replayed sped up every $`N`$, the curves refreshed every rollout
- [x] Redesigned as a game view anyone can follow: space scene, Earth below, station, chaser with an engine flame, closing banner
- [x] Status bar and legend moved off the scene, after both were found covering the chaser
- [x] PPO losses dropped from the window: they do not say whether the agent is improving
- [x] Tests: callback bookkeeping on episodes with known outcomes, status bar against the true state, all off-screen

## Step 5 — Training

- [x] `train.py`: parallel environments, checkpoints, CSV log, per-episode rows, a copy of the config per run
- [x] Config loading that rejects unknown keys and a shaping discount different from the training one
- [x] First run: **0 % docking** after 2 million steps
- [x] Diagnosis: with $`\Delta t = 1\ \text{s}`$ the agent looked 100 s ahead while an approach takes about 1000 s, and fuel cost more than approaching earned
- [x] Fix: $`\Delta t = 10\ \text{s}`$, 300-step episodes, $`w_r = 20`$, $`w_f = 2`$
- [x] Result: **200 / 200** docking on unseen starts, in the same 2 minutes

*Lesson.* The timescale of the decisions mattered more than any hyperparameter.
Also: the test comparing the YAML with the code defaults caught PyYAML reading
`6778.0e3` as a string.

## Step 6 — Classical baselines and evaluation

- [x] LQR on the exact discrete dynamics, parametrised by the time constant of its implicit glide slope and by a fuel weight
- [x] Ideal two-impulse transfer, minimised over its duration
- [x] `evaluate.py`: 54 LQR tunings swept, only those that always dock compete; table, plot and JSON
- [x] Tests: Riccati residual, closed-loop stability, slowest mode $`e^{-\Delta t/\tau}`$, two impulses landing exactly on the origin

*Lesson.* Two corrections. The LQR is not fuel-optimal here, since it penalises
$`|\mathbf{u}|^2`$ while the fuel paid is $`\sum|\mathbf{u}|`$. And the first LQR
tuning never docked, because its velocity weight hid a glide slope with a time
constant of 1000 s.

## Step 7 — Side-by-side comparison

- [x] Plan changed: the keyboard game mode was dropped, since a human pilot adds nothing measurable; `pygame` removed
- [x] Game view extracted into a reusable module
- [x] `play.py`: agent and LQR from the same start, same scale and clock, in a window or as a GIF
- [x] GIF at the top of the README

*Lesson.* On single starts the LQR sometimes docks first: the agent's advantage
holds for the median, and the replay shows it as it is.

## Step 8 — Training GIF

- [x] `train.py --record`: replays captured at chosen moments of the run plus the last one, joined into a GIF, headless too
- [x] Frames shrunk as they are taken; the exact theme colours added to the GIF palette
- [x] Fixed a headless replay that drew the first step instead of the last
- [x] GIF in the README, showing the agent learning

## Step 9 — README

- [x] Math rewritten for GitHub's renderer, every formula checked through its API
- [x] README reorganised as a technical write-up: physics, control problem, baselines, results, discussion, then installation, layout and tests

## Possible extensions

- [ ] Out-of-plane motion, the full three-dimensional problem
- [ ] Perturbations: differential drag and $`J_2`$
- [ ] Navigation noise on the observations
- [ ] An approach corridor or a keep-out zone around the target
- [ ] A policy trained to minimise $`\Delta v`$ without the time pressure, to compare with the two-impulse bound

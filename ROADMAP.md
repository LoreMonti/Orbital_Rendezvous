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

## Step 10 — A Markovian episode

- [x] The elapsed fraction of the episode, $`t/T_\mathrm{max}`$, added to the observation
- [x] The timeout made a true end of the episode, since the time limit is now part of the task (Pardo et al., 2018), still without penalty
- [x] Tests: the clock starts at zero and ticks once per step; at the timeout the shaping pays back exactly $`-\Phi(\mathbf{s})`$
- [x] Three training seeds, with and without the clock, on the same 200 unseen starts

| seed | without the clock | with the clock |
| --- | --- | --- |
| 0 | 200 / 200 | 199 / 200 |
| 1 | 200 / 200 | 197 / 200 |
| 2 | **0 / 200** | 200 / 200 |

Start by start, the retrained agent now beats the fastest reliable LQR on fuel
on every one of its 199 dockings, and is slower on only 3 of them.

*Lesson.* On a single seed the change looked like a small step back, 199
instead of 200. Three seeds told the real story: without the clock one run in
three parked 70 m from the target and waited out the episode. A single training
run is an anecdote.

## Step 11 — Trading time for fuel

- [x] Diagnosis: at $`\gamma = 0.99`$ docking late forfeits about 50 points of bonus, against 1.4 saved in fuel, so hurrying is worth 35 times the fuel
- [x] Training moved into the package (`training.py`), shared by `train.py` and the study; the default model reproduced weight for weight
- [x] Fuel curriculum: the fuel weight starts at 2 and rises to its target over the first half of training
- [x] Optional minimum thruster level (`thrust_deadzone`), off by default
- [x] `fuel_study.py`: a grid of $`\gamma`$ and $`w_f`$, three seeds each, in parallel, resumable, stopped cleanly by Ctrl-C
- [x] Costs aggregated over reliable seeds only (docking at least 95 %)
- [x] Result: $`\gamma = 0.999`$, $`w_f = 10`$ docks on 3 / 3 seeds with 0.83 m/s in 710 s, 15 % less fuel than the default agent and below the LQR front at that speed

*Lessons.* Four, each from a measurement rather than a guess.
A heavy fuel cost from the first step reopens the trap of Step 5; a curriculum
fixes it up to a weight of 10, not at 20.
At $`\gamma = 0.99`$ the fuel weight barely matters: the discount decides.
Exploration noise is taxed by the fuel cost, about 0.23 m/s per 1000 s, so a
slow approach looks expensive in training.
A minimum thruster level makes coasting free but ruins the final approach, since
its weakest firing is a hundred times the tidal acceleration near the target:
a negative result, kept as an option.
And a summary that averaged over every seed flattered two configurations with
runs that docked only from the easy starts; costs now come from reliable seeds.

## Step 12 — A Lagrange multiplier on fuel

- [x] The time-budget constraint first proposed was dropped before coding: the agent is already too fast, so "dock within T" would never bind
- [x] `FuelBudget` callback: dock while spending at most $`D`$; the fuel weight is the multiplier, updated by dual ascent on the deterministic policy's fuel, and held at zero until the agent docks in half of its test attempts
- [x] Budget mode in `fuel_study.py`; tests that retrace the worked example of dual ascent
- [x] Result, 3 seeds per budget: every run docks, the multiplier rises to 41–50 (the cap at $`D = 0.45`$), and fuel stays at 0.80–0.89 m/s: the budget is never met

*Lesson.* The multiplier fixed the trap of a heavy fuel cost, since the price
rises only after docking is learned, but not the fuel: a dearer fuel also makes
waiting dearer, because exploration noise burns fuel. Neither a hand-picked nor
a self-tuned weight is the lever; the noise is.

## Step 13 — An engine switch

- [x] `engine_switch` option: a third action; the engine fires only when it is positive, so "off" carries no noise while "on" keeps continuous thrust with no minimum
- [x] The LQR keeps the switch always on; `check_env` passes with the switch
- [x] With a fixed fuel weight of 10, the switch made the trap worse: 0 % docking, engine off 90 % of the time
- [x] With the Lagrange multiplier, 8 million steps, three seeds per budget: $`D = 0.75`$ met on 3 / 3 seeds, 0.74 m/s in 790 s, price settling at 24–34; $`D = 0.6`$ reliable on 1 / 3
- [x] A probe at $`D = 0.45`$: 0.58 m/s in 1100 s, engine off two steps in three, but 60 % docking
- [x] `fuel_summary.py`: the fuel story in one plot, 0.98 → 0.83 → 0.82 → 0.74 m/s

*Lesson.* The switch was the right lever, removing the tax on exploration noise,
and made a budget reachable for the first time. What is left is exploration
itself: a slow approach that docks every time exists, but PPO finds it only
some of the time. The fuel line of work stops here, with diminishing returns
(−15 %, then −11 %) and a clear account of each obstacle.

## Step 14 — An oriented target

- [x] Keep-out sphere of 20 m around the station, entered only inside a 15° approach cone around the docking axis (the V-bar); a violation is checked along the whole step and ends the attempt; `configs/ppo_corridor.yaml`
- [x] Baseline: the V-bar procedure, an LQR to a hold point 30 m out on the axis, then sliding along it with a steady radial thrust against Coriolis; 200 / 200, no violation, 1.06–1.23 m/s
- [x] Reference: the default agent violates the zone 199 times in 200, the fastest LQR 200 times
- [x] Attempt 1, the rule from the start: the agent stops approaching
- [x] Attempt 2, a potential towards the mouth of the cone at 15° from the start: the agent never learns to dock, even with violations free (checked against the straight potential: 97 % against 0 %)
- [x] Attempts 3–4, a Lagrange multiplier on violations in a penalty mode (`KeepOutBudget`), fast, then slow and relaxing: docking is learned, then collapses and does not return
- [x] Attempt 5, a curriculum narrowing the cone from 180° (`ConeCurriculum`), with a potential that follows the current cone around the sphere: both runs stop at 90°
- [x] `cone_summary.py` and `assets/cone_curriculum.json`

*Lesson.* The agent can shift its direction of arrival a little at a time, down
to the front half of the station; it cannot find, in small steps, the different
manoeuvre that a start behind the station needs, going around it. Helping it
with a clever potential did more harm than good twice. On the full corridor,
the classical procedure wins: the knowledge that solves it fits in a few lines.

## Possible extensions

In order of priority. Each would be a step of its own, with the same rules:
tests first, the README updated, a commit.

A note on the target's motion: the target *already* moves in this model. The
LVLH frame travels with it along its circular orbit, at about
$`7.7\ \text{km/s}`$ with respect to the Earth, and that motion is what produces
the $`-3n^2x`$ and $`\pm 2n\dot{y}`$ terms, the secular drift and the Coriolis
coupling. The target sits still at the centre of the view only because the view
is the target's own. What the real ISS adds is below: an oriented docking port,
perturbations, and, negligibly, a slightly eccentric orbit.

### 1. Fuel, if taken further

Steps 11 to 13 took the agent from 0.98 to 0.74 m/s and removed, one by one,
the discount, the stay-put trap and the tax on exploration noise. What is left
is finding a slow approach reliably.

- [ ] Longer training, and more seeds, at budgets between 0.6 and 0.75 m/s.
- [ ] A policy conditioned on the preference: the budget as an input drawn at
  random each episode, so one training learns the whole front.
- [ ] Less exploration noise late in training (an annealed or state-dependent
  standard deviation), combined with the engine switch.

### 2. The oriented target, if taken further

Step 14 stopped at a cone of 90°: the agent does not learn to go around the
station.

- [ ] A hybrid: the V-bar procedure to the hold point, the agent for the final
  approach along the corridor, where learned control has already shown its
  worth.
- [ ] A curriculum on the starting points instead of the cone: first in front
  of the station, then further and further behind.
- [ ] Imitation of the V-bar procedure as a starting policy, refined with RL.

### 3. Three dimensions

- [ ] Add the out-of-plane axis, $`\ddot{z} + n^2 z = u_z/m`$. On its own it is a
  decoupled oscillator and adds little; together with an oriented docking port
  it couples the three axes, so it follows item 2.

### 4. Robustness

- [ ] Navigation noise on the observed state, and thrust errors in magnitude and
  direction.
- [ ] Measure how the agent and the LQR degrade as the noise grows.

### 5. Perturbations

- [ ] Differential atmospheric drag and the $`J_2`$ term of the Earth's
  oblateness, which make the relative motion depart from the Clohessy-Wiltshire
  model. The exact propagation would give way to numerical integration, and the
  closed form would become the test reference for the unperturbed limit.
- [ ] An eccentric target orbit (Tschauner-Hempel equations). For the ISS,
  $`e \approx 0.0003`$, so this comes last.

### 6. A sounder comparison

- [ ] More training seeds, to put error bars on every number in the README
  (three were run in Step 10, enough to catch a failed run but not for a
  statistic).
- [ ] SAC as a second algorithm, on the same environment and budget.

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

## Step 15 — Two curricula for the corridor

- [x] `start_angle_range_deg` option: starts drawn within an angle of the docking axis; with every direction allowed, a seed selects the same start as before, so every earlier result holds
- [x] Curriculum logic shared in `MasteryCurriculum`; `StartCurriculum` widens the starts, tested on the outer 30° of the range only
- [x] Attempt 1, the start curriculum alone at a 15° cone, 8 million steps, two seeds: **no docking in 27 000 episodes** per seed, not even from in front of the port
- [x] Diagnosis: the Coriolis term $`2n|\dot{y}|`$ pushes an approach along the V-bar out of the cone; the default agent, from the front, docks 0 times in 100 with a 15° cone and 83 with a 60° cone
- [x] Fix: the two curricula in sequence, the cone narrowed on front starts first (`after=`), then the starts widened; `training.start_curriculum` in `configs/ppo_corridor.yaml`
- [x] Result, 32 million steps, three seeds: the cone reaches 15° on 3 / 3 seeds in 1.6–2.0 million steps; the starts reach 145°, 155° and 55°; 153, 121 and 71 dockings in 200 against 200 for the V-bar procedure
- [x] `corridor_eval.py`: dockings by direction of the start, and costs compared with the V-bar procedure on the starts each agent docks
- [x] `curriculum_summary.py` and `assets/start_curriculum.json`; the run's `config.yaml` now records the seed actually used

*Lesson.* The premise of the plan was wrong: a start inside the cone does not
make the straight approach easy, because in orbit the approach is not straight.
Holding the corridor against the Coriolis term was a skill of its own, and
Step 14's curriculum, restricted to starts in front of the port, taught it on
every seed. Going around the station was then learned in small steps of the
start, up to 145–155° on two seeds of three, but not from directly behind, and
twice as much training did not move it. On the starts they dock, the agents are
faster or cheaper than the procedure, never both, and never as reliable.

## Steps 16–20 — A side study: learning from a teacher

- [x] **Set aside** in [`experiments/teacher_student`](experiments/teacher_student), with its code, configurations, tests and results: it answered whether a network can fly the corridor (596 of 600 by imitation), not whether reinforcement learning can discover it alone, the question of this project

### Step 16 — Two learned pilots on the classical plan

- [x] Diagnosis of the Step 15 agents behind the station: from 160–180° they fail 18 times in 18, the best one flying straight into the back of the sphere, and they only ever go around on the side they start from
- [x] Hypothesis: the optimal action jumps between the two sides at 180°, which a continuous network cannot do
- [x] Diagnostic, starts on one side only, four runs: one reached 175° and then lost it, two never docked, one learned and collapsed; the jump matters, but losing what was learned matters more
- [x] `GoToEnv`: fly to a goal point and stop, the default task with the target moved; the goal and the absolute position both observed, since holding still at $`x`$ takes $`u_x = -3n^2x\,m`$
- [x] `HierarchicalPilot`: the V-bar planner (`side_waypoint`, now shared with the procedure), a go-to pilot to the waypoint and the hold point, a final-approach pilot down the corridor
- [x] `BestModel`: the best policy on validation starts is kept, not the last one
- [x] `configs/ppo_goto.yaml`, `configs/ppo_final_approach.yaml`; `corridor_eval.py --go-to --final` flies every pairing; `pilots_summary.py`
- [x] Result, three seeds of each pilot: 100 % on validation for all six; **200 / 200 with no violation for all nine pairings**, 47 / 47 from behind; 1.01–1.31 m/s in 1070–1850 s, against 1.06–1.23 m/s in 1400–1775 s for the procedure

*Lesson.* Two causes were tangled, and a cheap diagnostic separated them
before any new method was built: the plan should decide what a smooth
network cannot, and a learner should get a fixed task and be kept at its best.
With both, the same PPO that could not hold the corridor from behind docks
every time. On cost it matches the classical procedure rather than beating
it, so the win is reliability with learned control, not a better controller.

### Step 17 — A learned planner

- [x] `waypoint_menu`: 16 directions around the station at 40 and 60 m; the go-to pilot retrained with the menu among its goals (`configs/ppo_goto_menu.yaml`), 100 % on validation on 3 / 3 seeds, 200 / 200 with the rule's plan
- [x] `PlannerEnv`: one step per approach, a discrete choice from the menu, since the best side jumps at 180°; the frozen pilots fly the rest; $`+100`$ for a docking, $`-100`$ otherwise, less $`w_f\,\Delta v`$
- [x] `planner_eval.py`: the rule, learned planners and an oracle that flies all 33 choices from every start
- [x] Oracle: 0.90 m/s against 0.93 for the rule, 3 % of room and no more
- [x] First planner, $`w_f = 10`$, two seeds: 200 / 200, but two choices of 33 and 8–15 % more fuel than the rule
- [x] With $`w_f = 50`$, three seeds: 200 / 200, no violation, 0.91–0.92 m/s, within 0.02 m/s of the oracle, five or six choices, 400 s slower since time is not in the reward

*Lesson.* The discrete choice made the jump that stopped the single agent,
and the planner kept the approach at 200 in 200 from its first run. What it
did not do at first was refine: a reward in which fuel was worth a hundredth
of docking let it stop at the first two choices that never failed. The oracle,
computed before any training, said how much there was to win, 3 %, and so how
to read the result: the learned planner matches the rule and takes the little
the rule leaves, it does not find a better plan.

#### Step 17b — Time in the cost, the rule as a baseline

- [x] Flaw of the Step 17 planner: in front of the port a detour through a point 60 m out on the axis, nearly free in fuel, 580 s longer
- [x] Cost $`J = 50\,\Delta v + 0.01\,T`$ in the reward, the oracle and the choice of the best model; with it the oracle flies straight to the hold point from 151 starts and beats the rule by 1 %
- [x] Reward relative to the rule flown from the same start, $`J_\text{rule} - J`$ (`configs/ppo_planner_relative.yaml`)
- [x] Result, two seeds: 199 and 200 of 200, $`J`$ 68.8 and 66.4, worse than the rule; each planner fixed its choices within 13 000 approaches

*Lesson.* The diagnosis was half right. The reward was noisy, but the
obstacle was exploration: once only a few safe choices were ever tried, no
reward, however clean, could show that the others were cheaper.

#### Step 17c — Learning the planner from the oracle

- [x] `oracle_costs`: every choice flown from a start; `imitate_oracle.py` labels 4000 training starts in parallel and reuses the labels
- [x] `imitation.py`: soft targets on the cost saved, failures at zero; the same network as the PPO planner, saved and evaluated as one
- [x] Result: straight ahead on 152 starts, $`J`$ 63.3, but 197 of 200: three corners cut too close to the sphere, each with a wider choice that docks as its second
- [x] Fix: the expected cost of the choice added to the loss, a failure priced at 200; $`\lambda`$ = 0, 1, 5, 20 gave 197, 198, 199, 200 dockings
- [x] With $`\lambda = 20`$, three seeds: **200 of 200, no violation, $`J`$ 63.6, the oracle's cost**, 1715 s against 1760 for the rule

*Lesson.* On a discrete decision whose options a simulator can fly, learning
from the whole search beat learning from trials. The oracle's cheapest choice
hugs the edge of what is allowed, so imitation alone learned to cut corners;
pricing a failure into the fit, as a reward would, taught it to keep a margin
at almost no cost.

### Step 18 — More waypoints? Measured, not built

- [x] A planner may return several waypoints; `beam_search`: plans of up to three waypoints, only the three cheapest that dock extended at each level, 219 flights per start instead of about 36 000
- [x] `plan_search.py` on the 200 test starts: the best plan never has a second or third waypoint, not even from behind the station; 151 plans fly straight to the hold point, 49 through one waypoint
- [x] Not built: a planner choosing how many waypoints would learn a constant; distillation (Step 20) removes the remaining rules at once instead

*Lesson.* The cheap measurement came first again, and this time it said stop.
One sphere and one corridor need one point to go around; the idea of learning
the length of the plan was sound, but the problem gives it nothing to learn.

### Step 20 — One network again: distillation

- [x] Steps 18 and 19 set aside: distillation removes the remaining rules at once, since the student has no menu, sequence or thresholds
- [x] `distillation.py`: the teacher is the system of Step 17c; its flights record the student's observations and the teacher's commands as labels, with noise on the thrust applied but not on the labels (DART)
- [x] Noise of 0.2 everywhere: the teacher docked 8 flights in 40; of 0.05: one in four failed. Noise of 0.1 only beyond 40 m, and 0.02 everywhere: 11 829 of 12 000 docked
- [x] Student with two heads on a shared trunk: a mode, straight or around either side, chosen once and kept, and a thrust that sees the mode
- [x] First student: 180 of 200, all twenty failures in the last metre, validation swinging between 0 and 84 %
- [x] Fix: steps within 20 m weigh ten times more, a cosine decay of the learning rate, the best student on validation kept; 94–100 % on validation
- [x] The first run of the fix froze the computer: the loader re-read the whole file of flights for every one of 12 000 flights, and two runs in parallel asked for hundreds of gigabytes. Arrays are now read once (`load_flights`), and a run peaks at 0.54 GB
- [x] Result, three seeds, run in parallel: **199, 197 and 200 of 200**, 596 of 600, 0.92 m/s in 1690–1720 s, cheaper than the procedure on every docked start

*Lesson.* The same network that could not learn the corridor from scratch
learned it from a teacher, once it was given a discrete choice where the
behaviour jumps and weight where precision matters. And a memory bug is a
bug like any other: measure a run's peak on a short trial before launching
long or parallel ones.

## Step 21 — Back to RL from scratch: a reverse curriculum

- [x] `ReverseCurriculum`: starts a few metres from the port first, then further out and further round, the smallest distance kept at 2 m; `set_start_region`; inside the keep-out sphere a start lies in the cone
- [x] First run: stuck at the first stage for 20 million steps. The starts kept the task's random velocity, about 6 cm/s, and left a cone 1 m wide in one or two steps; even an LQR docked 50 times in 200. Fix: the start velocity shrinks with the distance within 15 m, $`\mathbf{v}_0 \min(1, r_0/15\ \text{m})`$, so that drifting out of the cone takes about eight steps from anywhere; a simple controller then docks 172–188 in 200 on the first stages
- [x] Second run: the agent flew 125 m away from the port and waited out the episode: near a narrow cone violations came far more often than dockings
- [x] A timeout priced as a failure (`timeout_reward`): never docked, stopped at 5 million steps
- [x] The approach cone opened on the first stage and narrowed, as in Step 15: the cone reaches 15° in 1.2–1.6 million steps, the first stages follow
- [x] Eight stages: stuck at 40 m and 45° (from 10° in one stage) at 75–80 %. Twenty-one stages, 10° at a time: through 35° in 2–3 million steps, stuck again at 45°, 75 %
- [x] Violations on the rim, 19 m out, 22° off the axis. A shaping path with a margin (a sphere of 25 m, a mouth of 7°): still 75 %. Half the starts from the stage's frontier: 70 %, and one seed collapsed. A slower glide slope, $`\tau = 400\ \text{s}`$: 75–85 %, and the chaser still reached the rim at 0.20 m/s
- [x] SAC in place of PPO (`configs/sac_corridor_reverse.yaml`, 256×256, 3 million steps, two seeds): about 1000 steps/s against 8000 for PPO; the cone reached 15° at 2.25 million steps on one seed and 30° on the other, and neither reached the 45° stage

*Lesson.* Check that the first stage of a curriculum is solvable, by a
simple controller, before training on it. The cone opened first is again what
lets the agent learn to dock under the rule. The wall at 45° held against four
changes to the curriculum and the reward, and SAC did not reach it. It looked
physical, an arrival too fast to turn; Step 22 shows it is not.

## Step 22 — What the wall at 45° is

Two seeds of 10 million steps each, the curriculum of Step 21 otherwise unchanged.

- [x] A graded failure penalty (`rewards.graded_failure`, off by default; `configs/ppo_corridor_graded.yaml`): a crash or a violation costs $`-100\,[\alpha + (1-\alpha)\min(1, e_\theta + e_v)]`$, $`\alpha = 0.5`$, with $`e_\theta`$ the angle outside the cone over 30° and $`e_v`$ the speed above the glide slope over 0.10 m/s, both where the chaser entered the sphere; a runaway keeps −100. The idea comes from fine-grained training [Pirovano, Milanesio et al., 2025]: tell a near miss from a wide one. Result: 70–75 % at 45°, unchanged. The violations entered at 0.21–0.23 m/s, the speed error saturated, and the median miss was 1: the penalty stayed binary exactly where it mattered
- [x] A discount of 0.995 instead of 0.99 (`configs/ppo_corridor_gamma.yaml`), since with 0.99 a docking 27 steps away is worth 0.76 of the bonus and one 80 steps away 0.45, which pays for haste: 75–85 % at 45°, unchanged; the early stages were slower
- [x] The diagnosis, on 200 new starts between 28 and 40 m and 35–45°: the wall is one-sided. From $`x \lt 0`$ the agents docked 81–99 of 100; from $`x \gt 0`$, 5–23 of 100. Dockings also peaked at 0.20–0.22 m/s, so speed was never the cause
- [x] The trajectories: one manoeuvre learned, over the sphere at $`y \approx 25\ \text{m}`$ towards $`+x`$ and down the cone. From $`x \gt 0`$ it loops and cuts into the sphere about 30° off the axis
- [x] The test: the same network flown as in a mirror on the $`x \gt 0`$ starts ($`x, \dot{x}, u_x`$ with their signs flipped) docks 172 and 161 of 194, against 47 and 9 unmirrored. The Clohessy-Wiltshire equations are not symmetric under $`x \to -x`$, but over one approach the difference costs a few per cent, not seventy

*Lesson.* Two changes to the reward did not move the wall because the wall
was not in the reward. Its cause was in the learning: PPO found the
manoeuvre on one side and the network did not carry it to the other. The
measurement that found it, success split by side, was cheap and should have
come before the remedies; an average over both sides hid a 90 % and a 15 %.

## Step 23 — A mirror for the two sides

- [x] `mirror_symmetry` (off by default; `configs/ppo_corridor_mirror.yaml`): a start outside the keep-out sphere with $`x \gt 0`$ is shown to the agent as on the side $`x \lt 0`$, with $`x`$, $`\dot{x}`$ and $`u_x`$ flipped for the whole episode; the side is fixed at the start, since the manoeuvre crosses the axis on its way into the cone
- [x] First version, every start mirrored and the true side given as an extra input: stuck at the first stage, 60–65 % and 15 % at 4 million steps. Next to the port the Coriolis term pushes the chaser out of the cone always towards the same side; there the problem is not symmetric
- [x] Starts inside the sphere never mirrored, the side still an input: stuck at the first stage again, although the task there was unchanged. The constant extra input changed the network and its training
- [x] No extra input: within 20 m the training is now the one of Step 21, bit for bit, a test pins it. Two seeds of 10 million steps: **seed 1 passed the wall**, the first agent from scratch to do it, and reached 85° at 60 m; from new starts at 45° it docks 89/99 from $`x \lt 0`$ and 97/101 from $`x \gt 0`$, at 75° 74/99 and 94/101. Seed 0 stayed at 45°, its two sides now even but weak, 33/99 and 48/101
- [x] Three seeds of 32 million steps, as in Step 15: the curriculum reached 105°, 155° and 155° (Step 15: 55°, 145°, 155°); seed 1's training success fell to nearly zero at 26 million steps and partly came back
- [x] On the 200 unseen starts of Step 15 (`scripts/corridor_eval.py`, which now flies the V-bar procedure without the mirror: it commands the true thrust, and through the mirror it docked 106 of 200): **152, 126 and 45 of 200** (best on validation: 143, 120, 58), against 153, 121 and 71 in Step 15. By direction, for the best seed: 97/97 within 90°, 44/56 between 90° and 135° (Step 15: 51), 11/47 from behind the station (Step 15: 5). The procedure docks 200 of 200

*Lesson.* The mirror removed the wall it was built for: the curriculum now
passes 45° on every seed, and the two sides dock alike. On the whole task it
did not pay: the best seed docks 152 of 200, as Step 15 did, and from behind
the station it still fails four times in five. The wall at 45° was one
obstacle of several, and the next one, behind the station, is the one Step 15
already found. Two smaller lessons: a symmetry helps only where it holds,
since next to the port it does not and imposing it there broke the easy stage;
and a change meant to be invisible must be made invisible, as an extra input
that looked harmless was not. The evaluation had its own trap: a classical
controller flown through the agent's mirror.

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

Reinforcement learning from scratch reached 153 of 200 (Step 15). The side
study located what stops it: behind the station the best side jumps, which a
continuous policy cannot do, and PPO loses manoeuvres it has learned; starts
on one side only, which remove the jump, went to 175° before the stage was
lost. Each has a remedy that stays within reinforcement learning.

- [ ] A policy with a discrete choice of side, taken by the policy itself at
  the start and kept, next to its continuous thrust, with the two curricula of
  Step 15.
- [ ] The best policy on validation kept (`BestModel`), and a learning rate
  that decays, so that a stage reached is not lost.
- [ ] If successes stay too rare: an off-policy algorithm with Hindsight
  Experience Replay, which relabels each failed approach as a success towards
  the point it reached.

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

# FONPR Modernization — Specification Set

This document enumerates the specifications required to drive the modernization
sprint: a training simulator, an honest evaluation harness, a consolidated
software stack, and a reproducible local deployment. Each spec section states
its decisions, defaults, and acceptance criteria. Items marked **[DECISION]**
need owner sign-off before or during implementation; each carries a
recommended default so work is never blocked on them.

Guiding principle: **fix the science before the software.** Every spec below
serves one of two goals — make the agent trainable (simulator) or make it
judgeable (baselines + evaluation).

---

## S1. Simulation Environment (`FONPR_SimEnv`)

The core deliverable. An offline, seedable, fast Gymnasium environment that is
API-identical to the live `FONPR_Env`, so a policy trained in simulation can be
deployed against the live advisor/actuator loop without modification.

### S1.1 Interface contract

| Item | Spec |
|---|---|
| API | Gymnasium `Env` (`reset(seed, options)`, `step(action)` returning `(obs, reward, terminated, truncated, info)`) |
| Observation space | `Box(shape=(samples, 3), dtype=float32)` — throughput (bytes/s), large-instance-on flag, small-instance-on flag. Identical to live env. With `time.include_time_features` (ADR-0002, default off): shape `(samples, 5)`, appending per-tick `sin/cos` of time-of-day as columns 3–4. |
| Action space | `Discrete(3)`: 0 = NOOP, 1 = transition to Large, 2 = transition to Small. Identical to live env. |
| `info` dict keys | `offered_load`, `served_load`, `slo_violation` (violation minutes), `instance_type`, `in_transition`, `step_cost_usd`, `step_penalty_usd`, `step_energy_wh` (S13; 0.0 under the price model), `offered_series`, `served_series`, `capacity_series` (per-tick arrays, consumed by the oracle and plots), `action_applied` — required, stable names. (`step_revenue_usd` was removed with the revenue term, ADR-0001/D6.) |
| Determinism | Same seed + same config ⇒ bit-identical trajectories. Enforced by test. |

### S1.2 Traffic model (offered load)

Synthetic offered-load generator, composable from:

1. **Diurnal component**: sinusoid with configurable base load, amplitude,
   period (default 24 sim-hours), and phase.
2. **Noise component**: multiplicative Gaussian noise (default σ = 5% of
   instantaneous load), plus optional AR(1) correlation.
3. **Burst process**: Poisson arrivals (default rate: 2/day) of load spikes
   with configurable magnitude distribution (default: lognormal, 1.5–4× base)
   and duration distribution (default: 10–45 sim-minutes).
4. **Drift component** (optional, for non-stationarity scenarios): slow linear
   or step change in base load across episodes.

All parameters live in a single config dataclass (S1.5). All randomness flows
from the env seed.

### S1.3 Plant model (how the cluster responds)

* **Capacity**: each instance type has a throughput capacity (bytes/s).
  Defaults calibrated so the Large instance comfortably serves peak diurnal
  load and the Small instance serves ~40% of peak.
* **Saturation**: `served = min(offered, capacity)`. Load above capacity is
  dropped and counts toward SLO violation (S2).
* **Transition dynamics**: an instance transition takes a configurable lag
  (default 5 sim-minutes, representing commit → Flux reconcile → reschedule).
  During transition, capacity is degraded (default: min of the two types) —
  this is the cost a proactive agent should learn to anticipate.
* **Cost accrual**: per-step infrastructure cost from the pricing table
  (S2.3), charged for whichever instance(s) are running, including both
  during transition overlap.

### S1.4 Time model

* One `step()` = one observation period (default 15 sim-minutes, matching the
  live env's cadence).
* Observation window = `window` sim-minutes back-sampled at `sample_rate`
  (defaults mirror live env).
* Episode length: default 7 sim-days (672 steps). `truncated=True` at episode
  end; never `terminated` (continuous task, matching live semantics).
* Wall-clock requirement: a 500k-step training rollout completes in **single-
  digit minutes on one laptop core**; hard floor 5,000 env steps/second
  single-core (measured at implementation: ~9,000/s).

### S1.5 Configuration

* Single `SimConfig` dataclass, YAML-loadable, with documented defaults for
  every parameter above. The eval harness (S4) records the full config next
  to every result.

### S1.6 Trace-replay mode

* Alternative offered-load source: replay from file instead of synthesis.
* **[DECISION] Trace format** — recommended default: **Parquet** with schema
  `(timestamp: datetime64[ns, UTC], throughput_bytes_per_sec: float64)`, CSV
  accepted as import format and converted. Includes a
  `fonpr trace pull` utility that exports the schema from a live Prometheus
  endpoint (reusing the existing advisor queries).
* Resampling rule: linear interpolation onto the sim timestep grid; gaps
  longer than 3× the timestep are an error, not silently interpolated.

### S1.7 Acceptance criteria

* Passes Gymnasium `check_env`.
* Determinism test passes (S8).
* A random policy and each baseline (S3) run to episode completion.
* Throughput benchmark meets S1.4 speed target.

---

## S2. Reward & Economics

**Ratified (ADR-0001): the revenue-per-byte concept is eliminated.** There is
no synthetic revenue term anywhere in the system. The objective is
**cost-minimal SLO compliance**: run the network as cheaply as possible while
meeting service-level objectives. This is what an operator actually optimizes,
and it removes an invented constant from every result.

### S2.1 Reward definition

Reward per step = `−(infra_cost + slo_penalty)`, where:

* **SLO definition (sim)**: a step is in violation when
  `served / offered < slo_target` (default 0.995). Violation minutes
  accumulate.
* **Penalty structure**: per-violation-minute penalty in dollars (default:
  `slo_penalty_rate` = 20× the hourly cost of the Large instance, prorated
  per minute — an SLA-credit-like structure where breaching is always far
  more expensive than provisioning). Configurable.
* **[DECISION D5] SLO threshold and penalty magnitude** — defaults above are
  ratified as working values; revisable by the domain owner via ADR as real
  SLA structures become available.
* **Live-env mapping (Phase 2, spec'd now so the sim reward is forward
  compatible)**: served/offered proxy from UPF throughput vs. UE-side offered
  load; extension points reserved for AMF registration success rate and
  session-establishment latency once exporters for open5gs metrics are in
  the local stack (S9).

Note the reward is always ≤ 0; policies are compared on total operating cost
(S4), where lower is better. The NOOP-on-Small baseline remains meaningful:
minimal infra cost, heavy penalties at peak load.

### S2.3 Cost model

* Pricing table moves from code (`ec2_cost_calculator`) to config, keyed by
  instance-type label, in $/hour. Unknown type = hard error (current
  behavior preserved).
* Transition double-billing: while both node groups are active, both accrue
  cost. This is deliberate — it is the economic pressure against action churn.

---

## S3. Baseline Policies

All policies implement a common interface: `Policy.act(obs) -> action`, no
learning state required. Baselines are first-class citizens of the eval
harness — the RL agent is judged only relative to them.

1. **B0 — NOOP**: always action 0. Floor reference.
2. **B1 — Threshold heuristic (V0-style)**: scale up when observed
   throughput > `up_threshold` × small capacity for `up_patience` steps;
   scale down below `down_threshold` × small capacity for `down_patience`
   steps. Hysteresis band and cooldown (default: no action within 4 steps of
   the last) are mandatory config.
3. **B2 — Reactive autoscaler-equivalent**: target-utilization rule
   (default 70%) on the current instance's capacity, single-step decision, no
   patience — models HPA-like behavior.
4. **B3 — Forecast-then-act**: seasonal-naive forecaster (predict next
   horizon = same time yesterday) with optional Holt-Winters upgrade;
   choose the cheapest instance whose capacity covers the forecast × safety
   margin (default 1.15).
5. **Oracle (upper bound)**: hindsight-optimal sizing computed on the full
   episode trace via dynamic programming over the 3-action space, including
   transition costs. Used for regret; never presented as a deployable policy.

Acceptance: each baseline has unit tests over hand-constructed traces with
known-correct decisions.

---

## S4. Evaluation Harness

### S4.1 Metrics (per episode)

* Total operating cost ($ = infra cost + SLO penalties) — headline, lower is
  better.
* SLO violation minutes and violation fraction.
* Action churn (count of non-NOOP actions).
* Regret vs. oracle ($ and %).
* Mean served/offered ratio.

### S4.2 Protocol

* Scenario suite: `steady`, `diurnal`, `diurnal+bursty` (default headline),
  `drift`, and (when traces exist) `trace-replay`.
* Econ variant: every scenario runs under the price-table econ by default,
  or under the S13 power model with `energy: true` in the eval config
  (`fonpr eval --energy`). Costs remain USD either way; `energy_kwh` is
  nonzero (and reported) only for energy-variant runs. Traffic, plant, and
  seeds are identical across variants, so demand traces stay comparable.
* N = 20 evaluation seeds per scenario, disjoint from training seeds.
* Report mean ± 95% CI across seeds. No single-seed claims, anywhere.

### S4.3 Outputs & CLI

* One command: `python -m fonpr.eval --config <yaml>` →
  `results/<run-id>/` containing `results.csv`, `results.md` (comparison
  table), `plots/*.png` (profit curves, load-vs-capacity timeline per
  policy), and the frozen config + git SHA for reproducibility.
* The README's headline table is generated by this command, never
  hand-edited.

### S4.4 Acceptance criterion for the whole project (the honest bar)

The learned agent is declared useful **only if** it beats B1 *and* B3 on
total operating cost with non-overlapping confidence intervals on the
`diurnal+bursty` scenario. If B3 wins, that result is reported with equal
prominence — a negative finding is a valid outcome of this work.

---

## S5. Learned Agent

* **[DECISION] RL library** — recommended: **Stable-Baselines3** (maintained,
  boring, sufficient; the networks are tiny and RLlib's distributed machinery
  is dead weight here). Alternative: CleanRL if maximal legibility is
  preferred over ergonomics.
* **[DECISION] Algorithm** — recommended: **DQN** as the ported agent
  (discrete 3-action space is DQN's home turf; SAC's continuous-control
  strengths are irrelevant here). Note: DQN's discreteness constraint applies
  to the *action* space only — continuous observations are exactly what the
  Q-network's function approximation handles (see ADR-0001). The SAC/RLlib
  and DQN/tf-agents code paths are archived, not deleted (S7).
* Network: MLP, 2 × 64 hidden units (config-exposed). Observation flattened.
* Training budget: 500k sim steps default; checkpoint every 50k; final +
  best-eval checkpoints saved in SB3 `.zip` format with the `SimConfig`
  embedded alongside.
* Seeds: 5 training seeds minimum; eval per S4.2.

---

## S6. Actuation Layer

* `Actuator` abstract base: `apply(requested_actions: dict) -> ActuationResult`,
  idempotent per call; `ActuationResult` carries success flag, applied values,
  and backend reference (e.g., commit SHA).
* Backends:
  1. **`GitHubFluxActuator`** — current behavior (PyGithub commit to values
     file; Flux reconciles). Unchanged semantics.
  2. **`HelmActuator`** — direct `helm upgrade --reuse-values --set ...`
     against the current kube-context. For local dev.
  3. **`DryRunActuator`** — logs the would-be change, returns success.
     This is **shadow mode**, and it is the default in every config template.
* Token sourcing precedence: `GH_TOKEN` env var → token file path →
  AWS Secrets Manager (retained as one optional backend, no longer a hard
  dependency; `boto3` becomes an optional extra).
* Error semantics: actuation failure is surfaced to the agent loop as a
  no-op step with a logged warning — never a crash of the control loop.

---

## S7. Packaging & Repository Layout

* Single `pyproject.toml`; the five `requirements*.txt` files and per-agent
  Dockerfiles collapse to one image with the agent selected by CLI arg.
* **[DECISION] Python floor** — recommended: **3.11** (3.12 preferred if all
  deps clear; no support below 3.11).
* Dependency policy: Gymnasium, SB3, numpy/pandas, prometheus-api-client,
  PyGithub, PyYAML. **Removed from the install path**: tensorflow, tf-agents,
  dm-reverb/reverb, ray[rllib], google-vizier, nose.
* Layout:
  ```
  fonpr/
    sim/          # S1: SimEnv, traffic + plant models, trace replay
    envs/         # live FONPR_Env (refactored onto shared interfaces)
    policies/     # S3 baselines + S5 agent wrappers
    eval/         # S4 harness
    actuators/    # S6
    advisors/     # existing, + fixture-recording support
    utilities/
  archive/        # tf_infrastructure, agent_bbo, agent_dqn (tf-agents),
                  # agent_sac (rllib) — preserved verbatim, excluded from
                  # packaging and CI
  docs/
  tests/
  ```
* Tooling: `ruff` (lint + format), type hints required on all new/refactored
  public functions.
* Entry points: `fonpr train`, `fonpr eval`, `fonpr run-agent`,
  `fonpr trace pull`.

---

## S8. Testing & CI

* **Advisor**: recorded Prometheus HTTP fixtures (JSON responses checked into
  `tests/fixtures/`) — no live server needed. Fixture recording script
  included.
* **Action handler / actuators**: mocked GitHub repo object; asserts exact
  YAML mutation; `DryRunActuator` round-trip.
* **SimEnv**: determinism (same seed ⇒ identical trajectory), Gymnasium
  `check_env`, plant-model edge cases (saturation, transition overlap
  billing).
* **Baselines**: hand-built traces with known-correct decisions per policy.
* **Smoke train**: DQN for 2k steps in < 60 s must produce a loadable
  checkpoint.
* CI: GitHub Actions on PR + main — ruff, pytest, smoke train. Python
  {floor, floor+1}. Target < 5 minutes total.
* Coverage bar: new/refactored modules ≥ 80%; `archive/` excluded.

---

## S9. Local Stack (`make local-stack`)

Reproducible laptop deployment of the full live loop (the target the sim
trains for). Assumes Linux or WSL2, Docker, 16 GB RAM minimum.

* **kind** cluster, 3 nodes: control plane + 2 workers labeled
  `node.kubernetes.io/instance-type=m4.xlarge` and `=t3.medium` respectively
  (the live env reads exactly this label, so observation code runs
  unmodified).
* **open5gs** via Gradiant/Openverso Helm charts (pinned version in the
  Makefile), 5G SA core profile.
* **UERANSIM** gNB + configurable UE count generating iperf-driven user-plane
  traffic through the UPF; a `make load-profile` target replays a diurnal
  pattern in accelerated time.
* **kube-prometheus-stack** pinned version; scrape interval 30 s.
* **Verification gate**: `make verify` runs every query in
  `fonpr/utilities/prom_queries.py` against the local Prometheus and fails
  unless all return non-empty series.
* Resource budget: full stack ≤ 10 GB RAM. Documented teardown
  (`make clean-stack`).

---

## S10. Decision Register

All decisions below were **ratified by the owner on 2026-08-01** and recorded
in ADR-0001. They are closed; changing one requires a superseding ADR.

| # | Decision | Ratified value |
|---|---|---|
| D1 | RL library | Stable-Baselines3 |
| D2 | Ported algorithm | ~~DQN~~ → **PPO** (superseded by ADR-0003; DQN retained for comparison, SAC + BBO archived) |
| D3 | Python floor | 3.11 |
| D4 | Trace file format | Parquet (CSV import) |
| D5 | SLO target + penalty constants | 0.995 / 20× Large-instance hourly cost, per violation-minute (working values; owner may revise via ADR) |
| D6 | Revenue-per-byte | **Eliminated entirely.** Objective is cost-minimal SLO compliance (S2). |
| D7 | Instance capacity calibration (sim) | Large = 1.2× peak diurnal, Small = 0.4× (working values) |
| D8 | Legacy agents | Archived in `archive/`, excluded from packaging and CI |

## S11. Non-Goals (this sprint)

* No multi-dimensional action space (Phase 5 of the roadmap).
* No offline RL / production trace training (Phase 3) — trace-replay *replays*
  load; it does not train from logged actions.
* No autonomous writes to any production values file: shadow mode
  (`DryRunActuator`) is the ceiling for anything cluster-facing this sprint.
* No changes to the deployed DockerHub images or the existing k8s manifests
  until the new stack passes S4's acceptance bar.

## S12. Live Control Loop (`fonpr run-agent`)

The deployment counterpart of one sim step: Advisor → Policy → Actuator on
a fixed cadence. Any Policy — baseline or trained checkpoint — runs here
unchanged, because the observation contract is S1.1's exactly.

* **LoopConfig**: frozen dataclass + strict YAML (canonical pattern);
  embeds a `SimConfig` so live thresholds use the same vocabulary the
  policies were benchmarked with. `actuator: dry-run` is the default and
  the only value in any committed config template.
* **ThroughputAdvisor**: builds `(samples, 3)` observations from live
  Prometheus — UPF served throughput (range query, interpolated onto the
  sample grid) plus instance-type flags from the UPF's node labels. Empty
  query results raise; they never silently produce zeros.
* **Action mapping**: NOOP → no actuation; LARGE/SMALL → a request pinning
  the UPF to the corresponding instance-type node group via nodeSelector.
* **Loop semantics**: one sanctioned broad catch — a failed iteration
  (observation or actuation) is logged and skipped, never kills the loop
  and never crashes into actuation. The loop returns a per-iteration
  audit log. `--once` runs a single iteration for verification.
* **Verification**: correctness against a real cluster is gated by S9's
  `make verify` (this environment cannot host one); the advisor and loop
  are contract-tested against recorded fixtures per S8.

## S13. Energy Variant (the application bet)

The identical control problem with infrastructure cost derived from
**measured power** instead of a cloud price table. The objective invariant
(cost-minimal SLO compliance, CLAUDE.md / ADR-0001/D6) is untouched: only
the derivation of `infra_cost` changes, so every policy, the oracle, and
the eval harness work unmodified.

* **PowerConfig** (optional `econ.power` section; template
  `configs/sim-energy.yaml`): per-instance-type `idle_watts` / `max_watts`
  and `electricity_usd_per_kwh`. Load-proportional server model:
  `watts = idle + (max − idle) × utilization`, utilization = served / the
  serving node's nominal capacity. During a transition the serving node
  draws load-proportional power while the co-billed node idles.
* **Costing is single-sourced** (`fonpr/sim/costing.py`) and consumed by
  both the env and the oracle DP, so regret stays exact under either cost
  model — enforced by an oracle/env consistency test.
* **SLO penalty** anchors to the hungriest tier's max-draw hourly cost
  (D5's multiplier unchanged), preserving the breach ≫ provisioning
  structure in watt-denominated economics.
* **Reporting**: `info.step_energy_wh` (0.0 under the price model);
  eval metrics gain `energy_kwh`. Results become kWh-denominated exactly
  when the config says so — no separate code path. The S4 harness runs the
  full scenario suite under this econ via `energy: true` / `fonpr eval
  --energy`, with an Energy (kWh) column added to `results.md`.
* **Calibration**: default watts are working placeholders. The lab rig
  measures real draw (smart plug / RAPL) under `make load-profile` and
  replaces them; that closes the "every term measurable" loop.
* **Why this exists**: cell-sleep / capacity-scaling energy saving is the
  industry's proven instance of this exact control problem; this section
  makes FONPR's twin speak its language natively.

## S14. Pool Plant Variant — replica-count actions (ADR-0004)

A homogeneous pool of small nodes where the action selects the **target
node count** directly. Selected by the optional `plant.pool` config
section (template `configs/sim-pool.yaml`); when absent, the binary
two-tier plant (S1.3) is used unchanged, keeping every recorded benchmark
reproducible.

* **PoolConfig**: `node_type`, `node_capacity_bytes_per_sec`,
  `min_nodes` >= 1, `max_nodes`, `initial_nodes`, `transition_lag_minutes`.
  Defaults: t3.medium at 8e6 B/s, counts 1..8, initial 1, lag 5 min —
  3 nodes match the legacy large tier's capacity, so D7 calibration holds.
* **Action space**: `Discrete(max_nodes - min_nodes + 1)`; action `a`
  means target count `min_nodes + a`. Choosing the current count is the
  no-op (there is no separate NOOP action); `action_applied` is True only
  when a resize actually starts.
* **Transition semantics** (mirrors S1.3's economics): a resize from `n`
  to `m` takes the lag; during it, capacity stays at the *old* count
  (`n x c` — booting nodes are not ready; draining nodes still serve) and
  billing covers `max(n, m)` nodes (booting nodes bill from launch;
  draining nodes bill until drained). After the lag, capacity and billing
  are `m`. Resize requests during a transition are ignored.
* **Observation**: `(samples, 3)` float32 — [served throughput,
  current_count / max_nodes, target_count / max_nodes] (the two count
  columns are equal outside a transition). ADR-0002 time features append
  as columns 3-4 exactly as in the binary variant.
* **Costing**: single-sourced in `fonpr/sim/costing.py`. Price model:
  `billed_count x hourly(node_type)`. Energy model (S13): serving nodes
  share load evenly (`util = served / (active_count x c)`), each drawing
  load-proportional power; transitional extra nodes idle.
* **Baselines** (same names, so S4 reporting is variant-agnostic):
  noop holds the initial count; threshold steps +/-1 on pool-utilization
  hysteresis with patience and cooldown; reactive is the literal HPA
  formula `ceil(throughput / (target_util x c))`; forecast sizes the
  count to seasonal-naive forecast x margin.
* **Oracle**: identical backward DP over the K count states with the
  transition semantics above; the planned-vs-realized consistency
  assertion (1e-6) applies unchanged.
* **Harness**: `energy`-style switch — `pool: true` in the eval config /
  `fonpr eval --pool`; composes with `--energy`. Scenario traffic, seeds,
  and protocol are identical, but costs are NOT comparable to the binary
  variant (different fleet economics): pool results compare only within
  pool bundles.
* **Hypothesis under test** (the reason this section exists): graded
  actions turn cyclic scale-down into a chain of single-step deviations
  with immediate reward. A 5-seed PPO campaign per S5 decides; if
  learners still converge to max-provisioning, that negative result is
  reported with equal prominence per S4.4.

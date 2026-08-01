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
| Observation space | `Box(shape=(samples, 3), dtype=float32)` — throughput (bytes/s), large-instance-on flag, small-instance-on flag. Identical to live env. |
| Action space | `Discrete(3)`: 0 = NOOP, 1 = transition to Large, 2 = transition to Small. Identical to live env. |
| `info` dict keys | `offered_load`, `served_load`, `slo_violation`, `instance_type`, `in_transition`, `step_cost_usd`, `step_revenue_usd` — required, stable names. |
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
* Wall-clock target: ≥ 10,000 steps/second single-core, so a full training
  run completes in minutes.

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

### S2.1 Reward v1 — parity mode

Reproduces the current live-env formula (revenue-per-byte × served throughput
− infra cost per step) so simulator results are comparable to the original
design. Constants sourced from the existing code, surfaced into config.

### S2.2 Reward v2 — SLO-based (the real objective)

Reward = `revenue − infra_cost − slo_penalty`, where:

* **SLO definition (sim)**: a step is in violation when
  `served / offered < slo_target` (default 0.995). Violation minutes
  accumulate.
* **Penalty structure**: per-violation-minute penalty (default: 20× the
  per-minute revenue at base load — an SLA-credit-like structure where
  breaching costs far more than serving). Configurable.
* **[DECISION] SLO thresholds and penalty magnitudes** are domain calls —
  defaults above are placeholders for the owner to ratify or replace.
* **Live-env mapping (Phase 2, spec'd now so the sim reward is forward
  compatible)**: served/offered proxy from UPF throughput vs. UE-side offered
  load; extension points reserved for AMF registration success rate and
  session-establishment latency once exporters for open5gs metrics are in
  the local stack (S9).

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

* Cumulative profit ($) — headline.
* SLO violation minutes and violation fraction.
* Action churn (count of non-NOOP actions).
* Regret vs. oracle ($ and %).
* Mean served/offered ratio.

### S4.2 Protocol

* Scenario suite: `steady`, `diurnal`, `diurnal+bursty` (default headline),
  `drift`, and (when traces exist) `trace-replay`.
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
cumulative profit with non-overlapping confidence intervals on the
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
  strengths are irrelevant here). The SAC/RLlib and DQN/tf-agents code paths
  are archived, not deleted (S7).
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

Consolidated list of the **[DECISION]** items. Defaults apply unless
overridden; none block the start of work.

| # | Decision | Recommended default | Needs |
|---|---|---|---|
| D1 | RL library | Stable-Baselines3 | sign-off |
| D2 | Ported algorithm | DQN (archive SAC + BBO) | sign-off |
| D3 | Python floor | 3.11 | sign-off |
| D4 | Trace file format | Parquet (CSV import) | sign-off |
| D5 | SLO target + penalty constants | 0.995 / 20× per-minute revenue | **domain owner input** |
| D6 | Revenue-per-byte (v1 parity constant) | carry existing value into config | domain owner input |
| D7 | Instance capacity calibration (sim) | Large = 1.2× peak diurnal, Small = 0.4× | domain owner input |
| D8 | Archive vs. delete legacy agents | archive (excluded from packaging/CI) | sign-off |

## S11. Non-Goals (this sprint)

* No multi-dimensional action space (Phase 5 of the roadmap).
* No offline RL / production trace training (Phase 3) — trace-replay *replays*
  load; it does not train from logged actions.
* No autonomous writes to any production values file: shadow mode
  (`DryRunActuator`) is the ceiling for anything cluster-facing this sprint.
* No changes to the deployed DockerHub images or the existing k8s manifests
  until the new stack passes S4's acceptance bar.

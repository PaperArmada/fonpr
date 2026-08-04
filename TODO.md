# TODO — ideas parked to protect scope discipline

Per CLAUDE.md: improvement ideas noticed mid-task are recorded here, not
implemented as drive-bys. Each becomes a spec-section-scoped work package
when scheduled.

- Refactor `fonpr/advisors`, `fonpr/action_handler`, `fonpr/utilities` to
  package-relative imports, module-level loggers, and type hints; remove them
  from the ruff exclusion list in `pyproject.toml` as they are cleaned.
- Port `fonpr/envs/live_env.py` onto the shared seam interfaces (S6 actuator
  injection, S2 reward) once the sim stack is proven.
- Replace `tests/test_advisor.py` nose-style test with recorded-fixture tests
  per S8; drop the `nose` dependency.
- Collapse `Dockerfile_V0`/`requirements_v0.txt` into the single-image build
  once the V0 agent is re-expressed as a Policy over the live env.
- S5 multi-seed training campaign (5 seeds) to upgrade the 2026-08-02
  preliminary benchmark to a citable result.
- ~~DQN training reliability study~~ resolved by ADR-0003 (PPO default,
  9/10 vs 4/10 convergence). Remaining research question: why no learner
  finds cyclic scale-down even with a clock signal — probe longer
  training, entropy schedule, or reward shaping; or conclude the
  Discrete(3) action space is the limit and prioritize roadmap Phase 5.
- ~~Energy scenario family in the eval harness~~ done (`fonpr eval
  --energy`; first watt-denominated benchmark recorded in
  docs/benchmarks/2026-08-02-energy-baselines/). Follow-up (parked per
  ADR-0005/C3, rung 5): re-run after a lab rig replaces placeholder watts
  with measured draw.
- ~~ADR-0004 replica-count action space~~ accepted, implemented (S14),
  and hypothesis-tested (docs/benchmarks/2026-08-02-campaign-pool-*).
  Open follow-ups: (a) owner decision on whether S4.4 should specify the
  paired per-eval-seed test for shared-seed protocols; (b) the surviving
  research question — learners are reactive, never anticipatory: no
  cyclic scale-down on clean diurnal even with a clock and graded
  actions; (c) pool + energy (--pool --energy) campaign = the cell-sleep
  setting proper.
- ~~Milan (Telecom Italia Big Data Challenge) trace converter~~ done
  (S1.6: `fonpr trace convert-milan`, ODbL attribution, tests). Follow-up
  is the recorded campaign below.
- Milan trace campaign (ADR-0005 rung 4): run the S4 protocol over
  converted Milan traces (business, residential, citywide clusters) and
  commit the first real-demand benchmark bundle. Requires the source TSVs
  (Harvard Dataverse doi:10.7910/DVN/EGZHFV).
- ~~MPC baseline (ADR-0005 rung 2; extends S3)~~ done (B4:
  `fonpr/policies/mpc.py`, S3 amendment, oracle-equivalence and
  saturation-escape contract tests).
- Miscalibration-robustness campaign (ADR-0005 rung 3; extends S4):
  evaluate frozen policies under perturbed twins — capacity calibration
  (D7 ± 30%), transition lag, observation noise on the served-throughput
  channel — and report ranking stability, not just point costs. Spec
  section lands with the implementation.
- S9 local stack (`make local-stack`): parked per ADR-0005/C3
  (simulation-first; hardware waits for rung 4). Machine notes from the
  2026-08-04 viability probe: WSL2 box has Docker 29.x, 31 GB RAM, TUN
  present; blockers logged were cgroup v1, no kind/helm, and the
  EKS-specific `node_sizing_query` in `prom_queries.py` that can never
  return non-empty on kind (S9 verify-gate conflict needing an owner
  call when unparked).
- S12 live-loop actuation mapping for pool actions (S14): translate a
  target node count to a replicaCount/ASG-size actuation request; the
  current mapping only covers binary LARGE/SMALL nodeSelector swaps.

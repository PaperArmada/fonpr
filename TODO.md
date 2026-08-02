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
  docs/benchmarks/2026-08-02-energy-baselines/). Follow-up: re-run after
  the lab rig replaces placeholder watts with measured draw.
- ~~ADR-0004 replica-count action space~~ accepted, implemented (S14),
  and hypothesis-tested (docs/benchmarks/2026-08-02-campaign-pool-*).
  Open follow-ups: (a) owner decision on whether S4.4 should specify the
  paired per-eval-seed test for shared-seed protocols; (b) the surviving
  research question — learners are reactive, never anticipatory: no
  cyclic scale-down on clean diurnal even with a clock and graded
  actions; (c) pool + energy (--pool --energy) campaign = the cell-sleep
  setting proper.
- Milan (Telecom Italia Big Data Challenge) trace converter:
  `fonpr trace convert-milan` — aggregate cell clusters (business,
  residential, citywide, event-day), interpolate 10-min source to tick
  granularity, scale peak to capacity calibration, emit trace-schema
  Parquet with train/eval week split; ODbL attribution note.
- S9 local stack (`make local-stack`): needs a Docker-capable machine;
  build and verify there rather than committing an untested Makefile.
- S12 live-loop actuation mapping for pool actions (S14): translate a
  target node count to a replicaCount/ASG-size actuation request; the
  current mapping only covers binary LARGE/SMALL nodeSelector swaps.

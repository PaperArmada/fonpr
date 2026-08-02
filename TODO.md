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

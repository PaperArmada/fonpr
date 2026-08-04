# FONPR — Operating Doctrine

RL/control agents for cost-optimal operation of a cloud-native 5G core.
`docs/SPECS.md` defines **what** is being built. This file defines **how**
work is done here. These rules are invariant: do not relitigate them in-session.
Changing one requires a superseding ADR in `docs/adr/` with owner sign-off,
landed *before* any code that depends on the change.

## Authority hierarchy

1. `docs/adr/` — closed decisions, append-only. Never contradict an accepted
   ADR. Ratified decisions live in ADR-0001; check it before proposing any
   library, algorithm, or architecture choice.
2. `docs/SPECS.md` — the behavioral contract. If code and spec disagree, the
   code is wrong. Spec amendments land in the same commit as (never after)
   the code implementing them.
3. This file — process rules.
4. Docstrings and comments — explanation only, never authority.

## Architecture invariants

- The system is four seams: **Advisor** (observe) → **Env** (frame) →
  **Policy** (decide) → **Actuator** (act). All work happens *inside* a seam.
  Changing a seam's interface is an ADR-level event.
- `fonpr/sim` is the training ground; the live env is a deployment target.
  Nothing ever trains against live infrastructure.
- `DryRunActuator` is the default in every config template. Autonomous writes
  to any live values file are forbidden absent explicit, per-instance owner
  instruction.
- The objective is **cost-minimal SLO compliance**. There is no revenue term
  anywhere in the system (ADR-0001/D6). Do not reintroduce one.
- All randomness flows from explicit seeds. Same seed + same config ⇒
  identical output. A change that breaks determinism is a bug regardless of
  what it improves.
- No performance claims exist outside `fonpr eval` output artifacts. Result
  tables and plots are generated, never hand-edited.

## One way to do each thing

This codebase deliberately has one blessed pattern per concern. Imitate the
canonical module; do not introduce a second pattern:

| Concern | The one way |
|---|---|
| Config | Frozen dataclass + YAML loader (canonical: `fonpr/sim/config.py`) |
| CLI | Subcommand under the single `fonpr` entry point |
| Logging | Stdlib `logging`, module-level logger; `print` is forbidden in library code |
| Errors | Raise specific exceptions; only the agent control loop catches broadly |
| Tests | `pytest`; fixtures in `tests/fixtures/`; no network access in tests |
| Randomness | `numpy.random.Generator` passed down from the env seed; no global RNG |

## Scope discipline

- One work package (= one spec section) per branch/PR. Name the spec section
  in the PR body.
- No drive-by refactors, renames, or "while I was here" changes. Record the
  idea in `TODO.md` and move on.
- Adding or upgrading a dependency is an ADR-level event. Prefer stdlib and
  the existing dependency set.
- `archive/` is read-only history: never modify it, import from it, or
  "fix" it. It is excluded from packaging, CI, and coverage.

## Definition of done

Code + tests + spec conformance + `ruff` clean + CI green. A feature without
tests does not exist. A result without a seed, config, git SHA, and
environment (package versions, ADR-0005) attached does not exist.

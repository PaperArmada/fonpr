# ADR-0004: Replica-count action space (pool plant v2)

- **Status**: Accepted
- **Date**: 2026-08-02
- **Decider**: Project owner (approved in session: "Approved; please
  proceed", following the evidence summary below)

## Context

Two independent findings converge on the same limitation:

1. **The RL finding** (ADR-0003 campaigns): a reliable learner (PPO,
   9/10 seeds) with a clock signal (ADR-0002) still converges to
   always-large instead of cyclic scale-down. Under `Discrete(3)` binary
   sizing, the good policy requires one giant, risky leap (a 60% capacity
   cut with transition exposure) that only pays off if reversed hours
   later — an exploration cliff with no gradient toward it.
2. **The energy finding** (2026-08-02 watt-denominated benchmark): under
   load-proportional power the always-large regret shrinks 10x, because
   idle floors dominate and the binary actuator cannot shed them in
   increments. The binding lever is actuator granularity, not schedule.

Both point at the action space, not the observation (ADR-0002 already
falsified observation enrichment as the constraint).

## Decision

Add a **pool plant variant** (S14): a homogeneous pool of small nodes
where the action selects the target node count directly
(`Discrete(max_nodes - min_nodes + 1)`). Selected by an optional
`plant.pool` config section, mirroring the `econ.power` pattern.

- The binary two-tier plant remains the default and is untouched — every
  recorded benchmark stays reproducible bit-for-bit.
- Baselines and the hindsight oracle generalize (HPA-style formulas; DP
  over K count states), so the S4 protocol and S4.4 bar apply unchanged.
- "Shed one replica at night" becomes a single-step deviation with an
  immediate, small, observable reward — the hypothesis under test is that
  this dissolves the exploration cliff. If learners still converge to
  max-provisioning under graded actions, that is a deeper negative result
  and is reported with equal prominence (S4.4).

## Consequences

- Two plant variants now exist behind one seam; the pool variant is the
  research frontier, the binary variant is the frozen reference.
- Continuous actions (and SAC's re-entry, ADR-0003) remain roadmap
  Phase 5, now gated on pool-variant evidence: go continuous only if
  count granularity proves to be the remaining constraint.
- The pool observation replaces the two instance flags with
  normalized current/target counts (columns 1-2); time features stay
  appended at columns 3-4. Policies are variant-specific; the Policy
  seam interface is unchanged.
- Combined with S13 (`--energy --pool`), this is the native language of
  cell-sleep economics — watts x sleepable nodes.

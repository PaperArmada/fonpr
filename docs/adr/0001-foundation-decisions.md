# ADR-0001: Foundation decisions for the modernization effort

- **Status**: Accepted
- **Date**: 2026-08-01
- **Decider**: Project owner (ratified in review of docs/SPECS.md)

## Context

The 2023 codebase explored four agent paradigms across five dependency stacks
with no shared objective definition and no evaluation baseline. Modernization
requires closing the option space so that incremental work — human- or
LLM-executed — composes instead of diverging. These decisions were proposed in
`docs/SPECS.md` (S10) and ratified by the owner.

## Decision

| # | Decision | Value | Rationale |
|---|---|---|---|
| D1 | RL library | Stable-Baselines3 | Maintained, minimal, sufficient for MLP-scale policies; RLlib's distributed machinery and tf-agents' Reverb dependency are dead weight at this scale. |
| D2 | Algorithm | DQN | The action space is `Discrete(3)` — DQN's native domain. Its discreteness constraint applies to *actions only*: continuous observations are handled by the Q-network's function approximation (this is the entire point of DQN over tabular Q-learning). SAC's continuous-action machinery solves a problem this project does not have. |
| D3 | Python floor | 3.11 | Modern typing and perf; nothing retained requires older. |
| D4 | Trace format | Parquet (CSV accepted at import) | Typed, compact, pandas-native. |
| D5 | SLO constants | `slo_target = 0.995`; penalty = 20× Large-instance hourly cost, prorated per violation-minute | Working values with an SLA-credit structure (breach ≫ provisioning cost). Owner may revise via superseding ADR as real SLA structures become available. |
| D6 | Revenue model | **Revenue-per-byte is eliminated entirely.** The objective is cost-minimal SLO compliance: reward = −(infra cost + SLO penalty). | The revenue constant was invented, not observed; it added a free parameter that could manufacture any desired result. Cost + penalty are both grounded in real quantities (cloud pricing, SLA credits). Owner-confirmed: no attachment to the original concept. |
| D7 | Sim capacity calibration | Large = 1.2× peak diurnal load; Small = 0.4× | Working values ensuring the sizing decision is non-trivial at peak. |
| D8 | Legacy agents | tf-agents/Reverb DQN, RLlib SAC, and Vizier BBO move to `archive/`, excluded from packaging, CI, and coverage | Preserved history without carrying unmaintained, Linux-locked dependencies in the install path. |

## Consequences

- Exactly one learning stack (SB3/DQN) is built, tested, and benchmarked.
- Every result is denominated in dollars of operating cost, with no synthetic
  revenue parameter to tune.
- Reintroducing a revenue term, a second RL framework, or a legacy agent into
  the active codebase requires a superseding ADR — it is not a code-review
  discussion.
- If a future action space goes multi-dimensional/continuous (roadmap Phase
  5), D2 is the ADR to supersede.

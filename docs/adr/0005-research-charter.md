# ADR-0005: Research charter: twin-first evaluation of RL for telco operations

- **Status**: Accepted
- **Date**: 2026-08-04
- **Decider**: Project owner

## Context

The specs define build contracts and an in-sim acceptance bar (S4.4), but the
question the project exists to answer is recorded nowhere: whether RL methods
are viable and advantageous for real-world workloads on cellular
infrastructure, in terms that create operational value for providers. Without
that statement, "done" collapses to S4.4, which no in-sim result can carry to
a real plant on its own. The 2026-08-02 campaigns supply the first rung: the
pool learner beats both required baselines on the headline scenario at N=60.

Two forces make the charter timely. Research now continues across
heterogeneous environments (cloud sessions and a local workstation), so
result trust needs an explicit environment leg. And the twin's known fidelity
gaps (working-value calibration per D5/D7, a clean observation channel, a
throughput-only SLO) mean in-sim advantage claims need a defined path to
operational meaning.

## Decision

Adopt the following charter. It contextualizes S4.4, which stands unchanged
as rung 1, and alters no seam interface.

| # | Element | Content |
|---|---|---|
| C1 | Thesis | RL control is viable and advantageous for cost-minimal SLO-compliant operation of cellular infrastructure, in terms that survive transfer toward real plants. |
| C2 | Evidence ladder | (1) the S4.4 in-sim bar; (2) advantage over the strongest deployable non-RL competitor, causal MPC (forecast feeding the oracle's DP over the twin); (3) ranking stability under twin miscalibration (capacity, transition lag, observation noise); (4) advantage on real demand traces (Milan, S1.6); (5) live verification (S9/S12) and physical calibration (S13). A claim of operational value must name its rung. |
| C3 | Simulation-first scope | Rungs 1-4 are pure simulation and constitute the current phase. Live-stack bring-up and physical power measurement are out of scope until rung 4 is climbed. |
| C4 | Environment reproducibility | uv is the package manager; `uv.lock` is committed; torch resolves from the CPU wheel index; every result bundle records Python and key package versions in `run_meta.yaml` alongside seed, config, and git SHA. |

## Consequences

- "Useful" for the project means climbing the ladder, not passing S4.4 alone;
  negative findings at any rung are deliverables of equal standing.
- Three work packages join `TODO.md`: an MPC baseline (extends S3), a
  miscalibration-robustness campaign (extends S4), and a recorded Milan
  trace campaign (S4.2 trace-replay).
- The definition of done gains an environment leg: a result without seed,
  config, git SHA, and environment attached does not exist.
- Cross-environment trust is evidence, not assumption: the first bridge
  reproduced the committed pool-confirm60 baseline and oracle rows
  digit-for-digit on the local workstation (2026-08-04).
- Superseding this ADR is required to reorder the ladder, put hardware work
  before rung 4, or drop a rung.

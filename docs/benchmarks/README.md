# Benchmark record

Committed `fonpr eval` bundles (generated, never hand-edited). Each bundle
carries its git SHA, config, and seeds in `run_meta.yaml`.

## 2026-08-02 — preliminary, DQN seed 0

First full benchmark through the new stack: 4 scenarios x 20 eval seeds,
baselines + DQN (500k steps, **one training seed**) + oracle.

**S4.4 verdict: acceptance bar NOT met (preliminary).** The DQN does not
beat both the threshold heuristic and the forecaster with non-overlapping
CIs on `diurnal_bursty` — all four non-trivial policies overlap there
($46–49 against an oracle bound of $41). On the clean `diurnal` scenario
the ordering is decisive: forecast $27.91 < reactive $30.68 < dqn $33.94 <
threshold $34.90, oracle $27.16.

The timeline plot shows why: this DQN converged to "go large immediately
and stay" — it eliminates SLO risk (5 violation minutes, matching the
oracle) but never harvests the night-time savings the forecaster captures.
A rational local optimum under an SLA-credit penalty 20x the instance cost,
and precisely the honest negative finding the harness exists to surface.

Caveats: one training seed (S5 requires 5 for any non-preliminary claim);
default DQN hyperparameters; observation window is a single 15-minute step,
which gives the agent no time-of-day signal — a plausible structural cause
worth investigating before hyperparameter tuning.

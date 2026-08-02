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

## 2026-08-02 — full campaign: 5 seeds x 2 observation variants

S5-conformant campaign (`2026-08-02-campaign-plain/`, `-time/`): 5 training
seeds per variant, 500k steps each, default SB3 DQN hyperparameters;
4 scenarios x 20 eval seeds; oracle costs verified bit-identical across
variants (traffic and plant are unaffected by observation enrichment).

**S4.4 verdict: bar decisively NOT met.** Mean total cost on `diurnal`
per training seed (baselines: forecast 27.91, reactive 30.68, threshold
34.90, oracle 27.16):

| Seed | plain | time-features |
|---|---|---|
| 0 | 33.9 | 480.4 |
| 1 | 222.2 | 469.0 |
| 2 | 462.6 | 480.2 |
| 3 | 33.9 | 327.2 |
| 4 | 33.9 | 33.9 |

Three findings, in order of importance:

1. **Training instability dominates everything else.** Plain: 3/5 seeds
   converge to the always-large policy; 2/5 collapse toward NOOP-like
   behavior costing 6-14x more. The single-seed preliminary run above drew
   a lucky seed — the multi-seed protocol exists precisely because of this.
2. **ADR-0002's hypothesis is not supported at this training budget.**
   No seed in either variant learned night-time scale-down (no diurnal
   cost lands between always-large 33.9 and forecast 27.9). The clock
   signal was never exploited.
3. **Time features made training *less* stable, not more** (1/5 vs 3/5
   reaching even the always-large optimum; one seed thrashes at ~93
   actions/episode). Wider observations demand more, not less,
   optimization reliability at a fixed budget.

Conclusion: the binding constraint is DQN training reliability under this
reward structure (large sparse penalties, always-negative returns), not
observation content. Next steps parked in TODO.md: exploration schedule /
hyperparameter study, and evaluating PPO as the more stable on-policy
alternative (algorithm change supersedes ADR-0001/D2 if adopted).
Forecast-then-act remains the standing champion; the honest headline is
unchanged — a 24-hour memory beats a 500k-step learner on this problem.

## 2026-08-02 — PPO campaign: reliability solved, exploration question sharpened

Identical protocol to the DQN campaign
(`2026-08-02-campaign-ppo-plain/`, `-ppo-time/`). Mean total cost on
`diurnal` per training seed:

| Seed | PPO plain | PPO time-features |
|---|---|---|
| 0 | 33.94 | 34.01 |
| 1 | 465.70 | 33.94 |
| 2 | 33.94 | 33.94 |
| 3 | 33.94 | 33.94 |
| 4 | 33.94 | 33.94 |

1. **Training reliability is solved.** 9/10 PPO seeds converge to the
   always-large optimum vs 4/10 for DQN; with time features, **5/5** —
   and converged seeds are bit-consistent (33.94 ± 0.00). This is the
   evidence base for ADR-0003 (PPO replaces DQN as default).
2. **The S4.4 bar is still not met.** Converged learners beat the
   threshold heuristic (33.94 vs 34.90) but lose to reactive (30.68) and
   forecast (27.91) on diurnal, and tie-or-lose on diurnal_bursty (47.69
   vs forecast 47.32, reactive 46.15).
3. **The open question is now precise**: even a reliable learner with a
   clock signal converges to always-large instead of cyclic scale-down.
   Optimizer stability is no longer the suspect; exploration/credit
   assignment toward coordinated multi-step deviations is. Candidate
   probes: longer training, entropy schedule, reward shaping on idle
   headroom, or accepting that this action space is too coarse and moving
   to roadmap Phase 5.

Forecast-then-act remains the standing champion across all recorded
campaigns.

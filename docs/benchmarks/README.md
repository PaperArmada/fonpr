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

## 2026-08-02 — first watt-denominated benchmark (S13 energy variant)

Full S4 protocol under the power-model econ
(`2026-08-02-energy-baselines/`): 4 scenarios x 20 eval seeds, baselines +
oracle + the converged PPO seed-0 checkpoint (trained under the *price*
model — this row is a zero-shot econ-transfer test, not an energy-trained
agent). Placeholder watts per S13; absolute dollars will change when the
lab rig calibrates real draw.

Headline numbers (diurnal / diurnal_bursty, total cost USD):
oracle 3.88 / 6.23, forecast 3.92 / 7.18, reactive 4.03 / 6.64,
ppo 4.16 / 6.50, threshold 4.32 / 6.66, noop 81.63 / 82.74.

Three findings:

1. **Energy economics compress the scale-down prize by an order of
   magnitude.** Under the price table the large:small cost ratio is ~4.8:1;
   under load-proportional power the *idle* ratio is 2:1 and utilization
   dominates, so always-large regret on diurnal falls from ~$7/week
   (25% of oracle) to $0.44/week (11%). Cyclic scale-down is still optimal
   (the oracle takes ~15 actions/week) — it is just worth 10x less. The
   granularity of the actuator, not the schedule, becomes the binding
   lever, which is evidence for the replica-count action-space direction.
2. **The price-trained PPO transfers cleanly and tops the non-oracle table
   on the headline scenario** (diurnal_bursty: 6.50 vs reactive 6.64,
   threshold 6.66, forecast 7.18; lowest regret 0.27). All CIs overlap, so
   S4.4 is NOT met and no superiority claim attaches — but the always-large
   policy it learned is near-optimal under burst-exposed watt economics,
   where forecasting's aggressive scale-down buys little and costs
   violation minutes. First scenario family where the learner is not
   behind the forecaster.
3. **Cost-minimal is not energy-minimal, and the harness now shows the
   gap.** NOOP burns the fewest kWh (19.4) while destroying the SLO; the
   oracle spends 31.9-34.9 kWh. Any future "green" objective is a
   *constraint trade* (kWh vs violation minutes), not a cheaper point on
   the same axis — the energy column exists so that trade is measured, not
   asserted.

## 2026-08-04 — B4 MPC baseline: the deployable ceiling measured, rung 2 cleared

First bundles recorded on the local workstation (ADR-0005/C4: `run_meta.yaml`
now carries the environment; the baseline and oracle rows are bit-identical
to the 2026-08-02 cloud bundles, so cross-bundle comparison is exact).

B4 (S3: B3's seasonal-naive forecaster feeding the oracle's step-cost DP
over the twin's own cost model, receding 24 h horizon, B3's margin) ran
under the full S4 pool protocol (`2026-08-04-mpc-pool-plain/`, 4 scenarios
x 20 seeds) and the confirm60 protocol
(`2026-08-04-mpc-pool-confirm60/`, diurnal_bursty x 60 seeds, identical
traffic to `2026-08-02-campaign-pool-confirm60/`).

Headline (diurnal_bursty, N=60): mpc 38.62 ± 1.53, forecast 38.66 ± 1.53,
reactive 35.32 ± 1.17, oracle 17.36. Recorded PPO seeds on the same
traffic: 33.01-34.20.

1. **Rung 2 (ADR-0005) is cleared on the headline scenario.** Every PPO
   training seed separates from B4 with non-overlapping unpaired 95% CIs
   (worst PPO upper 35.19 vs mpc lower 37.09) — the strictest reading.
   The learner is not merely beating autoscalers; it beats forecast-fed
   DP planning over the twin's own economics.
2. **Transition-aware scheduling is worth almost nothing in this plant.**
   B4 tracks B3 everywhere: exactly the oracle on steady (14.31 — the
   planner's sanity anchor), a 0.4% conservatism premium on clean diurnal
   (15.97 vs 15.90: pricing violations against margined levels pre-scales
   one step early), a wash on bursty and drift with slightly fewer
   actions. With a 5-minute lag and cheap co-billing, *when* you resize
   barely matters; what you can *foresee* does. The oracle's remaining
   edge over every deployable policy is burst foresight, which is not
   deployable.
3. **The forecast family loses to plain reactive on burst-dominated
   traffic** (B2 35.32 vs B3 38.66 / B4 38.62): reacting to actual load
   beats replaying yesterday's bursts, which arrive as phantom forecasts
   at the wrong steps. Against B2 the learner is better on the mean for
   all five seeds but CI-clear for only one — the shared-seed variance
   issue again; the paired-test protocol decision parked in `TODO.md`
   would resolve it.

## 2026-08-02 — pool campaign (S14/ADR-0004): the learner beats the baselines

The ADR-0004 hypothesis test: PPO x 5 training seeds x 2 observation
variants, 500k steps, on the replica-count pool plant
(`2026-08-02-campaign-pool-plain/`, `-pool-time/`); full S4 protocol,
20 eval seeds. Pool costs compare only within pool bundles (S4.2).

Headline scenario `diurnal_bursty` (oracle 17.29): every PPO seed lands
32.03-33.23, ahead of reactive 33.88, forecast 36.97, threshold 40.99.
Clean `diurnal` (oracle 15.29): forecast 15.90 < reactive 19.49 <
threshold 20.35 < every PPO seed 21.36-22.06.

1. **The exploration-cliff diagnosis is vindicated on the headline
   scenario.** With graded actions, all 10/10 seeds converge (tight
   clusters, no collapse — PPO reliability holds at 8 actions) and the
   "go large once" pathology is gone: learners take ~20-25 resize
   actions per episode (plain) and absorb bursts at a third fewer
   violation minutes than the forecaster. First learners in this
   project's history to top both required baselines on the headline
   scenario mean.
2. **S4.4 verdict — split, reported precisely.** Against threshold (B1):
   unpaired 95% CIs are decisively non-overlapping for all 10 seeds —
   that half of the bar is met. Against forecast (B3): unpaired CIs
   overlap marginally (worst PPO upper 34.16 vs forecast lower 33.65,
   a $0.10-$0.51 overlap) because both policies share the same offered
   traces and the CIs carry common between-seed traffic variance. The
   paired per-eval-seed test removes that shared variance: every one of
   the 10 seeds beats forecast by +$3.74 to +$4.94/week with the 95% CI
   of the paired difference excluding zero (14-16/20 eval-seed wins),
   and beats threshold 20/20. Under the letter of S4.2's N=20 unpaired
   CIs the bar is not yet formally met; under the paired analysis it is.
   Whether S4.4 should name the paired test (statistically correct for
   a shared-seed protocol) is an owner decision, not an in-session one.

   **Confirmatory run at N=60** (`2026-08-02-campaign-pool-confirm60/`;
   same protocol, seed offset, and scenario — a superset of the 20):
   every plain PPO seed separates from BOTH baselines with
   non-overlapping unpaired 95% CIs (worst PPO upper 35.19 vs forecast
   lower 37.13 and threshold lower 41.40; oracle 17.36). Under the
   strictest reading of S4.4 — unpaired intervals, all five training
   seeds individually — the learner beats B1 and B3 on diurnal_bursty.
   The N=20 marginal overlap was between-seed traffic variance, exactly
   as the paired analysis said. Formalizing a confirmatory-N rule in
   S4.2 (default N=20; marginal-overlap verdicts resolved by a
   pre-registered higher-N run, never a lower one) awaits owner
   sign-off.
3. **The clean-diurnal cyclic harvest is STILL unclaimed.** On pure
   diurnal traffic every learner parks near 3 nodes (~3 actions,
   regret ~$6.07/week) instead of tracking the cycle; time features
   again change nothing (and add churn). Graded actions fixed *reactive*
   dynamism, not *anticipatory scheduling*. The open question survives
   in sharper form: the learner acts when the state punishes it within
   a step, but never learns to act ahead of a predictable pattern —
   which is exactly the forecaster's one trick, and on clean diurnal it
   still wins by it.

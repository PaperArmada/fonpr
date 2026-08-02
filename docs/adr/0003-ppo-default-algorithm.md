# ADR-0003: PPO replaces DQN as the default training algorithm

- **Status**: Accepted (supersedes ADR-0001/D2)
- **Date**: 2026-08-02
- **Decider**: Project owner (approved the PPO study and evidence-based
  adoption in session; evidence below)

## Context

The 2026-08-02 DQN campaign (5 seeds x 2 observation variants, 500k steps,
default hyperparameters) showed severe training instability: only 4/10
seeds reached even the always-large local optimum; the rest collapsed to
NOOP-like policies costing 6-14x more. The diagnosed failure family is
off-policy bootstrapped value learning under sparse penalty cliffs. PPO —
on-policy, clipped updates, advantage estimation, natively categorical —
was run through the identical campaign protocol as the candidate
replacement. SAC was ruled out: SB3's SAC cannot express a Discrete action
space, and its off-policy bootstrapped critics share DQN's failure family.

## Decision

PPO is the default algorithm for `fonpr train` and the reference learner
in benchmarks. DQN remains available (`--algo dqn`) for comparison studies
but carries no default anywhere.

Evidence (identical protocol, mean total cost on `diurnal` per training
seed; baselines: forecast 27.91, threshold 34.90, oracle 27.16):

| Convergence to local optimum | DQN | PPO |
|---|---|---|
| plain observation | 3/5 | 4/5 |
| time-features observation | 1/5 | **5/5** |

Converged PPO seeds are bit-consistent across seeds (33.94 +/- 0.00 on
diurnal) — reliability, not just a better average.

## Consequences

- Sample efficiency is deliberately traded away; the simulator's ~9k
  steps/sec makes that trade nearly free.
- The open research question is sharpened, not answered: even reliable
  PPO with a clock signal converges to always-large rather than cyclic
  scale-down (S4.4 bar still unmet; forecast-then-act remains champion).
  The binding constraint is now exploration/credit assignment toward
  coordinated multi-step deviations, not optimizer stability — future
  work targets that (or richer action spaces per roadmap Phase 5).
- When Phase 5 moves to continuous actions, the algorithm question
  reopens (SAC vs continuous PPO) via a superseding ADR.

# ADR-0002: Optional time-of-day observation features

- **Status**: Accepted
- **Date**: 2026-08-02
- **Decider**: Project owner (direction approved in session; see benchmark
  2026-08-02 rationale)

## Context

The 2026-08-02 preliminary benchmark showed the DQN converging to an
always-large policy while the seasonal-naive forecaster captured night-time
savings. The agent's observation is a single 15-minute window of
[throughput, large-on, small-on]; it contains no signal for *where in the
daily cycle* the system is, so the agent structurally cannot learn diurnal
behavior. Before tuning hyperparameters, test the structural fix.

## Decision

Add an **opt-in** observation variant: when `time.include_time_features` is
set, the observation gains two columns, `sin(2*pi*tod)` and `cos(2*pi*tod)`
(tod = fraction of the 24h day), appended per tick as columns 3 and 4.

- Default remains **off**: the (samples, 3) contract of ADR-ratified
  interfaces is unchanged unless explicitly enabled.
- Columns are appended, never inserted: existing column indices (throughput,
  large-on, small-on) keep their meaning in both variants.
- Encoding is sin/cos, not raw hour, so midnight is continuous.

## Consequences

- A policy trained with time features can only deploy against an env with
  the same flag; checkpoints already embed their SimConfig, which makes the
  mismatch detectable. The live env must gain the same optional columns
  before such a policy ships (wall-clock time is trivially available there).
- The 5-seed campaign runs both variants; the benchmark reports them as
  separate policy families.
- If the variant wins decisively, a future ADR may make it the default.

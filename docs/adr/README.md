# Architecture Decision Records

Append-only log of closed decisions. An accepted ADR is not open for
relitigation in code review, chat sessions, or implementation — the only way
to change one is a new ADR that explicitly supersedes it, signed off by the
owner and landed before dependent code.

New ADRs: copy `0000-template.md` to the next number, keep it under a page.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-foundation-decisions.md) | Foundation decisions for the modernization effort | Accepted |
| [0002](0002-time-of-day-observation.md) | Optional time-of-day observation features | Accepted |
| [0003](0003-ppo-default-algorithm.md) | PPO replaces DQN as the default training algorithm | Accepted (supersedes 0001/D2) |
| [0004](0004-replica-count-action-space.md) | Replica-count action space (pool plant v2) | Accepted |

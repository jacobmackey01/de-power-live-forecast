# Scoring ledger

**Append-only.** Written by a job separate from the one that makes predictions,
which never writes to `predictions/`.

Scoring runs once outturn has settled. For each delivery day it records realised
prices, the three pre-registered call outcomes, and running aggregates against
the baselines declared in `PREREGISTRATION.md`.

Aggregates here are descriptive. The formal assessment against the pre-registered
success criteria happens once, at the end of the evaluation window on
2026-10-31, and is published on 2026-11-01. Reading the running numbers and
stopping early is precluded by the pre-registration.

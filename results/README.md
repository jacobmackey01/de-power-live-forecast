# Scoring ledger

**Append-only.** Written by a job separate from the one that makes predictions,
which never writes to `predictions/`.

Scoring runs once outturn has settled. For each delivery day it records realised
prices, the three pre-registered call outcomes, and running aggregates against
the baselines declared in `PREREGISTRATION.md`.

Aggregates here are descriptive. The formal assessment against the pre-registered
success criteria happens once, at the end of the evaluation window on
2026-10-31. Reading the running numbers and stopping early is precluded by the
pre-registration.

The assessment is `FINAL_REPORT.md` (with `final_report.json`), written by
`python -m de_power_live.final_report` in the scoring job once every window day
is SCORED or MISSED. The pre-registration named 2026-11-01 as the publication
date; the scoring job's 30-hour settle margin means the last window day can
first be scored on 2026-11-02, so the report lands then. Days scored after
2026-10-31 are kept out of the window's figure, summary aggregates and report.

Two historical commits, `a8f62529` and `cb640a4e`, were mislabelled as
sealed forecasts. Their ledger contents are authoritative: both record
`MISSED` delivery days, and neither commit is amended.

# Pre-registration, phase 2: live day-ahead forecasting record (DE-LU), model v2

**Status: FROZEN on its commit. Nothing below may be edited after the first v2
prediction file exists.**

This document is committed and pushed *before* any v2 prediction is generated.
Its git commit timestamp is the evidence that the calls below were declared in
advance rather than chosen afterwards from whatever worked.

---

## 0. Relation to phase 1

- Phase 1 (model v1, window 2026-08-03 to 2026-10-31) is closed and judged only
  by `PREREGISTRATION.md` and `results/FINAL_REPORT.md`. Its window is not
  extended, and nothing here changes how it was judged. Result: call A PASS
  (skill against last week +0.384, 95% CI [+0.304, +0.453]), call B PASS
  (PR-AUC 0.872 against 0.232, 63 negative hours), call C NULL (coverage 0.367
  and 0.620); 81 days scored, 9 missed, including the 2026-10-25 clock-change
  day.
- Phase 2 is a separate phase with its own window, calls and report. Its design
  comes from `docs/v1-lessons.md` and `research/v2/README.md`. Every change is
  declared below as a design decision, so none can be presented later as a
  discovery.
- v1 keeps sealing, unchanged, as a benchmark. Its phase-2 results are reported
  separately and never pooled with v2's.

## 1. What this is, and what it is not

Every day, before the day-ahead auction closes, the system publishes a sealed
v2 forecast for the next delivery day. A separate job scores it after
settlement and appends to `results/v2/ledger.jsonl`.

**It is not a claim of tradeable edge.** Every call compares against a naive
baseline, not against the market's own expectation. No P&L is claimed,
simulated or implied.

The deliverable that can be guaranteed is the record, not the result. A null
will be reported as a null.

## 2. Information cutoff rule

Only data published strictly before the seal may enter a prediction, and the
seal must fall before 12:00 Europe/Berlin on day D for delivery day D+1.

v2 uses two sources:

- **SMARD day-ahead prices**, for delivery days up to and including D. Day D
  cleared at noon on D-1, so all of it is known at the seal. The seal is
  refused if the price series already holds anything for D+1, or lacks D.
- **Open-Meteo forecasts** for D and D+1 at the eight sites and weights frozen
  with v1. Probe, 2026-11-11 09:05 Berlin: the live request returned every hour
  from 00:00 UTC on 10 Nov to 23:00 UTC on 13 Nov at 8 of 8 sites.

v2 does **not** use:

- **SMARD forecast series.** Banned in phase 1 because D+1 is populated only
  after the auction. Still banned.
- **Realised generation or load.** The same probe found solar and onshore-wind
  outturns published only up to 08:00 the previous morning, a day behind, while
  training history has them complete. A model trained on them would be served
  a blank at every live seal. v1 has this skew (`docs/v1-lessons.md`,
  section 8).
- **Fuel or carbon prices.** Yesterday's power price already carries their
  level, and no source with a verified publication time was found.

Every HTTP response feeding a prediction is hashed (SHA-256) into the sealed
file.

## 3. The pre-registered calls

All calls are declared together. The final report presents all of them
regardless of outcome.

### 3.1 The forecast

- **Target:** DE-LU day-ahead price, EUR/MWh, for every hour of local power day
  D+1.
- **Model:** one gradient-boosted regressor predicting
  `(price - same hour on D) / scale`. `scale` is the mean of the seven daily
  mean prices ending on D, floored at 30 EUR/MWh. The forecast is the price at
  the same hour on D plus the predicted change times the scale.
- **No price level among the inputs.** Every price input is a difference,
  divided by the scale. v1 saw raw levels and learned that high prices fall
  back to its training level; from 1 to 6 November 2026 it forecast 92 EUR/MWh
  too low on average.
- **Fitted once and frozen** (section 4). Not refitted during the window.

### 3.2 Lags on the local calendar

Every lag is "the same wall-clock hour k days earlier", never a UTC offset.

- A 25-hour day's two 02:00 hours share one value, their mean, when read as a
  lag. As a target, both receive the same forecast.
- A 23-hour day has no 02:00. Read as a lag, its 02:00 is the mean of its 01:00
  and 03:00.

### Call A: price level

- **Tested baseline:** the same wall-clock hour on D ("yesterday").
- **Metric:** MAE pooled over all scored hours; skill `1 - MAE_model / MAE_baseline`.
- **Criterion:** skill > 0 with a 95% CI excluding zero. Day-block bootstrap,
  10,000 draws, block = one power day, seed 20270303.
- **Reported regardless:** skill against the same hour on D-6 ("last week",
  phase 1's baseline) with its CI and no verdict; MAE, RMSE, bias; skill by
  calendar month; the paired difference between v2's and v1's MAE on the days
  both sealed, with a day-block CI and no verdict.

In phase 1 the yesterday baseline was a footnote, and v1 did not beat it with a
CI excluding zero. It is the harder of the two naive baselines and is now the
test.

### Call B: spike hours

- **Event:** the hour's price is above the 90th percentile of the hourly prices
  of the 28 delivery days before D+1. The threshold is computed at the seal and
  stored in the sealed file.
- **Forecast:** the share of the error pool (call C) that, added to the point
  forecast, puts the hour above the threshold.
- **Baseline:** for each wall-clock hour, how often it was above the same
  threshold over the same 28 days. Stored in the sealed file.
- **Metric and criterion:** PR-AUC (average precision) beats the baseline with
  a 95% day-block bootstrap CI excluding zero.
- **Power rule:** fewer than 30 events is UNDERPOWERED and reported
  descriptively.
- **Reported regardless:** Brier score and a reliability table. Phase 1's
  targets, below 0 and below 10 EUR/MWh, with event counts and PR-AUC against a
  month-by-hour climatology frozen from training data, and no verdict. The two
  previous winters had 33 and 9 hours below zero between 21 November and 28
  February, so these cannot be relied on to reach 30.

Declared and rejected: "one of the day's three cheapest hours". In development
it lost to a plain count of which hours had been cheapest over the last 28
days (0.716 against 0.814).

### Call C: interval calibration

- **Target:** central 50% and 80% intervals for each hourly price.
- **Rule:** quantiles 0.10, 0.25, 0.50, 0.75, 0.90 of the error pool, times the
  scale, added to the point forecast.
- **Error pool:** v2's own forecast errors, divided by the scale each was
  sealed with, over the 28 days before D+1. A day v2 sealed live contributes
  its live errors. A day before the first live seal contributes the warm-start
  errors frozen with the model: out-of-sample errors of an earlier fit on the
  28 days up to the training cut-off. A missed day contributes nothing. Fewer
  than 200 errors and the seal is refused.
- **Criterion:** pooled point coverage within 5 percentage points of nominal at
  both levels.
- **Declared secondary result, no verdict:** coverage by calendar month at both
  levels. In development the 80% interval's monthly coverage ranged from 0.73
  to 0.95 while the pooled figure was 0.78. A monthly verdict would be likely
  to fail and is not what is being claimed.
- **Reported regardless:** binomial and day-block CIs, mean width, pinball loss
  by quantile, counts between quantiles.

v1 fixed its interval width once, on summer data, and its 80% coverage fell
from 0.81 in August to 0.50 in October.

## 4. Model versioning and the frozen hash

- Every prediction records `model_version`, a SHA-256 of the artefacts
  (`price.ubj`, `warm_start.json`) and a SHA-256 of the source that determines
  a forecast: `v2/features.py`, `v2/model.py`, `v2/calls.py`. The seal is
  refused if any of them differs from `models/v2/MANIFEST.json`.
- The hash deliberately excludes HTTP and retry plumbing (`v2/inputs.py`,
  `v2/predict.py`, `v2/score.py`). In phase 1 a retry fix could not be made
  mid-window because the data client sat inside the hash. Payload hashes in
  each sealed file still record exactly what the model saw.
- The model is fitted on one thread with a fixed seed. A test confirms two fits
  on the same data produce the same bytes.
- Predictions are never regenerated. A change to a hashed file is a new version
  (v3, ...), and the final report shows every version that produced a live
  prediction, separately and pooled.

## 5. Missed days and operations

- A day with no seal before gate closure is recorded `MISSED` with the error
  that caused it, and cannot be backfilled. It counts against the record and
  carries no score.
- v1 and v2 seal in separate jobs. One failing does not excuse the other.
- Three scheduled starts a night (23:22, 00:52, 02:22 UTC) and a seal window at
  07:40 Europe/Berlin, as documented in `docs/seal-timing.md`. A run that fails
  while the gate is open records the miss only if it is the last scheduled run
  or a manual one.
- No v2 input depends on the seal hour, so a late seal uses the same
  information as a punctual one.
- A margin alarm checks, after every v2 seal and once a day, how long before
  the gate the newest v2 forecast was sealed, and fails below 90 minutes. It
  is the same alarm v1 has, pointed at `predictions/v2/`.

## 6. Evaluation window and stopping rule

- **Window:** the delivery date of the first sealed v2 prediction through
  **2027-02-28** inclusive. No early stopping, no extension.
- **Report date, set by the pipeline:** the last window day ends at 23:00 UTC
  on 2027-02-28. Scoring waits 30 hours for settlement, so it can first be
  scored on 2027-03-02. With a day of slack, the report is due by
  **2027-03-03**. If it lands later, it says so and why.
- **A late start.** The end is fixed, so a late first seal shortens the window.
  If the first sealed v2 prediction is for a delivery day after 2026-12-15,
  phase 2 still runs to 2027-02-28 and is reported in full, but every call is
  reported descriptively, with no verdict.
- **No clock change in the window.** The next is 2027-03-28. The local-calendar
  lags are therefore tested in phase 2 only by rehearsal on synthetic and
  recorded inputs (`tests/test_v2_pipeline.py`, `v2/rehearse.py`), not live.
  This document does not claim otherwise.
- **After the window: 2027-03-28.** v2 keeps sealing after the window closes.
  The report will be followed by an addendum stating whether v2 sealed and
  scored the 23-hour day 2027-03-28. It is an operational check with no
  verdict, declared now so that it cannot be added or dropped depending on how
  the day goes.

## 7. What would count as a null

- **A null:** pooled skill against yesterday is zero or negative, or its CI
  spans zero.
- **B null:** PR-AUC does not beat the baseline with a CI excluding zero.
  UNDERPOWERED below 30 events.
- **C null:** pooled coverage at either level misses nominal by more than 5
  points.

There are three verdicts and no correction for multiple comparisons. All three
are listed here in advance and all are reported.

A null on every call is acceptable. It would still show a system that ran
unattended, sealed before the fact, and reported against its declarations.

## 8. Integrity mechanisms

Carried over from phase 1:

1. `predictions/` is write-once; CI fails if any existing file is changed or
   deleted.
2. Predictions come from scheduled GitHub Actions jobs, so timestamps are
   GitHub's.
3. Scoring is a separate job that never writes to `predictions/`.
4. Every input payload is hashed into the sealed file.

New in phase 2:

5. CI fails if a v2 prediction exists while this document is a draft, or if
   this document changes at or after the first v2 prediction.
6. `results/v2/ledger.jsonl` is append-only, checked on every push and across
   the whole history.
7. The final report was written and tested before the first v2 prediction. The
   running summary is the same code in provisional mode, so the two cannot
   disagree on a definition.
8. Every clock-change day through the end of 2027 is run through the real seal
   path in the test suite.

## 9. What was looked at before freezing

The design was chosen on a replay of 3 August to 10 November 2026 and checked
on 21 November 2025 to 28 February 2026, both through the production seal and
scoring code (`research/v2/out/`). On those periods v2's skill against
yesterday was +0.437 and +0.396, spike PR-AUC 0.869 and 0.760, and pooled
coverage 0.474/0.780 and 0.498/0.792.

These are development figures. They chose the design and are not evidence for
it. Nothing sealed before this document's freeze commit counts toward any
verdict.

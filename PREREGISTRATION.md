# Pre-registration: live day-ahead forecasting track record (DE-LU)

**Status: FROZEN on first commit. Nothing below may be edited after the first prediction file exists.**

This document is committed and pushed *before* any prediction is generated. Its
git commit timestamp is the evidence that the calls below were declared in
advance rather than selected afterwards from whatever happened to work.

---

## 1. What this is, and what it is not

This is a live, forward-tested forecasting record for the German-Luxembourg
(DE-LU) day-ahead electricity market. Every trading day, before the day-ahead
auction closes, the system publishes a sealed forecast for the following
delivery day. After the outcome settles, a separate job scores it and appends to
a public ledger.

**It is not a claim of tradeable edge.** Calls A and B below measure forecast
skill against *naive and climatological baselines*, not against the market's own
expectation. Beating a same-hour-last-week baseline is a statement about the
baseline, not about the market. Any positive result here should be read as
"this model extracts signal from weather that simple persistence does not",
which is a much weaker claim than "this would have made money". No P&L is
claimed, simulated, or implied anywhere in this repository.

The deliverable I can guarantee is the *record*, not the result. The result may
well be null. It will be reported either way.

---

## 2. Information cutoff rule

**Only data with a publication time strictly earlier than the seal timestamp may
enter a prediction.** The seal timestamp must fall before 12:00 Europe/Berlin on
day D for a delivery day D+1.

This rule exists because of a specific, verified trap. On 2026-08-01 at 10:38
Europe/Berlin — 82 minutes before the auction close — SMARD's forecast series
(`price_da`, `load_fc`, `renewables_fc`, `residual_load_fc`, and the individual
wind and solar forecasts) were populated only through 23:00 of the *current*
day. Zero of tomorrow's 24 hours were available. SMARD publishes D+1 forecasts
only *after* the auction has already cleared.

Consequently:

- **SMARD forecast series are permanently banned as D+1 drivers.** Using them
  would mean forecasting with information published after the event being
  forecast — a look-ahead leak that would silently invalidate the entire record.
- SMARD is used for three things only: historical training data, realised lags
  up to the cutoff, and scoring after settlement.
- Forward drivers come from **Open-Meteo** numerical weather forecasts, which
  publish a genuine 3-day forward horizon and were confirmed on 2026-08-01 to
  provide 24/24 hours of D+1 coverage at all seven modelled sites.

Every HTTP response feeding a prediction is hashed (SHA-256) and the hash is
stored inside the sealed prediction file.

---

## 3. The three pre-registered calls

All three are declared here, together, before any data is collected. Reporting
only the ones that succeed is therefore not available as an option: the final
report must present all three regardless of outcome.

### Call A — Day-ahead price level

- **Target:** DE-LU day-ahead clearing price, EUR/MWh, for every hour of the
  local power day D+1.
- **Primary baseline (B1):** same hour, seven days prior (D-7 persistence).
- **Secondary baseline (B2):** same hour, one day prior (D-1 persistence).
- **Metric:** pooled mean absolute error, and skill `1 - MAE_model / MAE_B1`.
- **Success criterion:** pooled MAE skill against B1 is positive, with a 95%
  block-bootstrap confidence interval (block = one power day) excluding zero.
- **Reported regardless:** MAE, RMSE, bias, and skill against both baselines,
  with intervals.

### Call B — Negative-price hours

- **Target:** binary, per hour of D+1 — will this hour clear strictly below
  0 EUR/MWh?
- **Baseline:** climatological base rate conditioned on hour-of-day and month,
  estimated on training data only.
- **Metric:** precision-recall AUC (primary), Brier score, and a reliability
  curve.
- **Success criterion:** PR-AUC exceeds the climatological baseline with a 95%
  bootstrap interval excluding zero.
- **Power condition, declared in advance:** if fewer than 30 negative hours
  occur in the evaluation window, the test is declared **underpowered**, results
  are reported descriptively only, and no success or failure is claimed. In that
  case the pre-declared fallback target is hours clearing below
  **10 EUR/MWh**, evaluated identically. Both thresholds are fixed now; neither
  may be re-tuned later.
- **Baseline, frozen before the window:** the climatology is a 12x24 table of
  P(price < threshold) by calendar month and local hour, estimated on training
  data only and stored in `models/<version>/MANIFEST.json`. It is not
  re-estimated at scoring time. This matters because the table is a much
  stronger baseline than a single pooled rate: for v1, August 13:00 local
  carries a 38.3% historical negative rate against 2.1% at 03:00, so the model
  must beat a baseline that already knows solar crushes midday.
- **Scoring the fallback target:** the model has a single P(price < 0) head. For
  the <10 EUR/MWh fallback that score is used as a *ranking* score rather than a
  calibrated probability. PR-AUC depends only on the ranking and is invariant to
  monotone transformations, so this is valid for the pre-registered comparison;
  it would not be valid for Brier score or a reliability curve, which are
  therefore reported for the primary threshold only.

### Call C — Interval calibration

- **Target:** central 50% and 80% prediction intervals for the D+1 hourly price.
- **Metric:** empirical coverage against nominal coverage, pinball loss, and a
  PIT histogram.
- **Success criterion:** empirical coverage falls within 5 percentage points of
  nominal for *both* levels, assessed with a binomial confidence interval.
- **Rationale:** forecasting the level is hard; forecasting one's own error
  distribution is more tractable and is what position sizing actually depends
  on. This is the call most likely to yield a clean positive, and it is declared
  as such here rather than discovered later.

---

## 4. Model versioning

- Every prediction records `model_version` plus a SHA-256 of the model source
  and its frozen parameters.
- **Predictions are never regenerated.** A code change takes effect for future
  predictions only and increments the version.
- The final report must include results for **every model version that ever
  produced a live prediction**, both separately and pooled. Silently replacing a
  weak model with a stronger one and reporting only the latter is precluded by
  this rule.
- Model v1 is frozen at the commit referenced in `models/v1/MANIFEST.json`.

---

## 5. Missed days

If the daily job fails for any reason — outage, API change, expired workflow —
that day is written to the ledger as `MISSED` and **cannot be backfilled**. A
prediction sealed after the auction close is not a prediction.

The final report states the missed-day count alongside the results. A record
with gaps is a weaker record, and hiding the gaps would make it a dishonest one.

---

## 6. Evaluation window and stopping rule

- **Window:** first sealed prediction through **2026-10-31** inclusive.
- **Report published:** 2026-11-01.
- **No early stopping.** The window does not shorten if results look good, nor
  extend if they look poor.
- The window deliberately contains the DST transition on **2026-10-25**, a
  25-hour local power day. Handling it correctly, live, is part of what is being
  tested.

---

## 7. What would count as a null

Stated in advance so it cannot be reframed afterwards:

- **Call A null:** skill against B1 is zero or negative, or its confidence
  interval spans zero.
- **Call B null:** PR-AUC does not exceed the climatological baseline with a
  CI excluding zero — or the window yields fewer than 30 events at both declared
  thresholds, in which case it is underpowered rather than null.
- **Call C null:** empirical coverage for either interval level misses nominal
  by more than 5 percentage points.

A null on all three is a possible and acceptable outcome. It would still
demonstrate a system that ran unattended, sealed its calls before the fact, and
reported against them honestly — which is the point of the exercise.

---

## 8. Integrity mechanisms

1. `predictions/` is **write-once**. A CI job fails the build if any existing
   file under `predictions/` is modified or deleted.
2. Predictions are generated by a scheduled GitHub Actions job, so the commit
   timestamps are GitHub's rather than a local clock under my control.
3. Scoring is a separate job that appends to `results/` and never writes to
   `predictions/`.
4. Every input payload is hashed into the sealed file.
5. Filter identities are re-verified at runtime (`5097 == 123 + 3791 + 125` and
   `715 == 122 - 5097`); a breach aborts the run rather than silently feeding
   the model a relabelled quantity.

---

*Frozen: 2026-08-01. Author: Jacob Mackey.*

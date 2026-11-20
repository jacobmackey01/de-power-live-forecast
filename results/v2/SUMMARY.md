# Running record, phase 2: DE-LU day-ahead forecasting record, model v2

Evaluation window **2026-11-13 to 2027-02-28**, judged against the criteria frozen in [`PREREGISTRATION_PHASE2.md`](../../PREREGISTRATION_PHASE2.md) before the first v2 prediction. Computed by `python -m de_power_live.v2.final_report` from [`ledger.jsonl`](ledger.jsonl) alone.

**Provisional.** 101 window day(s) are not yet scored. The numbers below are the record so far, on the final report's own definitions; no verdict is final until the window closes.

## The record

- Calendar days in the window: **108**
- Scored: **6** days, 144 hours
- Missed: **1** (2026-11-21)
  - 2026-11-21: predict-v2 job failed in run 44716634213: de_power_live.v2.features.LeakageError: newest price is for 2026-11-19, but the seal day 2026-11-20 cleared yesterday and is the anchor of every forecast
- Seal margin before gate closure: minimum 74 min, median 257 min
- Model version(s): `v2` (source `942c4edc75b2`)
- Bootstrap: 10,000 draws, block = one power day, seed 20270303

## Verdicts

| Call | Criterion | Result | So far |
|---|---|---|---|
| **A** price level | MAE skill vs same hour yesterday > 0, 95% CI excluding 0 | +0.570 [+0.453, +0.669] | **PASS** |
| **B** spike hours | PR-AUC beats the hour-of-day baseline, CI excluding 0, at least 30 events | no spike hour yet (0 events) | **UNDERPOWERED** |
| **C** intervals | 50% and 80% coverage within 5 points of nominal | 50%: 0.569, 80%: 0.889 | **NULL** |

## Call A: price level

| | MAE (EUR/MWh) | Skill of model | 95% CI |
|---|---|---|---|
| Model | 23.34 | | |
| Same hour yesterday (tested) | 54.26 | +0.570 | [+0.453, +0.669] |
| Same hour last week (reported) | 123.32 | +0.811 | [+0.745, +0.857] |

Model RMSE 29.98 EUR/MWh, bias +0.43 EUR/MWh (negative = forecast below outturn).

| Month | Days | Model MAE | Yesterday MAE | Skill |
|---|---|---|---|---|
| 2026-11 | 6 | 23.34 | 54.26 | +0.570 |

Against v1 on the 6 days both sealed (descriptive, no verdict): v2's MAE is -15.05 EUR/MWh relative to v1's, 95% CI [-28.23, -1.87].

## Call B: spike hours

A spike hour is one whose price is above the 90th percentile of the hourly prices of the 28 delivery days before it. The baseline is how often that wall-clock hour was above the same threshold over the same days. Both were sealed with the forecast.

Brier score: model 0.0090, baseline 0.0203. No spike hour occurred, so PR-AUC is undefined.

| Forecast P(spike) | Hours | Mean forecast | Observed rate |
|---|---|---|---|
| 0.0-0.1 | 135 | 0.028 | 0.000 |
| 0.1-0.2 | 3 | 0.142 | 0.000 |
| 0.2-0.3 | 1 | 0.280 | 0.000 |
| 0.3-0.4 | 2 | 0.362 | 0.000 |
| 0.4-0.5 | 2 | 0.456 | 0.000 |
| 0.5-0.6 | 1 | 0.542 | 0.000 |

Phase 1's targets, reported without a verdict:

| Target | Events | PR-AUC model | PR-AUC climatology |
|---|---|---|---|
| < 0 EUR/MWh | 0 | n/a | n/a |
| < 10 EUR/MWh | 0 | n/a | n/a |

## Call C: interval calibration

| Interval | Coverage | Binomial 95% CI | Day-block 95% CI | Mean width | Within 5 points |
|---|---|---|---|---|---|
| 50% | 0.569 | [0.488, 0.647] | [0.493, 0.653] | 45.5 | no |
| 80% | 0.889 | [0.827, 0.930] | [0.806, 0.965] | 112.0 | no |

The verdict uses pooled point coverage. Coverage by month is a declared secondary result:

| Month | 50% | 80% |
|---|---|---|
| 2026-11 | 0.569 | 0.889 |

| Band | Hours | Expected |
|---|---|---|
| < q0.1 | 14 | 14 |
| q0.1-q0.25 | 21 | 22 |
| q0.25-q0.5 | 42 | 36 |
| q0.5-q0.75 | 40 | 36 |
| q0.75-q0.9 | 25 | 22 |
| > q0.9 | 2 | 14 |

Mean pinball loss by quantile: 0.1: 5.67, 0.25: 9.83, 0.5: 11.47, 0.75: 9.42, 0.9: 6.64.

## What this does and does not show

All calls compare against naive baselines, not against what the market had already priced in. No tradeable edge and no P&L is claimed.

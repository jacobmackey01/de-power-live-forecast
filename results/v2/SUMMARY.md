# Running record, phase 2: DE-LU day-ahead forecasting record, model v2

Evaluation window **2026-11-13 to 2027-02-28**, judged against the criteria frozen in [`PREREGISTRATION_PHASE2.md`](../../PREREGISTRATION_PHASE2.md) before the first v2 prediction. Computed by `python -m de_power_live.v2.final_report` from [`ledger.jsonl`](ledger.jsonl) alone.

**Provisional.** 104 window day(s) are not yet scored. The numbers below are the record so far, on the final report's own definitions; no verdict is final until the window closes.

## The record

- Calendar days in the window: **108**
- Scored: **4** days, 96 hours
- Missed: **0** (none)
- Seal margin before gate closure: minimum 74 min, median 256 min
- Model version(s): `v2` (source `942c4edc75b2`)
- Bootstrap: 10,000 draws, block = one power day, seed 20270303

## Verdicts

| Call | Criterion | Result | So far |
|---|---|---|---|
| **A** price level | MAE skill vs same hour yesterday > 0, 95% CI excluding 0 | +0.547 [+0.413, +0.694] | **PASS** |
| **B** spike hours | PR-AUC beats the hour-of-day baseline, CI excluding 0, at least 30 events | no spike hour yet (0 events) | **UNDERPOWERED** |
| **C** intervals | 50% and 80% coverage within 5 points of nominal | 50%: 0.594, 80%: 0.896 | **NULL** |

## Call A: price level

| | MAE (EUR/MWh) | Skill of model | 95% CI |
|---|---|---|---|
| Model | 23.40 | | |
| Same hour yesterday (tested) | 51.70 | +0.547 | [+0.413, +0.694] |
| Same hour last week (reported) | 107.40 | +0.782 | [+0.707, +0.861] |

Model RMSE 30.38 EUR/MWh, bias -1.23 EUR/MWh (negative = forecast below outturn).

| Month | Days | Model MAE | Yesterday MAE | Skill |
|---|---|---|---|---|
| 2026-11 | 4 | 23.40 | 51.70 | +0.547 |

Against v1 on the 4 days both sealed (descriptive, no verdict): v2's MAE is -14.39 EUR/MWh relative to v1's, 95% CI [-34.17, +5.38].

## Call B: spike hours

A spike hour is one whose price is above the 90th percentile of the hourly prices of the 28 delivery days before it. The baseline is how often that wall-clock hour was above the same threshold over the same days. Both were sealed with the forecast.

Brier score: model 0.0079, baseline 0.0203. No spike hour occurred, so PR-AUC is undefined.

| Forecast P(spike) | Hours | Mean forecast | Observed rate |
|---|---|---|---|
| 0.0-0.1 | 91 | 0.030 | 0.000 |
| 0.1-0.2 | 2 | 0.156 | 0.000 |
| 0.2-0.3 | 1 | 0.280 | 0.000 |
| 0.4-0.5 | 1 | 0.469 | 0.000 |
| 0.5-0.6 | 1 | 0.542 | 0.000 |

Phase 1's targets, reported without a verdict:

| Target | Events | PR-AUC model | PR-AUC climatology |
|---|---|---|---|
| < 0 EUR/MWh | 0 | n/a | n/a |
| < 10 EUR/MWh | 0 | n/a | n/a |

## Call C: interval calibration

| Interval | Coverage | Binomial 95% CI | Day-block 95% CI | Mean width | Within 5 points |
|---|---|---|---|---|---|
| 50% | 0.594 | [0.494, 0.687] | [0.479, 0.708] | 48.9 | no |
| 80% | 0.896 | [0.819, 0.942] | [0.781, 1.000] | 119.9 | no |

The verdict uses pooled point coverage. Coverage by month is a declared secondary result:

| Month | 50% | 80% |
|---|---|---|
| 2026-11 | 0.594 | 0.896 |

| Band | Hours | Expected |
|---|---|---|
| < q0.1 | 10 | 10 |
| q0.1-q0.25 | 10 | 14 |
| q0.25-q0.5 | 28 | 24 |
| q0.5-q0.75 | 29 | 24 |
| q0.75-q0.9 | 19 | 14 |
| > q0.9 | 0 | 10 |

Mean pinball loss by quantile: 0.1: 6.13, 0.25: 10.30, 0.5: 11.42, 0.75: 9.18, 0.9: 6.86.

## What this does and does not show

All calls compare against naive baselines, not against what the market had already priced in. No tradeable edge and no P&L is claimed.

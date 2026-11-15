# Running record, phase 2: DE-LU day-ahead forecasting record, model v2

Evaluation window **2026-11-13 to 2027-02-28**, judged against the criteria frozen in [`PREREGISTRATION_PHASE2.md`](../../PREREGISTRATION_PHASE2.md) before the first v2 prediction. Computed by `python -m de_power_live.v2.final_report` from [`ledger.jsonl`](ledger.jsonl) alone.

**Provisional.** 107 window day(s) are not yet scored. The numbers below are the record so far, on the final report's own definitions; no verdict is final until the window closes.

## The record

- Calendar days in the window: **108**
- Scored: **1** days, 24 hours
- Missed: **0** (none)
- Seal margin before gate closure: minimum 74 min, median 74 min
- Model version(s): `v2` (source `942c4edc75b2`)
- Bootstrap: 10,000 draws, block = one power day, seed 20270303

## Verdicts

| Call | Criterion | Result | So far |
|---|---|---|---|
| **A** price level | MAE skill vs same hour yesterday > 0, 95% CI excluding 0 | +0.381 [+0.381, +0.381] | **PASS** |
| **B** spike hours | PR-AUC beats the hour-of-day baseline, CI excluding 0, at least 30 events | no spike hour yet (0 events) | **UNDERPOWERED** |
| **C** intervals | 50% and 80% coverage within 5 points of nominal | 50%: 0.500, 80%: 1.000 | **NULL** |

## Call A: price level

| | MAE (EUR/MWh) | Skill of model | 95% CI |
|---|---|---|---|
| Model | 28.89 | | |
| Same hour yesterday (tested) | 46.70 | +0.381 | [+0.381, +0.381] |
| Same hour last week (reported) | 113.90 | +0.746 | [+0.746, +0.746] |

Model RMSE 35.06 EUR/MWh, bias -21.18 EUR/MWh (negative = forecast below outturn).

| Month | Days | Model MAE | Yesterday MAE | Skill |
|---|---|---|---|---|
| 2026-11 | 1 | 28.89 | 46.70 | +0.381 |

Against v1 on the 1 days both sealed (descriptive, no verdict): v2's MAE is +1.40 EUR/MWh relative to v1's, 95% CI [+1.40, +1.40].

## Call B: spike hours

A spike hour is one whose price is above the 90th percentile of the hourly prices of the 28 delivery days before it. The baseline is how often that wall-clock hour was above the same threshold over the same days. Both were sealed with the forecast.

Brier score: model 0.0011, baseline 0.0203. No spike hour occurred, so PR-AUC is undefined.

| Forecast P(spike) | Hours | Mean forecast | Observed rate |
|---|---|---|---|
| 0.0-0.1 | 24 | 0.025 | 0.000 |

Phase 1's targets, reported without a verdict:

| Target | Events | PR-AUC model | PR-AUC climatology |
|---|---|---|---|
| < 0 EUR/MWh | 0 | n/a | n/a |
| < 10 EUR/MWh | 0 | n/a | n/a |

## Call C: interval calibration

| Interval | Coverage | Binomial 95% CI | Day-block 95% CI | Mean width | Within 5 points |
|---|---|---|---|---|---|
| 50% | 0.500 | [0.314, 0.686] | [0.500, 0.500] | 54.5 | yes |
| 80% | 1.000 | [0.862, 1.000] | [1.000, 1.000] | 134.5 | no |

The verdict uses pooled point coverage. Coverage by month is a declared secondary result:

| Month | 50% | 80% |
|---|---|---|
| 2026-11 | 0.500 | 1.000 |

| Band | Hours | Expected |
|---|---|---|
| < q0.1 | 0 | 2 |
| q0.1-q0.25 | 2 | 4 |
| q0.25-q0.5 | 5 | 6 |
| q0.5-q0.75 | 7 | 6 |
| q0.75-q0.9 | 10 | 4 |
| > q0.9 | 0 | 2 |

Mean pinball loss by quantile: 0.1: 7.41, 0.25: 11.49, 0.5: 13.19, 0.75: 9.28, 0.9: 6.04.

## What this does and does not show

All calls compare against naive baselines, not against what the market had already priced in. No tradeable edge and no P&L is claimed.

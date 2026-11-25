# Running record, phase 2: DE-LU day-ahead forecasting record, model v2

Evaluation window **2026-11-13 to 2027-02-28**, judged against the criteria frozen in [`PREREGISTRATION_PHASE2.md`](../../PREREGISTRATION_PHASE2.md) before the first v2 prediction. Computed by `python -m de_power_live.v2.final_report` from [`ledger.jsonl`](ledger.jsonl) alone.

**Provisional.** 95 window day(s) are not yet scored. The numbers below are the record so far, on the final report's own definitions; no verdict is final until the window closes.

## The record

- Calendar days in the window: **108**
- Scored: **8** days, 192 hours
- Missed: **5** (2026-11-21, 2026-11-22, 2026-11-23, 2026-11-24, 2026-11-25)
  - 2026-11-21: predict-v2 job failed in run 44716634213: de_power_live.v2.features.LeakageError: newest price is for 2026-11-19, but the seal day 2026-11-20 cleared yesterday and is the anchor of every forecast
  - 2026-11-22: predict-v2 job failed in run 44891900663: de_power_live.v2.features.LeakageError: newest price is for 2026-11-19, but the seal day 2026-11-21 cleared yesterday and is the anchor of every forecast
  - 2026-11-23: predict-v2 job failed in run 45009722842: de_power_live.v2.inputs.InputError: 48 hour(s) of day-ahead prices are missing in the last 30 days, first at 2026-11-19T23:00:00+00:00
  - 2026-11-24: predict-v2 job failed in run 45160154543: de_power_live.v2.inputs.InputError: 48 hour(s) of day-ahead prices are missing in the last 30 days, first at 2026-11-19T23:00:00+00:00
  - 2026-11-25: predict-v2 job failed in run 45395140758: de_power_live.v2.inputs.InputError: 48 hour(s) of day-ahead prices are missing in the last 30 days, first at 2026-11-19T23:00:00+00:00
- Seal margin before gate closure: minimum 74 min, median 258 min
- Model version(s): `v2` (source `942c4edc75b2`)
- Bootstrap: 10,000 draws, block = one power day, seed 20270303

## Verdicts

| Call | Criterion | Result | So far |
|---|---|---|---|
| **A** price level | MAE skill vs same hour yesterday > 0, 95% CI excluding 0 | +0.558 [+0.338, +0.683] | **PASS** |
| **B** spike hours | PR-AUC beats the hour-of-day baseline, CI excluding 0, at least 30 events | no spike hour yet (0 events) | **UNDERPOWERED** |
| **C** intervals | 50% and 80% coverage within 5 points of nominal | 50%: 0.521, 80%: 0.880 | **NULL** |

## Call A: price level

| | MAE (EUR/MWh) | Skill of model | 95% CI |
|---|---|---|---|
| Model | 22.98 | | |
| Same hour yesterday (tested) | 52.05 | +0.558 | [+0.338, +0.683] |
| Same hour last week (reported) | 103.59 | +0.778 | [+0.679, +0.837] |

Model RMSE 28.93 EUR/MWh, bias +5.29 EUR/MWh (negative = forecast below outturn).

| Month | Days | Model MAE | Yesterday MAE | Skill |
|---|---|---|---|---|
| 2026-11 | 8 | 22.98 | 52.05 | +0.558 |

Against v1 on the 8 days both sealed (descriptive, no verdict): v2's MAE is -17.81 EUR/MWh relative to v1's, 95% CI [-28.97, -6.46].

## Call B: spike hours

A spike hour is one whose price is above the 90th percentile of the hourly prices of the 28 delivery days before it. The baseline is how often that wall-clock hour was above the same threshold over the same days. Both were sealed with the forecast.

Brier score: model 0.0083, baseline 0.0203. No spike hour occurred, so PR-AUC is undefined.

| Forecast P(spike) | Hours | Mean forecast | Observed rate |
|---|---|---|---|
| 0.0-0.1 | 174 | 0.031 | 0.000 |
| 0.1-0.2 | 11 | 0.138 | 0.000 |
| 0.2-0.3 | 2 | 0.245 | 0.000 |
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
| 50% | 0.521 | [0.450, 0.590] | [0.432, 0.609] | 42.5 | yes |
| 80% | 0.880 | [0.827, 0.919] | [0.797, 0.953] | 104.6 | no |

The verdict uses pooled point coverage. Coverage by month is a declared secondary result:

| Month | 50% | 80% |
|---|---|---|
| 2026-11 | 0.521 | 0.880 |

| Band | Hours | Expected |
|---|---|---|
| < q0.1 | 21 | 19 |
| q0.1-q0.25 | 43 | 29 |
| q0.25-q0.5 | 56 | 48 |
| q0.5-q0.75 | 44 | 48 |
| q0.75-q0.9 | 26 | 29 |
| > q0.9 | 2 | 19 |

Mean pinball loss by quantile: 0.1: 5.00, 0.25: 9.24, 0.5: 11.24, 0.75: 9.43, 0.9: 6.59.

## What this does and does not show

All calls compare against naive baselines, not against what the market had already priced in. No tradeable edge and no P&L is claimed.

# v2 candidate: replay results

Written 2026-11-11. Everything here is **development data**. The replay window
was looked at while the design was chosen, so these numbers justify a design
and are not evidence for it. The only evidence v2 will have is the record it
seals after its pre-registration is frozen.

The five price variants below were written down before the first run and run
once. Nothing was tuned afterwards.

**The definitive figures are in "Through the production code" below.** The
sections before it are the first research pass, which used three inputs that
turned out not to be available at a live seal. Removing them changed nothing
material, but the production figures are the ones to quote.

## What v2 changes

v1 predicted price minus last week's price, but it also saw raw price levels,
and it learned that a high price falls back to the level of its training years
(about 90 EUR/MWh). When prices stepped up to about 250 in late October, every
forecast came out too low.

v2 removes the level from the model:

- **Target:** the change from the same local hour yesterday.
- **Price inputs:** differences only (versus yesterday, versus last week,
  versus the day's mean). No raw price.
- **Scale:** the target and the price inputs are divided by the average price of
  the last seven days, so a change means the same at 90 EUR/MWh as at 250.
- **Weather inputs:** tomorrow's forecast and its change from today, since the
  target is a change.
- **Calendar:** day type of both days, plus the 24 December to 1 January period.
- **Lags on the local calendar**, so 23- and 25-hour days need no special case.
- **Seal hour:** trained at 07:30 Berlin, the hour it will seal at.

## Price level (call A)

Average error in EUR/MWh over the 100 replay days from 3 August, and skill
against "same hour yesterday" (0 = no better, 1 = perfect).

| | Aug | Sep | Oct | Nov | All | Skill vs yesterday |
|---|---|---|---|---|---|---|
| Same hour yesterday | 26.4 | 49.4 | 46.7 | 29.6 | 39.9 | 0 |
| Same hour last week | 31.9 | 61.6 | 76.2 | 79.7 | 59.3 | -0.49 |
| v1 as sealed live (91 of the days) | 20.4 | 38.4 | 47.8 | 73.7 | 40.4 | +0.005 |
| v1 design, refitted weekly | 18.8 | 34.1 | 36.1 | 43.3 | 31.2 | +0.22 |
| v2 unscaled, weekly refit | 16.4 | 28.2 | 26.3 | 27.3 | 24.1 | +0.40 |
| **v2 scaled, fitted once** | 16.5 | 25.9 | 24.4 | 22.4 | 22.4 | **+0.44** |
| v2 scaled, weekly refit | 16.6 | 25.6 | 24.3 | 23.3 | 22.4 | +0.44 |
| v2 scaled, weekly refit on the last 365 days | 16.9 | 25.4 | 23.0 | 22.7 | 21.9 | +0.45 |

The last row scored best and was not chosen: its lead over the single fit is
well inside the noise (see below), and a single fit is simpler to freeze.

- v2 scaled, fitted once on data to 1 August: skill +0.440, 95% interval
  [+0.382, +0.489] (`out/price_replay.log`).
- Against v1 on the 91 days both have: 17.7 EUR/MWh lower error, 95% interval
  [13.4, 22.6] (`out/rehearsal_aug_nov_2026.md`).
- November is the weakest month: +0.24 over the 10 days in the table.
- Three figures were computed from the hourly replay file after the run and
  are not in a committed log: October and November together +0.431
  [+0.338, +0.509]; November over 11 days +0.236 [+0.003, +0.343]; v2 better
  than yesterday's price on 83% of days. Treat them as unverified.
- The level bias is gone: when yesterday's price was above 250 EUR/MWh, v2 was
  about 1 too low on average, where v1 run through the same replay was 60 too
  low.

What the table suggests. These are readings of a replay, not controlled
experiments: the variants differ from v1 in target, scaling, lags, calendar and
weather inputs at once, so no single change is isolated.

1. **The price level looks like the main problem.** Refitting v1's design
   every week helps (+0.22) but still loses badly in November, and the
   level-free variants do not.
2. **No gain from refitting was detected.** The paired difference in daily
   error between a weekly refit and a single fit is -0.26 EUR/MWh
   [-0.78, +0.27] on this window and +0.06 [-0.32, +0.46] last winter
   (`out/followup_checks.log`). Both intervals span zero. That is no evidence
   of a difference; it is not a demonstration that the two are equivalent.
3. **Scaling scored better when the level moved** (+0.44 against +0.40) and
   the same last winter, when it did not.

## Last winter, as a check

The same design, unchanged, over 21 November 2025 to 28 February 2026: the
calendar span phase 2 will cover, in a period the design was not built on.
Prices were stable at about 100 EUR/MWh.

| | Error | Skill vs yesterday |
|---|---|---|
| Same hour yesterday | 22.8 | 0 |
| Same hour last week | 27.2 | -0.19 |
| v2 scaled, fitted once | 13.5 | +0.406 [+0.333, +0.474] |

## Intervals (call C)

The interval for each day is the forecast plus the spread of the last 28 days
of forecast errors (in scaled units), all known at the seal.

| | 80% interval covers | 50% interval covers | Width of 80% |
|---|---|---|---|
| v1 as sealed, phase 1 | 0.620 | 0.367 | 96 |
| v2, replay from 3 August | 0.781 | 0.486 | 72 |
| v2, last winter | 0.796 | 0.512 | 43 |
| Yesterday's price as the centre, replay | 0.792 | 0.492 | 142 |

Pooled coverage is inside the 5-point band in both periods. By month it drifts
more: 0.73 in September and 0.94 in November for the 80% interval. A pooled
verdict would pass; a per-month verdict would not.

Splitting by time of day, or using 14 or 56 days instead of 28, gave no
consistent improvement.

## Low-price and spike hours (call B)

How often phase 1's targets occurred between 21 November and 28 February:

| Winter | Hours below 0 | Hours below 10 |
|---|---|---|
| 2024/25 | 33 | 93 |
| 2025/26 | 9 | 38 |

At today's price level neither is likely to reach 30. Three replacements were
tried on the replay (PR-AUC, higher is better):

| Target | Events | v2 | Best simple baseline | Verdict |
|---|---|---|---|---|
| Price above the 90th percentile of the last 28 days | 471 | 0.867 | 0.616 | v2 better, +0.251 [+0.188, +0.318] |
| Price below the 10th percentile of the last 28 days | 198 | 0.859 | 0.260 | v2 better, but 0 events in November |
| One of the day's 3 cheapest hours | 303 | 0.716 | 0.814 | **v2 worse** |

- The "3 cheapest hours" target, which the pre-registration draft recommended,
  is beaten by a plain count of how often each hour was among the cheapest in
  the last 28 days. It should not be the call.
- The spike target stays powered in a rising or flat market (261 events last
  winter, PR-AUC 0.760 against 0.230) and its probabilities are close to
  calibrated without a separate recalibration step.

## An input that would not have been there

The research pass included three inputs built from realised generation and
load over the 24 hours before the seal, as v1 has. A live fetch on 2026-11-11
at 09:00 Berlin found solar and onshore-wind outturns published only up to
08:00 the previous morning. Training history has them complete, so a model
trained on them would have been served a blank at every live seal.

Skill against yesterday, fitted once, from `run_checks.py`
(`out/followup_checks.log`):

| | 3 Aug to 10 Nov 2026 | Last winter |
|---|---|---|
| With the inputs, complete, as replayed | +0.440 | +0.404 |
| Trained with them, solar and onshore wind blank at the seal | +0.439 | +0.355 |
| Without them | +0.430 | +0.385 |

- **They help a little when present.** Leaving them out costs about 0.4
  EUR/MWh: +0.39 [-0.27, +1.13] on this window, +0.42 [+0.02, +0.84] last
  winter.
- **Served blank, a model trained on them does worse than one that never had
  them.** Last winter the blank cost 1.1 EUR/MWh [0.55, 1.65], about 8% of the
  error. On this window it made no detectable difference.

So v2 gives up a small replay gain to avoid an input it would not have had
live. v1 has the skew (`docs/v1-lessons.md`, section 8).

An earlier version of this section, and the comment at the top of the frozen
`v2/features.py`, said the inputs "add nothing measurable" and that the blank
costs "about a quarter of the skill over a winter" (+0.385 to +0.288). Those
figures came from a run that was not committed and are not reproduced by the
committed experiment above. The direction is the same; the sizes were
overstated in one case and understated in the other.

## Through the production code

The same two periods, run day by day through the code that will seal and score
live (`python -m de_power_live.v2.rehearse`), with the model fitted once before
each period and the intervals warm-started as they will be at go-live. Full
output in `out/rehearsal_*.md`.

| | 3 Aug to 10 Nov 2026 | 21 Nov 2025 to 28 Feb 2026 |
|---|---|---|
| Days, hours | 100, 2401 | 100, 2400 |
| A: skill vs yesterday | +0.437 [+0.374, +0.493] | +0.396 [+0.321, +0.465] |
| B: spike PR-AUC vs baseline | 0.869 vs 0.615 (458 events) | 0.760 vs 0.230 (261 events) |
| C: coverage, 50% and 80% | 0.474, 0.780 | 0.498, 0.792 |
| C: 80% coverage by month | 0.79, 0.73, 0.76, 0.95 | 0.73, 0.82, 0.78, 0.80 |

The first period includes the 25-hour day 2026-10-25, which v1 could not seal.
November 2026 is the weakest month for call A: +0.186 over 10 days.

## What the replay cannot tell us

- **Weather. The replay knew more than a live seal would have.** It uses
  Open-Meteo's Historical Forecast archive, which stitches together the first
  few hours of each successive model run. For a delivery day that is weather
  forecast a few hours ahead. A live seal has the forecast made the morning
  before, 17 to 41 hours ahead. So every replay figure in this folder was
  produced with better weather than the model will have live, and the model
  was trained on that better weather too. **The size of the effect on v2 is
  unknown.** Frozen v1 run through the same replay scores 40.3 against 40.4 as
  sealed live, but v1 leans on weather less than v2 does and that comparison
  changes other inputs at the same time, so it does not bound the effect on
  v2. Expect live skill to be lower than the replay's. A matched experiment
  with forecasts as issued belongs to a later model version.
- **A different regime.** The window contains one large step up in prices. A
  step down, or a winter scarcity spike, is untested.
- **Selection.** Five variants were compared on this window. The one chosen is
  the simplest of the three scaled variants, not the best-scoring, and the
  gaps between those three are well inside the noise.
- **Seal hour.** The first research pass (`lib.py`) assumed a 07:30 seal.
  Production seals at 07:40. No v2 input depends on the hour.

## Decisions this points to

| Decision | Recommendation | Why |
|---|---|---|
| Headline hurdle for call A | Same hour yesterday | Last week is beaten by 0.62 and tells us nothing |
| Model | Scaled change-from-yesterday, fitted once and frozen | No gain from refitting was detected; one frozen artefact keeps phase 1's integrity check |
| Fuel prices | Leave out | Yesterday's price already carries the fuel level |
| Intervals | Last 28 days of scaled errors, pooled | Inside the band in both periods; simplest to audit |
| Call C verdict | Pooled coverage within 5 points, monthly coverage reported | Monthly coverage drifts more than 5 points |
| Call B target | Price above the 90th percentile of the last 28 days | Powered in winter; phase 1's targets reported without a verdict |
| Call B baseline | How often that hour exceeded the threshold in the last 28 days | The strongest simple baseline tried |

## Files

- `lib.py`: features, model, replay, as used in the first research pass.
- `run_price.py`, `run_calls.py`, `run_winter.py`: the three experiments.
- `run_checks.py`: two follow-up checks asked for by the independent audit of
  2026-11-12 (weekly against single fit, and the outturn inputs).
- `out/*.log`: their full output.
- `out/rehearsal_*.md`: the production-code rehearsals.
- `fetch.py`: downloads the history into `cache/` (not committed).

The production code is in `src/de_power_live/v2/`.

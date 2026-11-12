# v1: what went wrong, and what v2 does differently

Written on 2026-10-28, before the evaluation window closed. The operational
findings below are settled. The model findings describe patterns in the running
ledger and are confirmed or corrected by [`results/FINAL_REPORT.md`](../results/FINAL_REPORT.md)
once the window closes. Nothing here changes how phase 1 is judged: v1 is
assessed exactly as `PREREGISTRATION.md` froze it.

v2 runs as a separate phase with its own pre-registration, frozen before its
first prediction and running to 2027-02-28. v1 keeps sealing alongside it as a
benchmark.

---

## 1. The clock change: the guard held, the feature did not

The pre-registration put the 2026-10-25 clock change inside the test. v1 did
not produce a forecast for that day.

**What happened.** On 2026-10-24 the seal job stopped with:

```
LeakageError: price lag of 24h on target 2026-10-25 22:00:00+00:00 reaches
delivery day 2026-10-25, whose auction had not cleared at cutoff
```

`features.py` builds its price lags as fixed UTC offsets: "the price 24 hours
earlier". On a 24-hour day that always lands on the previous day. The
2026-10-25 power day has 25 hours, so for its last hour (23:00 local, 22:00 UTC)
24 hours earlier is 00:00 local on 2026-10-25 itself, a price nobody could know
at the seal. The leakage guard caught it and refused to seal, and per the
pre-registration the day was recorded MISSED and never backfilled.

So the safeguard did its job: no contaminated forecast entered the record. But
the test was whether the system handled the day correctly, live, and it did
not. The fix could not go in mid-window because `features.py` is inside the
frozen model hash.

**Why it was not caught.** The tests checked that 2026-10-25 has 25 target
hours. Nothing ran a 25-hour day through the whole pipeline, feature building
and leakage guard included, before the day arrived.

**v2:**

- Define lags on the local calendar: "same local hour on D-1 / D-7", matched by
  local wall-clock hour, with an explicit rule for the repeated and the missing
  hour. Never `t - 24h` in UTC.
- Add a dress-rehearsal test that runs the full `build_prediction` path on
  recorded inputs for every 23- and 25-hour day between now and the end of
  phase 2, leakage guard included. It fails CI if any of them would not seal.
- Write the exception into the MISSED row, not just "predict job failed in run
  N", so the record explains itself without needing the Actions logs.
- Phase 2 (to 2027-02-28) contains no clock change; the next is 2027-03-28. The
  fix therefore cannot be proved live inside phase 2, and the phase-2
  pre-registration should say so rather than imply otherwise.

## 2. Missed days were all plumbing, not modelling

All nine missed days had operational causes:

- **2026-08-28 to 2026-09-04 (8 days):** GitHub's scheduled-run queue delayed
  the seal job by 4.5 to 12 hours, past gate closure. The fix (an early cron
  that waits for a fixed local seal window) arrived mid-window.
- **2026-10-25:** the clock change above.

On 2026-10-27 the scoring job also died on a single dropped SMARD connection:
`smard.py` retries `URLError` and timeouts but not `RemoteDisconnected`. A
retry around the seal and score steps went into the workflows on 2026-10-27
and 2026-10-28, because `smard.py` itself is inside the frozen hash.

**v2:**

- Start with the early-cron + seal-window design from day one.
- Catch every transient network failure in the data clients (connection
  resets, remote disconnects, HTTP 5xx), and test that they are retried.
- Narrow the frozen hash to what determines a forecast (features, model,
  parameters), not the HTTP plumbing, so an operational fix does not count as
  a model change.
- Alert when a seal has not landed by a set local time, so a stuck run is
  noticed while there is still time to dispatch it by hand.

## 3. The close-out was not built before the window opened

The window had a fixed end and a fixed report date, but no code produced the
verdict, and nothing kept post-window days out of the window's numbers. The
final report, the window cut-off and a provisional mode were added on
2026-10-28, three days before the end.

The running summary also reported the mean of daily skill scores (+0.24 on
2026-10-26) while the pre-registered metric is skill pooled over all hours
(+0.37 on the same data). Both are defensible statistics, but only one is the
one that was declared.

The pre-registration named 2026-11-01 as the report date, but the scoring job's
own 30-hour settle margin means the last window day cannot be scored before
2026-11-02 05:00 UTC.

**v2:**

- Build and test the final report against synthetic data before the first
  prediction. The running summary calls the same functions, so the two can
  never disagree on a definition.
- Set the report date from the pipeline: window end plus the settle margin plus
  a day of slack.

## 4. Intervals were well calibrated for summer only (call C)

The 80% interval caught about 81% of outturns in August, then about 58% in
September and about 53% in October (as it stood on 2026-10-28, when this was
written; the final October figure is 50%). Outturns landed above the 90th percentile
roughly three times as often as they should. The mid-October price rise was
forecast far too low (average daily bias near -48 EUR/MWh from 2026-10-15
to the time of writing; -51 over 15 to 31 October once the month was complete), and from 2026-10-16 to 2026-10-18 no hour fell inside the 80%
interval.

The split-conformal calibration was fitted once, on the last 20% of training
data, which was a summer of solar-driven prices. When autumn changed what
drives prices, the interval width did not move with it.

**v2:**

- Recalibrate the intervals on a trailing window of live residuals known at the
  seal (adaptive conformal), declared and frozen in the pre-registration, so
  the width can follow a change of season.
- Scale width with conditions that predict error (residual load, recent price
  volatility) rather than one width for all hours.
- Report coverage by month as a declared secondary result, so a seasonal drift
  is visible in the design, not only in hindsight.

## 5. Price level beat last week, but not yesterday in October (call A)

Against the pre-registered D-7 baseline the model was clearly better
throughout. Against D-1 it was better in August and September but about level
in October. Over a November-to-February phase, autumn and winter prices are
driven more by fuel, temperature and imports than by weather-driven
renewables, and v1 sees only weather.

**v2:**

- Add drivers that are published before the seal: front-month gas (TTF) and
  carbon (EUA) settlements from the previous day, and temperature-driven load
  features. Each needs a verified publication time, like the SMARD probe in
  the README.
- Use D-1 as a declared second hurdle rather than a footnote, since it is the
  harder of the two naive baselines.

## 6. Negative prices will be rare in winter (call B)

All 63 negative hours in the window fell in August and September; October had
none. A November-to-February phase may not reach 30 negative hours, and the
< 10 EUR/MWh fallback may not either. The model's probabilities between 0.5
and 0.9 were also too high: those hours turned negative less often than
forecast.

**v2:**

- Choose call B's target for the season the phase covers, and declare in
  advance whether it is likely to be powered. Low-price hours relative to the
  day (for example, the cheapest decile) stay powered in any season.
- Recalibrate the probability head (isotonic or Platt, fitted on training
  data) so the middle of the range means what it says.

## 7. Smaller things

- Running the test suite locally needs Python 3.12 (the pinned NumPy) and an
  OpenMP runtime for xgboost. Document both, or ship a lockfile / dev
  container.
- Exact re-training reproducibility is still not guaranteed (README,
  "A reproducibility caveat"). Pin `nthread=1` for the frozen fit in v2 so the
  artefact can be rebuilt bit for bit.

## 8. Three live inputs were not what the model was trained on

Found on 2026-11-11, while building v2. v1 has three inputs built from realised
generation and load over the 24 hours before the seal. In training they are a
full 24-hour mean. Live, they are not: a fetch at 09:00 Berlin found solar and
onshore-wind outturns published only up to 08:00 the previous morning, a day
behind. So at every live seal:

- the solar input was the mean of the two or three hours that happened to be
  published, not of a day;
- the wind input counted onshore wind as zero for almost the whole window,
  because the code fills a missing onshore value with zero before adding
  offshore.

Nothing failed, and no backtest could show it, because by the time history is
re-read the data is complete. The effect on v1's accuracy looks small: v1 run
on complete history scores the same as v1 as sealed (40.3 against 40.4 EUR/MWh
over the same 91 days). It is still a train/serve skew that sat in the live
record for the whole of phase 1.

**v2:** uses no outturn data at all. In the replay those inputs help a little
when present (about 0.4 EUR/MWh), and a model trained with them and served a
blank does worse than one that never had them (about 1.1 EUR/MWh over last
winter). Figures and their intervals are in `research/v2/README.md`. A probe of what is actually
published at the seal hour belongs next to every input, not only the ones known
to be late.

## 9. The price level looks like the main problem with call A

Section 5 put October's weak result down to missing fuel and temperature
drivers. The November record showed something more basic. v1 predicts price
minus last week's price, but it also sees raw price levels, and it learned that
a high price falls back to the level of its training years, about 90 EUR/MWh.
When prices stepped up to about 250, its average error went from 0 (yesterday's
price under 50) to -70 (over 250), and from 1 to 6 November it was 92 EUR/MWh
too low on average, six times worse than repeating yesterday's price.

**v2:** predicts the change from the same hour yesterday, sees only price
differences, and scales both by the last week's average price. In the replay,
refitting v1's design every week does not remove the bias and the level-free
design does. That supports the explanation without proving it: v2 differs from
v1 in several ways at once (`research/v2/README.md`).


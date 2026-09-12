# Seal timing: what moving the seal hour costs

The predict job seals at **07:30 Europe/Berlin**. The model was fit at
**11:00 Europe/Berlin** (`SEAL_LOCAL_HOUR` in `dataset.py`). This note records
what that 3h30 gap does, why it was accepted, and what would make it
unacceptable.

It is written down because the honest version is easy to lose. The workflow
comment this replaces asserted that "sealing earlier costs no forecast
quality", which is true of SMARD and false in general, and that assertion would
have justified moving the seal to any hour at all.

## What the pre-registration requires

PREREGISTRATION.md section 2 imposes exactly one timing constraint: the seal
timestamp must fall before 12:00 Europe/Berlin on day D for delivery day D+1.
07:30 satisfies it, and every sealed file records its own `sealed_at_utc` and
`minutes_before_close`, so the choice is visible in the record rather than
asserted here.

Section 4 is not engaged. No model changes; nothing is refit or reversioned.

So this is compliant. Compliant is not the same as free.

## What actually changes

`build_features` takes a `cutoff_utc` and uses it in two distinct ways.

**Unaffected — most of the matrix.** Price features filter on *delivery day*,
not timestamp:

```python
cutoff_local_day = cutoff_utc.tz_convert(MARKET_TZ).normalize()
prices_known = prices.loc[price_delivery_days <= cutoff_local_day]
```

Day D's prices are published around 12:45 on D-1, so every price feature is
identical at 07:30 and at 11:00 on day D. The same holds for the calendar
features, which are deterministic, and the forward weather features, which are
indexed to the target hours rather than the cutoff. That is the large majority
of `FEATURE_COLUMNS`.

**Affected — three columns.** The settled-outturn aggregates use a 24-hour
window that closes at the cutoff:

```python
window_start = cutoff_utc - pd.Timedelta(hours=24)
```

covering `wind_actual_mean_pre_cutoff`, `solar_actual_mean_pre_cutoff` and
`load_actual_mean_pre_cutoff`. Sealing at 07:30 instead of 11:00 slides that
window 3h30 earlier than the window every training row was built with.

## Why the gap is bounded at four hours

`tests/test_predict_schedule.py` enforces `0 <= gap <= 4`. The bound is a
containment measure, not a proof of safety. Three things inform it:

1. **The affected features are 24-hour means.** A 24-hour window spans one full
   diurnal cycle wherever it starts, so shifting its phase changes the mean by
   the difference between the 3h30 dropped and the 3h30 added — a partial-day
   perturbation of a daily average, not a change of regime. This is an argument
   that the effect is small. It is not a measurement of it.

2. **The effect cannot be measured without breaking something else.** Quantifying
   it properly would mean refitting at the new cutoff and comparing, and
   section 4 forbids changing the model mid-window. Backtesting the frozen model
   against re-derived features would answer a different question, since the
   scored record is live and prospective by construction. So the honest position
   is that the shift is unquantified, and the bound keeps it small enough that
   it stays plausibly second-order.

3. **The gap is one-sided.** The test requires `gap >= 0`: the seal may move
   earlier than the training cutoff but never later, because later means closer
   to the gate, and the gate is what the whole schedule exists to beat.

The lower bound of 06:00 local, enforced separately, exists because Open-Meteo's
00Z run lands around 03:00–04:00 UTC. Sealing before it would serve the previous
evening's numerical weather run — strictly older forward information for no
scheduling benefit at all, which is the one version of this trade with nothing
on the other side.

## What is being traded for what

**This change does not improve forecast quality, and no claim is made that it
does.** It trades a small, bounded, unquantified shift in three of the model's
inputs for a large reduction in the chance of having no forecast at all.

The asymmetry is what justifies it. A missed day is a total loss: the ledger
records `MISSED`, the day counts against the record under section 5, and no
amount of later work recovers it — backfilling is forbidden. Eight consecutive
days were lost this way over 2026-08-28..09-04. A 3h30 window shift degrades
three inputs by an amount that is probably small.

If the queue delay ever recedes and a 10:00–11:00 local seal becomes reliable
again, moving the window back is strictly correct and requires only editing
`SEAL_WINDOW_LOCAL`. The test bound permits it; nothing else needs to change.

## What would make this unacceptable

- Evidence that the outturn aggregates carry more weight than assumed — a
  feature-importance or ablation result putting them among the dominant inputs.
- Any need to push the window below 06:00 local, which crosses from "shifted
  aggregates" into "older weather", a real and directional loss.
- Scoring showing systematic degradation from 2026-09-09 onward that does not
  appear before it. The ledger makes this checkable after the fact: seals before
  and after the change carry different `sealed_at_utc` hours and are otherwise
  produced by the identical frozen model.

That last one is the reason to keep this note. If the record does degrade, the
seal-hour change is the first thing that should be suspected, and it should not
require rediscovering which three features it touched.

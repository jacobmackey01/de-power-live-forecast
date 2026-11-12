# Phase 2: what you should know before trusting this record

The phase-2 pre-registration, the sealed forecasts, the frozen model and the
ledger cannot be edited. So where one of them is wrong, imprecise or weaker
than it sounds, the place to say so is here.

An independent audit on 2026-11-12, the day v2 went live, found most of what
follows. It found no evidence of leakage or tampering. It found ways a day
could be lost, ways a verdict could come out wrong, and claims that went
further than their evidence. Everything below that could be fixed without
touching a frozen file was fixed on 2026-11-13, before any v2 day had been
scored. What could not be fixed is stated.

## 1. The first forecast was built twice

- At 09:42:23 UTC on 12 November, [run 43193733116](https://github.com/jacobmackey01/de-power-live-forecast/actions/runs/43193733116)
  built a forecast for 13 November (mean 142.41 EUR/MWh) and then failed to
  commit it, because of a bug in the commit step. That forecast was never
  published and its file was not kept. Only the summary in the run log
  survives.
- At 09:46:30 UTC, [run 43194317044](https://github.com/jacobmackey01/de-power-live-forecast/actions/runs/43194317044)
  built another (mean 140.55 EUR/MWh). That is the published
  `predictions/v2/2026-11-13.json`.
- Both were built before the 11:00 UTC gate, with no price for 13 November in
  existence. The second differs from the first because the weather forecast
  had updated in the four minutes between them.
- Both were manual runs, started by hand on go-live day. The commit message
  says "Sealed by scheduled run"; that was the template's wording and is
  wrong for this commit.

The pre-registration says predictions are never regenerated. No published
prediction has been. But a forecast was built, lost and built again, and that
is close enough to the rule to state. Since 13 November every run uploads the
forecast it builds as a workflow artifact before trying to commit it, so what
each run built, and when, is on record for 90 days whatever happens next.

## 2. "A late seal uses the same information as a punctual one" is not quite true

Section 5 of the pre-registration says this. What is true is that no v2 input
is defined by the clock: nothing is "the last 24 hours before the seal". But
the weather forecast is updated through the morning, so two seals a few
minutes apart can differ, as the two above did.

## 3. The replay had better weather than a live seal

The design was chosen on a replay that used Open-Meteo's Historical Forecast
archive. That archive stitches together the first hours of each successive
model run, so for any delivery day it holds weather forecast a few hours
ahead. A live seal has the forecast made the morning before. The model was
also trained on the archive.

So the development figures in section 9 of the pre-registration (+0.437 and
+0.396 skill, and the rest) were produced with information a live seal does not
have, and **the size of that advantage is unknown**. Live skill should be
expected to come in lower. The live record is not affected: every live
forecast uses only the forecast available at its seal.

## 4. Conventions in the frozen code that the pre-registration does not spell out

- **The spike threshold on a clock-change day.** The pre-registration says the
  threshold is the 90th percentile of "the hourly prices of the 28 delivery
  days" before. The frozen code takes it over 24 wall-clock hours per day,
  with the repeated 02:00 of a 25-hour day averaged into one. For the first
  forecast that gives 322.927 EUR/MWh; over the literal 673 hourly prices it
  would be 322.824. The difference exists only while 25 October is inside the
  28-day window, which is until the forecast for 22 November.
- **The error pool** is counted the same way: 24 values per day.
- **Fewer than 28 days.** The frozen code will set a threshold from as few as
  21 complete days. Since 13 November a forecast is refused if any hour of
  prices is missing in the last 30 days, so this cannot now be reached.
- **A missing 02:00.** The frozen code fills a missing 02:00 from 01:00 and
  03:00 on any day, not only a 23-hour one. The same refusal makes this
  unreachable on an ordinary day.

## 5. v2 can stop and be unable to restart

The interval rule needs at least 200 forecast errors from the last 28 days,
and errors only come from days v2 forecast. After a long enough outage, fewer
than about nine forecast days in the last 28, v2 refuses to forecast, and
cannot recover by itself because it has no new errors to recover with.

This is a property of the frozen rule. It will not be worked around by
backfilling forecasts or topping up the pool. If it happens, phase 2 ends
there and the remaining days are MISSED. The audit job warns when the pool
falls below 336 errors.

## 6. What changed on 13 November, before any v2 day was scored

None of these touches a frozen file. Each has a test in
`tests/test_v2_audit.py`.

**In the report**

- Coverage exactly five points from nominal now counts as inside the band, as
  the pre-registration says. It was being rejected by a floating-point
  comparison. This could only have turned a pass into a null.
- The window opens on the delivery day of the first sealed forecast,
  2026-11-13, taken as a fixed fact. It was being read from the first ledger
  row, which could have been a MISSED day and would then have defeated the
  late-start rule.
- A day that appears twice in the ledger is refused. Before, the last row won,
  so appending a row could have replaced a score.
- Brier score and reliability are reported even if no spike hour occurs.
- If more than one model version seals in the window, each is reported
  separately as well as pooled.
- The final report is written once and carries the time it was generated and
  whether it was late.

**In sealing**

- The gate is checked again when the forecast is finished, and before every
  push. A forecast finished less than five minutes before the gate is refused.
  This is stricter than the pre-registration requires.
- A forecast is refused if any hour of prices is missing in the last 30 days,
  if any weather site fails, or if any model input is missing.
- The gate on a clock-change day is now noon on the wall clock. It was twelve
  elapsed hours after local midnight, which is an hour early on a 25-hour day
  and an hour late on a 23-hour day. As a result v1's sealed file for
  2026-10-26 records its gate as 10:00 UTC, not 11:00, and its margin as 145
  minutes, not 205. The error was on the safe side that day and has not
  occurred on a 23-hour day.

**In keeping the record whole**

- The scoring job now checks every delivery day from 13 November. A day whose
  gate has closed with no forecast and no MISSED row is entered as MISSED.
  Before, a seal job that never ran, or failed outside its seal step, left no
  trace. This adds to the rule in section 5 of the pre-registration; it does
  not replace it.
- An audit of the v2 record runs after every job that writes to it. Every
  scored row must repeat its sealed forecast exactly. Every sealed file must
  hold exactly the hours of its delivery day, once each, sealed before its own
  gate by the frozen model, and must have entered the repository before its
  gate.
- The margin alarm now fails if the newest forecast is not for the day that
  was due. Before, a model that had stopped sealing kept passing.

## 7. What the integrity checks do not prove

- **They detect; they do not prevent.** The `main` branch has no protection
  rule, so the repository owner can push to it directly. The checks would show
  a violation afterwards. They cannot stop one, and the owner could change or
  disable them.
- **Timestamps inside a file are the file's own claim.** The independent
  evidence of when a forecast existed is the time of the GitHub Actions run
  that built it and the commit that published it.
- **Input hashes cannot be checked from this repository.** Each sealed file
  records a hash of every response it used. The responses themselves are not
  kept, so nobody can re-inspect what the model saw.
- **The model hash covers the source that determines a forecast**
  (`v2/features.py`, `v2/model.py`, `v2/calls.py`), not the code that fetches
  inputs or assembles the file.
- **Rewriting history** would remove evidence from the very history the checks
  read. Only a copy of the repository held by someone else would show it.

## 8. Reproducibility

The frozen model has been rebuilt to the same bytes twice, once by its author
and once by the auditor, both on an Apple Silicon Mac from the cached training
data. It has not been rebuilt on Linux, and the training data is not published
with a hash. The test in the suite shows that two fits in one environment give
the same bytes, which is a narrower claim than "anyone can rebuild it".

## 9. Smaller points

- The comment at the top of the frozen `v2/features.py` says the outturn
  inputs "add nothing measurable" and that serving them blank "costs about a
  quarter of the skill over a winter". The committed experiment gives smaller
  numbers: about 0.4 EUR/MWh gained when present and about 1.1 lost when blank
  (`research/v2/README.md`). The decision to leave them out is unchanged.
- Bootstrap intervals for the descriptive results use fewer draws than the
  10,000 used for the three verdicts; the report's JSON says how many.
- The addendum on the 23-hour day 2027-03-28, promised in section 6 of the
  pre-registration, is not automated. It has to be written by hand.

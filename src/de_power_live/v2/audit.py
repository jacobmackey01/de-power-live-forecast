"""Does the phase-2 record follow from its sealed files? Read-only.

    python -m de_power_live.v2.audit

The integrity workflows already hold predictions/ to write-once and the ledger
to append-only. Those are checks on how the files change. They say nothing
about what the files contain, and an independent audit on 2026-11-12 showed
what that leaves open: a second ledger row for a day already scored is a pure
append, and a report that read the last row per day would have let it replace
the score. A sealed file with 24 copies of one hour passed the seal guard.

So this module checks content. It writes nothing and repairs nothing: a
discrepancy fails the run and stays on disk for a person to look at.

Standard library only, so that nothing has to be installed before it can run
and no dependency failure can silence it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .. import AUCTION_CLOSE_LOCAL_HOUR, MARKET_TZ
from . import FIRST_DELIVERY_DAY, LEDGER_PATH, MODEL_VERSION, MODELS_DIR, PREDICTIONS_DIR, REPO_ROOT

VALID_STATUSES = {"SCORED", "MISSED"}
QUANTILE_KEYS = ("0.1", "0.25", "0.5", "0.75", "0.9")
# Scoring waits 30 hours after a delivery day ends. A sealed day still unscored
# this long after it ended means scoring has stopped, not that it is waiting.
UNSCORED_AFTER = timedelta(hours=30 + 72)
# A day is due once its gate has closed. The scoring job records a MISSED row
# for a due day with no forecast; this long after the gate, one of the two
# must exist.
UNACCOUNTED_AFTER = timedelta(hours=36)
# The interval rule refuses to seal below 200 recent errors, and errors only
# come from sealed days. Warn while there is still time to notice.
POOL_WARNING = 336
MARGIN_TOLERANCE_MINUTES = 0.11


class LedgerError(ValueError):
    """Raised when the ledger cannot support a report: it is refused, not repaired."""


# ---- The calendar -----------------------------------------------------------


def expected_hours(day: date) -> list[datetime]:
    """Every hour of the local power day, in UTC: 23, 24 or 25 of them."""
    zone = ZoneInfo(MARKET_TZ)
    start = datetime.combine(day, time(), zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time(), zone).astimezone(timezone.utc)
    out, hour = [], start
    while hour < end:
        out.append(hour)
        hour += timedelta(hours=1)
    return out


def gate_utc(day: date) -> datetime:
    """Gate closure for delivery on ``day``: 12:00 local on the day before."""
    return datetime.combine(
        day - timedelta(days=1), time(AUCTION_CLOSE_LOCAL_HOUR), ZoneInfo(MARKET_TZ)
    ).astimezone(timezone.utc)


def days_due(now: datetime, first: date = FIRST_DELIVERY_DAY, grace: timedelta = timedelta(0)) -> list[date]:
    """Delivery days from ``first`` whose gate closed at least ``grace`` ago."""
    out, day = [], first
    while gate_utc(day) + grace <= now:
        out.append(day)
        day += timedelta(days=1)
    return out


# ---- The ledger ---------------------------------------------------------------


def read_ledger(path: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    if not path.exists():
        return rows, problems
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"ledger line {number} is not valid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            problems.append(f"ledger line {number} is not an object")
            continue
        rows.append(row)
    return rows, problems


def check_ledger(ledger: list[dict]) -> None:
    """Refuse a ledger with a repeated day or an unknown status.

    Appending is all a second row for an existing day needs. A reader that kept
    the last row per day would let that row replace a score with nothing in the
    history looking edited. So a repeated day is an error, whoever appended it
    and whatever it says.
    """
    seen: dict[str, int] = {}
    for line, row in enumerate(ledger, start=1):
        day, status = row.get("delivery_date_local"), row.get("status")
        if not isinstance(day, str):
            raise LedgerError(f"ledger row {line} has no delivery_date_local")
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise LedgerError(f"ledger row {line} has an invalid date {day!r}") from exc
        if status not in VALID_STATUSES:
            raise LedgerError(f"ledger row {line} ({day}) has status {status!r}")
        if day in seen:
            raise LedgerError(
                f"ledger rows {seen[day]} and {line} are both for {day}. A day is scored once; "
                "a second row cannot correct or replace the first."
            )
        seen[day] = line


# ---- A sealed file --------------------------------------------------------------


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse(stamp) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else None


def check_seal(payload: dict, day: str, manifest: dict | None = None) -> list[str]:
    """Everything a sealed v2 file must be for the record to rest on it."""
    problems: list[str] = []
    try:
        delivery = date.fromisoformat(day)
    except ValueError:
        return [f"{day}: not a delivery date"]

    if payload.get("delivery_date_local") != day:
        problems.append(f"{day}: file declares delivery day {payload.get('delivery_date_local')!r}")
    if payload.get("model_version") != MODEL_VERSION:
        problems.append(f"{day}: model_version is {payload.get('model_version')!r}")

    sealed, closes = _parse(payload.get("sealed_at_utc")), _parse(payload.get("auction_closes_utc"))
    if sealed is None or closes is None:
        problems.append(f"{day}: sealed_at_utc or auction_closes_utc is missing or has no timezone")
    else:
        if closes != gate_utc(delivery):
            problems.append(f"{day}: auction_closes_utc is {closes.isoformat()}, not the gate for this day")
        if sealed >= gate_utc(delivery):
            problems.append(f"{day}: sealed at {sealed.isoformat()}, at or after the gate")
        margin = payload.get("minutes_before_close")
        true_margin = (gate_utc(delivery) - sealed).total_seconds() / 60
        if not _finite(margin) or abs(margin - true_margin) > MARGIN_TOLERANCE_MINUTES:
            problems.append(
                f"{day}: minutes_before_close is {margin!r} but the timestamps give {true_margin:.1f}"
            )

    calls = payload.get("predictions")
    wanted = expected_hours(delivery)
    if not isinstance(calls, list) or not calls:
        return [*problems, f"{day}: no predictions"]
    stamps = [_parse(c.get("hour_utc")) if isinstance(c, dict) else None for c in calls]
    if stamps != wanted:
        problems.append(
            f"{day}: hours are not exactly the {len(wanted)} hours of the delivery day, in order "
            f"(found {len(calls)}, {len(set(map(str, stamps)))} distinct)"
        )
    if payload.get("n_hours") != len(calls):
        problems.append(f"{day}: n_hours is {payload.get('n_hours')!r} but {len(calls)} hours are present")

    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            problems.append(f"{day} hour {index}: not an object")
            continue
        for key in ("call_a_price_eur_mwh", "baseline_same_hour_yesterday"):
            if not _finite(call.get(key)):
                problems.append(f"{day} hour {index}: {key} is {call.get(key)!r}")
        for key in ("call_b_prob_spike", "baseline_spike", "call_b_prob_negative", "call_b_prob_below_10",
                    "baseline_negative", "baseline_below_10"):
            value = call.get(key)
            if not _finite(value) or not 0.0 <= value <= 1.0:
                problems.append(f"{day} hour {index}: {key} is {value!r}, not a probability")
        quantiles = call.get("call_c_quantiles")
        values = [quantiles.get(k) for k in QUANTILE_KEYS] if isinstance(quantiles, dict) else []
        if len(values) != len(QUANTILE_KEYS) or not all(_finite(v) for v in values):
            problems.append(f"{day} hour {index}: quantiles are missing or not numbers")
        elif values != sorted(values):
            problems.append(f"{day} hour {index}: quantiles are not in order")
        if len(problems) > 12:
            problems.append(f"{day}: further problems not listed")
            break

    for key in ("scale_eur_mwh", "spike_threshold_eur_mwh"):
        if not _finite(payload.get(key)):
            problems.append(f"{day}: {key} is {payload.get(key)!r}")
    if _finite(payload.get("scale_eur_mwh")) and payload["scale_eur_mwh"] <= 0:
        problems.append(f"{day}: scale is not positive")

    if manifest is not None:
        if payload.get("model_source_sha256") != manifest.get("source_sha256"):
            problems.append(f"{day}: model_source_sha256 is not the frozen model's")
        if payload.get("model_artefact_sha256") != manifest.get("artefact_sha256"):
            problems.append(f"{day}: model_artefact_sha256 is not the frozen model's")
    return problems


# ---- The record as a whole -------------------------------------------------------


def sealed_days(predictions_dir: Path) -> list[str]:
    if not predictions_dir.is_dir():
        return []
    return sorted(p.stem for p in predictions_dir.glob("????-??-??.json"))


def check_scored_row(row: dict, payload: dict) -> list[str]:
    """A SCORED row must repeat its sealed calls exactly and add only the price."""
    day = row["delivery_date_local"]
    problems = []
    for key in ("sealed_at_utc", "minutes_before_close", "model_version", "model_source_sha256",
                "spike_threshold_eur_mwh", "scale_eur_mwh"):
        if row.get(key) != payload.get(key):
            problems.append(f"{day}: ledger {key} differs from the sealed file")
    hours, calls = row.get("hours"), payload.get("predictions", [])
    if not isinstance(hours, list) or len(hours) != len(calls) or row.get("n_hours") != len(calls):
        return [*problems, f"{day}: ledger has {len(hours or [])} hours, the sealed file {len(calls)}"]
    threshold = payload.get("spike_threshold_eur_mwh")
    for index, (hour, call) in enumerate(zip(hours, calls)):
        want = {
            "hour_utc": call.get("hour_utc"),
            "pred": call.get("call_a_price_eur_mwh"),
            "d1": call.get("baseline_same_hour_yesterday"),
            "d7": call.get("baseline_same_hour_last_week"),
            "q": [call.get("call_c_quantiles", {}).get(k) for k in QUANTILE_KEYS],
            "p_spike": call.get("call_b_prob_spike"),
            "base_spike": call.get("baseline_spike"),
            "p_negative": call.get("call_b_prob_negative"),
            "base_negative": call.get("baseline_negative"),
            "p_below_10": call.get("call_b_prob_below_10"),
            "base_below_10": call.get("baseline_below_10"),
        }
        for key, value in want.items():
            if hour.get(key) != value:
                problems.append(f"{day} hour {index}: ledger {key} differs from the sealed call")
        price = hour.get("price")
        if not _finite(price):
            problems.append(f"{day} hour {index}: price is {price!r}")
        elif _finite(threshold):
            for key, outcome in (("spike", price > threshold), ("negative", price < 0.0), ("below_10", price < 10.0)):
                if hour.get(key) != int(outcome):
                    problems.append(f"{day} hour {index}: {key} flag does not follow from the price")
        if len(problems) > 12:
            problems.append(f"{day}: further problems not listed")
            break
    return problems


def publication_times(predictions_dir: Path, repo: Path = REPO_ROOT) -> dict[str, datetime] | None:
    """When each sealed file first entered the history, by committer time.

    Returns None if git cannot answer (not a repository, or a shallow clone
    that does not reach the commit), so that the caller can say the check was
    not made rather than pass it.
    """
    out: dict[str, datetime] = {}
    for day in sealed_days(predictions_dir):
        path = (predictions_dir / f"{day}.json").resolve()
        try:
            result = subprocess.run(
                ["git", "log", "--diff-filter=A", "--format=%cI", "--", str(path.relative_to(repo.resolve()))],
                cwd=repo, capture_output=True, text=True, check=True,
            )
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None
        stamps = result.stdout.split()
        if not stamps:
            return None
        out[day] = datetime.fromisoformat(stamps[-1].replace("Z", "+00:00"))
    return out


def audit(
    predictions_dir: Path = PREDICTIONS_DIR,
    ledger_path: Path = LEDGER_PATH,
    models_dir: Path = MODELS_DIR,
    now: datetime | None = None,
    first: date = FIRST_DELIVERY_DAY,
    published: dict[str, datetime] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (problems, warnings). Problems fail the audit; warnings do not."""
    now = now or datetime.now(timezone.utc)
    problems: list[str] = []
    warnings: list[str] = []

    manifest = None
    manifest_path = models_dir / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sealed = sealed_days(predictions_dir)
    payloads: dict[str, dict] = {}
    for day in sealed:
        try:
            payload = json.loads((predictions_dir / f"{day}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{day}: sealed file cannot be read: {exc}")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{day}: sealed file is not an object")
            continue
        payloads[day] = payload
        problems += check_seal(payload, day, manifest)
        if date.fromisoformat(day) < first:
            problems.append(f"{day}: sealed file is for a day before the window opened on {first}")

    if published is not None:
        for day, stamp in published.items():
            if day in payloads and stamp >= gate_utc(date.fromisoformat(day)):
                problems.append(
                    f"{day}: the sealed file entered the repository at {stamp.isoformat()}, at or after "
                    "the gate. Its own timestamp is not evidence that it was published in time."
                )

    rows, line_problems = read_ledger(ledger_path)
    problems += line_problems
    try:
        check_ledger(rows)
    except LedgerError as exc:
        problems.append(str(exc))
        return problems, warnings
    by_day = {r["delivery_date_local"]: r for r in rows}

    for day, row in by_day.items():
        if row["status"] == "SCORED":
            if day not in payloads:
                problems.append(f"{day}: SCORED in the ledger but there is no readable sealed file")
            else:
                problems += check_scored_row(row, payloads[day])
        elif day in sealed:
            problems.append(f"{day}: MISSED in the ledger but a sealed file exists")

    for day in days_due(now, first, UNACCOUNTED_AFTER):
        key = day.isoformat()
        if key not in sealed and key not in by_day:
            problems.append(
                f"{key}: its gate closed more than {UNACCOUNTED_AFTER.total_seconds() / 3600:.0f}h ago and "
                "there is neither a sealed forecast nor a MISSED row. The day is missing from the record."
            )

    zone = ZoneInfo(MARKET_TZ)
    for day in sealed:
        ended = datetime.combine(date.fromisoformat(day) + timedelta(days=1), time(), zone).astimezone(timezone.utc)
        if day not in by_day and now > ended + UNSCORED_AFTER:
            warnings.append(f"{day}: sealed and long since settled, but not scored. Has scoring stopped?")

    if sealed and sealed[-1] in payloads:
        pool = payloads[sealed[-1]].get("error_pool", {})
        if _finite(pool.get("n_errors")) and pool["n_errors"] < POOL_WARNING:
            warnings.append(
                f"{sealed[-1]}: only {pool['n_errors']} recent forecast errors behind the intervals. "
                "Below 200 the model refuses to seal, and it cannot recover from that by itself."
            )
    return problems, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the phase-2 record against its sealed files")
    parser.add_argument("--no-git", action="store_true", help="skip the publication-time check")
    args = parser.parse_args(argv)

    published = None if args.no_git else publication_times(PREDICTIONS_DIR)
    problems, warnings = audit(published=published)
    if published is None and not args.no_git and sealed_days(PREDICTIONS_DIR):
        warnings.append("publication times could not be read from git; that check was not made")

    for warning in warnings:
        print(f"::warning::{warning}")
    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        print(f"\n{len(problems)} problem(s). Nothing has been changed.")
        return 1
    rows, _ = read_ledger(LEDGER_PATH)
    print(f"OK: {len(sealed_days(PREDICTIONS_DIR))} sealed v2 forecast(s) and {len(rows)} ledger row(s) agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

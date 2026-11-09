"""Decide what a predict run does about the seal window, then do it.

The predict workflow is started by several scheduled crons, because one cron
is one ticket in GitHub's scheduled-run queue and that queue has delivered runs
anywhere from on time to twelve hours late. Each run that finds tomorrow still
unsealed asks this module what to do, and gets one of three answers:

``seal_now``   the window has already opened. Return at once; the caller seals.
``wait``       the window is close enough to sleep until. Sleep, then return;
               the caller seals.
``too_early``  the window is further away than one job can sleep. Return at
               once and say so; the caller seals nothing, and a later scheduled
               run takes the day.

``too_early`` is what makes an early cron safe to add. A run that arrives
punctually at 00:12 local cannot sleep until 07:30 inside one job, and it must
not seal on arrival either: docs/seal-timing.md explains why a seal before
06:00 local uses the previous evening's weather run. So it steps aside. The
last scheduled cron is placed so that it is never too early
(tests/test_predict_schedule.py holds that), which means at least one run
always reaches the seal.

Standard library only, like seal_guard, so the wait cannot fail for reasons
unrelated to the clock.

    python -m de_power_live.seal_wait --window 07:30 --max-wait-minutes 270
"""

from __future__ import annotations

import argparse
import sys
import time as time_module
from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from . import MARKET_TZ

SEAL_NOW = "seal_now"
WAIT = "wait"
TOO_EARLY = "too_early"


@dataclass(frozen=True)
class Decision:
    action: str
    wait_seconds: int
    window_utc: datetime


def parse_window(window: str) -> time:
    """Parse ``HH:MM`` local market time. Anything else is an operator error."""
    try:
        hour_text, minute_text = window.split(":")
        return time(int(hour_text), int(minute_text))
    except ValueError as exc:
        raise ValueError(f"seal window must be HH:MM, got {window!r}") from exc


def decide(now_utc: datetime, window: str, max_wait_minutes: int) -> Decision:
    """What a run starting at ``now_utc`` should do about today's seal window.

    The window is ``window`` local market time on the local calendar day that
    ``now_utc`` falls in. The wait is measured in elapsed seconds, in UTC:
    subtracting two local datetimes that share a tzinfo ignores the offset, and
    would be an hour out on both clock-change mornings.
    """
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    if max_wait_minutes <= 0:
        raise ValueError("max_wait_minutes must be positive")

    market = ZoneInfo(MARKET_TZ)
    local_day = now_utc.astimezone(market).date()
    window_utc = datetime.combine(local_day, parse_window(window), tzinfo=market).astimezone(
        timezone.utc
    )

    wait_seconds = int((window_utc - now_utc.astimezone(timezone.utc)).total_seconds())
    if wait_seconds <= 0:
        return Decision(SEAL_NOW, 0, window_utc)
    if wait_seconds > max_wait_minutes * 60:
        return Decision(TOO_EARLY, wait_seconds, window_utc)
    return Decision(WAIT, wait_seconds, window_utc)


def sleep_until(target_utc: datetime) -> None:
    """Sleep in short steps against the wall clock, so one long sleep that
    drifts or returns early cannot move the seal."""
    while True:
        remaining = (target_utc - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        time_module.sleep(min(remaining, 60.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for the seal window, or step aside")
    parser.add_argument("--window", required=True, help="seal window, HH:MM Europe/Berlin")
    parser.add_argument(
        "--max-wait-minutes",
        type=int,
        required=True,
        help="longest sleep this job can afford before its timeout",
    )
    args = parser.parse_args()

    decision = decide(datetime.now(timezone.utc), args.window, args.max_wait_minutes)
    window_text = f"{args.window} {MARKET_TZ} ({decision.window_utc:%H:%M} UTC)"

    if decision.action == TOO_EARLY:
        print(
            f"The seal window at {window_text} is {decision.wait_seconds // 60} min away, "
            f"more than the {args.max_wait_minutes} min this job can sleep. "
            "Sealing nothing; a later scheduled run takes this day."
        )
        print("too_early=true")
        return 0

    if decision.action == WAIT:
        print(f"Sleeping {decision.wait_seconds}s until {window_text}.")
        sys.stdout.flush()
        sleep_until(decision.window_utc)
        print("Seal window reached.")
    else:
        print(f"Already past {window_text}; sealing now.")

    print("too_early=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

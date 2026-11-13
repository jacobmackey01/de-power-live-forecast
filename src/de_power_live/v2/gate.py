"""Is the gate for a delivery day still open? For the seal job's commit step.

    python -m de_power_live.v2.gate 2026-11-14
    python -m de_power_live.v2.gate 2026-11-14 --need-minutes 2

Exits 0 while the auction for that day is at least ``--need-minutes`` from
closing (default 0: any time before the gate), 1 otherwise. The seal job asks
this before it commits a forecast and before every push, with a margin, because
a push that starts a second before the gate does not finish before it. It asks
once more after the push, with no margin, to learn whether the push did.

A forecast is only a forecast if it is public before the gate, and a timestamp
inside the file is not evidence that it was.

Standard library only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

from .audit import gate_utc


def minutes_left(day: date, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    return (gate_utc(day) - now).total_seconds() / 60


def is_open(day: date, need_minutes: float = 0.0, now: datetime | None = None) -> bool:
    """True while more than ``need_minutes`` remain before the gate for ``day``."""
    return minutes_left(day, now) > need_minutes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Is the gate for a delivery day still open?")
    parser.add_argument("day", help="delivery day, YYYY-MM-DD")
    parser.add_argument(
        "--need-minutes", type=float, default=0.0,
        help="exit 1 unless more than this many minutes remain before the gate",
    )
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit:
        return 2
    try:
        day = date.fromisoformat(args.day)
    except ValueError:
        print(f"not a date: {args.day!r}", file=sys.stderr)
        return 2
    if args.need_minutes < 0:
        print("--need-minutes cannot be negative", file=sys.stderr)
        return 2
    left = minutes_left(day)
    if is_open(day, args.need_minutes):
        print(f"gate for {day} is open: {left:.1f} min left")
        return 0
    if left > 0:
        print(f"gate for {day} closes in {left:.1f} min, less than the {args.need_minutes:g} needed")
    else:
        print(f"gate for {day} closed {-left:.1f} min ago")
    return 1


if __name__ == "__main__":
    sys.exit(main())

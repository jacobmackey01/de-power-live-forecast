"""Is the gate for a delivery day still open? For the seal job's commit step.

    python -m de_power_live.v2.gate 2026-11-14

Exits 0 while the auction for that day has not closed, 1 once it has. The seal
job asks this immediately before it commits a forecast and again before every
push: a forecast is only a forecast if it is public before the gate, and a
timestamp inside the file is not evidence that it was.

Standard library only.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone

from .audit import gate_utc


def minutes_left(day: date, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    return (gate_utc(day) - now).total_seconds() / 60


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m de_power_live.v2.gate YYYY-MM-DD", file=sys.stderr)
        return 2
    try:
        day = date.fromisoformat(args[0])
    except ValueError:
        print(f"not a date: {args[0]!r}", file=sys.stderr)
        return 2
    left = minutes_left(day)
    if left > 0:
        print(f"gate for {day} is open: {left:.1f} min left")
        return 0
    print(f"gate for {day} closed {-left:.1f} min ago")
    return 1


if __name__ == "__main__":
    sys.exit(main())

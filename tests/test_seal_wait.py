"""Tests for the seal-window wait.

Three crons now start the predict job, and this decision is what keeps them
from tripping over each other: a run sleeps until the window, seals at once if
the window has passed, or steps aside if the window is further away than one
job can sleep. Getting the arithmetic wrong in either direction is a miss. Too
strict and every run steps aside; too loose and a run sleeps into its timeout.

The clock-change mornings are tested explicitly, because v1 lost 2026-10-25 to
a calculation that was only ever exercised on 24-hour days.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from de_power_live.seal_wait import SEAL_NOW, TOO_EARLY, WAIT, decide, parse_window

WINDOW = "07:30"
MAX_WAIT = 270


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def test_a_punctual_last_cron_waits_in_winter():
    # 02:12 UTC is 03:12 CET; the window is 06:30 UTC.
    decision = decide(utc("2026-11-10T02:12:00"), WINDOW, MAX_WAIT)
    assert decision.action == WAIT
    assert decision.wait_seconds == 258 * 60
    assert decision.window_utc == utc("2026-11-10T06:30:00")


def test_a_punctual_last_cron_waits_in_summer():
    # 02:12 UTC is 04:12 CEST; the window is 05:30 UTC.
    decision = decide(utc("2027-06-15T02:12:00"), WINDOW, MAX_WAIT)
    assert decision.action == WAIT
    assert decision.wait_seconds == 198 * 60
    assert decision.window_utc == utc("2027-06-15T05:30:00")


def test_a_punctual_early_cron_steps_aside():
    # 23:12 UTC on the 9th is 00:12 CET on the 10th: 438 minutes before the window.
    decision = decide(utc("2026-11-09T23:12:00"), WINDOW, MAX_WAIT)
    assert decision.action == TOO_EARLY
    assert decision.wait_seconds == 438 * 60
    # It is aimed at the 10th's window, not the 9th's.
    assert decision.window_utc == utc("2026-11-10T06:30:00")


def test_the_same_early_cron_waits_once_the_queue_has_delayed_it():
    decision = decide(utc("2026-11-10T02:12:00"), WINDOW, MAX_WAIT)
    assert decision.action == WAIT


def test_a_run_delayed_past_the_window_seals_now():
    # The 2026-11-09 run: started 09:42 UTC, 10:42 CET.
    decision = decide(utc("2026-11-09T09:42:00"), WINDOW, MAX_WAIT)
    assert decision.action == SEAL_NOW
    assert decision.wait_seconds == 0


def test_the_boundaries_fall_on_the_safe_side():
    window_utc = utc("2026-11-10T06:30:00")
    assert decide(window_utc, WINDOW, MAX_WAIT).action == SEAL_NOW
    assert decide(utc("2026-11-10T06:29:59"), WINDOW, MAX_WAIT).action == WAIT
    # Exactly the longest affordable sleep still waits; one second more does not.
    assert decide(utc("2026-11-10T02:00:00"), WINDOW, MAX_WAIT).action == WAIT
    assert decide(utc("2026-11-10T01:59:59"), WINDOW, MAX_WAIT).action == TOO_EARLY


def test_the_spring_clock_change_morning_is_an_hour_shorter():
    # 2027-03-28: 00:30 CET to 07:30 CEST is six elapsed hours, not seven.
    decision = decide(utc("2027-03-27T23:30:00"), WINDOW, 24 * 60)
    assert decision.window_utc == utc("2027-03-28T05:30:00")
    assert decision.wait_seconds == 6 * 3600


def test_the_autumn_clock_change_morning_is_an_hour_longer():
    # 2026-10-25: 00:30 CEST to 07:30 CET is eight elapsed hours, not seven.
    decision = decide(utc("2026-10-24T22:30:00"), WINDOW, 24 * 60)
    assert decision.window_utc == utc("2026-10-25T06:30:00")
    assert decision.wait_seconds == 8 * 3600


def test_the_last_cron_still_waits_on_both_clock_change_mornings():
    for start in ("2026-10-25T02:12:00", "2027-03-28T02:12:00", "2027-10-31T02:12:00"):
        assert decide(utc(start), WINDOW, MAX_WAIT).action == WAIT


def test_a_naive_clock_is_refused():
    with pytest.raises(ValueError):
        decide(datetime(2026, 11, 10, 2, 12), WINDOW, MAX_WAIT)


def test_a_malformed_window_is_refused():
    for bad in ("0730", "7.30", "25:00", ""):
        with pytest.raises(ValueError):
            parse_window(bad)


def test_a_non_positive_wait_budget_is_refused():
    with pytest.raises(ValueError):
        decide(utc("2026-11-10T02:12:00"), WINDOW, 0)

"""Tests for the predict schedule.

The seal window is a tuning knob sitting directly on the integrity path, and
the temptation whenever the scheduling queue misbehaves is to drag it earlier
until the misses stop. That trade is not free, and these tests encode the two
limits on it so the reasoning survives the next bad week.

Parsing is done with :mod:`re` rather than a YAML library: the workflow is not
a project dependency, and adding one to test it would be a poor bargain.
"""

from __future__ import annotations

import re
from pathlib import Path

from de_power_live import AUCTION_CLOSE_LOCAL_HOUR
from de_power_live.dataset import SEAL_LOCAL_HOUR

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "predict.yml"

WINDOW_RE = re.compile(r'SEAL_WINDOW_LOCAL:\s*"(\d{1,2}):(\d{2})"')
CRON_RE = re.compile(r'cron:\s*"(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+"')


def _workflow() -> str:
    return PREDICT_WORKFLOW.read_text(encoding="utf-8")


def seal_window_local_hours() -> float:
    match = WINDOW_RE.search(_workflow())
    assert match, "predict.yml no longer declares SEAL_WINDOW_LOCAL"
    return int(match.group(1)) + int(match.group(2)) / 60.0


def test_seal_window_leaves_real_margin_before_the_gate():
    """Two hours is the floor. Below that a queue spike inside the job is fatal."""
    margin = AUCTION_CLOSE_LOCAL_HOUR - seal_window_local_hours()
    assert margin >= 2.0, f"only {margin:.2f}h of margin between seal window and gate"


def test_seal_window_stays_near_the_training_cutoff():
    """The frozen model was fit at SEAL_LOCAL_HOUR.

    build_features derives the three *_actual_mean_pre_cutoff columns from a 24h
    window closing at the cutoff, so a live seal far from the training hour
    feeds the model an input distribution it was never fit on. Price features
    are immune - they filter on delivery day, not timestamp - but these are not.
    """
    gap = SEAL_LOCAL_HOUR - seal_window_local_hours()
    assert 0 <= gap <= 4, (
        f"seal window is {gap:.2f}h from the {SEAL_LOCAL_HOUR}:00 training cutoff; "
        "moving it further drifts the outturn aggregates away from training"
    )


def test_seal_window_is_late_enough_for_the_00z_weather_run():
    """Open-Meteo is the only forward driver PREREGISTRATION.md section 2 allows.

    The 00Z NWP run publishes around 03:00-04:00 UTC, so a window earlier than
    about 06:00 local would seal on the previous evening's run for no gain.
    """
    assert seal_window_local_hours() >= 6.0


def test_cron_starts_early_enough_to_reach_the_window():
    """The job sleeps until the window, so it must be queued well before it."""
    match = CRON_RE.search(_workflow())
    assert match, "predict.yml no longer declares a cron schedule"
    minute, hour = int(match.group(1)), int(match.group(2))
    # Winter is the worst case for the sleep: local time is UTC+1, so the job
    # starts one hour later in local terms than it does in summer.
    start_local = hour + 1 + minute / 60.0
    assert start_local < seal_window_local_hours(), (
        "cron fires after the seal window; the wait step would never sleep"
    )


def test_cron_is_off_the_hour():
    """On-the-hour crons sit in the most contended slot on the platform."""
    match = CRON_RE.search(_workflow())
    assert match
    assert int(match.group(1)) != 0


def test_predict_is_invoked_without_a_date_override():
    """--date would let a run seal a day other than tomorrow. It has no place here."""
    workflow = _workflow()
    assert "de_power_live.predict" in workflow
    assert "--date" not in workflow


def test_the_gate_guard_is_still_reachable():
    """A skip guard that swallowed every case would silently stop sealing."""
    workflow = _workflow()
    assert "skip=false" in workflow
    assert "steps.guard.outputs.skip == 'false'" in workflow

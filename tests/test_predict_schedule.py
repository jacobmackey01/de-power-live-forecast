"""Tests for the predict schedule.

The seal window is a tuning knob sitting directly on the integrity path, and
the temptation whenever the scheduling queue misbehaves is to drag it earlier
until the misses stop. That trade is real but bounded, and docs/seal-timing.md
sets out what it costs. These tests hold the bounds so the reasoning survives
the next bad week.

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
SEAL_TIMING_DOC = REPO_ROOT / "docs" / "seal-timing.md"

WINDOW_RE = re.compile(r'SEAL_WINDOW_LOCAL:\s*"(\d{1,2}):(\d{2})"')
CRON_RE = re.compile(r'cron:\s*"(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+"')
TIMEOUT_RE = re.compile(r"timeout-minutes:\s*(\d+)")
INPUT_RE = re.compile(r"^      (\w+):$", re.MULTILINE)

# Europe/Berlin is UTC+1 in winter and UTC+2 in summer. Both must hold: a cron
# that only works in one of them fails silently for half the year, and the half
# it fails in is the half with the tighter gate.
BERLIN_UTC_OFFSETS = (1, 2)

# Lower bound on the seal window. Open-Meteo's 00Z run lands around 03:00-04:00
# UTC; sealing before it serves the previous evening's weather for no gain.
EARLIEST_DEFENSIBLE_WINDOW = 6.0

# Upper bound on the distance from the training cutoff. See docs/seal-timing.md.
MAX_GAP_FROM_TRAINING_CUTOFF = 4.0

# Slack for checkout, pip install and the seal itself, on top of the sleep.
JOB_OVERHEAD_MINUTES = 20


def _workflow() -> str:
    return PREDICT_WORKFLOW.read_text(encoding="utf-8")


def seal_window_local_hours() -> float:
    match = WINDOW_RE.search(_workflow())
    assert match, "predict.yml no longer declares SEAL_WINDOW_LOCAL"
    return int(match.group(1)) + int(match.group(2)) / 60.0


def cron_minute_hour() -> tuple[int, int]:
    match = CRON_RE.search(_workflow())
    assert match, "predict.yml no longer declares a cron schedule"
    return int(match.group(1)), int(match.group(2))


# ---- The window itself --------------------------------------------------


def test_seal_window_leaves_real_margin_before_the_gate():
    """Two hours is the floor. Below that a stall inside the job is fatal."""
    margin = AUCTION_CLOSE_LOCAL_HOUR - seal_window_local_hours()
    assert margin >= 2.0, f"only {margin:.2f}h of margin between seal window and gate"


def test_seal_window_stays_near_the_training_cutoff():
    """The frozen model was fit at SEAL_LOCAL_HOUR.

    build_features derives the three *_actual_mean_pre_cutoff columns from a 24h
    window closing at the cutoff, so a live seal far from the training hour
    feeds the model an input distribution it was never fit on. Price features
    are immune - they filter on delivery day, not timestamp - but these are not.

    The bound is containment, not a proof of safety. docs/seal-timing.md says so
    explicitly, and this test exists to keep the gap inside it.
    """
    gap = SEAL_LOCAL_HOUR - seal_window_local_hours()
    assert gap >= 0, (
        f"seal window is {-gap:.2f}h later than the training cutoff, "
        "which moves it toward the gate for no benefit"
    )
    assert gap <= MAX_GAP_FROM_TRAINING_CUTOFF, (
        f"seal window is {gap:.2f}h from the {SEAL_LOCAL_HOUR}:00 training cutoff, "
        f"beyond the {MAX_GAP_FROM_TRAINING_CUTOFF:.0f}h bound in docs/seal-timing.md"
    )


def test_seal_window_is_late_enough_for_the_00z_weather_run():
    """Open-Meteo is the only forward driver PREREGISTRATION.md section 2 allows."""
    assert seal_window_local_hours() >= EARLIEST_DEFENSIBLE_WINDOW


def test_the_trade_off_is_documented():
    """The bound above is only defensible if the reasoning is written down."""
    assert SEAL_TIMING_DOC.exists(), "docs/seal-timing.md is missing"
    text = SEAL_TIMING_DOC.read_text(encoding="utf-8")
    assert "SEAL_LOCAL_HOUR" in text
    assert "_actual_mean_pre_cutoff" in text


# ---- The cron, in both halves of the year -------------------------------


def test_cron_reaches_the_window_in_both_winter_and_summer():
    """Summer is the binding case: the same UTC cron starts an hour later local.

    Testing only UTC+1 would let a cron pass in winter while, in summer, firing
    after the seal window - at which point the wait step never sleeps and the
    seal lands at whatever hour the queue happens to deliver.
    """
    minute, hour = cron_minute_hour()
    window = seal_window_local_hours()
    assert hour + max(BERLIN_UTC_OFFSETS) < 24, "cron wraps past local midnight"
    for offset in BERLIN_UTC_OFFSETS:
        start_local = hour + offset + minute / 60.0
        assert start_local < window, (
            f"at UTC+{offset} the job starts {start_local:.2f} local, after the "
            f"{window:.2f} seal window; the wait step would never sleep"
        )


def test_the_longest_possible_sleep_fits_in_the_job_timeout():
    """Winter is the binding case here: the earliest local start, longest sleep.

    A sleep that outruns timeout-minutes turns a punctual run into a cancelled
    one, which is a miss with extra steps.
    """
    minute, hour = cron_minute_hour()
    window = seal_window_local_hours()
    timeout = TIMEOUT_RE.search(_workflow())
    assert timeout, "predict.yml no longer declares timeout-minutes"
    timeout_minutes = int(timeout.group(1))

    earliest_start = hour + min(BERLIN_UTC_OFFSETS) + minute / 60.0
    longest_sleep = (window - earliest_start) * 60.0
    assert longest_sleep + JOB_OVERHEAD_MINUTES <= timeout_minutes, (
        f"longest sleep is {longest_sleep:.0f} min, which with overhead exceeds "
        f"the {timeout_minutes} min timeout"
    )
    # GitHub cancels any job past six hours regardless of what is declared.
    assert timeout_minutes < 360


def test_cron_is_off_the_hour():
    """On-the-hour crons sit in the most contended slot on the platform."""
    minute, _ = cron_minute_hour()
    assert minute != 0


# ---- Inputs and guards --------------------------------------------------


def test_skip_wait_is_actually_wired_into_the_wait():
    """A declared-but-unread input is worse than none: it looks like a control."""
    workflow = _workflow()
    assert "skip_wait:" in workflow, "skip_wait input was removed but is still referenced"
    assert "github.event.inputs.skip_wait != 'true'" in workflow, (
        "skip_wait is declared but the wait step does not consult it, so manual "
        "runs always skip the wait regardless of what is passed"
    )


def test_every_declared_dispatch_input_is_referenced():
    """Generalises the bug above: an input nothing reads is a dead control."""
    workflow = _workflow()
    dispatch = workflow.split("workflow_dispatch:", 1)
    assert len(dispatch) == 2, "predict.yml no longer declares workflow_dispatch"
    body = dispatch[1].split("permissions:", 1)[0]
    declared = set(INPUT_RE.findall(body))
    assert declared, "no workflow_dispatch inputs found; adjust the parser"
    for name in declared:
        assert f"inputs.{name}" in workflow, f"input {name!r} is declared but never read"


def test_scheduled_runs_still_wait():
    """The wait must not be conditioned on the event being a dispatch.

    github.event.inputs is null on a schedule, so the comparison against 'true'
    is false and the wait runs. Pinned here because the obvious change when
    someone next reads that condition is to add an event_name check that
    accidentally excludes the only case that matters.
    """
    assert "github.event_name == 'schedule'" not in _workflow()


def test_the_guard_validates_rather_than_stats_the_file():
    """A filename is not a seal. The guard must parse before it skips."""
    workflow = _workflow()
    assert "de_power_live.seal_guard" in workflow
    assert 'predictions/$TARGET.json"' not in workflow, (
        "guard is back to testing for file existence, which treats a truncated "
        "or mismatched file as a successful seal"
    )
    # Without pipefail the guard's non-zero exit is masked by tee.
    assert "set -o pipefail" in workflow


def test_predict_is_invoked_without_a_date_override():
    """--date would let a run seal a day other than tomorrow. It has no place here."""
    workflow = _workflow()
    assert "de_power_live.predict" in workflow
    assert "--date" not in workflow


def test_the_workflow_reads_the_verdict_the_guard_actually_emits():
    """A contract test across the YAML/Python boundary.

    The workflow greps seal_guard's stdout for the verdict and feeds it to
    GITHUB_OUTPUT. Nothing else checks that the pattern it greps for still
    matches what the module prints, and a silent mismatch here would leave
    steps.guard.outputs.skip empty - which is neither 'true' nor 'false', so
    every downstream step would skip and the job would go green having done
    nothing at all.
    """
    workflow = _workflow()
    pattern = re.search(r"grep -E '([^']+)'", workflow)
    assert pattern, "the guard step no longer greps a verdict out of seal_guard"
    verdict = re.compile(pattern.group(1).strip("^$"))
    assert verdict.fullmatch("skip=true")
    assert verdict.fullmatch("skip=false")
    assert not verdict.fullmatch("skip=maybe")


def test_the_proceed_case_is_still_reachable():
    """A guard that could never say 'proceed' would silently stop sealing."""
    workflow = _workflow()
    assert "steps.guard.outputs.skip == 'false'" in workflow

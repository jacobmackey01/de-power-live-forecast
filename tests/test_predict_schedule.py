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
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from de_power_live import AUCTION_CLOSE_LOCAL_HOUR, MARKET_TZ
from de_power_live.dataset import SEAL_LOCAL_HOUR
from de_power_live.seal_wait import TOO_EARLY, decide

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
# Every test below runs against both seal workflows. predict-v2.yml is derived
# from predict.yml and must keep every property these tests hold.
SEAL_WORKFLOWS = ("predict.yml", "predict-v2.yml")
PREDICT_WORKFLOW = WORKFLOWS / "predict.yml"
SEAL_TIMING_DOC = REPO_ROOT / "docs" / "seal-timing.md"

WINDOW_RE = re.compile(r'SEAL_WINDOW_LOCAL:\s*"(\d{1,2}):(\d{2})"')
CRON_RE = re.compile(r'cron:\s*"(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+"')
TIMEOUT_RE = re.compile(r"timeout-minutes:\s*(\d+)")
MAX_WAIT_RE = re.compile(r'MAX_WAIT_MINUTES:\s*"(\d+)"')
LAST_CRON_RE = re.compile(r'LAST_SCHEDULED_CRON:\s*"(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+"')
GATE_RE = re.compile(r'GATE_LOCAL:\s*"(\d{1,2}):(\d{2})"')
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

# The earliest cron exists to survive a slow queue. This is the uniform delay
# the schedule must absorb, in both seasons, with the gate-margin alarm's 90
# minutes still in hand. The 2026-08-27..09-03 delays ran from 4h30 to 12h.
QUEUE_DELAY_TO_ABSORB_HOURS = 9.0
MARGIN_TO_KEEP_MINUTES = 90


@pytest.fixture(autouse=True, params=SEAL_WORKFLOWS)
def _each_seal_workflow(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "PREDICT_WORKFLOW", WORKFLOWS / request.param)


def _workflow() -> str:
    return PREDICT_WORKFLOW.read_text(encoding="utf-8")


def seal_window_local_hours() -> float:
    match = WINDOW_RE.search(_workflow())
    assert match, "predict.yml no longer declares SEAL_WINDOW_LOCAL"
    return int(match.group(1)) + int(match.group(2)) / 60.0


def crons() -> list[tuple[int, int]]:
    """Every scheduled cron as (minute, hour), in UTC, as written."""
    found = [(int(m), int(h)) for m, h in CRON_RE.findall(_workflow())]
    assert found, "predict.yml no longer declares a cron schedule"
    return found


def local_start_hours(cron: tuple[int, int], offset: int) -> float:
    """Local wall-clock hour a punctual run starts at, counted from midnight."""
    minute, hour = cron
    return (hour + offset) % 24 + minute / 60.0


def last_cron() -> tuple[int, int]:
    """The cron that starts latest in the local morning."""
    return max(crons(), key=lambda cron: local_start_hours(cron, min(BERLIN_UTC_OFFSETS)))


def max_wait_minutes() -> int:
    match = MAX_WAIT_RE.search(_workflow())
    assert match, "predict.yml no longer declares MAX_WAIT_MINUTES"
    return int(match.group(1))


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


def test_every_cron_starts_between_local_midnight_and_the_window():
    """Checked in both halves of the year, for every cron.

    After local midnight, because the workflow computes the delivery day as
    "tomorrow" at the moment the run starts: a run that could start the evening
    before would aim at a day whose gate has already closed. Before the window,
    because a cron that fires after it never sleeps and seals at whatever hour
    the queue happens to deliver. Testing only UTC+1 would let a cron pass in
    winter and break either rule in summer.
    """
    window = seal_window_local_hours()
    for cron in crons():
        minute, hour = cron
        for offset in BERLIN_UTC_OFFSETS:
            # A UTC evening cron is only acceptable if it has already crossed
            # local midnight; one in the UTC morning must not have wrapped.
            crossed_midnight = hour + offset >= 24
            assert crossed_midnight or hour < 12, (
                f"cron {hour:02d}:{minute:02d} UTC starts the evening before at UTC+{offset}"
            )
            start_local = local_start_hours(cron, offset)
            assert start_local < window, (
                f"at UTC+{offset} cron {hour:02d}:{minute:02d} UTC starts {start_local:.2f} "
                f"local, after the {window:.2f} seal window; it would never sleep"
            )


def test_the_longest_permitted_sleep_fits_in_the_job_timeout():
    """A sleep that outruns timeout-minutes turns a patient run into a cancelled
    one, which is a miss with extra steps."""
    timeout = TIMEOUT_RE.search(_workflow())
    assert timeout, "predict.yml no longer declares timeout-minutes"
    timeout_minutes = int(timeout.group(1))

    assert max_wait_minutes() + JOB_OVERHEAD_MINUTES <= timeout_minutes, (
        f"a {max_wait_minutes()} min sleep plus overhead exceeds the "
        f"{timeout_minutes} min timeout"
    )
    # GitHub cancels any job past six hours regardless of what is declared.
    assert timeout_minutes < 360


def test_the_last_cron_is_never_too_early():
    """Winter is the binding case: the earliest local start, the longest sleep.

    Earlier crons may step aside as too early, but only because a later one is
    coming. If the last cron could also be too early, a punctual queue would
    seal nothing at all.
    """
    window = seal_window_local_hours()
    for offset in BERLIN_UTC_OFFSETS:
        sleep = (window - local_start_hours(last_cron(), offset)) * 60.0
        assert 0 < sleep <= max_wait_minutes(), (
            f"at UTC+{offset} a punctual last cron would need to sleep {sleep:.0f} min, "
            f"beyond the {max_wait_minutes()} min a job may wait"
        )


def test_the_declared_last_cron_is_the_last_cron():
    """LAST_SCHEDULED_CRON decides which run may record a miss before the gate.

    If it named a cron that is not in the schedule, no scheduled run would ever
    match it, and a day on which every attempt failed before the gate would
    leave the record without a MISSED row. If it named an earlier cron, that
    run would write the day off while later attempts were still queued.
    """
    match = LAST_CRON_RE.search(_workflow())
    assert match, "predict.yml no longer declares LAST_SCHEDULED_CRON"
    declared = (int(match.group(1)), int(match.group(2)))
    assert declared in crons()
    for offset in BERLIN_UTC_OFFSETS:
        latest = max(crons(), key=lambda cron: local_start_hours(cron, offset))
        assert declared == latest, f"at UTC+{offset} the last cron is {latest}, not {declared}"


def test_every_cron_is_off_the_hour():
    """On-the-hour crons sit in the most contended slot on the platform."""
    for minute, _ in crons():
        assert minute != 0


def _seal_margin_minutes(delay_hours: float, day: date) -> float | None:
    """Margin to the gate if every cron is queued for ``delay_hours``.

    Replays the schedule through seal_wait.decide for the seal morning ``day``:
    runs start in cron order, the first one that is not too early seals at the
    later of its start and the window, and the rest are no-ops. Returns None
    if no run seals before the gate.
    """
    market = ZoneInfo(MARKET_TZ)
    gate = datetime.combine(day, time(AUCTION_CLOSE_LOCAL_HOUR), tzinfo=market).astimezone(
        timezone.utc
    )
    window_text = WINDOW_RE.search(_workflow())
    window = f"{int(window_text.group(1)):02d}:{window_text.group(2)}"

    starts = []
    for minute, hour in crons():
        # A cron in the UTC evening fires on the previous UTC date.
        fire_day = day - timedelta(days=1) if hour >= 12 else day
        fired = datetime.combine(fire_day, time(hour, minute), tzinfo=timezone.utc)
        starts.append(fired + timedelta(hours=delay_hours))

    for start in sorted(starts):
        decision = decide(start, window, max_wait_minutes())
        if decision.action == TOO_EARLY:
            continue
        sealed_at = max(start, decision.window_utc)
        if sealed_at >= gate:
            return None
        return (gate - sealed_at).total_seconds() / 60.0
    return None


def test_the_schedule_absorbs_a_slow_queue_in_both_seasons():
    """The reason there is more than one cron.

    With the single 02:12 UTC cron, the 7h30 delay seen on 2026-11-09 left 76
    minutes. The schedule must seal with the alarm margin intact for every
    uniform delay from none up to QUEUE_DELAY_TO_ABSORB_HOURS, winter and
    summer, including the punctual case where the early crons step aside.
    """
    for day in (date(2026, 12, 1), date(2027, 7, 1)):
        delay = 0.0
        while delay <= QUEUE_DELAY_TO_ABSORB_HOURS:
            margin = _seal_margin_minutes(delay, day)
            assert margin is not None, f"{day}: no run seals with a {delay:.2f}h queue delay"
            assert margin >= MARGIN_TO_KEEP_MINUTES, (
                f"{day}: a {delay:.2f}h queue delay leaves only {margin:.0f} min"
            )
            delay += 0.25


def test_a_punctual_queue_seals_at_the_window():
    """The early crons must cost nothing when the queue behaves."""
    window = seal_window_local_hours()
    expected = (AUCTION_CLOSE_LOCAL_HOUR - window) * 60.0
    for day in (date(2026, 12, 1), date(2027, 7, 1)):
        assert _seal_margin_minutes(0.0, day) == expected


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
    assert re.search(r"python -m de_power_live\.(v2\.)?predict\b", workflow)
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


# ---- Stepping aside, and who may record a miss --------------------------


def test_the_seal_step_respects_too_early():
    """A too-early run that sealed anyway would seal on the previous evening's
    weather run, hours before the window."""
    workflow = _workflow()
    assert "de_power_live.seal_wait" in workflow
    assert "steps.wait.outputs.too_early != 'true'" in workflow


def test_the_workflow_reads_the_verdict_the_wait_actually_emits(capsys, monkeypatch):
    """The same contract as the guard's, for the wait step.

    If the grep stopped matching what seal_wait prints, grep would exit
    non-zero and the run would fail before sealing - on every cron.
    """
    workflow = _workflow()
    pattern = re.search(r"grep -E '(\^too_early[^']+)'", workflow)
    assert pattern, "the wait step no longer greps a verdict out of seal_wait"
    verdict = re.compile(pattern.group(1))

    from de_power_live import seal_wait

    monkeypatch.setattr(seal_wait, "sleep_until", lambda target: None)
    for window in ("00:00", "23:59"):  # already past, and (almost always) too early
        monkeypatch.setattr(
            "sys.argv", ["seal_wait", "--window", window, "--max-wait-minutes", "1"]
        )
        assert seal_wait.main() == 0
        lines = capsys.readouterr().out.splitlines()
        assert sum(1 for line in lines if verdict.fullmatch(line)) == 1, lines


def test_only_a_final_attempt_records_a_miss_before_the_gate():
    """An early run that fails must leave the day to the runs behind it."""
    workflow = _workflow()
    assert '"$EVENT_SCHEDULE" != "$LAST_SCHEDULED_CRON"' in workflow
    assert "steps.miss.outputs.recorded == 'true'" in workflow
    # Event values go through env, not into the script body.
    assert "${{ github.event.schedule }}\"" not in workflow


def test_the_gate_in_the_workflow_is_the_gate_in_the_code():
    match = GATE_RE.search(_workflow())
    assert match, "predict.yml no longer declares GATE_LOCAL"
    assert (int(match.group(1)), int(match.group(2))) == (AUCTION_CLOSE_LOCAL_HOUR, 0)

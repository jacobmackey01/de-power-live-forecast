"""What the v2 workflows must and must not do. The schedule itself is covered
by tests/test_predict_schedule.py, which runs against both seal workflows."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_v2_seals_only_once_the_model_is_frozen():
    """Merged before the freeze, the job must do nothing: no forecast from a
    model that does not exist, and no MISSED row for failing to make one."""
    workflow = _read("predict-v2.yml")
    assert "models/v2/MANIFEST.json" in workflow
    guard = workflow[workflow.index("- name: Check whether tomorrow still needs sealing"):]
    assert "if: steps.frozen.outputs.active == 'true'" in guard.split("run:", 1)[0]


def test_v2_and_v1_write_to_different_places():
    v2 = _read("predict-v2.yml")
    assert "--predictions-dir predictions/v2 --ledger results/v2/ledger.jsonl" in v2
    assert "for path in predictions/v2 results/v2; do" in v2
    assert "python -m de_power_live.v2.predict" in v2
    assert "python -m de_power_live.predict" not in v2
    assert "de_power_live.v2" not in _read("predict.yml")
    assert "de_power_live.v2" not in _read("score.yml")


def test_v2_and_v1_do_not_share_a_seal_window_or_a_cron():
    def schedule(name):
        text = _read(name)
        return set(re.findall(r'cron:\s*"([^"]+)"', text)), re.search(r'SEAL_WINDOW_LOCAL:\s*"([^"]+)"', text).group(1)

    v1_crons, v1_window = schedule("predict.yml")
    v2_crons, v2_window = schedule("predict-v2.yml")
    assert not v1_crons & v2_crons
    assert v1_window != v2_window


def test_a_v2_miss_records_why():
    workflow = _read("predict-v2.yml")
    assert '2> "$RUNNER_TEMP/seal.err"' in workflow
    assert 'tail -n 1 "$RUNNER_TEMP/seal.err"' in workflow
    assert "de_power_live.v2.score --record-miss" in workflow


def test_every_job_that_writes_to_main_retries_a_rejected_push():
    """Four jobs now push to main. A bare `git push` loses the commit, and with
    it a sealed forecast, whenever another job got there first."""
    for name in ("predict.yml", "predict-v2.yml", "score.yml", "score-v2.yml"):
        workflow = _read(name)
        assert "git pull --rebase" in workflow, name
        assert "if git push; then" in workflow, name
        assert not re.search(r"^\s*git push\s*$", workflow, re.MULTILINE), name


def test_v2_scoring_never_commits_predictions():
    workflow = _read("score-v2.yml")
    assert "git add results/v2" in workflow
    assert "git add predictions" not in workflow
    assert "scoring changed predictions/" in workflow


def test_integrity_covers_the_phase_2_record():
    integrity = _read("integrity.yml")
    assert "PREREGISTRATION_PHASE2.md" in integrity
    assert "results/v2/ledger.jsonl" in integrity
    audit = _read("integrity-audit.yml")
    assert "predict-v2, score-v2" in audit
    assert "results/v2/ledger.jsonl" in audit
    assert "PREREGISTRATION_PHASE2.md" in audit


def test_the_margin_alarm_covers_v2():
    alarm = _read("gate-margin.yml")
    assert "workflows: [predict, predict-v2]" in alarm
    assert "--predictions-dir predictions/v2 --allow-empty" in alarm
    # The v2 check must run even when the v1 check has just failed.
    v2_step = alarm[alarm.index("- name: Check v2 margin to gate closure"):]
    assert "if: always()" in v2_step.split("run:", 1)[0]


def test_a_dry_run_cannot_seal_or_record_a_miss():
    """The rehearsal exists to be run on the live runner without touching the
    record. If the guard still ran, a dry run would go on to seal for real."""
    workflow = _read("predict-v2.yml")
    assert "python -m de_power_live.v2.predict --dry-run" in workflow
    guard = workflow[workflow.index("- name: Check whether tomorrow still needs sealing"):]
    assert "github.event.inputs.dry_run != 'true'" in guard.split("run:", 1)[0]
    # Every writing step depends on the guard having run.
    for name in ("Wait for the seal window", "Seal tomorrow's forecast"):
        step = workflow[workflow.index(f"- name: {name}"):].split("run:", 1)[0]
        assert "steps.guard.outputs.skip == 'false'" in step
    for name in ("Record a miss if sealing failed", "Commit"):
        step = workflow[workflow.index(f"- name: {name}"):].split("run:", 1)[0]
        assert "steps.seal.outcome" in step


# ---- The commit step, actually run ------------------------------------------------


def _commit_script() -> str:
    """The Commit step of predict-v2.yml as a shell script."""
    workflow = _read("predict-v2.yml")
    block = workflow[workflow.index("      - name: Commit"):]
    script = textwrap.dedent(block.split("run: |\n", 1)[1])
    assert "${{" not in script, "the commit script takes its inputs from env, not from interpolation"
    return script


def _push_buffer() -> str:
    """PUSH_BUFFER_MINUTES as the workflow sets it for every step."""
    match = re.search(r'^  PUSH_BUFFER_MINUTES: "(\d+)"$', _read("predict-v2.yml"), flags=re.M)
    assert match, "predict-v2.yml must set PUSH_BUFFER_MINUTES"
    return match.group(1)


def _run_commit(cwd: Path, seal_outcome: str, target: date) -> subprocess.CompletedProcess:
    """Run the commit step as the workflow would, with `python` able to import the package."""
    src = Path(__file__).resolve().parents[1] / "src"
    env = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(src),
        "SEAL_OUTCOME": seal_outcome,
        "EVENT_NAME": "schedule",
        "RUN_ID": "1",
        "TARGET": target.isoformat(),
        "PUSH_BUFFER_MINUTES": _push_buffer(),
    }
    return subprocess.run(["bash", "-e", "-c", _commit_script()], cwd=cwd, capture_output=True, text=True, env=env)


def _berlin_today() -> date:
    return datetime.now(ZoneInfo("Europe/Berlin")).date()


# The gate for the day after tomorrow is tomorrow at noon: always still open.
# The gate for today was yesterday at noon: always closed.
def _open_target() -> date:
    return _berlin_today() + timedelta(days=2)


def _closed_target() -> date:
    return _berlin_today()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _scratch_repo(tmp_path: Path) -> Path:
    """A clone with one commit pushed to a bare remote, as the runner's checkout is."""
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(tmp_path, "clone", "-q", str(remote), str(work))
    _git(work, "config", "user.name", "t")
    _git(work, "config", "user.email", "t@example.invalid")
    (work / "README").write_text("x")
    _git(work, "add", "README")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "origin", "HEAD")
    return work


@pytest.mark.parametrize(
    "written,seal_outcome,expected_subject",
    [
        # The first seal: predictions/v2 exists, results/v2 does not. On
        # 2026-11-12 the first live seal failed here, with the forecast built
        # and nowhere to go, because git add was handed a path that did not exist.
        (["predictions/v2/2026-11-13.json"], "success", "Seal v2 forecast"),
        # The mirror: a miss before any seal, so only results/v2 exists.
        (["results/v2/ledger.jsonl", "results/v2/SUMMARY.md"], "failure", "Record missed v2 forecast"),
        (["predictions/v2/2026-11-13.json", "results/v2/ledger.jsonl"], "success", "Seal v2 forecast"),
    ],
)
def test_the_commit_step_commits_whatever_was_written(tmp_path, written, seal_outcome, expected_subject):
    work = _scratch_repo(tmp_path)

    for name in written:
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    result = _run_commit(work, seal_outcome, _open_target())
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(work, "log", "-1", "--format=%s").startswith(expected_subject)
    assert sorted(_git(work, "show", "--name-only", "--format=", "HEAD").split()) == sorted(written)
    # And it reached the remote, which is the whole point of the step.
    assert _git(work, "rev-parse", "HEAD") == _git(work, "rev-parse", "@{u}")
    assert _git(work, "status", "--porcelain") == ""


def test_the_commit_step_does_nothing_when_nothing_was_written(tmp_path):
    _git(tmp_path, "init", "-q")
    result = _run_commit(tmp_path, "success", _open_target())
    assert result.returncode == 0 and "nothing to commit" in result.stdout


def test_a_forecast_is_not_published_once_its_gate_has_closed(tmp_path):
    """Audit finding: the gate was checked when the build started and never
    again, so a forecast could be built in time and pushed after the auction.
    The file's own timestamp would still say it was early."""
    work = _scratch_repo(tmp_path)
    path = work / "predictions" / "v2" / f"{_closed_target()}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    before = _git(work, "rev-parse", "HEAD")

    result = _run_commit(work, "success", _closed_target())
    assert result.returncode == 1
    assert "cannot be published before it" in result.stdout
    assert _git(work, "rev-parse", "HEAD") == before  # nothing committed
    assert _git(work, "rev-parse", "@{u}") == before  # nothing pushed


def test_a_miss_is_still_recorded_after_the_gate(tmp_path):
    """The gate stops forecasts, not the record of their absence."""
    work = _scratch_repo(tmp_path)
    path = work / "results" / "v2" / "ledger.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"status": "MISSED"}\n')
    result = _run_commit(work, "failure", _closed_target())
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(work, "log", "-1", "--format=%s").startswith("Record missed v2 forecast")


def test_two_jobs_appending_to_the_v2_ledger_both_survive_a_rebase(tmp_path):
    """The seal job appends a MISSED row and the scoring job appends scores, to
    the same file, from separate checkouts. Without the union rule in
    .gitattributes the second push is rejected, the rebase conflicts, and one
    job's row is lost with its runner."""
    root = Path(__file__).resolve().parents[1]
    first = _scratch_repo(tmp_path)
    (first / ".gitattributes").write_text((root / ".gitattributes").read_text())
    ledger = first / "results" / "v2" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"delivery_date_local": "2026-11-13", "status": "SCORED"}\n')
    _git(first, "add", "-A")
    _git(first, "commit", "-q", "-m", "one day")
    _git(first, "push", "-q", "origin", "HEAD")

    second = tmp_path / "second"
    _git(tmp_path, "clone", "-q", str(tmp_path / "remote.git"), str(second))
    _git(second, "config", "user.name", "t")
    _git(second, "config", "user.email", "t@example.invalid")

    with ledger.open("a") as handle:
        handle.write('{"delivery_date_local": "2026-11-14", "status": "SCORED"}\n')
    _git(first, "commit", "-q", "-am", "score")
    _git(first, "push", "-q", "origin", "HEAD")

    with (second / "results" / "v2" / "ledger.jsonl").open("a") as handle:
        handle.write('{"delivery_date_local": "2026-11-15", "status": "MISSED"}\n')
    _git(second, "commit", "-q", "-am", "miss")
    rejected = subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=second, capture_output=True)
    assert rejected.returncode != 0
    _git(second, "pull", "-q", "--rebase")
    _git(second, "push", "-q", "origin", "HEAD")

    lines = (second / "results" / "v2" / "ledger.jsonl").read_text().splitlines()
    assert sorted(lines) == sorted(
        '{"delivery_date_local": "%s", "status": "%s"}' % pair
        for pair in (("2026-11-13", "SCORED"), ("2026-11-14", "SCORED"), ("2026-11-15", "MISSED"))
    )
    assert lines[0].endswith('"2026-11-13", "status": "SCORED"}')  # the old row is still the prefix


def test_a_built_forecast_is_kept_even_if_it_cannot_be_committed():
    workflow = _read("predict-v2.yml")
    step = workflow[workflow.index("- name: Keep a copy of the sealed forecast"):].split("- name:", 2)[1]
    assert "actions/upload-artifact@" in step
    assert "if: always() && steps.seal.outcome == 'success'" in step
    assert "predictions/v2/${{ steps.guard.outputs.target }}.json" in step
    # Before the commit, so a failing commit cannot prevent it.
    assert workflow.index("Keep a copy of the sealed forecast") < workflow.index("- name: Commit")


def test_the_alarm_input_never_reaches_the_script_as_text():
    alarm = _read("gate-margin.yml")
    for block in alarm.split("run: |")[1:]:
        assert "${{" not in block.split("- name:")[0]
    assert "--expect-current" in alarm


def test_a_push_must_start_with_time_to_spare_and_is_checked_afterwards():
    """Follow-up review, 2026-11-13: the gate check before a push passed with
    one second left, and nothing looked at the clock once the push returned."""
    script = _commit_script()
    assert int(_push_buffer()) >= 2
    guarded = 'python -m de_power_live.v2.gate "$TARGET" --need-minutes "$PUSH_BUFFER_MINUTES"'
    # Once before the commit and once before each push attempt.
    assert script.count(guarded) == 2
    push = script.index("if git push; then")
    assert script.rindex(guarded, 0, push) > script.index("for attempt in")
    # After a push that succeeded, the gate is asked again with no margin.
    after = script[push:]
    assert 'python -m de_power_live.v2.gate "$TARGET"; then' in after
    assert "LATE" in after


def test_the_audit_is_given_githubs_record_of_pushes():
    audit = _read("integrity-audit.yml")
    step = audit[audit.index("- name: The phase-2 record follows from its sealed forecasts"):]
    step = step[: step.index("\n      - name:", 10)]
    assert "/activity?ref=refs/heads/main" in step
    assert 'python -m de_power_live.v2.audit --pushes "$PUSHES"' in step
    assert "GH_TOKEN: ${{ github.token }}" in step
    # Read-only: the job's token cannot write, and the fetch is a GET.
    assert "contents: read" in audit and "--method" not in step and "-X" not in step

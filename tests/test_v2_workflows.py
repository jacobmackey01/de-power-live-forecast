"""What the v2 workflows must and must not do. The schedule itself is covered
by tests/test_predict_schedule.py, which runs against both seal workflows."""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

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


def _commit_script(seal_outcome: str) -> str:
    """The Commit step of predict-v2.yml as a shell script."""
    workflow = _read("predict-v2.yml")
    block = workflow[workflow.index("      - name: Commit"):]
    script = textwrap.dedent(block.split("run: |\n", 1)[1])
    return script.replace("${{ steps.seal.outcome }}", seal_outcome).replace("${{ github.run_id }}", "1")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


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
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(tmp_path, "clone", "-q", str(remote), str(work))
    _git(work, "config", "user.name", "t")
    _git(work, "config", "user.email", "t@example.invalid")
    (work / "README").write_text("x")
    _git(work, "add", "README")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "origin", "HEAD")

    for name in written:
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    result = subprocess.run(
        ["bash", "-e", "-c", _commit_script(seal_outcome)], cwd=work, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert _git(work, "log", "-1", "--format=%s").startswith(expected_subject)
    assert sorted(_git(work, "show", "--name-only", "--format=", "HEAD").split()) == sorted(written)
    # And it reached the remote, which is the whole point of the step.
    assert _git(work, "rev-parse", "HEAD") == _git(work, "rev-parse", "@{u}")
    assert _git(work, "status", "--porcelain") == ""


def test_the_commit_step_does_nothing_when_nothing_was_written(tmp_path):
    _git(tmp_path, "init", "-q")
    result = subprocess.run(["bash", "-e", "-c", _commit_script("success")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and "nothing to commit" in result.stdout


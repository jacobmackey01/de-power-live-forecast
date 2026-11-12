"""What the v2 workflows must and must not do. The schedule itself is covered
by tests/test_predict_schedule.py, which runs against both seal workflows."""

from __future__ import annotations

import re
from pathlib import Path

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
    assert "git add predictions/v2 results/v2" in v2
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


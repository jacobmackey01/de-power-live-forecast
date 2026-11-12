"""The v2 seal, scoring and report path, run end to end on synthetic history.

The point of these tests is the one docs/v1-lessons.md makes about 2026-10-25:
v1's tests checked that the day had 25 hours and never ran a 25-hour day
through the path that seals a forecast. Here every clock-change day from the
start of phase 2 to the end of 2027 goes through ``build_prediction``, leakage
guard and gate check included, then through scoring and the report.

Synthetic data, so the tests need no network and no cache. They test that the
path runs and holds its invariants, not that the model is any good.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from de_power_live import seal_guard
from de_power_live.predict import SealError, seal
from de_power_live.v2 import final_report
from de_power_live.v2.calls import POOL_DAYS, QUANTILES
from de_power_live.v2.features import LeakageError, nominal_seal_utc
from de_power_live.v2.model import ARTEFACT, ModelIntegrityError, V2Model, file_hash
from de_power_live.v2.predict import build_prediction
from de_power_live.v2.rehearse import inputs_as_of, rehearse
from de_power_live.v2.score import IncompleteOutturn, miss_entry, score_day
from de_power_live.v2.train import freeze

from conftest import BLOCKS, CLOCK_CHANGE_DAYS, FAST, TZ  # noqa: E402

# The autumn 2026 rehearsal block, as a report window.
WINDOW = dict(first=date(2026, 10, 14), end=date(2026, 10, 27))


def _all_rows(rehearsals) -> dict[str, dict]:
    return {row["delivery_date_local"]: row for block in rehearsals.values() for row in block["ledger"]}


def _payload(rehearsals, day: str) -> dict:
    for block in rehearsals.values():
        path = block["work_dir"] / f"{day}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise AssertionError(f"{day} was not rehearsed")


# ---- The dress rehearsal ----------------------------------------------------


@pytest.mark.parametrize("day,n_hours", CLOCK_CHANGE_DAYS.items())
def test_every_clock_change_day_seals_and_scores(rehearsals, day, n_hours):
    payload = _payload(rehearsals, day)
    assert payload["n_hours"] == n_hours == len(payload["predictions"])
    assert _all_rows(rehearsals)[day]["status"] == "SCORED"
    assert _all_rows(rehearsals)[day]["n_hours"] == n_hours


@pytest.mark.parametrize("day", ["2026-10-26", "2027-03-29", "2027-11-01"])
def test_the_day_after_a_clock_change_seals_too(rehearsals, day):
    """Its anchor is the odd-length day. This is the mirror of the bug v1 had."""
    payload = _payload(rehearsals, day)
    assert payload["n_hours"] == 24
    assert all(p["baseline_same_hour_yesterday"] is not None for p in payload["predictions"])


def test_the_repeated_hour_gets_the_same_call_twice(rehearsals):
    calls = _payload(rehearsals, "2026-10-25")["predictions"]
    twice = [c for c in calls if c["hour_local"].startswith("2026-10-25T02:00")]
    assert len(twice) == 2 and twice[0]["hour_utc"] != twice[1]["hour_utc"]
    for key in ("call_a_price_eur_mwh", "call_b_prob_spike", "call_c_quantiles"):
        assert twice[0][key] == twice[1][key]


def test_every_rehearsed_file_satisfies_the_seal_guard(rehearsals):
    """predict-v2.yml uses the same guard as v1. A v2 file it called invalid
    would fail every run; one it could not read would be sealed twice."""
    for block in rehearsals.values():
        for path in sorted(block["work_dir"].glob("*.json")):
            state, problems = seal_guard.inspect(path.stem, block["work_dir"])
            assert state == seal_guard.SEALED, problems


def test_quantiles_are_ordered_and_contain_sane_probabilities(rehearsals):
    for block in rehearsals.values():
        for path in block["work_dir"].glob("*.json"):
            for call in json.loads(path.read_text())["predictions"]:
                q = [call["call_c_quantiles"][str(level)] for level in QUANTILES]
                assert q == sorted(q)
                for key in ("call_b_prob_spike", "baseline_spike", "call_b_prob_negative"):
                    assert 0.0 <= call[key] <= 1.0


def test_live_errors_replace_warm_start_errors_day_by_day(rehearsals):
    block = rehearsals["autumn_2026"]
    first = json.loads((block["work_dir"] / "2026-10-14.json").read_text())["error_pool"]
    last = json.loads((block["work_dir"] / "2026-10-27.json").read_text())["error_pool"]
    # First seal: no live day yet. 13 Oct is neither sealed nor in the warm start.
    assert first["live_days"] == 0 and first["days_without_errors"] == ["2026-10-13"]
    assert last["live_days"] == 13
    assert last["live_days"] + last["warm_start_days"] + len(last["days_without_errors"]) == POOL_DAYS


# ---- Guards ---------------------------------------------------------------------


def test_a_forecast_is_refused_after_the_gate(history, rehearsals):
    prices, weather = history
    block = rehearsals["autumn_2026"]
    late = pd.Timestamp("2026-10-27 12:00", tz=TZ).tz_convert("UTC")
    with pytest.raises(SealError, match="gate closure"):
        build_prediction(inputs_as_of(prices, weather, late), late, block["model_dir"], block["work_dir"])


def test_a_forecast_is_refused_if_its_own_day_is_already_priced(history, rehearsals):
    """At a real seal, tomorrow's prices do not exist. If they do, this is not a forecast."""
    prices, weather = history
    block = rehearsals["autumn_2026"]
    now = nominal_seal_utc(pd.Timestamp("2026-10-28"))
    inputs = inputs_as_of(prices, weather, now)
    inputs.prices = prices[prices.index < pd.Timestamp("2026-10-29", tz="UTC")]
    with pytest.raises(LeakageError, match="on or after the target"):
        build_prediction(inputs, now, block["model_dir"], block["work_dir"])


def test_a_forecast_is_refused_without_the_seal_days_prices(history, rehearsals):
    prices, weather = history
    block = rehearsals["autumn_2026"]
    now = nominal_seal_utc(pd.Timestamp("2026-10-28"))
    inputs = inputs_as_of(prices, weather, now)
    inputs.prices = inputs.prices[inputs.prices.index < pd.Timestamp("2026-10-26", tz="UTC")]
    with pytest.raises(LeakageError, match="anchor"):
        build_prediction(inputs, now, block["model_dir"], block["work_dir"])


def test_a_day_cannot_be_sealed_twice(rehearsals):
    block = rehearsals["autumn_2026"]
    payload = json.loads((block["work_dir"] / "2026-10-20.json").read_text())
    with pytest.raises(SealError, match="write-once"):
        seal(payload, block["work_dir"])


def test_a_changed_artefact_is_refused(history, tmp_path):
    prices, weather = history
    freeze(prices, weather, pd.Timestamp("2026-10-12"), tmp_path, params=FAST)
    V2Model.load(tmp_path)  # verifies
    (tmp_path / ARTEFACT).write_bytes((tmp_path / ARTEFACT).read_bytes() + b" ")
    with pytest.raises(ModelIntegrityError, match="manifest hash"):
        V2Model.load(tmp_path)


def test_a_changed_hashed_source_is_refused(history, tmp_path, monkeypatch):
    prices, weather = history
    freeze(prices, weather, pd.Timestamp("2026-10-12"), tmp_path, params=FAST)
    monkeypatch.setattr("de_power_live.v2.model.source_hash", lambda: "0" * 64)
    with pytest.raises(ModelIntegrityError, match="new model version"):
        V2Model.load(tmp_path)


def test_the_frozen_model_can_be_rebuilt_bit_for_bit(history, tmp_path):
    """Two fits on the same data in the same environment give the same bytes:
    one thread and a fixed seed. This does not show that the model can be
    rebuilt on another platform; docs/phase2-disclosures.md says what has and
    has not been demonstrated."""
    prices, weather = history
    a = freeze(prices, weather, pd.Timestamp("2026-10-12"), tmp_path / "a", params=FAST)
    b = freeze(prices, weather, pd.Timestamp("2026-10-12"), tmp_path / "b", params=FAST)
    assert a["artefact_sha256"] == b["artefact_sha256"]
    assert file_hash(tmp_path / "a" / ARTEFACT) == a["artefact_sha256"][ARTEFACT]


# ---- Scoring and the report ---------------------------------------------------------


def test_scoring_waits_for_every_realised_price(history, rehearsals):
    prices, _ = history
    payload = _payload(rehearsals, "2026-10-20")
    with pytest.raises(IncompleteOutturn):
        score_day(
            payload,
            prices[prices.index < pd.Timestamp(payload["predictions"][-1]["hour_utc"])],
            check_model=False,
        )


def test_a_scored_row_carries_everything_the_report_needs(rehearsals):
    row = _all_rows(rehearsals)["2026-10-25"]
    hour = row["hours"][0]
    for key in ("price", "pred", "d1", "d7", "q", "spike", "p_spike", "base_spike", "negative", "p_negative", "base_negative"):
        assert key in hour
    assert len(hour["q"]) == len(QUANTILES)
    json.dumps(row)  # the ledger is JSON lines


def test_the_final_report_is_refused_until_the_window_is_complete(rehearsals):
    ledger = rehearsals["autumn_2026"]["ledger"]
    window = dict(first=date(2026, 10, 14), end=date(2026, 12, 31))
    with pytest.raises(final_report.WindowIncomplete):
        final_report.build_report(ledger, draws=50, **window)
    assert final_report.build_report(ledger, provisional=True, draws=50, **window)["provisional"] is True


def test_the_final_report_runs_on_a_complete_window(rehearsals, monkeypatch, tmp_path):
    ledger = list(rehearsals["autumn_2026"]["ledger"])
    ledger.insert(3, ledger.pop(0))  # row order in the file must not matter
    ledger.append(miss_entry("2026-10-28", "predict-v2 job failed in run 1: SealError: example"))
    report = final_report.build_report(
        ledger, v1_dir=tmp_path, draws=200, first=date(2026, 10, 14), end=date(2026, 10, 28)
    )

    assert report["days"]["calendar"] == 15 and report["days"]["scored"] == 14
    assert report["days"]["hours"] == 14 * 24 + 1  # one 25-hour day
    assert [m["day"] for m in report["days"]["missed"]] == ["2026-10-28"]
    for call in ("call_a", "call_b", "call_c"):
        assert report[call]["verdict"] in {"PASS", "NULL", "UNDERPOWERED"}
    assert report["versus_v1"] is None  # no v1 files in tmp_path

    text = final_report.render_markdown(report)
    assert "SealError: example" in text  # the miss explains itself
    assert "Provisional" not in text
    json.dumps(report, default=str)


def test_the_running_summary_and_the_final_report_agree(rehearsals, monkeypatch, tmp_path):
    """Phase 1's summary used a different definition of skill from its report.
    Here both are the same function, so the same ledger gives the same numbers."""
    ledger = rehearsals["autumn_2026"]["ledger"]
    final = final_report.build_report(ledger, v1_dir=tmp_path, draws=100, **WINDOW)
    running = final_report.build_report(ledger, provisional=True, v1_dir=tmp_path, draws=100, **WINDOW)
    assert final["call_a"] == running["call_a"]
    assert final["call_c"] == running["call_c"]


def test_pooled_skill_is_pooled_over_hours_not_averaged_over_days(rehearsals, monkeypatch, tmp_path):
    ledger = rehearsals["autumn_2026"]["ledger"]
    report = final_report.build_report(ledger, v1_dir=tmp_path, draws=50, **WINDOW)
    model = sum(abs(h["pred"] - h["price"]) for row in ledger for h in row["hours"])
    base = sum(abs(h["d1"] - h["price"]) for row in ledger for h in row["hours"])
    assert report["call_a"]["vs_yesterday"]["skill"] == pytest.approx(1 - model / base)


def test_a_late_start_is_reported_without_verdicts(rehearsals, monkeypatch, tmp_path):
    """The rehearsal blocks start in October, before the cut-off, so they get
    verdicts. Move the cut-off earlier and the same ledger must get none."""
    ledger = rehearsals["autumn_2026"]["ledger"]
    on_time = final_report.build_report(ledger, v1_dir=tmp_path, draws=100, **WINDOW)
    assert on_time["late_start"] is False
    assert on_time["call_a"]["verdict"] in {"PASS", "NULL"}

    monkeypatch.setattr(final_report, "LATEST_FIRST_DAY_FOR_VERDICTS", date(2026, 10, 13))
    late = final_report.build_report(ledger, v1_dir=tmp_path, draws=100, **WINDOW)
    assert late["late_start"] is True
    for call in ("call_a", "call_b", "call_c"):
        assert late[call]["verdict"] == final_report.NO_VERDICT
        assert late[call]["verdict_if_on_time"] == on_time[call]["verdict"]
    # The numbers themselves are untouched.
    assert late["call_a"]["vs_yesterday"] == on_time["call_a"]["vs_yesterday"]
    assert "**No verdicts.**" in final_report.render_markdown(late)
    assert "**No verdicts.**" not in final_report.render_markdown(on_time)


def test_the_late_start_cut_off_is_the_one_in_the_pre_registration():
    from pathlib import Path

    from de_power_live.v2 import EVALUATION_END, LATEST_FIRST_DAY_FOR_VERDICTS

    text = (Path(__file__).resolve().parents[1] / "PREREGISTRATION_PHASE2.md").read_text(encoding="utf-8")
    assert f"after {LATEST_FIRST_DAY_FOR_VERDICTS.isoformat()}" in text
    assert f"**{EVALUATION_END.isoformat()}**" in text


def test_an_empty_ledger_renders_without_error():
    report = final_report.build_report([], provisional=True)
    text = final_report.render_markdown(report)
    # The window is known from the first seal, whatever the ledger holds, and an
    # empty ledger means nothing is scored yet, not that nothing is sealed.
    assert "2026-11-13" in text and "No day has been scored yet" in text

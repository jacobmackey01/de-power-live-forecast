"""One test for each weakness the independent audit of 2026-11-12 demonstrated.

That audit did not find that the record had been tampered with. It found ways
it could be, and ways a day could vanish or a verdict come out wrong, and it
proved each one with a small experiment. These tests are those experiments,
turned round: each must now fail to do the damage.
"""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from de_power_live import gate_margin
from de_power_live.predict import SealError
from de_power_live.v2 import FIRST_DELIVERY_DAY, PREDICTIONS_DIR, audit, final_report, score
from de_power_live.v2.audit import LedgerError, check_seal, gate_utc
from de_power_live.v2.inputs import InputError, check_prices
from de_power_live.v2.predict import PUBLISH_BUFFER, build_prediction
from de_power_live.v2.rehearse import inputs_as_of
from de_power_live.v2.score import InvalidSeal, miss_entry, reconcile, score_day

TZ = "Europe/Berlin"
WINDOW = dict(first=date(2026, 10, 14), end=date(2026, 10, 27))


@pytest.fixture()
def block(rehearsals):
    """The autumn 2026 rehearsal: 14 sealed and scored days, one of 25 hours."""
    return rehearsals["autumn_2026"]


def _payload(block, day: str) -> dict:
    return json.loads((block["work_dir"] / f"{day}.json").read_text(encoding="utf-8"))


def _manifest(block) -> dict:
    return json.loads((block["model_dir"] / "MANIFEST.json").read_text(encoding="utf-8"))


def _record(block, tmp_path, rows=None):
    """A copy of the rehearsed record on disk, as the audit reads it."""
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for path in block["work_dir"].glob("*.json"):
        (predictions / path.name).write_text(path.read_text())
    ledger = tmp_path / "ledger.jsonl"
    rows = block["ledger"] if rows is None else rows
    ledger.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return dict(
        predictions_dir=predictions, ledger_path=ledger, models_dir=block["model_dir"],
        # Noon on the 28th: every rehearsed day is due and accounted for, and
        # the day after the block is not yet overdue.
        first=date(2026, 10, 14), now=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
    )


# ---- Finding 3: an appended row must not be able to replace a score -----------


def test_a_second_row_for_a_day_is_refused_not_preferred(block, tmp_path):
    """The audit appended a flattering copy of a scored day. The report kept the
    last row per day, and skill went from -0.17 to +1.00 with the ledger's old
    contents intact as a prefix."""
    ledger = list(block["ledger"])
    flattering = copy.deepcopy(ledger[3])
    for hour in flattering["hours"]:
        hour["pred"] = hour["price"]
    with pytest.raises(LedgerError, match="both for 2026-10-17"):
        final_report.build_report([*ledger, flattering], v1_dir=tmp_path, draws=20, **WINDOW)
    with pytest.raises(LedgerError):
        final_report.build_report([*ledger, flattering], provisional=True, v1_dir=tmp_path, draws=20, **WINDOW)

    problems, _ = audit.audit(**_record(block, tmp_path, [*ledger, flattering]))
    assert any("both for 2026-10-17" in p for p in problems)


def test_an_unknown_status_is_refused(block, tmp_path):
    ledger = [*block["ledger"], {"delivery_date_local": "2026-10-28", "status": "CORRECTED"}]
    with pytest.raises(LedgerError, match="status 'CORRECTED'"):
        final_report.build_report(ledger, provisional=True, v1_dir=tmp_path, draws=20, **WINDOW)


def test_a_scored_row_must_repeat_its_sealed_forecast(block, tmp_path):
    """Scores are computed from the ledger, so the ledger is held to the seal."""
    ledger = copy.deepcopy(block["ledger"])
    ledger[5]["hours"][7]["pred"] += 25.0
    problems, _ = audit.audit(**_record(block, tmp_path, ledger))
    assert any("2026-10-19 hour 7: ledger pred differs from the sealed call" in p for p in problems)


def test_the_untouched_record_passes_the_audit(block, tmp_path):
    problems, warnings = audit.audit(**_record(block, tmp_path))
    assert problems == [] and warnings == []


def test_a_missed_row_for_a_sealed_day_is_a_contradiction(block, tmp_path):
    ledger = [r for r in block["ledger"] if r["delivery_date_local"] != "2026-10-20"]
    ledger.append(miss_entry("2026-10-20", "example"))
    problems, _ = audit.audit(**_record(block, tmp_path, ledger))
    assert any("2026-10-20: MISSED in the ledger but a sealed file exists" in p for p in problems)


def test_a_missing_hour_in_a_scored_row_is_refused_by_the_report(block, tmp_path):
    """A gap must not drop out of a pooled sum without being counted."""
    ledger = copy.deepcopy(block["ledger"])
    ledger[2]["hours"][4]["pred"] = None
    with pytest.raises(LedgerError, match="missing or non-finite 'pred'"):
        final_report.build_report(ledger, provisional=True, v1_dir=tmp_path, draws=20, **WINDOW)


# ---- Finding 3, second half: what counts as a sealed file -----------------------


def test_a_real_seal_is_valid(block):
    for day in ("2026-10-20", "2026-10-25", "2026-10-26"):
        assert check_seal(_payload(block, day), day, _manifest(block)) == []


def test_24_copies_of_one_hour_are_not_a_forecast(block):
    payload = _payload(block, "2026-10-20")
    payload["predictions"] = [copy.deepcopy(payload["predictions"][0]) for _ in range(24)]
    assert any("not exactly the 24 hours" in p for p in check_seal(payload, "2026-10-20"))
    with pytest.raises(InvalidSeal):
        score_day(payload, pd.Series(dtype=float), check_model=False)


def test_a_timestamp_after_the_gate_is_caught_whatever_margin_it_claims(block):
    payload = _payload(block, "2026-10-20")
    payload["sealed_at_utc"] = (gate_utc(date(2026, 10, 20)) + timedelta(minutes=3)).isoformat()
    payload["minutes_before_close"] = 260.0
    problems = check_seal(payload, "2026-10-20")
    assert any("at or after the gate" in p for p in problems)
    assert any("minutes_before_close" in p for p in problems)


def test_other_ways_a_seal_can_be_wrong(block):
    day, manifest = "2026-10-20", _manifest(block)

    wrong_model = _payload(block, day)
    wrong_model["model_source_sha256"] = "0" * 64
    assert any("not the frozen model" in p for p in check_seal(wrong_model, day, manifest))

    crossed = _payload(block, day)
    crossed["predictions"][3]["call_c_quantiles"]["0.1"] = 9999.0
    assert any("hour 3: quantiles are not in order" in p for p in check_seal(crossed, day))

    short = _payload(block, day)
    del short["predictions"][10]
    assert any("not exactly the 24 hours" in p for p in check_seal(short, day))

    renamed = _payload(block, day)
    assert any("declares delivery day" in p for p in check_seal(renamed, "2026-10-21"))

    not_a_number = _payload(block, day)
    not_a_number["predictions"][0]["call_a_price_eur_mwh"] = float("nan")
    assert any("call_a_price_eur_mwh" in p for p in check_seal(not_a_number, day))


def test_the_25_hour_day_must_have_25_distinct_hours(block):
    payload = _payload(block, "2026-10-25")
    assert len(payload["predictions"]) == 25
    del payload["predictions"][2]
    assert any("not exactly the 25 hours" in p for p in check_seal(payload, "2026-10-25"))


def test_a_forecast_that_reached_the_repository_after_its_gate_is_reported(block, tmp_path):
    record = _record(block, tmp_path)
    published = {"2026-10-20": gate_utc(date(2026, 10, 20)) + timedelta(minutes=10)}
    problems, _ = audit.audit(**record, published=published)
    assert any("2026-10-20: the sealed file entered the repository" in p for p in problems)
    on_time = {"2026-10-20": gate_utc(date(2026, 10, 20)) - timedelta(minutes=10)}
    assert audit.audit(**record, published=on_time)[0] == []


# ---- Finding 2: the gate at the moment of sealing --------------------------------


def test_a_build_that_crosses_the_gate_is_refused(history, block):
    """The audit started a build at 10:59:59 UTC and finished it at 11:00:01.
    The gate had been checked once, at the start, and the forecast was written."""
    prices, weather = history
    close = gate_utc(date(2026, 10, 28))
    started = pd.Timestamp(close) - pd.Timedelta(seconds=1)
    with pytest.raises(SealError, match="at or after the gate"):
        build_prediction(
            inputs_as_of(prices, weather, started), started, block["model_dir"], block["work_dir"],
            sealed_at_utc=pd.Timestamp(close) + pd.Timedelta(seconds=1),
        )


def test_a_forecast_too_late_to_publish_is_refused(history, block):
    prices, weather = history
    close = pd.Timestamp(gate_utc(date(2026, 10, 28)))
    started = close - pd.Timedelta(minutes=30)
    with pytest.raises(SealError, match="too late to be published"):
        build_prediction(
            inputs_as_of(prices, weather, started), started, block["model_dir"], block["work_dir"],
            sealed_at_utc=close - PUBLISH_BUFFER + pd.Timedelta(seconds=1),
        )
    payload = build_prediction(
        inputs_as_of(prices, weather, started), started, block["model_dir"], block["work_dir"],
        sealed_at_utc=close - PUBLISH_BUFFER - pd.Timedelta(seconds=1),
    )
    assert payload["minutes_before_close"] >= 5.0


# ---- Findings 4 and 5: two verdict rules ------------------------------------------


def _coverage_days(inside_50: int, inside_80: int, n: int = 100) -> list[dict]:
    """One day of ``n`` hours with exactly that many inside each interval."""
    price = np.zeros(n)
    q = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), (n, 1))
    q[inside_80:] = [5.0, 6.0, 7.0, 8.0, 9.0]  # outside both intervals
    q[inside_50:inside_80, 1], q[inside_50:inside_80, 3] = 0.5, 0.9  # inside 80%, outside 50%
    return [{"price": price, "q": q}]


@pytest.mark.parametrize(
    "inside_50,inside_80,passes",
    [(55, 80, True), (45, 80, True), (50, 75, True), (50, 85, True), (56, 80, False), (50, 74, False), (50, 86, False)],
)
def test_exactly_five_points_from_nominal_is_inside_the_band(inside_50, inside_80, passes):
    """0.55 - 0.50 is not exactly 0.05 in floating point. Compared without
    slack it fell outside the band it is on the edge of, and turned what should
    be a pass into a null."""
    result = final_report.call_c(_coverage_days(inside_50, inside_80), ["2026-12"], np.zeros((10, 1), dtype=int))
    assert result["intervals"]["50"]["coverage"] == pytest.approx(inside_50 / 100)
    assert result["intervals"]["80"]["coverage"] == pytest.approx(inside_80 / 100)
    assert (result["verdict"] == "PASS") is passes


def test_the_window_opens_at_the_first_seal_not_the_first_ledger_row(block, tmp_path):
    """A MISSED row before any forecast existed used to open the window, which
    lengthened it and switched off the late-start rule."""
    ledger = [miss_entry("2026-10-01", "no forecast yet"), *block["ledger"]]
    report = final_report.build_report(ledger, v1_dir=tmp_path, draws=50, **WINDOW)
    assert report["window"]["first"] == "2026-10-14"
    assert report["days"]["calendar"] == 14 and report["days"]["missed"] == []

    late = dict(first=date(2026, 12, 16), end=date(2026, 12, 16))
    row = copy.deepcopy(block["ledger"][0])
    row["delivery_date_local"] = "2026-12-16"
    ledger = [miss_entry("2026-12-01", "no forecast yet"), row]
    report = final_report.build_report(ledger, v1_dir=tmp_path, draws=50, **late)
    assert report["late_start"] is True
    assert report["call_a"]["verdict"] == final_report.NO_VERDICT


def test_the_registered_window_opens_with_the_first_file_in_the_record():
    sealed = sorted(p.stem for p in PREDICTIONS_DIR.glob("????-??-??.json"))
    assert sealed, "the repository should hold at least the first v2 forecast"
    assert sealed[0] == FIRST_DELIVERY_DAY.isoformat()


# ---- Findings 1 and 7: no day may leave the record --------------------------------


def test_a_due_day_with_no_forecast_and_no_row_is_entered_as_missed(tmp_path):
    first = date(2026, 12, 1)
    (tmp_path / "2026-12-01.json").write_text("{}")
    ledger = [miss_entry("2026-12-02", "recorded at the time")]
    # 4 Dec 12:00 UTC: the gates for delivery on the 1st to the 5th have closed.
    now = datetime(2026, 12, 4, 12, 0, tzinfo=timezone.utc)
    added = reconcile(ledger, tmp_path, now, first)
    assert [row["delivery_date_local"] for row in added] == ["2026-12-03", "2026-12-04", "2026-12-05"]
    assert all(row["status"] == "MISSED" and "published before gate closure" in row["reason"] for row in added)
    # A second run adds nothing.
    assert reconcile([*ledger, *added], tmp_path, now, first) == []


def test_a_day_is_not_called_missed_until_well_after_its_gate(tmp_path):
    first = date(2026, 12, 1)
    gate = gate_utc(date(2026, 12, 1))  # 30 Nov, 11:00 UTC
    assert reconcile([], tmp_path, gate + timedelta(minutes=44), first) == []
    assert len(reconcile([], tmp_path, gate + timedelta(minutes=46), first)) == 1
    assert reconcile([], tmp_path, gate - timedelta(days=3), first) == []


def test_the_audit_reports_a_day_missing_from_the_record(block, tmp_path):
    """Before reconciliation existed, a seal job that never ran left nothing
    behind. If that ever happens again the audit says so."""
    ledger = [r for r in block["ledger"] if r["delivery_date_local"] != "2026-10-22"]
    record = _record(block, tmp_path, ledger)
    (record["predictions_dir"] / "2026-10-22.json").unlink()
    problems, _ = audit.audit(**record)
    assert any("2026-10-22" in p and "missing from the record" in p for p in problems)


def test_the_audit_warns_before_the_error_pool_runs_dry(block, tmp_path):
    """Below 200 recent errors the model refuses to seal, and it cannot get new
    errors without sealing. The warning has to come while there is time."""
    record = _record(block, tmp_path)
    newest = record["predictions_dir"] / "2026-10-27.json"
    payload = json.loads(newest.read_text())
    payload["error_pool"]["n_errors"] = 300
    newest.write_text(json.dumps(payload))
    ledger = copy.deepcopy(block["ledger"])
    _, warnings = audit.audit(**{**record, "ledger_path": record["ledger_path"]})
    assert any("only 300 recent forecast errors" in w for w in warnings)
    assert ledger  # unchanged


def test_the_alarm_notices_a_model_that_has_stopped_sealing():
    """The audit gave the alarm a 2020 seal with a healthy margin and it passed."""
    after_gate = datetime(2026, 12, 3, 11, 30, tzinfo=timezone.utc)  # 12:30 Berlin
    before_gate = datetime(2026, 12, 3, 9, 0, tzinfo=timezone.utc)  # 10:00 Berlin
    assert gate_margin.expected_delivery(after_gate) == "2026-12-04"
    assert gate_margin.expected_delivery(before_gate) == "2026-12-03"
    assert gate_margin.is_stale([("2020-01-01", 260.0)], before_gate)
    assert gate_margin.is_stale([("2026-12-03", 260.0)], after_gate)
    assert not gate_margin.is_stale([("2026-12-04", 260.0)], after_gate)
    assert not gate_margin.is_stale([("2026-12-03", 260.0)], before_gate)
    # Just after local midnight the day rolls and nothing new is due until noon.
    assert not gate_margin.is_stale([("2026-12-04", 260.0)], datetime(2026, 12, 3, 23, 30, tzinfo=timezone.utc))


def test_the_alarm_fails_on_a_stale_seal_only_when_asked_to(tmp_path):
    (tmp_path / "2020-01-01.json").write_text(json.dumps({"delivery_date_local": "2020-01-01", "minutes_before_close": 260.0}))
    assert gate_margin.main(["--predictions-dir", str(tmp_path)]) == 0
    assert gate_margin.main(["--predictions-dir", str(tmp_path), "--expect-current"]) == 1


# ---- Finding 6: inputs with holes ----------------------------------------------------


def _hourly(start: str, hours: int) -> pd.Series:
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.Series(100.0, index=index)


def test_a_hole_in_the_price_history_is_refused():
    prices = _hourly("2026-10-01", 24 * 40)
    assert check_prices(prices)["ok"] is True
    with pytest.raises(InputError, match="1 hour"):
        check_prices(prices.drop(prices.index[-30]))
    with pytest.raises(InputError, match="24 hour"):
        check_prices(prices.drop(prices.index[-200:-176]))
    with pytest.raises(InputError, match="repeat"):
        check_prices(pd.concat([prices, prices.iloc[-5:-4]]).sort_index())
    with pytest.raises(InputError, match="outside"):
        check_prices(prices.where(prices.index != prices.index[10], 9999.0))


def test_a_seal_is_refused_when_one_ordinary_hour_of_prices_is_missing(history, block):
    """The feature grid fills a missing 02:00 on any day, not only a 23-hour
    one. The audit removed an ordinary day's 02:00 and a forecast was sealed
    anyway, up to 15 EUR/MWh different."""
    prices, weather = history
    now = pd.Timestamp("2026-10-27 07:40", tz=TZ).tz_convert("UTC")
    inputs = inputs_as_of(prices, weather, now)
    hole = pd.Timestamp("2026-10-27 02:00", tz=TZ).tz_convert("UTC")
    inputs.prices = inputs.prices.drop(hole)
    with pytest.raises(SealError, match="price history is not complete"):
        build_prediction(inputs, now, block["model_dir"], block["work_dir"], sealed_at_utc=now)


def test_a_seal_is_refused_when_the_seal_days_weather_is_missing(history, block):
    prices, weather = history
    now = pd.Timestamp("2026-10-27 07:40", tz=TZ).tz_convert("UTC")
    inputs = inputs_as_of(prices, weather, now)
    seal_day = inputs.weather.index.tz_convert(TZ).normalize().tz_localize(None) == pd.Timestamp("2026-10-27")
    inputs.weather = inputs.weather[~seal_day]
    with pytest.raises(SealError, match="model input"):
        build_prediction(inputs, now, block["model_dir"], block["work_dir"], sealed_at_utc=now)


# ---- Finding 10: what the report promises regardless --------------------------------


def test_brier_and_reliability_are_reported_with_no_spike_at_all(block, tmp_path):
    ledger = copy.deepcopy(block["ledger"])
    for row in ledger:
        for hour in row["hours"]:
            hour["spike"] = 0
    report = final_report.build_report(ledger, v1_dir=tmp_path, draws=20, **WINDOW)
    spike = report["call_b"]["spike"]
    assert spike["events"] == 0 and report["call_b"]["verdict"] == "UNDERPOWERED"
    assert spike["brier_model"] >= 0 and spike["brier_baseline"] >= 0
    assert spike["reliability"] and "pr_auc_model" not in spike
    text = final_report.render_markdown(report)
    assert "Brier score: model" in text and "PR-AUC is undefined" in text


def test_each_model_version_is_reported_on_its_own(block, tmp_path):
    ledger = copy.deepcopy(block["ledger"])
    for row in ledger[7:]:
        row["model_version"], row["model_source_sha256"] = "v3", "f" * 64
    report = final_report.build_report(ledger, v1_dir=tmp_path, draws=20, **WINDOW)
    assert len(report["by_version"]) == 2
    assert sum(v["days"] for v in report["by_version"].values()) == 14
    assert "More than one model version sealed" in final_report.render_markdown(report)


def test_the_final_report_is_written_once_and_says_when(block, tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("".join(json.dumps(r) + "\n" for r in block["ledger"]))
    monkeypatch.setattr(final_report, "FIRST_DELIVERY_DAY", WINDOW["first"])
    monkeypatch.setattr(final_report, "EVALUATION_END", WINDOW["end"])
    monkeypatch.setattr(final_report, "BOOTSTRAP_DRAWS", 50)
    monkeypatch.setattr(final_report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(final_report, "REPORT_PATH", tmp_path / "FINAL_REPORT.md")
    monkeypatch.setattr(final_report, "JSON_PATH", tmp_path / "final_report.json")

    assert final_report.main(["--ledger", str(ledger_path)]) == 0
    first = (tmp_path / "FINAL_REPORT.md").read_text()
    assert "Generated 20" in first and "due by 2027-03-03" in first
    saved = json.loads((tmp_path / "final_report.json").read_text())
    assert saved["late"] is False and saved["generated_at_utc"]

    ledger_path.write_text("")  # whatever happens to the ledger afterwards
    assert final_report.main(["--ledger", str(ledger_path)]) == 0
    assert (tmp_path / "FINAL_REPORT.md").read_text() == first


def test_scoring_reaches_back_as_far_as_the_oldest_unscored_day():
    today = date(2027, 1, 20)
    assert score.price_blocks_for(date(2027, 1, 18), today) == score.MIN_PRICE_BLOCKS
    assert score.price_blocks_for(date(2026, 11, 13), today) == 12
    assert score.price_blocks_for(date(2020, 1, 1), today) == score.MAX_PRICE_BLOCKS

"""The phase-2 verdicts, computed from the v2 ledger alone.

    python -m de_power_live.v2.final_report                # writes once the window is complete
    python -m de_power_live.v2.final_report --provisional  # prints the record so far

Written and tested before the first v2 forecast was sealed. Phase 1 had a fixed
end date and no code that produced its verdict until three days before it; the
running summary also used a different definition of skill from the one that had
been declared. Here the running summary (results/v2/SUMMARY.md) is this report
in provisional mode, so the two cannot disagree.

Criteria, from PREREGISTRATION_PHASE2.md section 3:

A  pooled MAE skill against the same hour yesterday > 0, 95% CI excluding 0
B  PR-AUC for spike hours above the hour-of-day baseline, 95% CI excluding 0;
   UNDERPOWERED below 30 events
C  pooled coverage of the 50% and 80% intervals within 5 points of nominal
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from . import (
    EVALUATION_END,
    FIRST_DELIVERY_DAY,
    LATEST_FIRST_DAY_FOR_VERDICTS,
    LEDGER_PATH,
    REPO_ROOT,
    REPORT_DUE,
    RESULTS_DIR,
)
from .audit import LedgerError, check_ledger
from .calls import QUANTILES
from .score import load_ledger

REPORT_PATH = RESULTS_DIR / "FINAL_REPORT.md"
JSON_PATH = RESULTS_DIR / "final_report.json"
V1_PREDICTIONS_DIR = REPO_ROOT / "predictions"

BOOTSTRAP_SEED = 20270303
BOOTSTRAP_DRAWS = 10_000
MIN_EVENTS = 30
COVERAGE_TOLERANCE = 0.05
NOMINAL = {"80": 0.80, "50": 0.50}
INTERVAL_COLUMNS = {"80": (0, 4), "50": (1, 3)}
PIT_EXPECTED = np.diff([0.0, *QUANTILES, 1.0])
DESCRIPTIVE = (("negative", "< 0 EUR/MWh"), ("below_10", "< 10 EUR/MWh"))


NO_VERDICT = "NO VERDICT"


class WindowIncomplete(RuntimeError):
    """Raised when a final report is asked for before every window day is settled."""


def _interval(values: np.ndarray) -> list[float]:
    values = values[~np.isnan(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _wilson(hits: int, n: int, z: float = 1.959964) -> list[float]:
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def window_rows(
    ledger: list[dict], provisional: bool, first: date, end: date
) -> tuple[list[dict], list[str]]:
    """Ledger rows inside ``[first, end]`` in date order, and the days still open.

    ``first`` is the delivery day of the first sealed prediction. It is passed
    in, not read off the ledger: the earliest ledger row can be a MISSED day
    before any forecast existed, and taking the window from it would both
    lengthen the window and defeat the late-start rule.
    """
    check_ledger(ledger)
    by_day = {r["delivery_date_local"]: r for r in ledger}
    outstanding, day = [], first
    while day <= end:
        if day.isoformat() not in by_day:
            outstanding.append(day.isoformat())
        day += timedelta(days=1)
    if outstanding and not provisional:
        raise WindowIncomplete(
            f"{len(outstanding)} window day(s) not yet SCORED or MISSED, "
            f"from {outstanding[0]} to {outstanding[-1]}"
        )
    inside = [r for d, r in sorted(by_day.items()) if first <= date.fromisoformat(d) <= end]
    return inside, outstanding


def _arrays(row: dict) -> dict[str, np.ndarray]:
    hours = row["hours"]

    def column(key: str) -> np.ndarray:
        return np.array([np.nan if h[key] is None else h[key] for h in hours], dtype=float)

    out = {key: column(key) for key in ("price", "pred", "d1", "d7", "spike", "p_spike", "base_spike")}
    for name, _ in DESCRIPTIVE:
        for key in (name, f"p_{name}", f"base_{name}"):
            out[key] = column(key)
    out["q"] = np.array([h["q"] for h in hours], dtype=float)
    # Every scored hour must carry the realised price and all of the sealed
    # calls. A gap would otherwise drop out of a sum without being counted,
    # which changes a pooled figure silently.
    for key in ("price", "pred", "d1", "p_spike", "base_spike", "q"):
        if not np.isfinite(out[key]).all():
            raise LedgerError(f"{row['delivery_date_local']}: scored row has a missing or non-finite {key!r}")
    stamps = [h["hour_utc"] for h in hours]
    if len(set(stamps)) != len(stamps):
        raise LedgerError(f"{row['delivery_date_local']}: scored row repeats an hour")
    return out


def _skill(days: list[dict], base: str, draws: np.ndarray) -> dict:
    """Pooled MAE skill against ``base`` over the hours where the baseline exists."""
    model = np.array([np.nansum(np.where(np.isnan(d[base]), np.nan, np.abs(d["pred"] - d["price"]))) for d in days])
    other = np.array([np.nansum(np.abs(d[base] - d["price"])) for d in days])
    hours = np.array([np.sum(~np.isnan(d[base])) for d in days])
    boot = 1 - model[draws].sum(axis=1) / other[draws].sum(axis=1)
    return {
        "hours": int(hours.sum()),
        "mae_model": float(model.sum() / hours.sum()),
        "mae_baseline": float(other.sum() / hours.sum()),
        "skill": float(1 - model.sum() / other.sum()),
        "skill_ci95": _interval(boot),
    }


def call_a(days: list[dict], months: list[str], draws: np.ndarray) -> dict:
    yesterday = _skill(days, "d1", draws)
    last_week = _skill(days, "d7", draws)
    err = np.concatenate([d["pred"] - d["price"] for d in days])
    by_month = {}
    for month in sorted(set(months)):
        subset = [d for d, m in zip(days, months) if m == month]
        model = sum(np.abs(d["pred"] - d["price"]).sum() for d in subset)
        base = sum(np.abs(d["d1"] - d["price"]).sum() for d in subset)
        n = sum(len(d["price"]) for d in subset)
        by_month[month] = {
            "days": len(subset),
            "mae_model": float(model / n),
            "mae_yesterday": float(base / n),
            "skill_vs_yesterday": float(1 - model / base),
        }
    return {
        "vs_yesterday": yesterday,
        "vs_last_week": last_week,
        "rmse_model": float(np.sqrt(np.mean(err**2))),
        "bias_model": float(np.mean(err)),
        "by_month": by_month,
        "verdict": "PASS" if yesterday["skill_ci95"][0] > 0 else "NULL",
    }


def _pr_auc(days: list[dict], outcome: str, score: str, base: str, draws: np.ndarray) -> dict:
    y = np.concatenate([d[outcome] for d in days]).astype(int)
    s = np.concatenate([d[score] for d in days])
    b = np.concatenate([d[base] for d in days])
    # Brier is defined with no event at all, and the pre-registration promises
    # it whatever happens, so it is computed before the event count is looked at.
    entry = {
        "events": int(y.sum()),
        "hours": int(len(y)),
        "brier_model": float(np.mean((s - y) ** 2)),
        "brier_baseline": float(np.mean((b - y) ** 2)),
        "bootstrap_draws": int(len(draws)),
    }
    if y.sum() == 0:
        return entry
    entry["pr_auc_model"] = float(average_precision_score(y, s))
    entry["pr_auc_baseline"] = float(average_precision_score(y, b))
    entry["difference"] = entry["pr_auc_model"] - entry["pr_auc_baseline"]

    bounds = np.cumsum([0, *(len(d[outcome]) for d in days)])
    index = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(days))]

    def gap(sample: np.ndarray) -> float:
        rows = np.concatenate([index[i] for i in sample])
        if y[rows].sum() == 0:
            return np.nan  # PR-AUC is undefined without an event
        return average_precision_score(y[rows], s[rows]) - average_precision_score(y[rows], b[rows])

    boot = np.array([gap(sample) for sample in draws])
    entry["difference_ci95"] = _interval(boot)
    entry["draws_without_events"] = int(np.isnan(boot).sum())
    return entry


def _reliability(days: list[dict], outcome: str, score: str) -> list[dict]:
    y = np.concatenate([d[outcome] for d in days])
    s = np.concatenate([d[score] for d in days])
    edges = np.linspace(0, 1, 11)
    bins = np.clip(np.digitize(s, edges[1:-1]), 0, 9)
    return [
        {
            "bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
            "hours": int((bins == i).sum()),
            "mean_forecast": float(s[bins == i].mean()),
            "observed_rate": float(y[bins == i].mean()),
        }
        for i in range(10)
        if (bins == i).any()
    ]


def call_b(days: list[dict], draws: np.ndarray) -> dict:
    spike = _pr_auc(days, "spike", "p_spike", "base_spike", draws)
    spike["reliability"] = _reliability(days, "spike", "p_spike")
    if spike["events"] < MIN_EVENTS:
        verdict = "UNDERPOWERED"
    else:
        verdict = "PASS" if spike["difference_ci95"][0] > 0 else "NULL"
    descriptive = {
        name: {"label": label, **_pr_auc(days, name, f"p_{name}", f"base_{name}", draws[:1000])}
        for name, label in DESCRIPTIVE
    }
    return {"spike": spike, "descriptive": descriptive, "verdict": verdict}


def call_c(days: list[dict], months: list[str], draws: np.ndarray) -> dict:
    truth = np.concatenate([d["price"] for d in days])
    q = np.concatenate([d["q"] for d in days])
    intervals = {}
    for name, (lo, hi) in INTERVAL_COLUMNS.items():
        hits_by_day = np.array([np.sum((d["price"] >= d["q"][:, lo]) & (d["price"] <= d["q"][:, hi])) for d in days])
        hours_by_day = np.array([len(d["price"]) for d in days])
        coverage = float(hits_by_day.sum() / hours_by_day.sum())
        by_month = {}
        for month in sorted(set(months)):
            pick = np.array([m == month for m in months])
            by_month[month] = float(hits_by_day[pick].sum() / hours_by_day[pick].sum())
        intervals[name] = {
            "nominal": NOMINAL[name],
            "coverage": coverage,
            "binomial_ci95": _wilson(int(hits_by_day.sum()), int(hours_by_day.sum())),
            "day_block_ci95": _interval(hits_by_day[draws].sum(axis=1) / hours_by_day[draws].sum(axis=1)),
            "mean_width": float(np.mean(q[:, hi] - q[:, lo])),
            # Inclusive: exactly five points away is inside the band. The
            # slack is for floating point (0.55 - 0.50 is not exactly 0.05).
            "within_tolerance": bool(abs(coverage - NOMINAL[name]) <= COVERAGE_TOLERANCE + 1e-9),
            "by_month": by_month,
        }
    band = np.sum(truth[:, None] > q, axis=1)
    pit = [int((band == i).sum()) for i in range(len(QUANTILES) + 1)]
    pinball = {
        str(level): float(np.mean(np.maximum(level * (truth - q[:, i]), (level - 1) * (truth - q[:, i]))))
        for i, level in enumerate(QUANTILES)
    }
    passed = all(v["within_tolerance"] for v in intervals.values())
    return {
        "intervals": intervals,
        "pit_counts": pit,
        "pit_expected": [float(len(truth) * p) for p in PIT_EXPECTED],
        "pinball": pinball,
        "verdict": "PASS" if passed else "NULL",
    }


def versus_v1(rows: list[dict], v1_dir: Path, draws: int = BOOTSTRAP_DRAWS) -> dict | None:
    """Paired MAE difference on the hours both models sealed. Descriptive only."""
    diffs, hours = [], []
    for row in rows:
        path = v1_dir / f"{row['delivery_date_local']}.json"
        if not path.exists():
            continue
        v1 = {p["hour_utc"]: p["call_a_price_eur_mwh"] for p in json.loads(path.read_text(encoding="utf-8"))["predictions"]}
        shared = [h for h in row["hours"] if h["hour_utc"] in v1]
        if not shared:
            continue
        diffs.append(sum(abs(h["pred"] - h["price"]) - abs(v1[h["hour_utc"]] - h["price"]) for h in shared))
        hours.append(len(shared))
    if not diffs:
        return None
    diffs, hours = np.array(diffs), np.array(hours)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample = rng.integers(0, len(diffs), size=(draws, len(diffs)))
    return {
        "days": int(len(diffs)),
        "bootstrap_draws": int(draws),
        "mae_difference_v2_minus_v1": float(diffs.sum() / hours.sum()),
        "ci95": _interval(diffs[sample].sum(axis=1) / hours[sample].sum(axis=1)),
    }


def by_version(scored: list[dict], days: list[dict]) -> dict:
    """The headline figures for each model version on its own (section 4)."""
    out = {}
    labels = [f"{r['model_version']} ({r['model_source_sha256'][:12]})" for r in scored]
    for label in sorted(set(labels)):
        subset = [d for d, name in zip(days, labels) if name == label]
        model = sum(np.abs(d["pred"] - d["price"]).sum() for d in subset)
        base = sum(np.abs(d["d1"] - d["price"]).sum() for d in subset)
        hours = sum(len(d["price"]) for d in subset)
        truth = np.concatenate([d["price"] for d in subset])
        q = np.concatenate([d["q"] for d in subset])
        out[label] = {
            "days": len(subset),
            "hours": int(hours),
            "mae_model": float(model / hours),
            "mae_yesterday": float(base / hours),
            "skill_vs_yesterday": float(1 - model / base),
            "coverage_50": float(np.mean((truth >= q[:, 1]) & (truth <= q[:, 3]))),
            "coverage_80": float(np.mean((truth >= q[:, 0]) & (truth <= q[:, 4]))),
            "spike_events": int(sum(d["spike"].sum() for d in subset)),
        }
    return out


def build_report(
    ledger: list[dict],
    provisional: bool = False,
    v1_dir: Path = V1_PREDICTIONS_DIR,
    draws: int = BOOTSTRAP_DRAWS,
    first: date | None = None,
    end: date | None = None,
) -> dict:
    """The phase-2 report. ``first`` and ``end`` default to the registered
    window; a rehearsal over another period passes its own."""
    first = first or FIRST_DELIVERY_DAY
    end = end or EVALUATION_END
    rows, outstanding = window_rows(ledger, provisional, first, end)
    scored = [r for r in rows if r["status"] == "SCORED"]
    missed = [r for r in rows if r["status"] == "MISSED"]
    report: dict = {
        "provisional": provisional,
        "window": {"first": first.isoformat(), "end": end.isoformat()},
        "days": {
            "calendar": (end - first).days + 1,
            "scored": len(scored),
            "missed": [{"day": r["delivery_date_local"], "reason": r.get("reason", "")} for r in missed],
            "outstanding": len(outstanding),
        },
        "bootstrap": {"draws": draws, "seed": BOOTSTRAP_SEED, "block": "one power day"},
    }
    if not scored:
        return report

    days = [_arrays(r) for r in scored]
    months = [r["delivery_date_local"][:7] for r in scored]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.integers(0, len(days), size=(draws, len(days)))
    margins = [r["minutes_before_close"] for r in scored]
    report["days"]["hours"] = int(sum(len(d["price"]) for d in days))
    report["seal_margin_minutes"] = {"min": float(min(margins)), "median": float(np.median(margins))}
    report["model_versions"] = sorted({(r["model_version"], r["model_source_sha256"][:12]) for r in scored})
    report["by_version"] = by_version(scored, days)
    report["call_a"] = call_a(days, months, resamples)
    report["call_b"] = call_b(days, resamples)
    report["call_c"] = call_c(days, months, resamples)
    report["versus_v1"] = versus_v1(scored, v1_dir, draws)
    # Section 6: a window that starts too late is reported, not judged.
    report["late_start"] = first > LATEST_FIRST_DAY_FOR_VERDICTS
    if report["late_start"]:
        for call in ("call_a", "call_b", "call_c"):
            report[call]["verdict_if_on_time"] = report[call]["verdict"]
            report[call]["verdict"] = NO_VERDICT
    return report


# ---- rendering ---------------------------------------------------------------


def _ci(pair: list[float], fmt: str = "+.3f") -> str:
    return f"[{pair[0]:{fmt}}, {pair[1]:{fmt}}]"


def render_markdown(report: dict, title: str = "Final assessment, phase 2") -> str:
    out = [f"# {title}: DE-LU day-ahead forecasting record, model v2", ""]
    window, days = report["window"], report["days"]

    out += [
        f"Evaluation window **{window['first']} to {window['end']}**, judged against the criteria "
        "frozen in [`PREREGISTRATION_PHASE2.md`](../../PREREGISTRATION_PHASE2.md) before the first "
        "v2 prediction. Computed by `python -m de_power_live.v2.final_report` from "
        "[`ledger.jsonl`](ledger.jsonl) alone.",
        "",
    ]
    if report["provisional"]:
        out += [
            f"**Provisional.** {days['outstanding']} window day(s) are not yet scored. The numbers "
            "below are the record so far, on the final report's own definitions; no verdict is "
            "final until the window closes.",
            "",
        ]
    missed = ", ".join(m["day"] for m in days["missed"]) or "none"
    out += [
        "## The record",
        "",
        f"- Calendar days in the window: **{days['calendar']}**",
        f"- Scored: **{days['scored']}** days" + (f", {days['hours']} hours" if "hours" in days else ""),
        f"- Missed: **{len(days['missed'])}** ({missed})",
    ]
    for m in days["missed"]:
        out.append(f"  - {m['day']}: {m['reason']}")
    if "call_a" not in report:
        out += [
            "",
            "No day has been scored yet. A forecast is scored about 30 hours after its delivery day "
            "ends, so sealed forecasts can exist before anything appears here.",
            "",
        ]
        return "\n".join(out)
    margin = report["seal_margin_minutes"]
    out += [
        f"- Seal margin before gate closure: minimum {margin['min']:.0f} min, median {margin['median']:.0f} min",
        "- Model version(s): " + ", ".join(f"`{v}` (source `{h}`)" for v, h in report["model_versions"]),
        f"- Bootstrap: {report['bootstrap']['draws']:,} draws, block = one power day, seed {report['bootstrap']['seed']}",
    ]
    if report.get("generated_at_utc"):
        line = f"- Generated {report['generated_at_utc'][:16].replace('T', ' ')} UTC; due by {report['due_by']}."
        if report.get("late"):
            line += (
                " **This report is late.** It could not be written until every window day was "
                "scored or recorded missed; the last of them was settled after the due date."
            )
        out.append(line)
    out.append("")

    a, b, c = report["call_a"], report["call_b"], report["call_c"]
    label = "So far" if report["provisional"] else "Verdict"
    if report.get("late_start"):
        out += [
            f"**No verdicts.** The first v2 forecast was for {window['first']}, after "
            f"{LATEST_FIRST_DAY_FOR_VERDICTS.isoformat()}. The pre-registration (section 6) says a "
            "window that starts this late is reported descriptively only. The results below are "
            "complete, and none of them is a pass or a null.",
            "",
        ]
    spike = b["spike"]
    b_result = (
        f"{spike['pr_auc_model']:.3f} vs {spike['pr_auc_baseline']:.3f}, difference "
        f"{spike['difference']:+.3f} {_ci(spike['difference_ci95'])}"
        if "pr_auc_model" in spike
        else "no spike hour yet"
    )
    out += [
        "## Verdicts",
        "",
        f"| Call | Criterion | Result | {label} |",
        "|---|---|---|---|",
        f"| **A** price level | MAE skill vs same hour yesterday > 0, 95% CI excluding 0 | "
        f"{a['vs_yesterday']['skill']:+.3f} {_ci(a['vs_yesterday']['skill_ci95'])} | **{a['verdict']}** |",
        f"| **B** spike hours | PR-AUC beats the hour-of-day baseline, CI excluding 0, at least "
        f"{MIN_EVENTS} events | {b_result} ({spike['events']} events) | **{b['verdict']}** |",
        f"| **C** intervals | 50% and 80% coverage within 5 points of nominal | "
        f"50%: {c['intervals']['50']['coverage']:.3f}, 80%: {c['intervals']['80']['coverage']:.3f} | **{c['verdict']}** |",
        "",
        "## Call A: price level",
        "",
        "| | MAE (EUR/MWh) | Skill of model | 95% CI |",
        "|---|---|---|---|",
        f"| Model | {a['vs_yesterday']['mae_model']:.2f} | | |",
        f"| Same hour yesterday (tested) | {a['vs_yesterday']['mae_baseline']:.2f} | "
        f"{a['vs_yesterday']['skill']:+.3f} | {_ci(a['vs_yesterday']['skill_ci95'])} |",
        f"| Same hour last week (reported) | {a['vs_last_week']['mae_baseline']:.2f} | "
        f"{a['vs_last_week']['skill']:+.3f} | {_ci(a['vs_last_week']['skill_ci95'])} |",
        "",
        f"Model RMSE {a['rmse_model']:.2f} EUR/MWh, bias {a['bias_model']:+.2f} EUR/MWh "
        "(negative = forecast below outturn).",
        "",
        "| Month | Days | Model MAE | Yesterday MAE | Skill |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {month} | {m['days']} | {m['mae_model']:.2f} | {m['mae_yesterday']:.2f} | {m['skill_vs_yesterday']:+.3f} |"
        for month, m in a["by_month"].items()
    ]
    versions = report.get("by_version", {})
    if len(versions) > 1:
        out += [
            "",
            "More than one model version sealed in this window. Each on its own (the figures above pool them):",
            "",
            "| Version | Days | Model MAE | Yesterday MAE | Skill | 50% coverage | 80% coverage |",
            "|---|---|---|---|---|---|---|",
        ]
        out += [
            f"| `{name}` | {v['days']} | {v['mae_model']:.2f} | {v['mae_yesterday']:.2f} | "
            f"{v['skill_vs_yesterday']:+.3f} | {v['coverage_50']:.3f} | {v['coverage_80']:.3f} |"
            for name, v in versions.items()
        ]
    v1 = report.get("versus_v1")
    if v1:
        out += [
            "",
            f"Against v1 on the {v1['days']} days both sealed (descriptive, no verdict): v2's MAE is "
            f"{v1['mae_difference_v2_minus_v1']:+.2f} EUR/MWh relative to v1's, 95% CI {_ci(v1['ci95'], '+.2f')}.",
        ]

    out += ["", "## Call B: spike hours", ""]
    out.append(
        "A spike hour is one whose price is above the 90th percentile of the hourly prices of the "
        "28 delivery days before it. The baseline is how often that wall-clock hour was above the "
        "same threshold over the same days. Both were sealed with the forecast."
    )
    out += [
        "",
        f"Brier score: model {spike['brier_model']:.4f}, baseline {spike['brier_baseline']:.4f}."
        + (
            f" Bootstrap draws without an event: {spike['draws_without_events']}."
            if "draws_without_events" in spike
            else " No spike hour occurred, so PR-AUC is undefined."
        ),
        "",
        "| Forecast P(spike) | Hours | Mean forecast | Observed rate |",
        "|---|---|---|---|",
    ]
    out += [
        f"| {r['bin']} | {r['hours']} | {r['mean_forecast']:.3f} | {r['observed_rate']:.3f} |"
        for r in spike["reliability"]
    ]
    out += ["", "Phase 1's targets, reported without a verdict:", "", "| Target | Events | PR-AUC model | PR-AUC climatology |", "|---|---|---|---|"]
    for entry in b["descriptive"].values():
        scores = (
            f"{entry['pr_auc_model']:.3f} | {entry['pr_auc_baseline']:.3f}" if "pr_auc_model" in entry else "n/a | n/a"
        )
        out.append(f"| {entry['label']} | {entry['events']} | {scores} |")

    out += [
        "",
        "## Call C: interval calibration",
        "",
        "| Interval | Coverage | Binomial 95% CI | Day-block 95% CI | Mean width | Within 5 points |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("50", "80"):
        v = c["intervals"][name]
        out.append(
            f"| {name}% | {v['coverage']:.3f} | {_ci(v['binomial_ci95'], '.3f')} | "
            f"{_ci(v['day_block_ci95'], '.3f')} | {v['mean_width']:.1f} | {'yes' if v['within_tolerance'] else 'no'} |"
        )
    out += [
        "",
        "The verdict uses pooled point coverage. Coverage by month is a declared secondary result:",
        "",
        "| Month | 50% | 80% |",
        "|---|---|---|",
    ]
    out += [
        f"| {month} | {c['intervals']['50']['by_month'][month]:.3f} | {c['intervals']['80']['by_month'][month]:.3f} |"
        for month in c["intervals"]["80"]["by_month"]
    ]
    bands = ["< q0.1", "q0.1-q0.25", "q0.25-q0.5", "q0.5-q0.75", "q0.75-q0.9", "> q0.9"]
    out += ["", "| Band | Hours | Expected |", "|---|---|---|"]
    out += [f"| {band} | {n} | {e:.0f} |" for band, n, e in zip(bands, c["pit_counts"], c["pit_expected"])]
    out += [
        "",
        "Mean pinball loss by quantile: " + ", ".join(f"{k}: {v:.2f}" for k, v in c["pinball"].items()) + ".",
        "",
        "## What this does and does not show",
        "",
        "All calls compare against naive baselines, not against what the market had already "
        "priced in. No tradeable edge and no P&L is claimed.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-2 final assessment")
    parser.add_argument("--provisional", action="store_true", help="print the record so far; write nothing")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    args = parser.parse_args(argv)

    ledger = load_ledger(args.ledger)
    if args.provisional:
        print(render_markdown(build_report(ledger, provisional=True), title="Running record, phase 2"))
        return 0
    # Written once. A verdict that could be regenerated later is one that could
    # be regenerated differently.
    if REPORT_PATH.exists():
        print(f"{REPORT_PATH.name} already exists; leaving it as written")
        return 0
    try:
        report = build_report(ledger)
    except WindowIncomplete as exc:
        print(f"window not complete, nothing written: {exc}")
        return 0
    now = datetime.now(timezone.utc)
    report["generated_at_utc"] = now.isoformat()
    report["due_by"] = REPORT_DUE.isoformat()
    report["late"] = now.date() > REPORT_DUE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_markdown(report), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {REPORT_PATH} and {JSON_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

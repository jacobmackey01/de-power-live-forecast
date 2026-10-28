"""The formal assessment of the evaluation window, against PREREGISTRATION.md.

Runs after every scoring job. Until every delivery day from the first sealed
prediction through EVALUATION_END is either SCORED or MISSED in the ledger, it
reports how many days are outstanding and writes nothing: a verdict published
from a partial window would be the early read the pre-registration forbids.
Once the window is complete it writes ``results/FINAL_REPORT.md`` and
``results/final_report.json``.

Both outputs are a pure function of the committed ledger and sealed
predictions - no wall-clock values, and a fixed bootstrap seed - so re-running
it on any later day reproduces the same bytes, and days scored after the
window cannot move any number in it.

    python -m de_power_live.final_report               # write once complete
    python -m de_power_live.final_report --provisional # print now, write nothing
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import average_precision_score

from . import MARKET_TZ
from .track_record import EVALUATION_END, load_record

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "predictions"
RESULTS_DIR = REPO_ROOT / "results"
LEDGER_PATH = RESULTS_DIR / "ledger.jsonl"
REPORT_PATH = RESULTS_DIR / "FINAL_REPORT.md"
JSON_PATH = RESULTS_DIR / "final_report.json"

# Fixed so the published intervals are reproducible to the last digit.
BOOTSTRAP_SEED = 20261101
BOOTSTRAP_DRAWS = 10_000
MIN_EVENTS = 30
COVERAGE_TOLERANCE = 0.05
# Call B thresholds in the order PREREGISTRATION.md section 3 declares them.
CALL_B_TARGETS = (("negative", "< 0 EUR/MWh"), ("below_10", "< 10 EUR/MWh"))
# The levels every sealed prediction carries (model.QUANTILES). Restated rather
# than imported so the assessment never loads the model or its xgboost runtime;
# hourly_quantiles() checks each sealed file against it.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
# Expected share of outturns in each band between the sealed quantiles.
PIT_EXPECTED = np.diff([0.0, *QUANTILES, 1.0])


class WindowIncomplete(RuntimeError):
    """Raised when some window day is neither SCORED nor MISSED yet."""


# ---- inputs ------------------------------------------------------------------


def local_day_hours(day: date) -> int:
    """23, 24 or 25: the length of a local power day in the market time zone."""
    zone = ZoneInfo(MARKET_TZ)
    # Subtracting two datetimes that share a tzinfo ignores their offsets, so
    # the difference is taken in UTC.
    start = datetime.combine(day, time(), zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time(), zone).astimezone(timezone.utc)
    return round((end - start).total_seconds() / 3600)


def window_rows(
    ledger_path: Path, predictions_dir: Path, require_complete: bool = True
) -> tuple[list[dict], date]:
    """Ledger rows inside the window, after checking the window is complete."""
    load_record(ledger_path)  # same validation the published figure relies on
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sealed = sorted(p.stem for p in predictions_dir.glob("????-??-??.json"))
    first = date.fromisoformat(min([*sealed, *(r["delivery_date_local"] for r in rows)]))

    by_day = {r["delivery_date_local"]: r for r in rows}
    outstanding = []
    day = first
    while day <= EVALUATION_END:
        if day.isoformat() not in by_day:
            outstanding.append(day.isoformat())
        day += timedelta(days=1)
    if outstanding and require_complete:
        raise WindowIncomplete(
            f"{len(outstanding)} window day(s) not yet SCORED or MISSED: {', '.join(outstanding)}"
        )

    inside = [r for r in rows if date.fromisoformat(r["delivery_date_local"]) <= EVALUATION_END]
    return sorted(inside, key=lambda r: r["delivery_date_local"]), first


def hourly_quantiles(day: str, predictions_dir: Path) -> dict[str, list[float]]:
    payload = json.loads((predictions_dir / f"{day}.json").read_text(encoding="utf-8"))
    for p in payload["predictions"]:
        if sorted(map(float, p["call_c_quantiles"])) != list(QUANTILES):
            raise ValueError(f"{day}: sealed quantile levels differ from {QUANTILES}")
    return {
        p["hour_utc"]: [p["call_c_quantiles"][str(q)] for q in QUANTILES]
        for p in payload["predictions"]
    }


# ---- statistics --------------------------------------------------------------


def block_draws(n_days: int) -> np.ndarray:
    """Day-block bootstrap resamples: block = one power day, as pre-registered."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, n_days, size=(BOOTSTRAP_DRAWS, n_days))


def interval(values: np.ndarray) -> list[float]:
    values = values[~np.isnan(values)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def wilson(hits: int, n: int, z: float = 1.959964) -> list[float]:
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def call_a(rows: list[dict], draws: np.ndarray) -> dict:
    n = np.array([r["n_hours_scored"] for r in rows], dtype=float)
    a = [r["call_a"] for r in rows]
    err = np.array([x["mae_model"] for x in a]) * n
    err_b1 = np.array([x["mae_baseline_b1_lag168"] for x in a]) * n
    err_b2 = np.array([x["mae_baseline_b2_lag24"] for x in a]) * n

    def skill(base: np.ndarray) -> dict:
        boot = 1 - err[draws].sum(axis=1) / base[draws].sum(axis=1)
        return {"skill": float(1 - err.sum() / base.sum()), "ci95": interval(boot)}

    vs_b1, vs_b2 = skill(err_b1), skill(err_b2)
    return {
        "hours": int(n.sum()),
        "mae_model": float(err.sum() / n.sum()),
        "mae_b1": float(err_b1.sum() / n.sum()),
        "mae_b2": float(err_b2.sum() / n.sum()),
        "rmse_model": float(math.sqrt(sum(x["rmse_model"] ** 2 * h for x, h in zip(a, n)) / n.sum())),
        "bias_model": float(sum(x["bias_model"] * h for x, h in zip(a, n)) / n.sum()),
        "vs_b1": vs_b1,
        "vs_b2": vs_b2,
        "verdict": "PASS" if vs_b1["ci95"][0] > 0 else "NULL",
    }


def call_b(rows: list[dict], draws: np.ndarray) -> dict:
    per_day = [r["call_b"]["hours"] for r in rows]
    targets = {}
    for name, label in CALL_B_TARGETS:
        days = [
            (
                np.array([h["model_score"] for h in hours]),
                np.array([h[f"outcome_{name}"] for h in hours]),
                np.array([h[f"baseline_{name}"] for h in hours]),
            )
            for hours in per_day
        ]
        score, outcome, base = (np.concatenate(parts) for parts in zip(*days))

        def pr_auc_gap(sample) -> float:
            s, o, b = (np.concatenate([days[i][k] for i in sample]) for k in range(3))
            if o.sum() == 0:
                return np.nan  # PR-AUC is undefined without an event
            return average_precision_score(o, s) - average_precision_score(o, b)

        boot = np.array([pr_auc_gap(sample) for sample in draws])
        entry = {
            "label": label,
            "events": int(outcome.sum()),
            "hours": int(len(outcome)),
            "pr_auc_model": float(average_precision_score(outcome, score)),
            "pr_auc_climatology": float(average_precision_score(outcome, base)),
            "difference_ci95": interval(boot),
            "draws_without_events": int(np.isnan(boot).sum()),
        }
        entry["difference"] = entry["pr_auc_model"] - entry["pr_auc_climatology"]
        if name == "negative":
            # Brier and reliability only at the primary threshold: the model's
            # score is a calibrated P(price < 0), not a probability of < 10.
            entry["brier_model"] = float(np.mean((score - outcome) ** 2))
            entry["brier_climatology"] = float(np.mean((base - outcome) ** 2))
            edges = np.linspace(0, 1, 11)
            bins = np.clip(np.digitize(score, edges[1:-1]), 0, 9)
            entry["reliability"] = [
                {
                    "bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
                    "hours": int((bins == i).sum()),
                    "mean_forecast": float(score[bins == i].mean()),
                    "observed_rate": float(outcome[bins == i].mean()),
                }
                for i in range(10)
                if (bins == i).any()
            ]
        targets[name] = entry

    if targets["negative"]["events"] >= MIN_EVENTS:
        tested = "negative"
    elif targets["below_10"]["events"] >= MIN_EVENTS:
        tested = "below_10"
    else:
        tested = None

    if tested is None:
        verdict = "UNDERPOWERED"
    else:
        verdict = "PASS" if targets[tested]["difference_ci95"][0] > 0 else "NULL"
    return {"tested_target": tested, "targets": targets, "verdict": verdict}


def call_c(rows: list[dict], predictions_dir: Path, draws: np.ndarray) -> dict:
    """Coverage and PIT, recomputed hour by hour from the sealed quantiles."""
    hits = {"50": [], "80": []}
    pit = np.zeros(len(QUANTILES) + 1, dtype=int)
    pinball = {str(q): 0.0 for q in QUANTILES}
    hours_per_day = []

    for row in rows:
        quantiles = hourly_quantiles(row["delivery_date_local"], predictions_dir)
        day_hits = {"50": 0, "80": 0}
        for hour in row["call_b"]["hours"]:
            q = quantiles[hour["hour_utc"]]
            price = hour["price"]
            day_hits["80"] += int(q[0] <= price <= q[4])
            day_hits["50"] += int(q[1] <= price <= q[3])
            pit[int(np.searchsorted(q, price, side="right"))] += 1
            for level, value in zip(QUANTILES, q):
                diff = price - value
                pinball[str(level)] += max(level * diff, (level - 1) * diff)
        for level in hits:
            hits[level].append(day_hits[level])
            # The ledger's own coverage must agree with the hourly recomputation.
            recorded = round(row["call_c"][f"coverage_{level}"] * len(row["call_b"]["hours"]))
            if recorded != day_hits[level]:
                raise ValueError(
                    f"{row['delivery_date_local']}: ledger coverage_{level} implies "
                    f"{recorded} hits, sealed quantiles give {day_hits[level]}"
                )
        hours_per_day.append(len(row["call_b"]["hours"]))

    n = np.array(hours_per_day)
    total = int(n.sum())
    levels = {}
    for level, nominal in (("50", 0.5), ("80", 0.8)):
        h = np.array(hits[level])
        coverage = h.sum() / total
        binomial = wilson(int(h.sum()), total)
        levels[level] = {
            "nominal": nominal,
            "coverage": float(coverage),
            "binomial_ci95": binomial,
            "day_block_ci95": interval(h[draws].sum(axis=1) / n[draws].sum(axis=1)),
            "within_tolerance": bool(abs(coverage - nominal) <= COVERAGE_TOLERANCE),
            "ci_inside_band": bool(
                nominal - COVERAGE_TOLERANCE <= binomial[0]
                and binomial[1] <= nominal + COVERAGE_TOLERANCE
            ),
        }
    return {
        "hours": total,
        "levels": levels,
        "pinball": {k: v / total for k, v in pinball.items()},
        "pit_counts": pit.tolist(),
        "pit_expected": [float(x * total) for x in PIT_EXPECTED],
        "verdict": "PASS" if all(v["within_tolerance"] for v in levels.values()) else "NULL",
    }


def assess(rows: list[dict], predictions_dir: Path) -> dict:
    draws = block_draws(len(rows))
    return {
        "days": len(rows),
        "call_a": call_a(rows, draws),
        "call_b": call_b(rows, draws),
        "call_c": call_c(rows, predictions_dir, draws),
    }


def clock_change_days(rows: list[dict], first: date) -> list[dict]:
    """Every 23- or 25-hour day in the window, and what the record holds for it."""
    by_day = {r["delivery_date_local"]: r for r in rows}
    days = []
    day = first
    while day <= EVALUATION_END:
        hours = local_day_hours(day)
        if hours != 24:
            row = by_day.get(day.isoformat(), {})
            days.append(
                {
                    "date": day.isoformat(),
                    "local_hours": hours,
                    "status": row.get("status", "OUTSTANDING"),
                    "hours_scored": row.get("n_hours_scored"),
                    "reason": row.get("reason"),
                }
            )
        day += timedelta(days=1)
    return days


def build_report(
    ledger_path: Path = LEDGER_PATH,
    predictions_dir: Path = PREDICTIONS_DIR,
    require_complete: bool = True,
) -> dict:
    rows, first = window_rows(ledger_path, predictions_dir, require_complete)
    scored = [r for r in rows if r["status"] == "SCORED"]
    missed = [r["delivery_date_local"] for r in rows if r["status"] == "MISSED"]
    versions = sorted({r["model_version"] for r in scored})
    margins = [r["minutes_before_close"] for r in scored]
    return {
        "window": {"first": first.isoformat(), "last": EVALUATION_END.isoformat()},
        "clock_change_days": clock_change_days(rows, first),
        "calendar_days": (EVALUATION_END - first).days + 1,
        "scored_days": len(scored),
        "missed_days": missed,
        "seal_margin_minutes": {"min": min(margins), "median": float(np.median(margins))},
        "model_versions": {
            v: sorted({r["model_source_sha256"] for r in scored if r["model_version"] == v})
            for v in versions
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "block": "one power day"},
        "pooled": assess(scored, predictions_dir),
        "by_version": (
            {v: assess([r for r in scored if r["model_version"] == v], predictions_dir) for v in versions}
            if len(versions) > 1
            else {}
        ),
    }


# ---- rendering ---------------------------------------------------------------


def _ci(pair: list[float], fmt: str = "+.3f") -> str:
    return f"[{pair[0]:{fmt}}, {pair[1]:{fmt}}]"


def render_markdown(report: dict, title: str = "Final assessment") -> str:
    pooled = report["pooled"]
    a, b, c = pooled["call_a"], pooled["call_b"], pooled["call_c"]
    w = report["window"]
    versions = ", ".join(
        f"`{v}` (source `{shas[0][:12]}`)" for v, shas in report["model_versions"].items()
    )
    missed = report["missed_days"]

    lines = [
        f"# {title}: DE-LU day-ahead forecasting record",
        "",
        f"Evaluation window **{w['first']} to {w['last']}**, judged against the criteria frozen "
        "in [`PREREGISTRATION.md`](../PREREGISTRATION.md) before the first prediction.",
        "Generated by `python -m de_power_live.final_report` from the committed ledger and sealed",
        "predictions only. Days scored after the window are excluded and cannot change any number here.",
        "",
        "## Verdicts",
        "",
        "| Call | Criterion | Result | Verdict |",
        "|---|---|---|---|",
        f"| **A** price level | MAE skill vs same-hour-last-week > 0, 95% CI excluding 0 | "
        f"{a['vs_b1']['skill']:+.3f} {_ci(a['vs_b1']['ci95'])} | **{a['verdict']}** |",
    ]
    if b["tested_target"] is None:
        lines.append(
            "| **B** low-price hours | PR-AUC beats climatology, CI excluding 0 | "
            f"fewer than {MIN_EVENTS} events at both thresholds | **UNDERPOWERED** |"
        )
    else:
        t = b["targets"][b["tested_target"]]
        lines.append(
            f"| **B** hours {t['label']} | PR-AUC beats 12x24 climatology, CI excluding 0 | "
            f"{t['pr_auc_model']:.3f} vs {t['pr_auc_climatology']:.3f}, difference "
            f"{t['difference']:+.3f} {_ci(t['difference_ci95'])} | **{b['verdict']}** |"
        )
    lines += [
        f"| **C** intervals | 50% and 80% coverage within {COVERAGE_TOLERANCE * 100:.0f}pp of nominal | "
        f"50%: {c['levels']['50']['coverage']:.3f}, 80%: {c['levels']['80']['coverage']:.3f} | "
        f"**{c['verdict']}** |",
        "",
        "## The record",
        "",
        f"- Calendar days in the window: **{report['calendar_days']}**",
        f"- Scored: **{report['scored_days']}** days, {a['hours']} hours",
        f"- Missed: **{len(missed)}**" + (f" ({', '.join(missed)})" if missed else ""),
        "  - A missed day was recorded when no forecast was sealed before gate closure and was never "
        "backfilled. Missed days count against the record but carry no score.",
        f"- Seal margin before gate closure: minimum {report['seal_margin_minutes']['min']:.0f} min, "
        f"median {report['seal_margin_minutes']['median']:.0f} min",
        f"- Model version(s): {versions}",
        f"- Bootstrap: {report['bootstrap']['draws']:,} draws, block = one power day, "
        f"seed {report['bootstrap']['seed']}",
        "",
        "## Clock-change days",
        "",
        "The pre-registration (section 6) put the window's clock change inside the test: handling "
        "it correctly, live, was part of what was being evaluated.",
        "",
        "| Delivery day | Local hours | Outcome |",
        "|---|---|---|",
        *(
            f"| {d['date']} | {d['local_hours']} | "
            + (
                f"Scored, {d['hours_scored']} of {d['local_hours']} hours"
                if d["status"] == "SCORED"
                else f"**{d['status']}**" + (f": {d['reason']}" if d["reason"] else "")
            )
            + " |"
            for d in report["clock_change_days"]
        ),
        "",
        *(
            [
                "A clock-change day that was not scored means that part of the test was not passed. "
                "The cause and the fix are written up in "
                "[`docs/v1-lessons.md`](../docs/v1-lessons.md).",
                "",
            ]
            if any(d["status"] != "SCORED" for d in report["clock_change_days"])
            else []
        ),
        "## Call A: price level",
        "",
        "| | MAE (EUR/MWh) | Skill of model | 95% CI |",
        "|---|---|---|---|",
        f"| Model | {a['mae_model']:.2f} | | |",
        f"| B1 same hour, D-7 (primary) | {a['mae_b1']:.2f} | {a['vs_b1']['skill']:+.3f} | {_ci(a['vs_b1']['ci95'])} |",
        f"| B2 same hour, D-1 | {a['mae_b2']:.2f} | {a['vs_b2']['skill']:+.3f} | {_ci(a['vs_b2']['ci95'])} |",
        "",
        f"Model RMSE {a['rmse_model']:.2f} EUR/MWh, bias {a['bias_model']:+.2f} EUR/MWh "
        "(negative = forecast below outturn).",
        "",
        "## Call B: low-price hours",
        "",
        "| Threshold | Events | PR-AUC model | PR-AUC climatology | Difference | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for name, _ in CALL_B_TARGETS:
        t = b["targets"][name]
        marker = " (tested)" if name == b["tested_target"] else ""
        lines.append(
            f"| {t['label']}{marker} | {t['events']} | {t['pr_auc_model']:.3f} | "
            f"{t['pr_auc_climatology']:.3f} | {t['difference']:+.3f} | {_ci(t['difference_ci95'])} |"
        )
    neg = b["targets"]["negative"]
    lines += [
        "",
        f"The tested threshold follows the pre-declared power rule: < 0 EUR/MWh if it has at least "
        f"{MIN_EVENTS} events, otherwise < 10 EUR/MWh if that does, otherwise underpowered. "
        "PR-AUC is computed as average precision. Bootstrap draws that contained no event "
        f"are dropped ({neg['draws_without_events']} at < 0).",
        "",
        f"Brier score at < 0: model {neg['brier_model']:.4f}, climatology {neg['brier_climatology']:.4f}.",
        "",
        "| Forecast P(< 0) | Hours | Mean forecast | Observed rate |",
        "|---|---|---|---|",
        *(
            f"| {r['bin']} | {r['hours']} | {r['mean_forecast']:.3f} | {r['observed_rate']:.3f} |"
            for r in neg["reliability"]
        ),
        "",
        "## Call C: interval calibration",
        "",
        "| Interval | Coverage | Binomial 95% CI | Day-block 95% CI | Within 5pp |",
        "|---|---|---|---|---|",
    ]
    for level in ("50", "80"):
        v = c["levels"][level]
        lines.append(
            f"| {level}% | {v['coverage']:.3f} | {_ci(v['binomial_ci95'], '.3f')} | "
            f"{_ci(v['day_block_ci95'], '.3f')} | {'yes' if v['within_tolerance'] else 'no'} |"
        )
    pit_labels = ["< q0.1", "q0.1-q0.25", "q0.25-q0.5", "q0.5-q0.75", "q0.75-q0.9", "> q0.9"]
    lines += [
        "",
        "The verdict uses the point coverage against the 5pp band. The binomial interval treats hours "
        "as independent, which they are not; the day-block interval is the more honest width.",
        "",
        "Where outturns fell relative to the sealed quantiles (PIT):",
        "",
        "| Band | Hours | Expected |",
        "|---|---|---|",
        *(
            f"| {label} | {count} | {expected:.0f} |"
            for label, count, expected in zip(pit_labels, c["pit_counts"], c["pit_expected"])
        ),
        "",
        "Mean pinball loss by quantile: "
        + ", ".join(f"{k}: {v:.2f}" for k, v in c["pinball"].items())
        + ".",
        "",
    ]
    if report["by_version"]:
        lines += ["## By model version", ""]
        for v, res in report["by_version"].items():
            lines.append(
                f"- `{v}`: {res['days']} days; A {res['call_a']['verdict']}, "
                f"B {res['call_b']['verdict']}, C {res['call_c']['verdict']}"
            )
        lines.append("")
    else:
        lines += [
            "Only one model version produced live predictions, so the per-version and pooled results "
            "are the same numbers.",
            "",
        ]
    lines += [
        "## What this does and does not show",
        "",
        "Calls A and B compare against naive and climatological baselines, not against what the "
        "market had already priced in. No tradeable edge and no P&L is claimed.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="print the assessment of the days scored so far; writes nothing",
    )
    args = parser.parse_args(argv)

    if args.provisional:
        print(render_markdown(
            build_report(require_complete=False),
            title="PROVISIONAL, from the days scored so far, not the pre-registered assessment",
        ))
        return 0

    try:
        report = build_report()
    except WindowIncomplete as exc:
        print(f"Window not complete; no report written. {exc}")
        return 0

    JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)} and {JSON_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

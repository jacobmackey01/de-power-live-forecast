"""Create the deterministic public track-record figure from the scoring ledger.

The ledger is the only numerical input. This module deliberately uses no wall-clock
values, random state, or external data so the same committed ledger produces the
same SVG bytes on every machine.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"
OUTPUT_PATH = REPO_ROOT / "results" / "prospective_track_record.svg"

# The evaluation endpoint is a protocol choice from PREREGISTRATION.md, not a
# value copied from the live ledger.
EVALUATION_END = date(2026, 10, 31)
MIN_NEGATIVE_HOURS = 30

BACKGROUND = "#fffdf9"
TEXT = "#1f2933"
MUTED = "#5b6770"
RULE = "#d8d9d6"
GRID = "#e8e8e4"
MODEL = "#0b5563"
MODEL_LIGHT = "#2b8790"
B1 = "#9a4f00"
B2 = "#5e477e"
NOMINAL = "#8a9298"
FONT = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
SHORT_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
LONG_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


class TrackRecordError(ValueError):
    """Raised when the ledger cannot support an exact chart."""


@dataclass(frozen=True)
class DayRecord:
    delivery_date: date
    model_version: str
    hours: int
    mae_model: float
    mae_b1: float
    mae_b2: float
    brier_model_sum: float
    brier_baseline_sum: float
    coverage_50_hits: int
    coverage_80_hits: int
    negative_hours: int
    below_10_hours: int | None


@dataclass(frozen=True)
class TrackRecord:
    days: tuple[DayRecord, ...]
    pending_count: int
    missed_count: int
    model_versions: tuple[str, ...]
    first_date: date
    latest_date: date

    @property
    def scored_hours(self) -> int:
        return sum(day.hours for day in self.days)

    @property
    def negative_hours(self) -> int:
        return sum(day.negative_hours for day in self.days)

    @property
    def below_10_hours(self) -> int | None:
        values = [day.below_10_hours for day in self.days]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _fail(context: str, message: str) -> TrackRecordError:
    return TrackRecordError(f"{context}: {message}")


def _field(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise _fail(context, f"missing required field {key!r}")
    return mapping[key]


def _as_number(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(context, f"{field} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise _fail(context, f"{field} must be finite")
    return result


def _as_integer(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(context, f"{field} must be an integer")
    return value


def _bounded_number(
    mapping: dict[str, Any],
    field: str,
    context: str,
    low: float,
    high: float,
) -> float:
    value = _as_number(_field(mapping, field, context), field, context)
    if value < low or value > high:
        raise _fail(context, f"{field}={value} is outside [{low}, {high}]")
    return value


def _parse_date(value: Any, context: str) -> date:
    if not isinstance(value, str):
        raise _fail(context, "delivery_date_local must be a YYYY-MM-DD string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise _fail(context, f"invalid delivery_date_local {value!r}") from exc
    if parsed.isoformat() != value:
        raise _fail(context, f"delivery_date_local must use YYYY-MM-DD: {value!r}")
    return parsed


def _coverage_hits(value: float, hours: int, field: str, context: str) -> int:
    """Recover the exact integer hit count from score.py's five-decimal coverage."""
    hits = int(round(value * hours))
    if not 0 <= hits <= hours:
        raise _fail(context, f"{field} cannot represent a hit count")
    # score.py rounds daily coverage to five decimal places. Reject values that
    # cannot be the rounded representation of an integer numerator rather than
    # silently treating an arbitrary percentage as exact.
    if abs(value - (hits / hours)) > 0.000006:
        raise _fail(
            context,
            f"{field}={value} is not reconstructible from {hours} hourly outcomes",
        )
    return hits


def _parse_scored(row: dict[str, Any], line_number: int, delivery_date: date) -> DayRecord:
    context = f"line {line_number} ({delivery_date.isoformat()})"

    model_version = _field(row, "model_version", context)
    if not isinstance(model_version, str) or not model_version:
        raise _fail(context, "model_version must be a non-empty string")

    n_predicted = _as_integer(
        _field(row, "n_hours_predicted", context), "n_hours_predicted", context
    )
    n_scored = _as_integer(
        _field(row, "n_hours_scored", context), "n_hours_scored", context
    )
    if n_predicted <= 0 or n_scored <= 0:
        raise _fail(context, "hour counts must be positive")
    if n_scored > n_predicted:
        raise _fail(context, "n_hours_scored cannot exceed n_hours_predicted")

    call_a = _field(row, "call_a", context)
    if not isinstance(call_a, dict):
        raise _fail(context, "call_a must be an object")
    mae_model = _as_number(_field(call_a, "mae_model", context), "mae_model", context)
    mae_b1 = _as_number(
        _field(call_a, "mae_baseline_b1_lag168", context),
        "mae_baseline_b1_lag168",
        context,
    )
    mae_b2 = _as_number(
        _field(call_a, "mae_baseline_b2_lag24", context),
        "mae_baseline_b2_lag24",
        context,
    )
    if min(mae_model, mae_b1, mae_b2) < 0:
        raise _fail(context, "MAE values cannot be negative")

    call_b = _field(row, "call_b", context)
    if not isinstance(call_b, dict):
        raise _fail(context, "call_b must be an object")
    hourly = _field(call_b, "hours", context)
    if not isinstance(hourly, list):
        raise _fail(context, "call_b.hours must be an array")
    if len(hourly) != n_scored:
        raise _fail(
            context,
            f"call_b.hours has {len(hourly)} entries but n_hours_scored is {n_scored}",
        )

    brier_model_sum = 0.0
    brier_baseline_sum = 0.0
    negative_hours = 0
    below_10_values: list[int] = []

    for index, hour in enumerate(hourly):
        hour_context = f"{context} call_b.hours[{index}]"
        if not isinstance(hour, dict):
            raise _fail(hour_context, "hour record must be an object")
        model_score = _bounded_number(hour, "model_score", hour_context, 0.0, 1.0)
        baseline_score = _bounded_number(
            hour, "baseline_negative", hour_context, 0.0, 1.0
        )
        outcome = _as_integer(
            _field(hour, "outcome_negative", hour_context),
            "outcome_negative",
            hour_context,
        )
        if outcome not in (0, 1):
            raise _fail(hour_context, "outcome_negative must be 0 or 1")

        price = _as_number(_field(hour, "price", hour_context), "price", hour_context)
        if int(price < 0) != outcome:
            raise _fail(
                hour_context,
                "outcome_negative disagrees with the recorded realised price",
            )

        brier_model_sum += (model_score - outcome) ** 2
        brier_baseline_sum += (baseline_score - outcome) ** 2
        negative_hours += outcome

        if "outcome_below_10" in hour:
            below = _as_integer(
                hour["outcome_below_10"], "outcome_below_10", hour_context
            )
            if below not in (0, 1):
                raise _fail(hour_context, "outcome_below_10 must be 0 or 1")
            if below != int(price < 10):
                raise _fail(
                    hour_context,
                    "outcome_below_10 disagrees with the recorded realised price",
                )
            below_10_values.append(below)

    stored_negative = _as_integer(
        _field(call_b, "n_negative_hours", context),
        "n_negative_hours",
        context,
    )
    if stored_negative != negative_hours:
        raise _fail(
            context,
            f"n_negative_hours={stored_negative} disagrees with hourly outcomes={negative_hours}",
        )

    stored_brier = _bounded_number(call_b, "brier", context, 0.0, 1.0)
    if abs(stored_brier - (brier_model_sum / n_scored)) > 0.000011:
        raise _fail(context, "stored model Brier score disagrees with hourly records")

    below_10_hours: int | None = None
    if below_10_values:
        below_10_hours = sum(below_10_values)
        if "n_below_10_hours" in call_b:
            stored_below = _as_integer(
                call_b["n_below_10_hours"], "n_below_10_hours", context
            )
            if stored_below != below_10_hours:
                raise _fail(
                    context,
                    "n_below_10_hours disagrees with hourly outcomes",
                )
    elif "n_below_10_hours" in call_b:
        raise _fail(
            context,
            "n_below_10_hours is present but hourly fallback outcomes are incomplete",
        )

    call_c = _field(row, "call_c", context)
    if not isinstance(call_c, dict):
        raise _fail(context, "call_c must be an object")
    coverage_50 = _bounded_number(call_c, "coverage_50", context, 0.0, 1.0)
    coverage_80 = _bounded_number(call_c, "coverage_80", context, 0.0, 1.0)

    return DayRecord(
        delivery_date=delivery_date,
        model_version=model_version,
        hours=n_scored,
        mae_model=mae_model,
        mae_b1=mae_b1,
        mae_b2=mae_b2,
        brier_model_sum=brier_model_sum,
        brier_baseline_sum=brier_baseline_sum,
        coverage_50_hits=_coverage_hits(coverage_50, n_scored, "coverage_50", context),
        coverage_80_hits=_coverage_hits(coverage_80, n_scored, "coverage_80", context),
        negative_hours=negative_hours,
        below_10_hours=below_10_hours,
    )


def _read_entries(path: Path) -> list[tuple[int, dict[str, Any]]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrackRecordError(f"cannot read ledger {path}: {exc}") from exc
    if not raw.strip():
        raise TrackRecordError(f"ledger {path} is empty")

    entries: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise TrackRecordError(f"line {line_number}: blank JSONL lines are invalid")
        try:
            value = json.loads(line, parse_constant=_reject_constant)
        except (TypeError, ValueError) as exc:
            raise TrackRecordError(f"line {line_number}: malformed JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise TrackRecordError(f"line {line_number}: JSON value must be an object")
        entries.append((line_number, value))
    return entries


def load_record(path: Path = LEDGER_PATH) -> TrackRecord:
    entries = _read_entries(Path(path))
    seen_dates: dict[date, tuple[int, str]] = {}
    scored: list[DayRecord] = []
    pending_count = 0
    missed_count = 0

    for line_number, row in entries:
        context = f"line {line_number}"
        status = _field(row, "status", context)
        if not isinstance(status, str) or not status:
            raise _fail(context, "status must be a non-empty string")
        delivery_date = _parse_date(
            _field(row, "delivery_date_local", context), context
        )

        if delivery_date in seen_dates:
            previous_line, previous_status = seen_dates[delivery_date]
            if status == "SCORED" and previous_status == "SCORED":
                raise _fail(
                    context,
                    f"duplicate scored delivery date; first appears on line {previous_line}",
                )
            raise _fail(
                context,
                f"duplicate delivery date; first appears on line {previous_line}",
            )
        seen_dates[delivery_date] = (line_number, status)

        if status == "SCORED":
            scored.append(_parse_scored(row, line_number, delivery_date))
        elif status == "MISSED":
            missed_count += 1
        else:
            pending_count += 1

    if not scored:
        raise TrackRecordError("ledger contains no SCORED rows")

    scored.sort(key=lambda item: item.delivery_date)
    versions = tuple(sorted({item.model_version for item in scored}))
    return TrackRecord(
        days=tuple(scored),
        pending_count=pending_count,
        missed_count=missed_count,
        model_versions=versions,
        first_date=scored[0].delivery_date,
        latest_date=scored[-1].delivery_date,
    )


def cumulative_points(record: TrackRecord) -> list[dict[str, Any]]:
    """Return cumulative metrics using the scored-hour denominator throughout."""
    hours = 0
    mae_model_sum = 0.0
    mae_b1_sum = 0.0
    mae_b2_sum = 0.0
    brier_model_sum = 0.0
    brier_baseline_sum = 0.0
    coverage_50_hits = 0
    coverage_80_hits = 0
    negative_hours = 0
    below_10_hours: int | None = 0

    points: list[dict[str, Any]] = []
    for day in record.days:
        hours += day.hours
        mae_model_sum += day.mae_model * day.hours
        mae_b1_sum += day.mae_b1 * day.hours
        mae_b2_sum += day.mae_b2 * day.hours
        brier_model_sum += day.brier_model_sum
        brier_baseline_sum += day.brier_baseline_sum
        coverage_50_hits += day.coverage_50_hits
        coverage_80_hits += day.coverage_80_hits
        negative_hours += day.negative_hours

        if day.below_10_hours is None:
            below_10_hours = None
        elif below_10_hours is not None:
            below_10_hours += day.below_10_hours

        points.append(
            {
                "date": day.delivery_date,
                "hours": hours,
                "mae_model": mae_model_sum / hours,
                "mae_b1": mae_b1_sum / hours,
                "mae_b2": mae_b2_sum / hours,
                "brier_model": brier_model_sum / hours,
                "brier_baseline": brier_baseline_sum / hours,
                "coverage_50": coverage_50_hits / hours,
                "coverage_80": coverage_80_hits / hours,
                "negative_hours": negative_hours,
                "below_10_hours": below_10_hours,
            }
        )
    return points


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _coord(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def _text(
    x: float,
    y: float,
    value: str,
    *,
    fill: str = TEXT,
    size: int = 12,
    weight: str | None = None,
    anchor: str | None = None,
    transform: str | None = None,
    letter_spacing: str | None = None,
) -> str:
    attrs = [
        f'x="{_coord(x)}"',
        f'y="{_coord(y)}"',
        f'fill="{_esc(fill)}"',
        f'font-family="{_esc(FONT)}"',
        f'font-size="{size}px"',
    ]
    if weight:
        attrs.append(f'font-weight="{_esc(weight)}"')
    if anchor:
        attrs.append(f'text-anchor="{_esc(anchor)}"')
    if transform:
        attrs.append(f'transform="{_esc(transform)}"')
    if letter_spacing:
        attrs.append(f'letter-spacing="{_esc(letter_spacing)}"')
    return f"<text {' '.join(attrs)}>{_esc(value)}</text>"


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = RULE,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    attrs = [
        f'x1="{_coord(x1)}"',
        f'y1="{_coord(y1)}"',
        f'x2="{_coord(x2)}"',
        f'y2="{_coord(y2)}"',
        f'stroke="{_esc(stroke)}"',
        f'stroke-width="{_coord(width)}"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{_esc(dash)}"')
    return f"<line {' '.join(attrs)} />"


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = BACKGROUND,
    stroke: str = RULE,
    stroke_width: float = 1.0,
) -> str:
    return (
        f'<rect x="{_coord(x)}" y="{_coord(y)}" width="{_coord(width)}" '
        f'height="{_coord(height)}" fill="{_esc(fill)}" '
        f'stroke="{_esc(stroke)}" stroke-width="{_coord(stroke_width)}" />'
    )


def _path(points: list[dict[str, Any]], key: str, xs: list[float], y_max: float, top: float, bottom: float) -> str:
    commands = []
    for index, point in enumerate(points):
        y = bottom - (point[key] / y_max) * (bottom - top)
        commands.append(f"{'M' if index == 0 else 'L'} {_coord(xs[index])} {_coord(y)}")
    return " ".join(commands)


def _y_value(point: dict[str, Any], key: str, y_max: float, top: float, bottom: float) -> float:
    return bottom - (point[key] / y_max) * (bottom - top)


def _x_positions(count: int, left: float, right: float) -> list[float]:
    if count == 1:
        return [(left + right) / 2]
    return [left + (right - left) * index / (count - 1) for index in range(count)]


def _tick_indices(count: int) -> list[int]:
    candidates = [0, (count - 1) // 2, count - 1]
    return list(dict.fromkeys(candidates))


def _date_label(value: date) -> str:
    return f"{value.day} {SHORT_MONTHS[value.month - 1]}"


def _date_long(value: date) -> str:
    return f"{value.day} {LONG_MONTHS[value.month - 1]} {value.year}"


def _fmt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _place_labels(actual: list[float], top: float, bottom: float, gap: float = 18.0) -> list[float]:
    order = sorted(range(len(actual)), key=lambda index: (actual[index], index))
    placed = [0.0] * len(actual)
    previous = None
    for index in order:
        value = max(top + 10.0, min(bottom - 4.0, actual[index]))
        if previous is not None:
            value = max(value, previous + gap)
        placed[index] = value
        previous = value

    overflow = max(placed) - (bottom - 4.0)
    if overflow > 0:
        placed = [value - overflow for value in placed]
    underflow = (top + 10.0) - min(placed)
    if underflow > 0:
        placed = [value + underflow for value in placed]
    return placed


def _direct_labels(
    lines: list[str],
    points: list[dict[str, Any]],
    xs: list[float],
    series: list[tuple[str, str, str, int]],
    y_max: float,
    top: float,
    bottom: float,
    label_x: float,
) -> None:
    actual = [_y_value(points[-1], key, y_max, top, bottom) for _, key, _, _ in series]
    placed = _place_labels(actual, top, bottom)
    for (name, key, color, decimals), actual_y, label_y in zip(series, actual, placed):
        if abs(actual_y - label_y) > 0.5:
            lines.append(
                _line(xs[-1] + 2, actual_y, label_x - 7, label_y, stroke=color, width=0.8)
            )
        value = _fmt(points[-1][key], decimals)
        lines.append(
            _text(
                label_x,
                label_y + 4,
                f"{name} · {value}",
                fill=color,
                size=12,
                weight="600",
            )
        )


def _draw_axes(
    lines: list[str],
    points: list[dict[str, Any]],
    xs: list[float],
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    y_max: float,
    y_ticks: list[float],
    y_decimals: int,
    y_label: str,
    x_label: str,
) -> None:
    for tick in y_ticks:
        y = bottom - (tick / y_max) * (bottom - top)
        lines.append(_line(left, y, right, y, stroke=GRID, width=0.8))
        lines.append(
            _text(left - 12, y + 4, _fmt(tick, y_decimals), fill=MUTED, size=11, anchor="end")
        )
    lines.append(_line(left, top, left, bottom, stroke=RULE, width=1.0))
    lines.append(_line(left, bottom, right, bottom, stroke=RULE, width=1.0))

    for index in _tick_indices(len(points)):
        lines.append(_line(xs[index], bottom, xs[index], bottom + 4, stroke=RULE, width=1.0))
        lines.append(
            _text(xs[index], bottom + 22, _date_label(points[index]["date"]), fill=MUTED, size=11, anchor="middle")
        )

    lines.append(
        _text((left + right) / 2, bottom + 48, x_label, fill=MUTED, size=11, anchor="middle")
    )
    lines.append(
        _text(
            left - 62,
            (top + bottom) / 2,
            y_label,
            fill=MUTED,
            size=11,
            anchor="middle",
            transform=f"rotate(-90 {_coord(left - 62)} {_coord((top + bottom) / 2)})",
        )
    )


def _panel_header(
    lines: list[str],
    x: float,
    y: float,
    title: str,
    subtitle: str,
    meta: str,
) -> None:
    lines.append(_text(x, y, title, size=15, weight="650"))
    lines.append(_text(x, y + 21, subtitle, fill=MUTED, size=11))
    lines.append(_text(x + 470, y, meta, fill=MUTED, size=11, anchor="end"))


def _nice_a_max(points: list[dict[str, Any]]) -> float:
    highest = max(
        max(point["mae_model"], point["mae_b1"], point["mae_b2"]) for point in points
    )
    return max(10.0, math.ceil((highest * 1.15) / 10.0) * 10.0)


def _nice_brier_max(points: list[dict[str, Any]]) -> float:
    highest = max(max(point["brier_model"], point["brier_baseline"]) for point in points)
    return max(0.05, math.ceil((highest * 1.25) / 0.01) * 0.01)


def _support_line(record: TrackRecord) -> str:
    if record.negative_hours < MIN_NEGATIVE_HOURS:
        return (
            f"Support pending: {record.negative_hours}/{MIN_NEGATIVE_HOURS} negative-price hours; "
            "descriptive only."
        )
    if record.latest_date < EVALUATION_END:
        return (
            f"Support reached: {record.negative_hours} negative-price hours; "
            "window open through 31 October 2026."
        )
    return (
        f"Support reached: {record.negative_hours} negative-price hours; "
        "final assessment remains subject to the preregistered gate."
    )


def render_svg(record: TrackRecord) -> str:
    points = cumulative_points(record)
    xs_a = _x_positions(len(points), 112, 850)
    xs_b = _x_positions(len(points), 104, 420)
    xs_c = _x_positions(len(points), 673, 990)

    a_max = _nice_a_max(points)
    brier_max = _nice_brier_max(points)
    model_name = (
        f"Model ({record.model_versions[0]})"
        if len(record.model_versions) == 1
        else "Model (pooled versions)"
    )

    status_parts = [
        "Early prospective record" if record.latest_date < EVALUATION_END else "Prospective record",
        f"{len(record.days)} scored days",
        f"{record.scored_hours} hours",
        f"through {_date_long(record.latest_date)}",
    ]
    if record.missed_count:
        label = "missed day" if record.missed_count == 1 else "missed days"
        status_parts.append(f"{record.missed_count} {label}")
    if record.pending_count:
        label = "pending day" if record.pending_count == 1 else "pending days"
        status_parts.append(f"{record.pending_count} {label}")
    status = " · ".join(status_parts)

    below_text = (
        f" · {record.below_10_hours} below EUR 10"
        if record.below_10_hours is not None
        else ""
    )
    description = (
        f"Source-derived prospective record from {_date_long(record.first_date)} to "
        f"{_date_long(record.latest_date)}, with {len(record.days)} scored delivery days "
        f"and {record.scored_hours} scored hours. The sample is still accumulating."
    )

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 860" role="img" aria-labelledby="svg-title svg-desc">',
        '<title id="svg-title">Prospective DE-LU forecast track record</title>',
        f'<desc id="svg-desc">{_esc(description)}</desc>',
        _rect(0, 0, 1200, 860, fill=BACKGROUND, stroke=BACKGROUND, stroke_width=0),
        _text(48, 44, "Prospective DE-LU forecast track record", size=25, weight="650"),
        _text(48, 70, "Sealed before auction close · scored after settlement", fill=MUTED, size=13),
        _text(48, 96, status, fill=TEXT, size=12, weight="550"),
        _rect(48, 126, 1104, 326),
        _rect(48, 482, 535, 298),
        _rect(617, 482, 535, 298),
    ]

    # Panel A: price error.
    _panel_header(
        lines,
        70,
        157,
        "Panel A · Price forecast error",
        "Cumulative hourly MAE · lower is better",
        f"{len(record.days)} scored days · {record.scored_hours} scored hours",
    )
    lines.append(
        _text(
            70,
            196,
            "B1: same hour, seven days prior (D-7 persistence) · "
            "B2: same hour, one day prior (D-1 persistence)",
            fill=MUTED,
            size=11,
        )
    )
    _draw_axes(
        lines,
        points,
        xs_a,
        left=112,
        right=850,
        top=218,
        bottom=394,
        y_max=a_max,
        y_ticks=[a_max * index / 4 for index in range(5)],
        y_decimals=0,
        y_label="cumulative MAE (EUR/MWh)",
        x_label="scored delivery date",
    )
    series_a = [
        (model_name, "mae_model", MODEL, 2),
        ("B1 · same-hour 7-day", "mae_b1", B1, 2),
        ("B2 · same-hour previous-day", "mae_b2", B2, 2),
    ]
    for _, key, color, _ in series_a:
        lines.append(
            f'<path d="{_path(points, key, xs_a, a_max, 218, 394)}" fill="none" '
            f'stroke="{_esc(color)}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />'
        )
        lines.append(
            f'<circle cx="{_coord(xs_a[-1])}" cy="{_coord(_y_value(points[-1], key, a_max, 218, 394))}" '
            f'r="3" fill="{_esc(color)}" />'
        )
    _direct_labels(lines, points, xs_a, series_a, a_max, 218, 394, 870)

    # Panel B: negative-price probability.
    _panel_header(
        lines,
        70,
        513,
        "Panel B · Negative-price probability",
        "Cumulative Brier score · lower is better",
        f"{record.scored_hours} evaluated hours · {record.negative_hours} negative{below_text}",
    )
    lines.append(_text(70, 549, _support_line(record), fill=TEXT, size=11))
    lines.append(
        _text(
            70,
            563,
            "Descriptive early sample; no reliable forecasting claim is made before the gates.",
            fill=MUTED,
            size=10,
        )
    )
    _draw_axes(
        lines,
        points,
        xs_b,
        left=104,
        right=420,
        top=585,
        bottom=690,
        y_max=brier_max,
        y_ticks=[brier_max * index / 4 for index in range(5)],
        y_decimals=3,
        y_label="Brier score",
        x_label="scored delivery date",
    )
    series_b = [
        ("Model", "brier_model", MODEL, 4),
        ("Climatology", "brier_baseline", B1, 4),
    ]
    for _, key, color, _ in series_b:
        lines.append(
            f'<path d="{_path(points, key, xs_b, brier_max, 585, 690)}" fill="none" '
            f'stroke="{_esc(color)}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />'
        )
        lines.append(
            f'<circle cx="{_coord(xs_b[-1])}" cy="{_coord(_y_value(points[-1], key, brier_max, 585, 690))}" '
            f'r="3" fill="{_esc(color)}" />'
        )
    _direct_labels(lines, points, xs_b, series_b, brier_max, 585, 690, 438)

    # Panel C: interval coverage.
    _panel_header(
        lines,
        639,
        513,
        "Panel C · Interval coverage",
        "Empirical coverage against nominal reference",
        f"{record.scored_hours} evaluated hours",
    )
    lines.append(
        _text(
            639,
            549,
            "Coverage alone is not utility: wider intervals can over-cover.",
            fill=TEXT,
            size=11,
        )
    )
    lines.append(
        _text(
            639,
            563,
            "Pinball loss and interval width remain part of Call C.",
            fill=MUTED,
            size=10,
        )
    )
    _draw_axes(
        lines,
        points,
        xs_c,
        left=673,
        right=990,
        top=585,
        bottom=690,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        y_decimals=2,
        y_label="coverage",
        x_label="scored delivery date",
    )
    for nominal, label in ((0.5, "50% nominal"), (0.8, "80% nominal")):
        y = 690 - nominal * (690 - 585)
        lines.append(_line(673, y, 990, y, stroke=NOMINAL, width=1.0, dash="4 4"))
        lines.append(_text(681, y - 5, label, fill=NOMINAL, size=10))
    series_c = [
        ("50% empirical", "coverage_50", MODEL, 1),
        ("80% empirical", "coverage_80", MODEL_LIGHT, 1),
    ]
    for _, key, color, _ in series_c:
        lines.append(
            f'<path d="{_path(points, key, xs_c, 1.0, 585, 690)}" fill="none" '
            f'stroke="{_esc(color)}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />'
        )
        lines.append(
            f'<circle cx="{_coord(xs_c[-1])}" cy="{_coord(_y_value(points[-1], key, 1.0, 585, 690))}" '
            f'r="3" fill="{_esc(color)}" />'
        )
    _direct_labels(lines, points, xs_c, series_c, 1.0, 585, 690, 1008)

    lines.extend(
        [
            _line(48, 810, 1152, 810, stroke=RULE, width=1.0),
            _text(
                48,
                833,
                "Source: results/ledger.jsonl · SCORED rows only · cumulative values use scored-hour weights · "
                "missed and pending rows are excluded.",
                fill=MUTED,
                size=10,
            ),
            _text(
                48,
                850,
                "Sample still accumulating; no final claim is made before the preregistered gates are reached.",
                fill=MUTED,
                size=10,
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def write_svg(
    ledger_path: Path = LEDGER_PATH,
    output_path: Path = OUTPUT_PATH,
) -> TrackRecord:
    record = load_record(ledger_path)
    svg = render_svg(record)
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(svg)
    except OSError as exc:
        raise TrackRecordError(f"cannot write SVG {output_path}: {exc}") from exc
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the prospective track-record SVG")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    try:
        record = write_svg(args.ledger, args.output)
    except TrackRecordError as exc:
        print(f"track-record generation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"generated {args.output} from {len(record.days)} scored days "
        f"and {record.scored_hours} scored hours"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

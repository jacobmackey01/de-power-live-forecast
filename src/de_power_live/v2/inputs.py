"""Live inputs for a v2 seal. HTTP plumbing, deliberately outside the model hash.

The clients themselves (smard.py, weather.py) are inside v1's frozen hash and
are not edited here. v2 wraps them: one more day of weather, and a retry around
the whole fetch for the failures the clients do not catch themselves (SMARD
dropping a connection without replying killed a scoring run on 2026-10-27).

v2 reads one SMARD series, the day-ahead price. v1's runtime identity check
guards the forecast series against being relabelled; v2 uses none of them, so
it checks the price series itself instead.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from .. import MARKET_TZ
from ..dataset import target_hours_for
from ..smard import USER_AGENT, FetchRecord, SmardClient
from ..weather import API_URL, HOURLY_VARS, UNIT_PARAMS, OpenMeteoClient, Site
from .features import WEATHER_COLS

# Enough history for the 14-day lags, the 7-day scale and the 28-day windows.
SMARD_BLOCKS = 9
FETCH_ATTEMPTS = 3
FETCH_PAUSE_SECONDS = 20.0
ENERGY_CHARTS_PRICE_URL = "https://api.energy-charts.info/price"
PRICE_CENT = Decimal("0.01")

# Day-ahead prices are the only SMARD series v2 reads. They have been in
# [-500, 4000] EUR/MWh, the harmonised auction limits, for the whole history;
# anything outside means the series is not what it claims to be.
PRICE_BOUNDS = (-500.0, 4000.0)
# Every price the features, the scale, the 28-day spike threshold and the error
# pool can reach is within this many days of the newest one. All of them must
# be present. The history back to mid-2024 has no missing hour, so this refuses
# nothing that has ever actually occurred.
RECENT_DAYS = 30


class InputError(RuntimeError):
    """Raised when the inputs for a seal cannot be assembled."""


@dataclass
class V2WeatherClient(OpenMeteoClient):
    """The live forecast plus the previous day.

    The features need the seal day's weather from local midnight, which is the
    evening before in UTC. The v1 client starts at 00:00 UTC today.
    """

    past_days: int = 1

    def fetch_site(self, site: Site) -> pd.DataFrame:
        query = urllib.parse.urlencode(
            {
                "latitude": site.lat,
                "longitude": site.lon,
                "hourly": ",".join(HOURLY_VARS),
                "forecast_days": self.forecast_days,
                "past_days": self.past_days,
                "timezone": "UTC",
                **UNIT_PARAMS,
            }
        )
        return self._parse_hourly(self._get_json(f"{API_URL}?{query}"), site.name)


@dataclass
class LiveInputs:
    prices: pd.Series
    weather: pd.DataFrame
    price_check: dict
    weather_coverage: dict
    provenance: dict


def check_prices(prices: pd.Series) -> dict:
    """Refuse a price series that is empty, has a hole or a repeat, or is out of bounds."""
    if prices.empty:
        raise InputError("SMARD returned no day-ahead prices")
    low, high = float(prices.min()), float(prices.max())
    if low < PRICE_BOUNDS[0] or high > PRICE_BOUNDS[1]:
        raise InputError(f"day-ahead prices span [{low}, {high}], outside {PRICE_BOUNDS}")
    span = prices.index.max() - prices.index.min()
    if span < pd.Timedelta(days=RECENT_DAYS):
        # A short series has no hole in it and would pass the check below, with
        # 28-day windows it cannot fill. The frozen rules would carry on from as
        # few as 21 days; this refuses before they are asked.
        raise InputError(
            f"day-ahead prices reach back only {span.days} days; {RECENT_DAYS} are needed"
        )
    recent = prices[prices.index >= prices.index.max() - pd.Timedelta(days=RECENT_DAYS)]
    expected = pd.date_range(recent.index.min(), recent.index.max(), freq="h")
    absent = expected.difference(recent.index)
    missing = int(len(absent))
    if recent.index.has_duplicates:
        raise InputError("day-ahead prices repeat an hour")
    if missing:
        # Hours are counted in UTC, where no day has a gap, so any missing hour
        # is a real hole. The features would fill or skip it without complaint.
        raise InputError(
            f"{missing} hour(s) of day-ahead prices are missing in the last {RECENT_DAYS} days, "
            f"first at {absent[0].isoformat()}"
        )
    return {
        "n_prices": int(len(prices)),
        "first_utc": prices.index.min().isoformat(),
        "last_utc": prices.index.max().isoformat(),
        "min": low,
        "max": high,
        "missing_hours_recent": missing,
        "recent_days_checked": RECENT_DAYS,
        "ok": True,
    }


def _fill_missing_prices(prices: pd.Series, seal_day: pd.Timestamp) -> tuple[pd.Series, dict | None]:
    # Anchor the window to the last expected seal-day hour even if SMARD
    # has stopped publishing before then. Construct the grid in UTC for DST.
    last_hour = target_hours_for(seal_day)[-1]
    expected = pd.date_range(end=last_hour, periods=24 * RECENT_DAYS + 1, freq="h")
    missing = expected.difference(prices.index)
    if missing.empty:
        return prices, None
    if prices.index.has_duplicates or (prices.index > last_hour).any():
        raise InputError("SMARD prices repeat an hour or include a delivery day beyond the seal day")

    query = urllib.parse.urlencode(
        {
            "bzn": "DE-LU",
            "start": expected[0].tz_convert(MARKET_TZ).date().isoformat(),
            "end": seal_day.date().isoformat(),
        }
    )
    url = f"{ENERGY_CHARTS_PRICE_URL}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read()
    record = FetchRecord(
        url=url,
        sha256=hashlib.sha256(raw).hexdigest(),
        n_bytes=len(raw),
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    payload = json.loads(raw, parse_float=Decimal)
    if (
        payload.get("unit") != "EUR / MWh"
        or "Bundesnetzagentur | SMARD.de" not in payload.get("license_info", "")
    ):
        raise InputError("Energy-Charts response does not identify SMARD prices in EUR/MWh")

    stamps, values = payload["unix_seconds"], payload["price"]
    if len(stamps) != len(values) or any(type(stamp) is not int or stamp % 900 for stamp in stamps):
        raise InputError("Energy-Charts returned an invalid quarter-hour grid")
    index = pd.to_datetime(stamps, unit="s", utc=True)
    if index.has_duplicates:
        raise InputError("Energy-Charts returned repeated quarters")
    quarters = pd.Series(values, index=index, dtype=object).dropna()
    if (quarters.index >= last_hour + pd.Timedelta(hours=1)).any():
        raise InputError("Energy-Charts returned post-cutoff prices")
    if any(
        type(value) not in (int, Decimal) or not PRICE_BOUNDS[0] <= value <= PRICE_BOUNDS[1]
        for value in quarters
    ):
        raise InputError("Energy-Charts returned an invalid quarter-hour price")

    hourly = {}
    for hour in expected:
        values_at_hour = [
            quarters.get(hour + pd.Timedelta(minutes=offset)) for offset in (0, 15, 30, 45)
        ]
        if all(value is not None for value in values_at_hour):
            hourly[hour] = sum(values_at_hour, Decimal(0)) / 4

    overlap = prices.index.intersection(hourly)
    if overlap.empty or any(
        abs(hourly[hour] - Decimal(str(prices.loc[hour]))) > PRICE_CENT / 2 for hour in overlap
    ):
        raise InputError("Energy-Charts hourly means could not be reconciled with SMARD")

    unavailable = missing.difference(hourly)
    if len(unavailable):
        raise InputError(
            f"Energy-Charts lacks four quarter-hour prices for {len(unavailable)} missing SMARD hour(s), "
            f"first at {unavailable[0].isoformat()}"
        )
    recovered = pd.Series(
        [float(hourly[hour].quantize(PRICE_CENT, rounding=ROUND_HALF_UP)) for hour in missing],
        index=missing,
        name=prices.name,
    )
    return pd.concat([prices, recovered]).sort_index(), {
        "response": record.to_dict(),
        "filled_hours_utc": [hour.isoformat() for hour in missing],
        "overlap_hours_checked": len(overlap),
    }


def check_weather(weather: pd.DataFrame, target_day: pd.Timestamp) -> dict:
    """Refuse weather that does not cover every hour of the seal day and the target day.

    The model reads the forecast for both days. This looks at the hours as they
    arrived, in UTC, where no day has a gap: a 23-hour day is 23 real hours and
    a 25-hour day 25. It has to run on the raw frame, before the features are
    built, because the feature grid fills a missing 02:00 from the hours either
    side of it on any day, and a check made afterwards finds nothing missing.
    """
    target = pd.Timestamp(target_day).normalize()
    seal_day = target - pd.Timedelta(days=1)
    absent_columns = [col for col in WEATHER_COLS if col not in weather.columns]
    if absent_columns:
        raise InputError(f"weather has no column(s) {absent_columns}")
    if weather.index.has_duplicates:
        raise InputError("weather repeats an hour")
    needed = target_hours_for(seal_day).append(target_hours_for(target))
    absent = needed.difference(weather.index)
    if len(absent):
        raise InputError(
            f"{len(absent)} hour(s) of weather are missing for {seal_day.date()} and {target.date()}, "
            f"first at {absent[0].isoformat()}"
        )
    values = weather.loc[needed, list(WEATHER_COLS)].to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise InputError(
            f"{int(bad.sum())} weather value(s) are missing for {seal_day.date()} and {target.date()}, "
            f"first {WEATHER_COLS[col]} at {needed[row].isoformat()}"
        )
    return {
        "first_utc": needed[0].isoformat(),
        "last_utc": needed[-1].isoformat(),
        "hours_checked": int(len(needed)),
        "columns_checked": len(WEATHER_COLS),
        "ok": True,
    }


def _retry(label: str, call):
    last: Exception | None = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - any transient failure is retried
            last = exc
            if attempt < FETCH_ATTEMPTS:
                time.sleep(FETCH_PAUSE_SECONDS)
    raise InputError(f"{label} failed {FETCH_ATTEMPTS} times: {type(last).__name__}: {last}")


def fetch_live_inputs() -> LiveInputs:
    today = pd.Timestamp.now(tz=MARKET_TZ).normalize().tz_localize(None)

    def market():
        client = SmardClient()
        return client, client.fetch_series("price_da", n_blocks=SMARD_BLOCKS)

    smard, series = _retry("SMARD", market)
    primary_prices = series.dropna()
    prices, fallback = _retry("Price history", lambda: _fill_missing_prices(primary_prices, today))
    price_check = check_prices(prices)

    def forecast():
        client = V2WeatherClient()
        drivers, coverage = client.national_drivers()
        # The v1 client carries on with fewer sites and reweights the rest. The
        # frozen weights are part of the model, so v2 needs all of them.
        if coverage.get("sites_returned") != coverage.get("sites_requested"):
            raise InputError(f"weather returned for only some sites: {coverage.get('sites_failed')}")
        # Checked inside the retry, so that a response with an hour missing is
        # asked for again rather than given up on.
        check_weather(drivers, today + pd.Timedelta(days=1))
        return client, drivers, coverage

    meteo, weather, coverage = _retry("Open-Meteo", forecast)

    provenance = {"smard": smard.provenance(), "open_meteo": meteo.provenance()}
    if fallback is not None:
        provenance["energy_charts"] = fallback

    return LiveInputs(
        prices=prices,
        weather=weather,
        price_check=price_check,
        weather_coverage=coverage,
        provenance=provenance,
    )

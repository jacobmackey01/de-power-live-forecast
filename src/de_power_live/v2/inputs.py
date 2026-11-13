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

import time
import urllib.parse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import MARKET_TZ
from ..dataset import target_hours_for
from ..smard import SmardClient
from ..weather import API_URL, HOURLY_VARS, UNIT_PARAMS, OpenMeteoClient, Site
from .features import WEATHER_COLS

# Enough history for the 14-day lags, the 7-day scale and the 28-day windows.
SMARD_BLOCKS = 9
FETCH_ATTEMPTS = 3
FETCH_PAUSE_SECONDS = 20.0

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
    def market():
        client = SmardClient()
        return client, client.fetch_series("price_da", n_blocks=SMARD_BLOCKS)

    smard, series = _retry("SMARD", market)
    prices = series.dropna()
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
        today = pd.Timestamp.now(tz=MARKET_TZ).normalize().tz_localize(None)
        check_weather(drivers, today + pd.Timedelta(days=1))
        return client, drivers, coverage

    meteo, weather, coverage = _retry("Open-Meteo", forecast)

    return LiveInputs(
        prices=prices,
        weather=weather,
        price_check=price_check,
        weather_coverage=coverage,
        provenance={"smard": smard.provenance(), "open_meteo": meteo.provenance()},
    )

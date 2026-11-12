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

import pandas as pd

from ..smard import SmardClient
from ..weather import API_URL, HOURLY_VARS, UNIT_PARAMS, OpenMeteoClient, Site

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
        return client, drivers, coverage

    meteo, weather, coverage = _retry("Open-Meteo", forecast)

    return LiveInputs(
        prices=prices,
        weather=weather,
        price_check=price_check,
        weather_coverage=coverage,
        provenance={"smard": smard.provenance(), "open_meteo": meteo.provenance()},
    )

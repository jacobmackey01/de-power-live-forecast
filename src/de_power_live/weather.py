"""Open-Meteo client: the forward drivers for D+1.

Why this is the primary source rather than SMARD's published forecasts: SMARD
populates D+1 only *after* the day-ahead auction has cleared (verified
2026-08-01, see PREREGISTRATION.md section 2). Open-Meteo publishes a genuine
3-day forward horizon and was confirmed to supply 24/24 hours of D+1 at every
site below, with no API key.

Site weights approximate where German wind and solar capacity actually sits.
They are deliberately coarse and are frozen as part of model v1 - they are a
declared modelling choice, not a fitted parameter.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

API_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "de-power-live-forecast/0.1 (+https://github.com/jacobmackey01/de-power-live-forecast)"

HOURLY_VARS = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "cloud_cover",
    "surface_pressure",
]


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float
    wind_weight: float  # share of national wind capacity this site proxies
    solar_weight: float  # share of national solar capacity
    load_weight: float  # share of population, for temperature-driven demand


# Coarse proxies for the German capacity map: wind concentrated in the north and
# offshore, solar skewed south, load following population.
SITES: tuple[Site, ...] = (
    Site("bremerhaven_n", 53.55, 8.58, wind_weight=0.22, solar_weight=0.05, load_weight=0.08),
    Site("rostock_ne", 54.09, 12.14, wind_weight=0.16, solar_weight=0.04, load_weight=0.05),
    Site("magdeburg_e", 52.13, 11.62, wind_weight=0.18, solar_weight=0.10, load_weight=0.08),
    Site("german_bight_offshore", 54.00, 6.60, wind_weight=0.20, solar_weight=0.00, load_weight=0.00),
    Site("kassel_central", 51.31, 9.49, wind_weight=0.12, solar_weight=0.13, load_weight=0.14),
    Site("cologne_w", 50.94, 6.96, wind_weight=0.08, solar_weight=0.13, load_weight=0.25),
    Site("munich_s", 48.14, 11.58, wind_weight=0.02, solar_weight=0.30, load_weight=0.25),
    Site("freiburg_sw", 48.00, 7.85, wind_weight=0.02, solar_weight=0.25, load_weight=0.15),
)

# Generic onshore turbine power curve (m/s). Real fleets are messier, but the
# cubic-then-saturating shape is what matters: raw wind speed is a poor feature
# because the power relationship is strongly non-linear and bounded at both ends.
CUT_IN_MS = 3.0
RATED_MS = 12.0
CUT_OUT_MS = 25.0


class WeatherError(RuntimeError):
    """Raised when Open-Meteo cannot be reached or returns an unusable payload."""


@dataclass
class FetchRecord:
    url: str
    sha256: str
    n_bytes: int
    fetched_at_utc: str

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "sha256": self.sha256,
            "n_bytes": self.n_bytes,
            "fetched_at_utc": self.fetched_at_utc,
        }


def turbine_power_curve(wind_speed_ms: np.ndarray) -> np.ndarray:
    """Map hub-height wind speed to normalised turbine output in [0, 1].

    Cubic ramp between cut-in and rated, flat at rated, zero beyond cut-out.
    The cut-out shelf matters: storms *reduce* generation, and a linear feature
    would predict the opposite at exactly the hours where prices move most.
    """
    speed = np.asarray(wind_speed_ms, dtype=float)
    out = np.zeros_like(speed)

    ramp = (speed >= CUT_IN_MS) & (speed < RATED_MS)
    out[ramp] = (speed[ramp] ** 3 - CUT_IN_MS**3) / (RATED_MS**3 - CUT_IN_MS**3)

    rated = (speed >= RATED_MS) & (speed < CUT_OUT_MS)
    out[rated] = 1.0

    # At or beyond cut-out, and for NaN input, output stays 0 / NaN respectively.
    out[np.isnan(speed)] = np.nan
    return out


@dataclass
class OpenMeteoClient:
    """Fetches forward weather and reduces it to national driver features."""

    sites: tuple[Site, ...] = SITES
    forecast_days: int = 3
    timeout: float = 30.0
    max_retries: int = 4
    backoff_seconds: float = 2.0
    fetches: list[FetchRecord] = field(default_factory=list)

    def _get_json(self, url: str) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
        else:
            raise WeatherError(f"failed to fetch {url}: {last_exc}")

        self.fetches.append(
            FetchRecord(
                url=url,
                sha256=hashlib.sha256(raw).hexdigest(),
                n_bytes=len(raw),
                fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeatherError(f"malformed JSON from {url}: {exc}") from exc

    def fetch_site(self, site: Site) -> pd.DataFrame:
        """Hourly forward weather for one site, indexed by tz-aware UTC."""
        query = urllib.parse.urlencode(
            {
                "latitude": site.lat,
                "longitude": site.lon,
                "hourly": ",".join(HOURLY_VARS),
                "forecast_days": self.forecast_days,
                "timezone": "UTC",
            }
        )
        payload = self._get_json(f"{API_URL}?{query}")

        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            raise WeatherError(f"no hourly block for site {site.name}")

        missing = [v for v in HOURLY_VARS if v not in hourly]
        if missing:
            raise WeatherError(f"site {site.name} missing variables: {missing}")

        index = pd.to_datetime(hourly["time"], utc=True)
        frame = pd.DataFrame({v: hourly[v] for v in HOURLY_VARS}, index=index)
        frame.index.name = "timestamp_utc"
        return frame

    def national_drivers(self) -> tuple[pd.DataFrame, dict]:
        """Capacity-weighted national driver features.

        Returns the feature frame and a coverage report. Weights are renormalised
        over the sites that actually returned data, so a single failed site
        degrades precision rather than silently biasing the national total
        toward whichever regions happened to respond.
        """
        per_site: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}
        for site in self.sites:
            try:
                per_site[site.name] = self.fetch_site(site)
            except WeatherError as exc:
                failures[site.name] = str(exc)

        if not per_site:
            raise WeatherError("every site failed; cannot build drivers")

        ok = [s for s in self.sites if s.name in per_site]
        index = per_site[ok[0].name].index
        for name, frame in per_site.items():
            if not frame.index.equals(index):
                index = index.intersection(frame.index)
        if len(index) == 0:
            raise WeatherError("sites returned non-overlapping time indices")

        def weighted(attr: str, column: str, transform=None) -> pd.Series:
            total = sum(getattr(s, attr) for s in ok)
            if total <= 0:
                return pd.Series(np.nan, index=index)
            acc = pd.Series(0.0, index=index)
            for site in ok:
                w = getattr(site, attr) / total
                if w == 0:
                    continue
                values = per_site[site.name].loc[index, column].astype(float)
                if transform is not None:
                    values = pd.Series(transform(values.to_numpy()), index=index)
                acc = acc + w * values
            return acc

        drivers = pd.DataFrame(index=index)
        drivers["wind_cf"] = weighted("wind_weight", "wind_speed_100m", turbine_power_curve)
        drivers["wind_speed_100m"] = weighted("wind_weight", "wind_speed_100m")
        drivers["wind_gust_10m"] = weighted("wind_weight", "wind_gusts_10m")
        drivers["solar_radiation"] = weighted("solar_weight", "shortwave_radiation")
        drivers["solar_direct"] = weighted("solar_weight", "direct_radiation")
        drivers["cloud_cover"] = weighted("solar_weight", "cloud_cover")
        drivers["temperature"] = weighted("load_weight", "temperature_2m")
        drivers["surface_pressure"] = weighted("load_weight", "surface_pressure")

        # Ramp features: hour-on-hour change in the two generation drivers.
        drivers["wind_cf_ramp"] = drivers["wind_cf"].diff()
        drivers["solar_radiation_ramp"] = drivers["solar_radiation"].diff()

        coverage = {
            "sites_requested": len(self.sites),
            "sites_returned": len(per_site),
            "sites_failed": failures,
            "wind_weight_covered": round(sum(s.wind_weight for s in ok), 4),
            "solar_weight_covered": round(sum(s.solar_weight for s in ok), 4),
            "load_weight_covered": round(sum(s.load_weight for s in ok), 4),
            "n_hours": int(len(index)),
            "first_hour_utc": index[0].isoformat(),
            "last_hour_utc": index[-1].isoformat(),
        }
        return drivers, coverage

    def provenance(self) -> list[dict]:
        return [f.to_dict() for f in self.fetches]

"""Client for the SMARD (Bundesnetzagentur) chart_data API.

No API key is required. Every response is hashed on arrival so that the exact
inputs behind a sealed prediction can be proven after the fact.

Filter identification
---------------------
SMARD does not return usable labels in the ``meta`` block, so the filter IDs
below were verified arithmetically against a live payload (2026-08-01) rather
than taken from documentation:

    5097 == 123 + 3791 + 125      (renewables fc == wind onshore + offshore + solar)
    715  == 122 - 5097            (residual load fc == load fc - renewables fc)

Both identities held exactly, which pins 122 as total load forecast, 5097 as
total renewable generation forecast and 715 as residual load forecast. Filter
4359 was rejected: its values (~8 GW) are an order of magnitude below German
system load, so whatever it is, it is not the load forecast.

``verify_filter_identities`` re-checks those identities at runtime. It is wired
into the daily prediction job because a silent relabelling upstream would
otherwise poison the model without raising anything.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

BASE_URL = "https://www.smard.de/app/chart_data"
DEFAULT_REGION = "DE"
DEFAULT_RESOLUTION = "hour"

USER_AGENT = "de-power-live-forecast/0.1 (+https://github.com/jacobmackey01/de-power-live-forecast)"

# Verified filter IDs. See module docstring for how these were established.
FILTERS: dict[str, int] = {
    # Target
    "price_da": 4169,  # DE-LU day-ahead price, EUR/MWh
    # Day-ahead forecasts (published by TSOs, mirrored by SMARD)
    "load_fc": 122,  # total load forecast, MW
    "renewables_fc": 5097,  # total renewable generation forecast, MW
    "residual_load_fc": 715,  # residual load forecast, MW  (= load_fc - renewables_fc)
    "wind_onshore_fc": 123,  # MW
    "wind_offshore_fc": 3791,  # MW
    "solar_fc": 125,  # MW
    # Realised outturn (used for lags and for scoring)
    "load_actual": 410,  # MW
    "wind_onshore_actual": 4067,  # MW
    "wind_offshore_actual": 1225,  # MW
    "solar_actual": 4068,  # MW
}

# Components that must sum to the aggregate, used by verify_filter_identities.
_RENEWABLE_COMPONENTS = ("wind_onshore_fc", "wind_offshore_fc", "solar_fc")


class SmardError(RuntimeError):
    """Raised when SMARD cannot be reached or returns something unusable."""


@dataclass
class FetchRecord:
    """Provenance for a single HTTP response.

    Recorded alongside every prediction so the inputs available at seal time are
    verifiable later, rather than merely asserted.
    """

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


@dataclass
class SmardClient:
    """Reads SMARD hourly series and records provenance for each fetch."""

    region: str = DEFAULT_REGION
    resolution: str = DEFAULT_RESOLUTION
    timeout: float = 30.0
    max_retries: int = 4
    backoff_seconds: float = 2.0
    fetches: list[FetchRecord] = field(default_factory=list)

    # ---- HTTP -----------------------------------------------------------

    def _get_raw(self, url: str) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise SmardError(f"failed to fetch {url} after {self.max_retries} attempts: {last_exc}")

    def _get_json(self, url: str) -> dict:
        raw = self._get_raw(url)
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
            raise SmardError(f"malformed JSON from {url}: {exc}") from exc

    # ---- Series ---------------------------------------------------------

    def index_timestamps(self, filter_id: int) -> list[int]:
        """Epoch-ms start of each weekly block SMARD holds for this filter."""
        url = f"{BASE_URL}/{filter_id}/{self.region}/index_{self.resolution}.json"
        payload = self._get_json(url)
        stamps = payload.get("timestamps")
        if not stamps:
            raise SmardError(f"filter {filter_id} returned an empty index")
        return sorted(int(t) for t in stamps)

    def _block(self, filter_id: int, block_ts: int) -> list[list]:
        url = (
            f"{BASE_URL}/{filter_id}/{self.region}/"
            f"{filter_id}_{self.region}_{self.resolution}_{block_ts}.json"
        )
        payload = self._get_json(url)
        series = payload.get("series")
        if series is None:
            raise SmardError(f"filter {filter_id} block {block_ts} has no series")
        return series

    def fetch_series(self, name: str, n_blocks: int = 2) -> pd.Series:
        """Fetch the most recent ``n_blocks`` weekly blocks for a named filter.

        Returns an hourly series indexed by tz-aware UTC timestamps. Nulls are
        preserved rather than filled: an unpublished future hour and a genuine
        zero are different things, and collapsing them is how a forecast quietly
        starts predicting on data it does not have.
        """
        if name not in FILTERS:
            raise KeyError(f"unknown series name {name!r}; known: {sorted(FILTERS)}")
        filter_id = FILTERS[name]

        blocks = self.index_timestamps(filter_id)[-n_blocks:]
        points: list[list] = []
        for block_ts in blocks:
            points.extend(self._block(filter_id, block_ts))

        if not points:
            raise SmardError(f"no data points returned for {name}")

        frame = pd.DataFrame(points, columns=["ts_ms", "value"])
        frame = frame.drop_duplicates(subset="ts_ms", keep="last").sort_values("ts_ms")
        index = pd.to_datetime(frame["ts_ms"], unit="ms", utc=True)
        return pd.Series(frame["value"].to_numpy(), index=index, name=name)

    def fetch_frame(self, names: list[str], n_blocks: int = 2) -> pd.DataFrame:
        """Fetch several series and align them on a common UTC hourly index."""
        return pd.concat(
            [self.fetch_series(name, n_blocks=n_blocks) for name in names],
            axis=1,
        ).sort_index()

    # ---- Sanity ---------------------------------------------------------

    def verify_filter_identities(self, frame: pd.DataFrame, tolerance: float = 1.0) -> dict:
        """Re-check the arithmetic identities that pin the filter meanings.

        A silent upstream relabelling would otherwise feed the model a different
        quantity under the same name. Returns a report; callers decide whether a
        breach is fatal.
        """
        report: dict = {"checked_at_utc": datetime.now(timezone.utc).isoformat()}

        needed = {"renewables_fc", *_RENEWABLE_COMPONENTS}
        if needed.issubset(frame.columns):
            component_sum = frame[list(_RENEWABLE_COMPONENTS)].sum(axis=1, min_count=len(_RENEWABLE_COMPONENTS))
            gap = (frame["renewables_fc"] - component_sum).abs()
            worst = float(gap.max()) if gap.notna().any() else None
            report["renewables_identity"] = {
                "max_abs_error_mw": worst,
                "n_compared": int(gap.notna().sum()),
                "ok": worst is not None and worst <= tolerance,
            }

        if {"residual_load_fc", "load_fc", "renewables_fc"}.issubset(frame.columns):
            implied = frame["load_fc"] - frame["renewables_fc"]
            gap = (frame["residual_load_fc"] - implied).abs()
            worst = float(gap.max()) if gap.notna().any() else None
            report["residual_load_identity"] = {
                "max_abs_error_mw": worst,
                "n_compared": int(gap.notna().sum()),
                "ok": worst is not None and worst <= tolerance,
            }

        checks = [v for k, v in report.items() if isinstance(v, dict) and "ok" in v]
        report["all_ok"] = bool(checks) and all(c["ok"] for c in checks)
        return report

    def provenance(self) -> list[dict]:
        """Every response fetched by this client, in order, with hashes."""
        return [f.to_dict() for f in self.fetches]

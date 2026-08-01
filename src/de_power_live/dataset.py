"""Training-set construction.

Built day by day through the *same* ``build_features`` call the live path uses,
with the cutoff each day would actually have had. Assembling training features
with a separate, more convenient code path is how train/serve skew gets in: the
offline version quietly has access to something the online one does not, and
nothing fails until the record is live.

It is slower than a single vectorised pass over history. That is the price of
the two paths being provably the same function.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import AUCTION_CLOSE_LOCAL_HOUR, MARKET_TZ
from .features import build_features
from .smard import SmardClient
from .weather import OpenMeteoClient

# Seal an hour before the gate closes, matching the live schedule's safety margin.
SEAL_LOCAL_HOUR = AUCTION_CLOSE_LOCAL_HOUR - 1


def seal_time_for(target_day: pd.Timestamp) -> pd.Timestamp:
    """The UTC instant a forecast for ``target_day`` would have been sealed.

    Anchored in local market time then converted, so the UTC hour shifts with
    DST exactly as the real deadline does.
    """
    day_before = pd.Timestamp(target_day).tz_localize(None).normalize() - pd.Timedelta(days=1)
    local = day_before.tz_localize(MARKET_TZ) + pd.Timedelta(hours=SEAL_LOCAL_HOUR)
    return local.tz_convert("UTC")


def target_hours_for(target_day: pd.Timestamp) -> pd.DatetimeIndex:
    """Every hour of the local power day, in UTC.

    Length is 23, 24 or 25 depending on DST. Hard-coding 24 is the classic way
    to lose an hour in October and gain a phantom one in March.
    """
    day = pd.Timestamp(target_day).tz_localize(None).normalize()
    start = day.tz_localize(MARKET_TZ).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="h", inclusive="left")


@dataclass
class TrainingData:
    features: pd.DataFrame
    prices: pd.Series
    days_built: int
    days_skipped: dict[str, str]


def build_training_set(
    prices: pd.Series,
    weather: pd.DataFrame,
    actuals: pd.DataFrame,
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
) -> TrainingData:
    """Assemble features for every target day in ``[first_day, last_day]``."""
    frames: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}

    for day in pd.date_range(first_day, last_day, freq="D"):
        try:
            hours = target_hours_for(day)
            cutoff = seal_time_for(day)

            if not hours.isin(weather.index).all():
                skipped[str(day.date())] = "weather does not cover the full power day"
                continue

            realised = prices.reindex(hours)
            if realised.notna().sum() < len(hours) * 0.9:
                skipped[str(day.date())] = "outturn prices incomplete"
                continue

            frames.append(
                build_features(
                    weather=weather,
                    prices=prices,
                    actuals=actuals,
                    cutoff_utc=cutoff,
                    target_index=hours,
                )
            )
        except Exception as exc:  # noqa: BLE001
            skipped[str(day.date())] = f"{type(exc).__name__}: {exc}"

    if not frames:
        raise RuntimeError(f"no usable training days; skips: {skipped}")

    features = pd.concat(frames).sort_index()
    features = features[~features.index.duplicated(keep="first")]
    return TrainingData(
        features=features,
        prices=prices.reindex(features.index),
        days_built=len(frames),
        days_skipped=skipped,
    )


def fetch_history(
    years: float = 2.0,
    end_day: pd.Timestamp | None = None,
    cache: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    """Pull the raw history needed for training: prices, weather, actuals.

    Cached to disk by (years, end day). A full pull is several hundred sequential
    SMARD block requests; re-running it on every model iteration is slow for us
    and impolite to a free public API. The cache is gitignored - sealed
    predictions carry input hashes, which is the part that has to be auditable.
    """
    end = pd.Timestamp(end_day or pd.Timestamp.now(tz=MARKET_TZ).normalize()).tz_localize(None)
    start = end - pd.Timedelta(days=int(365 * years))

    cache_path = (
        Path(__file__).resolve().parents[2] / "cache" / f"history_{years}y_{end.date()}.pkl"
    )
    if cache and cache_path.exists():
        with cache_path.open("rb") as handle:
            prices, weather, actuals, provenance = pickle.load(handle)
        provenance = dict(provenance)
        provenance["served_from_cache"] = str(cache_path.name)
        return prices, weather, actuals, provenance

    smard = SmardClient()
    # 2 weekly blocks per week of history, plus slack for the lag warm-up.
    n_blocks = int(years * 53) + 3
    columns = [
        "price_da",
        "load_actual",
        "wind_onshore_actual",
        "wind_offshore_actual",
        "solar_actual",
    ]
    frame = smard.fetch_frame(columns, n_blocks=n_blocks)

    identities = smard.verify_filter_identities(
        smard.fetch_frame(
            [
                "load_fc",
                "renewables_fc",
                "residual_load_fc",
                "wind_onshore_fc",
                "wind_offshore_fc",
                "solar_fc",
            ],
            n_blocks=2,
        )
    )
    if not identities.get("all_ok"):
        raise RuntimeError(f"SMARD filter identities failed: {identities}")

    prices = frame["price_da"].dropna()
    actuals = pd.DataFrame(
        {
            "wind_total_actual": frame["wind_onshore_actual"].fillna(0)
            + frame["wind_offshore_actual"].fillna(0),
            "solar_actual": frame["solar_actual"],
            "load_actual": frame["load_actual"],
        }
    )

    meteo = OpenMeteoClient()
    weather, coverage = meteo.national_drivers(
        start_date=str(start.date()), end_date=str(end.date())
    )

    provenance = {
        "smard_fetches": len(smard.provenance()),
        "openmeteo_fetches": len(meteo.provenance()),
        "identity_check": identities,
        "weather_coverage": coverage,
        "price_span_utc": [prices.index.min().isoformat(), prices.index.max().isoformat()],
        "n_prices": int(len(prices)),
    }

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump((prices, weather, actuals, provenance), handle)

    return prices, weather, actuals, provenance

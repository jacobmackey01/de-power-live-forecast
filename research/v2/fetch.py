"""Pull the history the v2 replay runs on. Cached under cache/ (gitignored)."""
import sys, time
from de_power_live.dataset import fetch_history

for attempt in range(1, 6):
    try:
        prices, weather, actuals, prov = fetch_history(years=2.3)
        break
    except Exception as exc:  # SMARD drops connections; smard.py does not retry all of them
        print(f"attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(20)
else:
    sys.exit(1)
print("prices", prices.index.min(), prices.index.max(), len(prices))
print("weather", weather.index.min(), weather.index.max(), weather.shape, list(weather.columns))
print("actuals", actuals.index.min(), actuals.index.max(), actuals.notna().sum().to_dict())
print({k: v for k, v in prov.items() if k not in ("identity_check", "weather_coverage")})

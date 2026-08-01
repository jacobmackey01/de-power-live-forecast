"""Live pre-committed day-ahead forecasting for the DE-LU power market.

The calls this system makes are declared in PREREGISTRATION.md, which is frozen
as of the first sealed prediction. Read that first: it defines what is being
claimed, what is explicitly not being claimed, and the information cutoff rule
that keeps the record honest.
"""

__version__ = "0.1.0"

# Bumped whenever the model or its frozen parameters change. Every sealed
# prediction records this alongside a hash of the model source, and the final
# report must cover every version that ever produced a live prediction.
MODEL_VERSION = "v1"

BIDDING_ZONE = "DE-LU"
MARKET_TZ = "Europe/Berlin"

# Day-ahead auction gate closure. Seals must land strictly before this.
AUCTION_CLOSE_LOCAL_HOUR = 12

"""Model v2 and the phase-2 record.

Phase 2 is a separate, separately pre-registered phase (PREREGISTRATION_PHASE2.md).
Nothing in this package is imported by the v1 path, and nothing here changes a
file inside v1's frozen source hash: v1 keeps sealing unchanged as the benchmark.

What v2 is, in one paragraph. v1 predicted price minus last week's price but
also saw raw price levels, and learned that a high price falls back to the level
of its training years. v2 removes the level: the target is the change from the
same local hour yesterday, every price input is a difference, and both are
divided by the average price of the last seven days. docs/v1-lessons.md and
research/v2/README.md set out why.
"""

from datetime import date
from pathlib import Path

MODEL_VERSION = "v2"

REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_DIR = REPO_ROOT / "predictions" / "v2"
MODELS_DIR = REPO_ROOT / "models" / "v2"
RESULTS_DIR = REPO_ROOT / "results" / "v2"
LEDGER_PATH = RESULTS_DIR / "ledger.jsonl"

# The phase-2 evaluation window ends here, inclusive. No extension.
EVALUATION_END = date(2027, 2, 28)
# The window end is fixed, so a late start shortens it. If the first v2 forecast
# is for a delivery day after this, phase 2 is reported with no verdicts
# (PREREGISTRATION_PHASE2.md section 6).
LATEST_FIRST_DAY_FOR_VERDICTS = date(2026, 12, 15)

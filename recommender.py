# recommender.py — winner-preserving filter recommendations with anti-affinity pre-trim
from __future__ import annotations

import math, html, itertools as it
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# =============================================================================
# Defaults (UI can override)
# =============================================================================
WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV = "lottery_filters_batch_10.csv"
TODAY_POOL_CSV: Optional[str] = None

TARGET_MAX = 44
ALWAYS_KEEP_WINNER = True
MINIMIZE_BEYOND_TARGET = True
OUTPUT_DIR = Path(".")

# Anti-affinity default: trim OFF the top X% most seed-like combos
AFFINITY_EXCLUDE_TOP_PCT = 0.25

# Where dated pools live, e.g., pools/pool_2025-08-22.csv
POOL_ARCHIVE_DIR = Path("pools")

# --- NoPool integration knobs (historical signals; no simulated pool) ---
INCLUDE_NOPOOL_PANEL = True
NOPOOL_MIN_DAYS      = 60       # require both-applicable on ≥ this many days
NOPOOL_MIN_KEEP      = 75.0     # show pairs with winner-kept ≥ this %
NOPOOL_PARITY_SKEW   = 10.0     # show parity only if abs(50 - even%) ≥ this
NOPOOL_MAX_ROWS      = 20       # how many pairs to list in the panel

# =============================================================================
# Pool archive helpers
# =============================================================================
def get_pool_for_seed(seed_row: pd.Series, *, keep_permutations: bool = True) -> pd.DataFrame:
    """
    Load the EXACT pool your app had at the pre-CSV stage for this seed's date.
    Strategy:
      1) If seed_row has a Date/DrawDate, try pools/pool_YYYY-MM-DD.csv
      2) Else/fallback to TODAY_POOL_CS

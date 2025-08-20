# recommender.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import itertools as it
import html
from datetime import datetime

# =========================
# Defaults (UI can override)
# =========================
WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV = "lottery_filters_batch_10.csv"
TODAY_POOL_CSV: Optional[str] = None
TARGET_MAX = 44
ALWAYS_KEEP_WINNER = True
MINIMIZE_BEYOND_TARGET = True
OUTPUT_DIR = Path(".")

# =========================
# Helpers / domain mapping
# =========================
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
# App uses 1..5 VTRAC labels
VTRAC = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "V

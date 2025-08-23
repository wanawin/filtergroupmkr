# recommender.py — winner-prediction filter recommendations with anti-affinity pre-trim
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
# NOTE: kept for compatibility, but the app NEVER guards the winner now.
ALWAYS_KEEP_WINNER = False
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
      2) Else/fallback to TODAY_POOL_CSV or repo-root today_pool.csv
    Returns a DataFrame with a 'combo' column of 5-digit strings.
    """
    date_val = None
    for c in getattr(seed_row, "index", []):
        lc = str(c).lower()
        if lc in ("date", "drawdate", "draw_date"):
            try:
                date_val = pd.to_datetime(seed_row[c], errors="coerce").date()
            except Exception:
                date_val = None
            break

    df = None
    if date_val:
        arch = POOL_ARCHIVE_DIR / f"pool_{date_val}.csv"
        if arch.exists():
            df = pd.read_csv(arch)

    if df is None:
        fallback = (globals().get("TODAY_POOL_CSV") or "today_pool.csv")
        if not Path(fallback).exists():
            raise FileNotFoundError(
                f"No archived pool found for {date_val} and no {fallback} present. "
                f"Add pools/pool_<date>.csv or provide today_pool.csv."
            )
        df = pd.read_csv(fallback)

    if "combo" not in df.columns:
        if "Result" in df.columns:
            df = df.rename(columns={"Result": "combo"})
        else:
            raise RuntimeError("Pool CSV must contain 'combo' or 'Result' column.")

    df["combo"] = (
        df["combo"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    )
    df = df[df["combo"].str.fullmatch(r"\d{5}")]
    return df[["combo"]].copy()

# =============================================================================
# Domain helpers / env
# =============================================================================
MIRROR = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}
VTRAC  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in str(s)]

def sum_category(total: int) -> str:
    if 0 <= total <= 15:  return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def spread_band(spread: int) -> str:
    if spread <= 3: return "0-3"
    if spread <= 5: return "4-5"
    if spread <= 7: return "6-7"
    if spread <= 9: return "8-9"
    return "10+"

def hot_cold_due(history: List[List[int]], k_hotcold: int = 10):
    flat = [d for row in history[-k_hotcold:] for d in row]
    cnt = Counter(flat)
    hot, cold = set(), set()
    if cnt:
        most = cnt.most_common()
        topk = 6
        thresh = most[topk-1][1] if len(most) >= topk else most[-1][1]
        hot = {d for d,c in most if c >= thresh}
        least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
        coldk = 4
        if least:
            cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
            cold = {d for d,c in least if c <= cth}
    last2 = set(d for row in history[-2:] for d in row)
    due = set(range(10)) - last2
    return hot, cold, due

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def _parity_major_label(digs: List[int]) -> str:
    return "even>=3" if sum(1 for d in digs if d % 2 == 0) >= 3 else "even<=2"

def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
    seed   = winners[idx-1]
    winner = winners[idx]
    seed_list  = digits_of(seed)
    combo_list = sorted(digits_of(winner))

    seed_sum = sum(seed_list)
    combo_sum = sum(combo_list)
    history_digits = [digits_of(s) for s in winners[:idx]]
    hot, cold, due = hot_cold_due(history_digits, k_hotcold=10)

    env = {
        "combo": winner,
        "combo_digits": set(combo_list),

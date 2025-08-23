# recommender.py — winner-prediction filter recommendations + archetype-conditioned safety
from __future__ import annotations

import math, html, itertools as it
from dataclasses import dataclass
from collections import Counter, defaultdict
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
# Prediction mode: NEVER guard the (unknown) winner.
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
    ev = sum(1 for d in digs if d % 2 == 0)
    return f"{ev}E{5-ev}O"

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
        "combo_digits_list": combo_list,
        "combo_sum": combo_sum,
        "combo_sum_cat": sum_category(combo_sum),
        "combo_sum_category": sum_category(combo_sum),

        "seed": seed,
        "seed_digits": set(seed_list),
        "seed_digits_list": seed_list,
        "seed_sum": seed_sum,
        "seed_sum_category": sum_category(seed_sum),

        "spread_seed": max(seed_list) - min(seed_list),
        "spread_combo": max(combo_list) - min(combo_list),

        "seed_vtracs": set(VTRAC[d] for d in seed_list),
        "combo_vtracs": set(VTRAC[d] for d in combo_list),

        "hot_digits": sorted(hot),
        "cold_digits": sorted(cold),
        "due_digits": sorted(due),

        "mirror": MIRROR,
        "vtrac": VTRAC,

        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted, "Counter": Counter,
    }
    return env

# =============================================================================
# CSV loaders
# =============================================================================
@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str
    expression: str

def load_winners(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = "Result" if "Result" in df.columns else None
    if col is None:
        for c in df.columns:
            vals = df[c].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            if (vals.str.fullmatch(r"\d{5}")).all():
                col = c; break
    if col is None:
        raise ValueError("Winners CSV must have a 5-digit column (preferably named 'Result').")
    vals = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    vals = vals[vals.str.fullmatch(r"\d{5}")]
    return vals.tolist()

def load_filters(path: str) -> List[FilterDef]:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    req = ["id","name","enabled","applicable_if","expression"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"Filters CSV missing column: {r}")
    def to_bool(x):
        if isinstance(x, bool): return x
        if pd.isna(x): return False
        return str(x).strip().lower() in {"true","1","yes","y"}
    df["enabled"] = df["enabled"].map(to_bool)
    for col in ["applicable_if","expression"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('"""','"', regex=False).str.replace("'''","'", regex=False)
        df[col] = df[col].apply(
            lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"', "'"} else s
        )
    out: List[FilterDef] = []
    for _, r in df.iterrows():
        out.append(FilterDef(
            str(r["id"]).strip(),
            str(r["name"]).strip(),
            bool(r["enabled"]),
            str(r["applicable_if"]).strip(),
            str(r["expression"]).strip(),
        ))
    return out

# =============================================================================
# Affinity (used ONLY to rank by similarity for pre-trim)
# =============================================================================
def combo_affinity(env_now: Dict[str, object], combo: str) -> float:
    """
    Deterministic similarity score used only for ranking within today's pool.
    Higher = 'more similar' to seed. We will DROP the top X% (anti-affinity).
    """
    cd = digits_of(combo)
    seedd = env_now["seed_digits_list"]
    combo_sum, seed_sum = sum(cd), env_now["seed_sum"]
    sum_prox = math.exp(-abs(combo_sum - seed_sum) / 2.0)

    spread_c = max(cd) - min(cd)
    spread_s = env_now["spread_seed"]
    spread_prox = math.exp(-abs(spread_c - spread_s) / 2.0)

    struct_match = 1.0 if classify_structure(cd) == classify_structure(seedd) else 0.0
    parity_match = 1.0 if _parity_major_label(cd) == _parity_major_label(seedd) else 0.0

    hi8_seed = sum(1 for d in seedd if d >= 8)
    hi8_combo = sum(1 for d in cd if d >= 8)
    if

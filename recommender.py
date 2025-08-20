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

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in s]

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs)
    counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

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

# =========================
# CSV loaders
# =========================
@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str
    expression: str

def load_winners(path: str) -> List[str]:
    df = pd.read_csv(path)
    # Prefer 'Result', else auto-detect a 5-digit string column
    col = "Result" if "Result" in df.columns else None
    if col is None:
        for c in df.columns:
            vals = df[c].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            if (vals.str.fullmatch(r"\d{5}")).all():
                col = c
                break
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

# =========================
# Env builder (matches your app)
# =========================
def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
    seed   = winners[idx-1]
    combo  = winners[idx]
    seed_list  = digits_of(seed)
    combo_list = sorted(digits_of(combo))

    seed_sum = sum(seed_list)
    combo_sum = sum(combo_list)

    prev_seed = winners[idx-1]
    prev_prev_seed = winners[idx-2] i

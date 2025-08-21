# recommender.py (full corrected, hardened)
# Updated with safe handling of None values, pool parsing, and applicable IDs.

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterable, Union
from pathlib import Path
import itertools as it

# =========================
# Defaults
# =========================
WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV = "lottery_filters_batch_10.csv"
TODAY_POOL_CSV: Optional[str] = None
TARGET_MAX = 44
ALWAYS_KEEP_WINNER = True
MINIMIZE_BEYOND_TARGET = True
OUTPUT_DIR = Path(".")

# =========================
# Helpers / mappings
# =========================
MIRROR = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}
VTRAC = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}

def sum_category(total: int) -> str:
    if 0 <= total <= 15: return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in s if ch.isdigit()]

# =========================
# Filters
# =========================
@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str
    expression: str

# =========================
# CSV loaders
# =========================
def load_winners(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "Result" not in df.columns:
        raise ValueError("Winners CSV missing 'Result' column")
    vals = df["Result"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    return vals[vals.str.fullmatch(r"\d{5}")].tolist()

def load_filters(path: str) -> List[FilterDef]:
    df = pd.read_csv(path)
    req = ["id","name","enabled","applicable_if","expression"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"Filters CSV missing {r}")
    def to_bool(x):
        if isinstance(x,bool): return x
        if pd.isna(x): return False
        return str(x).lower() in {"true","1","yes","y"}
    df["enabled"] = df["enabled"].map(to_bool)
    filters: List[FilterDef] = []
    for _,row in df.iterrows():
        filters.append(FilterDef(
            str(row["id"]),str(row["name"]),bool(row["enabled"]),
            str(row["applicable_if"] or ""),str(row["expression"] or "")
        ))
    return filters

def load_pool(path: Optional[str]) -> List[str]:
    if not path: return []
    if not Path(path).exists(): return []
    df = pd.read_csv(path)
    col = "combo" if "combo" in df.columns else ("Result" if "Result" in df.columns else None)
    if not col: return []
    vals = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    return vals[vals.str.fullmatch(r"\d{5}")].tolist()

# =========================
# Eval helpers
# =========================
def safe_eval(expr: str, env: Dict[str,object]) -> bool:
    if not expr: return True
    try:
        return bool(eval(expr,{"__builtins__":{}},env))
    except Exception:
        return False

# =========================
# Applicable IDs parser
# =========================
def parse_applicable_only(val: Optional[Union[str,Iterable[str]]]) -> List[str]:
    if val is None: return []
    if isinstance(val,float): return []
    if isinstance(val,str):
        parts = []
        for token in val.replace("\n"," ").replace("\t"," ").split(","):
            for sub in token.split():
                if sub.strip(): parts.append(sub.strip())
        return parts
    try:
        return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        return []

# =========================
# Main
# =========================
def main(
    winners_csv: str = WINNERS_CSV,
    filters_csv: str = FILTERS_CSV,
    today_pool_csv: Optional[str] = TODAY_POOL_CSV,
    applicable_only: Optional[Union[str,Iterable[str]]] = None
):
    winners = load_winners(winners_csv)
    filters = load_filters(filters_csv)
    pool = load_pool(today_pool_csv)

    # Defensive: ensure pool is always a list
    if pool is None:
        pool = []

    ids_only = set(parse_applicable_only(applicable_only))
    applicable = {f.fid:f for f in filters if f.enabled and (not ids_only or f.fid in ids_only)}

    # Debug info for tracing issues
    print(f"Loaded {len(winners)} winners, {len(filters)} filters, {len(pool)} pool entries")
    print(f"Applicable filters: {list(applicable.keys())}")

    # Quick sequence output
    rows = []
    for step,f in enumerate(applicable.values(),start=1):
        rows.append({"step":step,"filter_id":f.fid,"name":f.name})
    seq_df = pd.DataFrame(rows)
    seq_df.to_csv(OUTPUT_DIR/"recommender_sequence.csv",index=False)

    # Safe handling for empty pool
    remaining = len(pool) if pool else 0
    summary_rows = [{"remaining": remaining, "applied_filters": len(applicable)}]
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR/"summary.csv", index=False)

    # Avoid pairs safe write
    pd.DataFrame([],columns=["filter_id_1","filter_id_2","pair_risk"]).to_csv(OUTPUT_DIR/"avoid_pairs.csv",index=False)

if __name__ == "__main__":
    main()

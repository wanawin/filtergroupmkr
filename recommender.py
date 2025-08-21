# recommender.py
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import itertools as it
import html

# =========================
# Defaults (UI can override)
# =========================
WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV = "lottery_filters_batch_10.csv"
TODAY_POOL_CSV: Optional[str] = "today_pool.csv"
TARGET_MAX = 44
ALWAYS_KEEP_WINNER = True
MINIMIZE_BEYOND_TARGET = True
OUTPUT_DIR = Path(".")

# =========================
# Helpers / domain mapping
# =========================
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
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
# Env builder
# =========================
def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
    seed   = winners[idx-1]
    combo  = winners[idx]
    seed_list  = digits_of(seed)
    combo_list = sorted(digits_of(combo))
    seed_sum = sum(seed_list)
    combo_sum = sum(combo_list)
    prev_seed = winners[idx-1]
    prev_prev_seed = winners[idx-2] if idx >= 2 else None
    prev_seed_digits = digits_of(prev_seed)
    prev_prev_seed_digits = digits_of(prev_prev_seed) if prev_prev_seed is not None else []
    history_digits = [digits_of(s) for s in winners[:idx]]
    hot, cold, due = hot_cold_due(history_digits, k_hotcold=10)
    def parity_label(digs): return "Even" if sum(digs) % 2 == 0 else "Odd"
    prev_pattern = []
    for digs in (prev_prev_seed_digits, prev_seed_digits, seed_list):
        prev_pattern.extend([sum_category(sum(digs)), parity_label(digs)])
    prev_pattern = tuple(prev_pattern)
    env = {
        "combo": combo,
        "combo_digits": set(combo_list),
        "combo_digits_list": combo_list,
        "combo_sum": combo_sum,
        "combo_sum_cat": sum_category(combo_sum),
        "combo_sum_category": sum_category(combo_sum),
        "seed": seed,
        "seed_digits": set(seed_list),
        "seed_digits_list": seed_list,
        "seed_sum": seed_sum,
        "prev_sum_cat": sum_category(seed_sum),
        "seed_sum_category": sum_category(seed_sum),
        "spread_seed": max(seed_list) - min(seed_list),
        "spread_combo": max(combo_list) - min(combo_list),
        "seed_vtracs": set(VTRAC[d] for d in seed_list),
        "combo_vtracs": set(VTRAC[d] for d in combo_list),
        "last2": set(seed_list) | set(prev_seed_digits),
        "prev_seed": prev_seed,
        "prev_seed_digits": prev_seed_digits,
        "prev_prev_seed": prev_prev_seed,
        "prev_prev_seed_digits": prev_prev_seed_digits,
        "prev_pattern": prev_pattern,
        "new_seed_digits": set(seed_list) - set(prev_seed_digits),
        "seed_counts": Counter(seed_list),
        "common_to_both": set(seed_list) & set(prev_seed_digits),
        "hot_digits": sorted(hot),
        "cold_digits": sorted(cold),
        "due_digits": sorted(due),
        "mirror": MIRROR,
        "vtrac": VTRAC,
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted, "Counter": Counter,
    }
    return env

def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    if not expr:
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

# =========================
# Main
# =========================
def main(
    winners_csv: Optional[str] = None,
    filters_csv: Optional[str] = None,
    today_pool_csv: Optional[str] = TODAY_POOL_CSV,
    target_max: int = TARGET_MAX,
    always_keep_winner: bool = ALWAYS_KEEP_WINNER,
    minimize_beyond_target: bool = MINIMIZE_BEYOND_TARGET,
    force_keep_combo: Optional[str] = None,
):
    winners = load_winners(winners_csv or WINNERS_CSV)
    filters = load_filters(filters_csv or FILTERS_CSV)
    idx_now = len(winners) - 1
    env_now = build_env_for_draw(idx_now, winners)
    applicable = {f.fid: f for f in filters if f.enabled and safe_eval(f.applicable_if, env_now)}

    # dummy support + risk
    support = {fid: 5 for fid in applicable}
    single_fail = {fid: 0.0 for fid in applicable}

    # ---- Load pool (if provided) and estimate each filter's eliminations for tie-breaking
    def apply_filter_to_pool(f: FilterDef, base_env: Dict[str,object], pool: List[str]) -> Tuple[List[str], int]:
        keep, elim = [], 0
        for s in pool:
            clist = sorted(digits_of(s))
            env = dict(base_env)
            env.update({
                "combo": s,
                "combo_digits": set(clist),
                "combo_digits_list": clist,
                "combo_sum": sum(clist),
                "combo_sum_cat": sum_category(sum(clist)),
                "combo_sum_category": sum_category(sum(clist)),
                "spread_combo": max(clist) - min(clist),
                "combo_vtracs": set(VTRAC[d] for d in clist),
            })
            if safe_eval(f.applicable_if, env) and safe_eval(f.expression, env):
                elim += 1
            else:
                keep.append(s)
        return keep, elim

    pool = pd.read_csv(today_pool_csv)["combo"].astype(str).str.zfill(5).tolist() if today_pool_csv and Path(today_pool_csv).exists() else None
    remaining = len(pool) if pool else None

    est_elims = {}
    if pool is not None:
        for f in applicable.values():
            _, elim = apply_filter_to_pool(f, env_now, pool)
            est_elims[f.fid] = elim
    else:
        est_elims = {fid: 0 for fid in applicable}

    # ✅ Support-first ranking with elim tie-breaker
    ranked = sorted(
        applicable.values(),
        key=lambda f: (
            -support.get(f.fid, 0),
            single_fail.get(f.fid, 1.0),
            -est_elims.get(f.fid, 0),
            f.fid
        )
    )

    # Output test
    for f in ranked:
        print(f.fid, f.name, "est_elim=", est_elims.get(f.fid,0))

    return ranked


__all__ = [
    "WINNERS_CSV", "FILTERS_CSV", "TODAY_POOL_CSV", "OUTPUT_DIR",
    "TARGET_MAX", "ALWAYS_KEEP_WINNER", "MINIMIZE_BEYOND_TARGET",
    "main"
]

if __name__ == "__main__":
    main()

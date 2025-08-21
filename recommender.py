# recommender.py
# Skeleton engine for the DCS Recommender — winner-preserving
# ---------------------------------------------------------------------
# This file exposes a single entry point:
#     main(winners_csv, filters_csv, ..., applicable_only=[...])
# which your Streamlit app imports as:  from recommender import main
#
# The skeleton includes:
# - CSV loaders (winners, filters, optional pool)
# - Environment builder for a draw (seed/prevs, parity, vtrac, hot/cold/due)
# - Safe expression evaluator for filter rules (stubs, ready to wire)
# - Optional use of profiler tables if present (case_history.csv / case_filter_stats.csv)
# - A minimal, winner-preserving reduction loop
# - Result artifacts saved in repo root (recommender_sequence.csv, do_not_apply.csv)

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import itertools as it
import pandas as pd
import html

# =========================
# Defaults / constants
# =========================
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
VTRAC  = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}

OUTPUT_DIR = Path(".")

# Filenames (optional profiler artifacts)
CASE_HISTORY = OUTPUT_DIR / "case_history.csv"
CASE_STATS   = OUTPUT_DIR / "case_filter_stats.csv"

# =========================
# Small helpers
# =========================
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
    """Very lightweight hot/cold/due, over last k draws of history."""
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
# Filter definition + loaders
# =========================
@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str  # python expression (safe_eval)
    expression: str     # python expression (safe_eval)

def _to_bool(x) -> bool:
    if isinstance(x, bool): return x
    if pd.isna(x): return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def load_winners(path: str) -> List[str]:
    df = pd.read_csv(path)
    # Prefer a 'Result' column; else auto-detect a 5-digit column
    if "Result" in df.columns:
        col = "Result"
    else:
        col = None
        for c in df.columns:
            vals = df[c].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            if (vals.str.fullmatch(r"\d{5}")).all():
                col = c; break
    if col is None:
        raise ValueError("Winners CSV must contain a 5-digit column (preferably 'Result').")
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
    df["enabled"] = df["enabled"].map(_to_bool)

    # dequote any outer quotes
    for col in ["applicable_if","expression"]:
        s = df[col].astype(str).str.strip()
        s = s.str.replace('"""','"', regex=False).str.replace("'''","'", regex=False)
        df[col] = s.apply(lambda t: t[1:-1] if len(t)>=2 and t[0]==t[-1] and t[0] in {'"', "'"} else t)

    out: List[FilterDef] = []
    for _, r in df.iterrows():
        out.append(FilterDef(
            fid=str(r["id"]).strip(),
            name=str(r["name"]).strip(),
            enabled=bool(r["enabled"]),
            applicable_if=str(r["applicable_if"]).strip(),
            expression=str(r["expression"]).strip(),
        ))
    return out

def load_pool(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = "combo" if "combo" in df.columns else ("Result" if "Result" in df.columns else None)
    if not col:
        raise ValueError("Pool file must have 'combo' or 'Result' column.")
    vals = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    vals = vals[vals.str.fullmatch(r"\d{5}")]
    return vals.tolist()

# =========================
# Env + safe_eval
# =========================
def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
    """Build the same env your filters expect. idx is the index of 'today' seed (>=1)."""
    seed   = winners[idx-1]
    combo  = winners[idx] if idx < len(winners) else "00000"  # dummy 'today' combo
    seed_list  = digits_of(seed)
    combo_list = sorted(digits_of(combo))

    prev_seed = winners[idx-1]
    prev_prev_seed = winners[idx-2] if idx >= 2 else None
    prev_seed_digits = digits_of(prev_seed)
    prev_prev_seed_digits = digits_of(prev_prev_seed) if prev_prev_seed is not None else []

    # history up to seed
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
        "combo_sum": sum(combo_list),
        "combo_sum_cat": sum_category(sum(combo_list)),
        "combo_sum_category": sum_category(sum(combo_list)),

        "seed": seed,
        "seed_digits": set(seed_list),
        "seed_digits_list": seed_list,
        "seed_sum": sum(seed_list),
        "prev_sum_cat": sum_category(sum(seed_list)),
        "seed_sum_category": sum_category(sum(seed_list)),

        "spread_seed": max(seed_list) - min(seed_list),
        "spread_combo": max(combo_list) - min(combo_list),

        "seed_vtracs": set(VTRAC[d] for d in seed_list),
        "combo_vtracs": set(VTRAC[d] for d in combo_list),

        "last2": set(seed_list) | set(prev_seed_digits),

        "prev_seed": prev_seed,
        "prev_seed_digits": prev_seed_digits,
        "prev_seed_digits_list": prev_seed_digits,
        "prev_prev_seed": prev_prev_seed,
        "prev_prev_seed_digits": prev_prev_seed_digits,
        "prev_prev_seed_digits_list": prev_prev_seed_digits,
        "prev_pattern": prev_pattern,

        "new_seed_digits": set(seed_list) - set(prev_seed_digits),
        "seed_counts": Counter(seed_list),
        "common_to_both": set(seed_list) & set(prev_seed_digits),

        "hot_digits": sorted(hot),
        "cold_digits": sorted(cold),
        "due_digits": sorted(due),
        "due_digits_2": due,

        "mirror": MIRROR,
        "vtrac": VTRAC,

        # safe builtins
        "any": any, "all": all, "len": len, "sum": sum,
        "max": max, "min": min, "set": set, "sorted": sorted, "Counter": Counter,
    }
    return env

def current_bucket(env: Dict[str, object]) -> Tuple[str,str,str,str]:
    seed_sum_cat = env["seed_sum_category"]
    seed_list = env["seed_digits_list"]
    struct = classify_structure(seed_list)
    sb = spread_band(env["spread_seed"])
    parity_major = "even>=3" if sum(1 for d in seed_list if d % 2 == 0) >= 3 else "even<=2"
    return seed_sum_cat, struct, sb, parity_major

def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    """Evaluate a boolean python expression in a constrained env.
       Any exception = False (acts as 'not eliminating')."""
    if not expr:
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

# =========================
# Profiler hooks (optional)
# =========================
def _load_case_stats() -> Optional[pd.DataFrame]:
    if CASE_STATS.exists():
        try:
            return pd.read_csv(CASE_STATS)
        except Exception:
            return None
    return None

def _risk_for_filter_ids(applicable_ids: Iterable[str]) -> Dict[str, float]:
    """Return estimated single-filter failure risk (0..1) for the given IDs using
       the optional profiler table. If missing, default to 0.0."""
    stats = _load_case_stats()
    risks: Dict[str, float] = {}
    if stats is None or stats.empty:
        for fid in applicable_ids:
            risks[fid] = 0.0
        return risks

    # Expected columns in case_filter_stats.csv (adjust if your profiler writes differently):
    #   filter_id, bucket_key, applicable_n, eliminated_winner_n, failure_rate
    # We’ll just take the global failure_rate per filter_id (min risk if multiple rows).
    g = stats.groupby("filter_id", as_index=False)["failure_rate"].mean()
    m = {row["filter_id"]: float(row["failure_rate"]) for _, row in g.iterrows()}
    for fid in applicable_ids:
        risks[fid] = float(m.get(fid, 0.0))
    return risks

# =========================
# Core application logic (minimal, winner-preserving)
# =========================
def _apply_filter_to_pool(f: FilterDef, base_env: Dict[str,object], pool: List[str]) -> Tuple[List[str], int]:
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

# =========================
# Public entry
# =========================
def main(
    winners_csv: str,
    filters_csv: str,
    today_pool_csv: Optional[str] = None,
    target_max: int = 44,
    always_keep_winner: bool = True,
    minimize_beyond_target: bool = True,
    force_keep_combo: Optional[str] = None,
    override_seed: Optional[str] = None,
    override_prev: Optional[str] = None,
    override_prevprev: Optional[str] = None,
    applicable_only: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """
    Minimal winner-preserving recommender loop.
    Returns a list of dict rows suitable for display as a table.
    Also writes 'recommender_sequence.csv' and 'do_not_apply.csv' if possible.
    """

    # ------- Load inputs -------
    if not Path(winners_csv).exists():
        raise FileNotFoundError(f"Missing winners CSV: {winners_csv}")
    if not Path(filters_csv).exists():
        raise FileNotFoundError(f"Missing filters CSV: {filters_csv}")

    winners = load_winners(winners_csv)
    filters = [f for f in load_filters(filters_csv) if f.enabled]

    # Optional: override seed & prevs to match UI inputs
    if override_seed:
        seed = str(override_seed).strip().zfill(5)
        prev = (override_prev or winners[-1]).strip().zfill(5)
        prevprev = (override_prevprev or winners[-2]).strip().zfill(5)
        winners = winners[:-2] + [prevprev, prev, seed, "00000"]  # last is dummy "today"

    if len(winners) < 2:
        raise ValueError("Need at least 2 draws in winners CSV.")

    idx_now = len(winners) - 1
    env_now = build_env_for_draw(idx_now, winners)

    # Determine today's applicable filters (by 'applicable_if' only)
    applicable = {}
    for f in filters:
        if safe_eval(f.applicable_if, env_now):
            applicable[f.fid] = f

    # Restrict to explicit list if provided
    if applicable_only:
        wanted = {fid.strip() for fid in applicable_only}
        applicable = {fid: f for fid, f in applicable.items() if fid in wanted}

    # Pull estimated risks from profiler (if available); default to 0
    risk_map = _risk_for_filter_ids(applicable.keys())

    # Rank by (lower risk first), then by id for determinism
    ranked = sorted(applicable.values(), key=lambda f: (risk_map.get(f.fid, 0.0), f.fid))

    # ------- Winner-preserving reduction -------
    pool = load_pool(today_pool_csv) if (today_pool_csv and Path(today_pool_csv).exists()) else None
    seq_rows: List[Dict[str, object]] = []
    remaining = len(pool) if pool is not None else None

    # pick a "keep" combo: explicit override else last real winner (if pool provided)
    winner_today = None
    if force_keep_combo:
        winner_today = str(force_keep_combo).strip().replace(" ", "")
    elif pool is not None:
        # if we overrode seed, winners[idx_now] is dummy; keep the provided keep_combo only
        last_real = winners[idx_now] if winners[idx_now].isdigit() else None
        winner_today = last_real

    for step, f in enumerate(ranked, start=1):
        est_risk = float(risk_map.get(f.fid, 0.0))

        if pool is not None:
            new_pool, elim = _apply_filter_to_pool(f, env_now, pool)
            # winner-preserving
            if always_keep_winner and winner_today and (winner_today not in new_pool):
                seq_rows.append({
                    "step": step, "filter_id": f.fid, "name": f.name,
                    "est_risk": round(est_risk, 4),
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "would_remove_winner"
                })
                continue
            if elim == 0:
                seq_rows.append({
                    "step": step, "filter_id": f.fid, "name": f.name,
                    "est_risk": round(est_risk, 4),
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "no_reduction"
                })
                continue
            # accept
            pool = new_pool
            remaining = len(pool)
            seq_rows.append({
                "step": step, "filter_id": f.fid, "name": f.name,
                "est_risk": round(est_risk, 4),
                "eliminated_now": elim, "remaining": remaining,
                "skipped_reason": None
            })
            if (remaining <= target_max) and (not minimize_beyond_target):
                break
        else:
            # record ordering only
            seq_rows.append({
                "step": step, "filter_id": f.fid, "name": f.name,
                "est_risk": round(est_risk, 4),
                "eliminated_now": None, "remaining": None,
                "skipped_reason": None
            })

    # ------- Artifacts -------
    try:
        pd.DataFrame(seq_rows).to_csv(OUTPUT_DIR / "recommender_sequence.csv", index=False)
    except Exception:
        pass

    # Very simple DO_NOT_APPLY list using a fixed threshold; replace with profiler tiering if desired
    try:
        if applicable:
            rows = []
            for fid, f in applicable.items():
                r = float(risk_map.get(fid, 0.0))
                tier = "DO_NOT_APPLY" if r >= 0.20 else ("APPLY_LATE" if r >= 0.10 else "SAFE")
                rows.append({"filter_id": fid, "name": f.name, "risk": round(r,4), "tier": tier})
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "do_not_apply.csv", index=False)
    except Exception:
        pass

    return seq_rows


# Allow running from CLI for quick smoke tests
if __name__ == "__main__":
    # Adjust file names here if your CSVs differ
    print("Running recommender skeleton...")
    rows = main(
        winners_csv="DC5_Midday_Full_Cleaned_Expanded.csv",
        filters_csv="lottery_filters_batch_10.csv",
        today_pool_csv="today_pool.csv",   # optional; comment out if not present
        target_max=44,
        always_keep_winner=True,
        minimize_beyond_target=True,
        force_keep_combo=None,
        override_seed=None,
        override_prev=None,
        override_prevprev=None,
        applicable_only=None,
    )
    print(f"Produced {len(rows)} rows; see recommender_sequence.csv if written.")

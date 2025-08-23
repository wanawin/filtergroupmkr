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
NOPOOL_MIN_DAYS      = 60
NOPOOL_MIN_KEEP      = 75.0
NOPOOL_PARITY_SKEW   = 10.0
NOPOOL_MAX_ROWS      = 20

# =============================================================================
# Pool archive helpers
# =============================================================================
def get_pool_for_seed(seed_row: pd.Series, *, keep_permutations: bool = True) -> pd.DataFrame:
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

    df["combo"] = df["combo"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
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
def _parity_major_label(digs: List[int]) -> str:
    return "even>=3" if sum(1 for d in digs if d % 2 == 0) >= 3 else "even<=2"

def classify_structure(digs: List[int]) -> str:
    c = Counter(digs); counts = sorted(c.values(), reverse=True)
    if counts == [5]:       return "quint"
    if counts == [4,1]:     return "quad"
    if counts == [3,2]:     return "triple_double"
    if counts == [3,1,1]:   return "triple"
    if counts == [2,2,1]:   return "double_double"
    if counts == [2,1,1,1]: return "double"
    return "single"

def combo_affinity(env_now: Dict[str, object], combo: str) -> float:
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
    if hi8_seed >= 3:
        hi8_score = math.exp(-max(0, hi8_combo - 1))
    else:
        hi8_score = math.exp(-abs(hi8_combo - hi8_seed) / 1.5)
    overlap = len(set(seedd) & set(cd))
    overlap_eq1 = 1.0 if overlap == 1 else 0.0
    W_SUM, W_SPD, W_STR, W_PAR, W_HI8, W_OV1 = 0.8, 0.3, 0.4, 0.15, 0.35, 0.25
    return (
        W_SUM * sum_prox + W_SPD * spread_prox + W_STR * struct_match +
        W_PAR * parity_match + W_HI8 * hi8_score + W_OV1 * overlap_eq1
    )

# =============================================================================
# Risk (history-only) + helpers
# =============================================================================
def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    if not expr:
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def historical_risk_for_applicable(
    filters: List[FilterDef],
    winners: List[str],
    idx_now: int,
    max_draws: Optional[int] = None,
    max_bucket_matches: Optional[int] = None,
    decay_half_life: Optional[int] = None,
):
    env_now = build_env_for_draw(idx_now, winners)
    start_idx = max(1, (idx_now - (max_draws or (idx_now - 1))))
    candidate_indices = list(range(start_idx, idx_now))

    def w(i: int) -> float:
        if not decay_half_life:
            return 1.0
        age = (idx_now - i)
        return 0.5 ** (age / float(decay_half_life))

    app_w = Counter()
    blk_w = Counter()
    coapp = Counter()
    both_block = Counter()

    for i in candidate_indices:
        env_i = build_env_for_draw(i, winners)
        applicable_ids = []
        blocked_ids = []
        wi = w(i)
        for f in filters:
            if not f.enabled:
                continue
            if safe_eval(f.applicable_if, env_i):
                applicable_ids.append(f.fid)
                app_w[f.fid] += wi
                if safe_eval(f.expression, env_i):
                    blk_w[f.fid] += wi
                    blocked_ids.append(f.fid)
        for a,b in it.combinations(sorted(applicable_ids), 2):
            coapp[(a,b)] += 1
        for a,b in it.combinations(sorted(blocked_ids), 2):
            both_block[(a,b)] += 1

    single_failure_rate = {fid: (blk_w[fid]/app_w[fid]) if app_w[fid] > 0 else 0.0 for fid in app_w}
    pair_risk = {}
    for (a,b), n in coapp.items():
        bb = both_block.get((a,b), 0)
        pair_risk[(a,b)] = bb / n if n > 0 else 0.0
    return single_failure_rate, pair_risk, candidate_indices

def today_applicable_filters(filters: List[FilterDef], winners: List[str]):
    if len(winners) < 2:
        raise SystemExit("Need at least 2 winners.")
    idx_now = len(winners) - 1
    env_now = build_env_for_draw(idx_now, winners)
    app: Dict[str, FilterDef] = {}
    for f in filters:
        if f.enabled and safe_eval(f.applicable_if, env_now):
            app[f.fid] = f
    return idx_now, app, env_now

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

# =============================================================================
# NoPool (historical signals without pool) — TODAY'S applicable pairs only
# =============================================================================
def nopool_today_pairs(
    winners: List[str],
    filters_all: Dict[str, FilterDef],
    idx_now: int,
    *, min_days: int = NOPOOL_MIN_DAYS,
    min_keep: float = NOPOOL_MIN_KEEP,
    parity_skew: float = NOPOOL_PARITY_SKEW,
    max_rows: int = NOPOOL_MAX_ROWS,
) -> pd.DataFrame:
    N = idx_now
    if N < 2:
        return pd.DataFrame()
    envs = [build_env_for_draw(i, winners) for i in range(1, idx_now+1)]
    env_now = envs[-1]
    todays_ids = [fid for fid, f in filters_all.items() if safe_eval(f.applicable_if, env_now)]
    if len(todays_ids) < 2:
        return pd.DataFrame()

    f_slice = {fid: filters_all[fid] for fid in todays_ids}
    comp_app = {}
    comp_blk = {}
    for fid, f in f_slice.items():
        try:  comp_app[fid] = compile(f.applicable_if if f.applicable_if else "True", "<app_if>", "eval")
        except Exception: comp_app[fid] = compile("False", "<app_if>", "eval")
        try:  comp_blk[fid] = compile(f.expression if f.expression else "False", "<expr>", "eval")
        except Exception: comp_blk[fid] = compile("False", "<expr>", "eval")

    F = len(f_slice)
    idx_map = {fid:i for i,fid in enumerate(todays_ids)}
    app = np.zeros((N, F), dtype=bool)
    blk = np.zeros((N, F), dtype=bool)
    even_arr = np.zeros(N, dtype=int)

    for i, env in enumerate(envs):
        even_arr[i] = 1 if (sum(env["combo_digits_list"]) % 2 == 0) else 0
        for fid in todays_ids:
            j = idx_map[fid]
            try:
                a = bool(eval(comp_app[fid], {"__builtins__": {}}, env))
            except Exception:
                a = False
            app[i, j] = a
            if a:
                try:
                    b = bool(eval(comp_blk[fid], {"__builtins__": {}}, env))
                except Exception:
                    b = False
                blk[i, j] = b

    recs = []
    for a_idx in range(F):
        for b_idx in range(a_idx+1, F):
            mask = app[:, a_idx] & app[:, b_idx]
            n = int(mask.sum())
            if n == 0:
                continue
            either_blocks = int(((blk[:, a_idx] | blk[:, b_idx]) & mask).sum())
            both_blocks  = int(((blk[:, a_idx] & blk[:, b_idx]) & mask).sum())
            kept_rate = (1 - (either_blocks / n)) * 100.0
            even_rate = float(even_arr[mask].mean() * 100.0)
            parity_note = ""
            if abs(even_rate - 50.0) >= float(parity_skew):
                tip = "EVEN" if even_rate >= 50.0 else "ODD"
                parity_note = f"{tip} skew: {even_rate:.1f}%"
            recs.append({
                "Filter A ID": todays_ids[a_idx],
                "Filter A Name": f_slice[todays_ids[a_idx]].name,
                "Filter B ID": todays_ids[b_idx],
                "Filter B Name": f_slice[todays_ids[b_idx]].name,
                "Days both applicable (history)": n,
                "Winner kept % (both applicable)": round(kept_rate, 3),
                "Either blocks winner %": round((either_blocks/n)*100.0, 3),
                "Both block winner %": round((both_blocks/n)*100.0, 3),
                "Parity note (only if meaningful)": parity_note,
            })
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df.sort_values(
        ["Winner kept % (both applicable)", "Days both applicable (history)"],
        ascending=[False, False]
    )
    df_viable = df[(df["Days both applicable (history)"] >= int(min_days)) &
                   (df["Winner kept % (both applicable)"] >= float(min_keep))].copy()
    df.to_csv(OUTPUT_DIR / "NoPool_today_pairs_ALL.csv", index=False)
    df_viable.head(max_rows).to_csv(OUTPUT_DIR / "NoPool_today_pairs_TOP.csv", index=False)
    return df_viable.head(max_rows)

# =============================================================================
# Column detector
# =============================================================================
def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =============================================================================
# Final Ordered SafeList (respect sequence)
# =============================================================================
def build_final_ordered_safelist(
    safe_source_path: str | Path = "do_not_apply.csv",
    avoid_pairs_path: str | Path = "avoid_pairs.csv",
    seq_path: str | Path = "recommender_sequence.csv",
    out_basename: str = "final_safe_ordered",
) -> Optional[Dict[str, object]]:
    safe_source_path  = Path(safe_source_path)
    avoid_pairs_path  = Path(avoid_pairs_path)
    seq_path          = Path(seq_path)

    if not safe_source_path.exists() or not avoid_pairs_path.exists():
        return None

    safe_df = pd.read_csv(safe_source_path)
    tier_col = _detect_col(safe_df, ["tier", "Tier", "TIER"])
    fid_col  = _detect_col(safe_df, ["filter_id", "Filter ID", "FILTER_ID", "fid", "FID", "id"])
    if not tier_col or not fid_col:
        return None

    safe_df[fid_col] = safe_df[fid_col].astype(str).str.strip()
    safe_only = safe_df[safe_df[tier_col].astype(str).str.upper() == "SAFE"].copy()
    safe_ids = set(safe_only[fid_col])

    metrics = (safe_only.rename(columns={fid_col: "filter_id"})
               .drop_duplicates("filter_id")
               .set_index("filter_id"))

    ap_df = pd.read_csv(avoid_pairs_path)
    fid1 = _detect_col(ap_df, ["filter_id_1", "FID1", "fid1"])
    fid2 = _detect_col(ap_df, ["filter_id_2", "FID2", "fid2"])
    if not fid1 or not fid2:
        return None
    ap_ids = set(ap_df[fid1].astype(str).str.strip()) | set(ap_df[fid2].astype(str).str.strip())
    revised_safe_ids = [fid for fid in safe_ids if fid not in ap_ids]

    in_seq_order, not_in_seq = [], []
    elim_map: Dict[str, float] = {}

    if seq_path.exists():
        seq_df = pd.read_csv(seq_path)
        seq_fid = _detect_col(seq_df, ["filter_id", "Filter ID", "FILTER_ID", "fid", "FID", "id"])
        if seq_fid:
            seq_df[seq_fid] = seq_df[seq_fid].astype(str).str.strip()
            seq_order = seq_df[seq_fid].tolist()
            elim_col = _detect_col(seq_df, ["eliminated_now", "eliminations", "elim_count", "elimination_count", "elims"])
            if elim_col:
                try:
                    elim_map = dict(zip(seq_df[seq_fid], pd.to_numeric(seq_df[elim_col], errors="coerce").fillna(0)))
                except Exception:
                    elim_map = {}
            in_seq_order = [fid for fid in seq_order if fid in revised_safe_ids]
            remaining = [fid for fid in revised_safe_ids if fid not in set(in_seq_order)]
        else:
            remaining = list(revised_safe_ids)
    else:
        remaining = list(revised_safe_ids)

    def aggression(fid: str) -> float:
        if fid in elim_map:
            try: return float(elim_map[fid])
            except Exception: pass
        if "support" in metrics.columns and fid in metrics.index:
            try: return float(metrics.loc[fid, "support"])
            except Exception: pass
        if "risk" in metrics.columns and fid in metrics.index:
            try: return float(metrics.loc[fid, "risk"])
            except Exception: pass
        return 0.0

    not_in_seq = sorted(remaining, key=aggression, reverse=True)
    final_ids = in_seq_order + not_in_seq

    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{out_basename}_{ts}"
    out_csv = OUTPUT_DIR / f"{base}.csv"
    out_txt = OUTPUT_DIR / f"{base}.txt"
    out_html = OUTPUT_DIR / f"{base}.html"

    final_df = pd.DataFrame({"filter_id": final_ids})
    if not metrics.empty:
        final_df = final_df.merge(metrics.reset_index(), on="filter_id", how="left")

    final_df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        for fid in final_ids:
            f.write(f"{fid}\n")
    final_df.to_html(out_html, index=False, escape=False)

    return {
        "counts": {
            "safe_only": len(safe_ids),
            "revised_safe": len(revised_safe_ids),
            "in_sequence": len(in_seq_order),
            "not_in_sequence": len(not_in_seq),
            "final_total": len(final_ids),
        },
        "outputs": {"csv": str(out_csv), "txt": str(out_txt), "html": str(out_html), "sequence_used": seq_path.exists()},
    }

# =============================================================================
# Safest-first SafeList (ignore sequence)
# =============================================================================
def build_final_safest_first_safelist(
    safe_source_path: str | Path = "do_not_apply.csv",
    avoid_pairs_path: str | Path = "avoid_pairs.csv",
    seq_path: str | Path = "recommender_sequence.csv",
    out_basename: str = "final_safe_safest_first",
) -> Optional[Dict[str, object]]:
    safe_source_path  = Path(safe_source_path)
    avoid_pairs_path  = Path(avoid_pairs_path)
    seq_path          = Path(seq_path)

    if not safe_source_path.exists() or not avoid_pairs_path.exists():
        return None

    safe_df = pd.read_csv(safe_source_path)
    tier_col = _detect_col(safe_df, ["tier", "Tier", "TIER"])
    fid_col  = _detect_col(safe_df, ["filter_id", "Filter ID", "FILTER_ID", "fid", "FID", "id"])
    if not tier_col or not fid_col:
        return None

    safe_df[fid_col] = safe_df[fid_col].astype(str).str.strip()
    safe_only = safe_df[safe_df[tier_col].astype(str).str.upper() == "SAFE"].copy()
    safe_ids = set(safe_only[fid_col])

    metrics = (safe_only.rename(columns={fid_col: "filter_id"})
               .drop_duplicates("filter_id")
               .set_index("filter_id"))

    ap_df = pd.read_csv(avoid_pairs_path)
    fid1 = _detect_col(ap_df, ["filter_id_1", "FID1", "fid1"])
    fid2 = _detect_col(ap_df, ["filter_id_2", "FID2", "fid2"])
    if not fid1 or not fid2:
        return None
    ap_ids = set(ap_df[fid1].astype(str).str.strip()) | set(ap_df[fid2].astype(str).str.strip())
    revised_safe_ids = [fid for fid in safe_ids if fid not in ap_ids]

    elim_map: Dict[str, float] = {}
    if seq_path.exists():
        try:
            seq_df = pd.read_csv(seq_path)
            seq_fid = _detect_col(seq_df, ["filter_id", "Filter ID", "FILTER_ID", "fid", "FID", "id"])
            elim_col = _detect_col(seq_df, ["eliminated_now", "eliminations", "elim_count", "elimination_count", "elims"])
            if seq_fid and elim_col:
                elim_map = dict(zip(
                    seq_df[seq_fid].astype(str).str.strip(),
                    pd.to_numeric(seq_df[elim_col], errors="coerce").fillna(0.0)
                ))
        except Exception:
            elim_map = {}

    def risk_of(fid: str) -> float:
        try: return float(metrics.loc[fid, "risk"])
        except Exception: return 1e9

    def support_of(fid: str) -> float:
        try: return float(metrics.loc[fid, "support"])
        except Exception: return -1e9

    def elims_of(fid: str) -> float:
        try: return float(elim_map.get(fid, 0.0))
        except Exception: return 0.0

    ordered = sorted(revised_safe_ids, key=lambda fid: (risk_of(fid), -support_of(fid), -elims_of(fid)))

    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{out_basename}_{ts}"
    out_csv = OUTPUT_DIR / f"{base}.csv"
    out_txt = OUTPUT_DIR / f"{base}.txt"
    out_html = OUTPUT_DIR / f"{base}.html"

    final_df = pd.DataFrame({"filter_id": ordered})
    if not metrics.empty:
        final_df = final_df.merge(metrics.reset_index(), on="filter_id", how="left")

    final_df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        for fid in ordered:
            f.write(f"{fid}\n")
    final_df.to_html(out_html, index=False, escape=False)

    return {
        "counts": {"safe_only": len(safe_ids), "revised_safe": len(revised_safe_ids), "final_total": len(ordered)},
        "outputs": {"csv": str(out_csv), "txt": str(out_txt), "html": str(out_html)},
        "order_mode": "safest_first",
    }

# =============================================================================
# Grouped-by-aggression SafeList (bins + safest-within-each)
# =============================================================================
def build_final_grouped_safelist(
    safe_source_path: str | Path = "do_not_apply.csv",
    avoid_pairs_path: str | Path = "avoid_pairs.csv",
    seq_path: str | Path = "recommender_sequence.csv",
    out_basename: str = "final_safe_grouped",
) -> Optional[Dict[str, object]]:
    # 1) inputs
    safe_source_path  = Path(safe_source_path)
    avoid_pairs_path  = Path(avoid_pairs_path)
    seq_path          = Path(seq_path)
    if not safe_source_path.exists() or not avoid_pairs_path.exists():
        return None

    safe_df = pd.read_csv(safe_source_path)
    tier_col = _detect_col(safe_df, ["tier", "Tier", "TIER"])
    fid_col  = _detect_col(safe_df, ["filter_id", "Filter ID", "FILTER_ID", "fid", "FID", "id"])
    if not tier_col or not fid_col:
        return None

    safe_df[fid_col] = safe_df[fid_col].astype(str).str.strip()
    safe_only = safe_df[safe_df[tier_col].astype(str).str.upper() == "SAFE"].copy()
    metrics = (safe_only.rename(columns={fid_col: "filter_id"})
               .drop_duplicates("filter_id")
               .set_index("filter_id"))
    safe_ids = set(metrics.index)

    # 2) remove avoid-pair ids
    ap_df = pd.read_csv(avoid_pairs_path)
    fid1 = _detect_col(ap_df, ["filter_id_1", "FID1", "fid1"])
    fid2 = _detect_col(ap_df, ["filter_id_2", "FID2", "fid2"])
    if not fid1 or not fid2:
        return None
    ap_ids = set(ap_df[fid1].astype(str).str.strip()) | set(ap_df[fid2].astype(str).str.strip())
    revised_ids = [fid for fid in safe_ids if fid not in ap_ids]

    # 3) eliminations map from sequence
    elim_map: Dict[str, float] = {}
    if seq_path.exists():
        try:
            seq_df = pd.read_csv(seq_path)
            seq_fid = _detect_col(seq_df, ["filter_id", "Filter ID", "FILTER_ID", "fid", "FID", "id"])
            elim_col = _detect_col(seq_df, ["eliminated_now", "eliminations", "elim_count", "elimination_count", "elims"])
            if seq_fid and elim_col:
                tmp = (seq_df
                       .assign(fid=seq_df[seq_fid].astype(str).str.strip(),
                               elims=pd.to_numeric(seq_df[elim_col], errors="coerce").fillna(0.0))
                       .groupby("fid", as_index=True)["elims"].sum())
                elim_map = tmp.to_dict()
        except Exception:
            elim_map = {}

    def risk_of(fid: str) -> float:
        try: return float(metrics.loc[fid, "risk"])
        except Exception: return 1e9

    def support_of(fid: str) -> float:
        try: return float(metrics.loc[fid, "support"])
        except Exception: return -1e9

    def elims_of(fid: str) -> float:
        return float(elim_map.get(fid, 0.0))

    # 4) bin edges and labels
    def bin_label(e: float) -> str:
        e = float(e)
        if e <= 0: return "0 (no impact)"
        if 1 <= e <= 60:   return "001–060 eliminations"
        if 61 <= e <= 100: return "061–100 eliminations"
        if 101 <= e <= 300:return "101–300 eliminations"
        if 301 <= e <= 500:return "301–500 eliminations"
        if 501 <= e <= 700:return "501–700 eliminations"
        return "701+ eliminations"

    # 5) score and group
    rows = []
    for fid in revised_ids:
        rows.append({
            "filter_id": fid,
            "risk": risk_of(fid),
            "support": support_of(fid),
            "elims": elims_of(fid),
            "group": bin_label(elims_of(fid)),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)

    # within-group: safest (risk asc), support desc, elims desc
    df = df.sort_values(["group","risk","support","elims"],
                        ascending=[True, True, False, False])

    # group order: most aggressive to least
    group_order = [
        "701+ eliminations",
        "501–700 eliminations",
        "301–500 eliminations",
        "101–300 eliminations",
        "061–100 eliminations",
        "001–060 eliminations",
        "0 (no impact)",
    ]
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    df = df.sort_values(["group","risk","support","elims"], ascending=[True, True, False, False])

    # join back extra metrics cols if present
    out_df = df.merge(metrics.reset_index(), on="filter_id", how="left")

    # outputs
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{out_basename}_{ts}"
    out_csv = OUTPUT_DIR / f"{base}.csv"
    out_txt = OUTPUT_DIR / f"{base}.txt"
    out_html = OUTPUT_DIR / f"{base}.html"

    out_df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        current = None
        for _, r in out_df.iterrows():
            if r["group"] != current:
                current = r["group"]
                f.write(f"\n### {current}\n")
            f.write(f"{r['filter_id']}\n")
    # HTML with group headers
    parts = []
    for grp, gdf in out_df.groupby("group", sort=False):
        hdr = f"<h3 style='margin:8px 0'>{html.escape(str(grp))}</h3>"
        parts.append(hdr + gdf.drop(columns=["group"]).to_html(index=False, border=0))
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write("<html><body>" + "\n".join(parts) + "</body></html>")

    return {"outputs": {"csv": str(out_csv), "txt": str(out_txt), "html": str(out_html)}, "groups": out_df["group"].unique().tolist()}

# =============================================================================
# Public entry
# =============================================================================
def main(
    winners_csv: Optional[str] = None,
    filters_csv: Optional[str] = None,
    today_pool_csv: Optional[str] = None,
    target_max: int = TARGET_MAX,
    always_keep_winner: bool = ALWAYS_KEEP_WINNER,
    minimize_beyond_target: bool = MINIMIZE_BEYOND_TARGET,
    force_keep_combo: Optional[str] = None,
    override_seed: Optional[str] = None,
    override_prev: Optional[str] = None,
    override_prevprev: Optional[str] = None,
    max_draws: Optional[int] = None,
    max_bucket_matches: Optional[int] = None,
    decay_half_life: Optional[int] = None,
    applicable_only: Optional[List[str]] = None,
    affinity_exclude_top_pct: Optional[float] = None,
    include_nopool_panel: bool = INCLUDE_NOPOOL_PANEL,
):
    winners_path = winners_csv or WINNERS_CSV
    filters_path = filters_csv or FILTERS_CSV

    present = ", ".join(sorted(p.name for p in Path(".").glob("*.csv")))
    if not Path(winners_path).exists():
        raise FileNotFoundError(f"Missing {winners_path}. CSVs here: {present}")
    if not Path(filters_path).exists():
        raise FileNotFoundError(f"Missing {filters_path}. CSVs here: {present}")

    winners_df = pd.read_csv(winners_path)
    winners = load_winners(winners_path)
    filters_list = load_filters(filters_path)

    # Build map id->filter
    filters_map: Dict[str, FilterDef] = {f.fid: f for f in filters_list if f.enabled}

    idx_now, applicable, env_now = today_applicable_filters(filters_list, winners)

    # If caller wants to limit to a subset of filters
    if applicable_only:
        applicable_only = {fid.strip() for fid in applicable_only}
        applicable = {fid: f for fid, f in applicable.items() if fid in applicable_only}

    # Historical risk for ranking order
    single_fail, pair_risk, idx_used = historical_risk_for_applicable(
        list(applicable.values()), winners, idx_now,
        max_draws=max_draws, max_bucket_matches=max_bucket_matches,
        decay_half_life=decay_half_life
    )
    support = {fid: 0 for fid in applicable}
    for i in idx_used:
        env_i = build_env_for_draw(i, winners)
        for fid, f in applicable.items():
            if safe_eval(f.applicable_if, env_i):
                support[fid] = support.get(fid, 0) + 1
    ranked = sorted(applicable.values(),
                    key=lambda f: (single_fail.get(f.fid, 1.0), -support.get(f.fid,0), f.fid))

    # Load pool (today) or archived
    if today_pool_csv:
        df_pool = pd.read_csv(today_pool_csv)
        if "combo" not in df_pool.columns:
            if "Result" in df_pool.columns: df_pool = df_pool.rename(columns={"Result":"combo"})
            else: raise ValueError("Pool CSV must have 'combo' or 'Result' column.")
        pool = df_pool["combo"].astype(str).str.replace(r"\D","",regex=True).str.zfill(5)
        pool = pool[pool.str.fullmatch(r"\d{5}")].tolist()
    else:
        seed_row = winners_df.iloc[idx_now-1]
        pool_df = get_pool_for_seed(seed_row)
        pool = pool_df["combo"].astype(str).tolist()

    remaining = len(pool)
    winner_today = winners[idx_now] if pool else None

    seq_rows = []

    # ===== Anti-affinity pre-trim =====
    if pool and affinity_exclude_top_pct and 0 < float(affinity_exclude_top_pct) < 1:
        aff = np.array([combo_affinity(env_now, c) for c in pool], dtype=float)
        ranks = pd.Series(aff).rank(method="average", ascending=True)
        thr = 1.0 - float(affinity_exclude_top_pct)
        aff_pct = (ranks - 1) / max(len(pool) - 1, 1)
        try:
            win_ix = pool.index(winner_today)
            win_pct = float(aff_pct.iloc[win_ix])
        except Exception:
            win_pct = float("nan")
        would_remove = (not np.isnan(win_pct)) and (win_pct >= thr)
        if always_keep_winner and would_remove:
            seq_rows.append({
                "step": 0,
                "filter_id": f"AFF_TOP{int(round(affinity_exclude_top_pct*100))}",
                "name": f"Anti-affinity: drop top {int(round(affinity_exclude_top_pct*100))}%",
                "eliminated_now": 0,
                "remaining": remaining,
                "skipped_reason": "would_remove_winner",
            })
        else:
            keep_mask = aff_pct < thr
            new_pool = [c for c, keep in zip(pool, keep_mask) if keep]
            elim = len(pool) - len(new_pool)
            pool = new_pool
            remaining = len(pool)
            seq_rows.append({
                "step": 0,
                "filter_id": f"AFF_TOP{int(round(affinity_exclude_top_pct*100))}",
                "name": f"Anti-affinity: drop top {int(round(affinity_exclude_top_pct*100))}%",
                "eliminated_now": elim,
                "remaining": remaining,
                "skipped_reason": None,
            })

    # ===== Pair conflicts (for "avoid pairs" table) =====
    avoid_rows = []
    for a, b in it.combinations(sorted(applicable.keys()), 2):
        pr = pair_risk.get((a,b)) or pair_risk.get((b,a)) or 0.0
        if pr > 0:
            co_app = both_blk = 0
            for i in idx_used:
                env_i = build_env_for_draw(i, winners)
                ai = safe_eval(applicable[a].applicable_if, env_i)
                bi = safe_eval(applicable[b].applicable_if, env_i)
                if ai and bi:
                    co_app += 1
                    if safe_eval(applicable[a].expression, env_i) and safe_eval(applicable[b].expression, env_i):
                        both_blk += 1
            if co_app > 0:
                avoid_rows.append({
                    "filter_id_1": a, "filter_id_2": b,
                    "co_applicable_n": co_app, "both_blocked_n": both_blk,
                    "pair_risk": round(both_blk / max(co_app, 1), 6)
                })
    pd.DataFrame(avoid_rows).sort_values(
        ["pair_risk","co_applicable_n"], ascending=[False,False]
    ).to_csv(OUTPUT_DIR / "avoid_pairs.csv", index=False)

    # ===== Apply filters in order, but never remove the winner =====
    base_env = env_now
    final_pool = pool[:] if pool else None

    for step, f in enumerate(ranked, start=1):
        fid = f.fid
        if final_pool is not None:
            new_pool, elim = apply_filter_to_pool(f, base_env, final_pool)
            if always_keep_winner and (winner_today in final_pool) and (winner_today not in new_pool):
                seq_rows.append({
                    "step": step, "filter_id": fid, "name": f.name,
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "would_remove_winner"
                })
                continue
            if elim == 0:
                seq_rows.append({
                    "step": step, "filter_id": fid, "name": f.name,
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "no_reduction"
                })
                continue
            final_pool = new_pool
            remaining = len(final_pool)
            seq_rows.append({
                "step": step, "filter_id": fid, "name": f.name,
                "eliminated_now": elim, "remaining": remaining,
                "skipped_reason": None
            })
            if (remaining <= target_max) and (not minimize_beyond_target):
                break
        else:
            seq_rows.append({
                "step": step, "filter_id": fid, "name": f.name,
                "eliminated_now": None, "remaining": None,
                "skipped_reason": None
            })

    # ===== Outputs (sequence) =====
    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(OUTPUT_DIR / "recommender_sequence.csv", index=False)
    if final_pool is not None:
        pd.DataFrame([r for r in seq_rows if r["eliminated_now"] not in (None,0)]).to_csv(
            OUTPUT_DIR / "pool_reduction_log.csv", index=False
        )

    # ===== Tier list (quick “do not apply” summary) =====
    rows = []
    for fid, f in applicable.items():
        risk = float(single_fail.get(fid, 0.0))
        sup  = int(support.get(fid, 0))
        if risk >= 0.20 and sup >= 6: tier = "DO_NOT_APPLY"
        elif risk >= 0.10:            tier = "APPLY_LATE"
        else:                          tier = "SAFE"
        rows.append({"filter_id": fid, "name": f.name, "risk": round(risk,4), "support": sup, "tier": tier})
    tier_order = {"DO_NOT_APPLY": 0, "APPLY_LATE": 1, "SAFE": 2}
    do_not_df = pd.DataFrame(rows)
    if not do_not_df.empty:
        do_not_df = do_not_df.sort_values(
            by=["tier","risk","support","filter_id"],
            ascending=[True, False, False, True],
            key=lambda s: s.map(tier_order) if s.name == "tier" else s
        )
        do_not_df.to_csv(OUTPUT_DIR / "do_not_apply.csv", index=False)

    # ===== NoPool panel =====
    nopool_df = pd.DataFrame()
    if include_nopool_panel:
        nopool_df = nopool_today_pairs(
            winners, filters_map, idx_now,
            min_days=NOPOOL_MIN_DAYS,
            min_keep=NOPOOL_MIN_KEEP,
            parity_skew=NOPOOL_PARITY_SKEW,
            max_rows=NOPOOL_MAX_ROWS,
        )

    # ===== Final SafeList builds =====
    final_safe_meta = build_final_ordered_safelist(
        safe_source_path=OUTPUT_DIR / "do_not_apply.csv",
        avoid_pairs_path=OUTPUT_DIR / "avoid_pairs.csv",
        seq_path=OUTPUT_DIR / "recommender_sequence.csv",
        out_basename="final_safe_ordered",
    )

    final_safe_safest_meta = build_final_safest_first_safelist(
        safe_source_path=OUTPUT_DIR / "do_not_apply.csv",
        avoid_pairs_path=OUTPUT_DIR / "avoid_pairs.csv",
        seq_path=OUTPUT_DIR / "recommender_sequence.csv",
        out_basename="final_safe_safest_first",
    )

    final_safe_grouped_meta = build_final_grouped_safelist(
        safe_source_path=OUTPUT_DIR / "do_not_apply.csv",
        avoid_pairs_path=OUTPUT_DIR / "avoid_pairs.csv",
        seq_path=OUTPUT_DIR / "recommender_sequence.csv",
        out_basename="final_safe_grouped",
    )

    # ===== One-pager (HTML) =====
    seed_list = base_env['seed_digits_list']
    parity_major = "even>=3" if sum(1 for d in seed_list if d%2==0) >= 3 else "even<=2"
    winner_kept = (winner_today in (final_pool or [])) if pool else True

    HTML_CSS = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>DC5 Recommender — One-Pager</title>
<style>
:root { --fg:#0f172a; --muted:#64748b; --bg:#fff; }
body { margin:0; font:14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Helvetica, Arial; color:var(--fg); background:var(--bg); }
.wrap { max-width: 900px; margin: 28px auto 64px; padding: 0 20px; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 18px; margin: 24px 0 8px; }
.card { border:1px solid #e2e8f0; border-radius:16px; padding:16px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.table { width:100%; border-collapse: collapse; }
.table th, .table td { text-align:left; padding:8px 10px; border-bottom:1px solid #e5e7eb; }
.table th { background:#f8fafc; font-weight:700; }
.kpi { background:#f8fafc; padding:8px 10px; border-radius:12px; border:1px solid #e2e8f0; display:inline-block; margin-right:8px; }
.badge { display:inline-block; padding:6px 10px; border-radius:12px; background:#eef2ff; margin-right:8px; }
.badge.ok { background:#ecfdf5; }
.badge.bad { background:#fef2f2; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
</style></head><body><div class='wrap'>"""

    applied = pd.DataFrame(seq_rows)
    applied = applied[(applied["eliminated_now"].notna()) & (applied["eliminated_now"]>0)]
    skipped = pd.DataFrame(seq_rows)
    skipped = skipped[(skipped["skipped_reason"].notna()) & (skipped["skipped_reason"]!="")]

    def df_to_html_table(df: pd.DataFrame, columns: list, empty_msg: str, limit: int = None):
        if df is None or df.empty:
            return "<p class='muted mono'>{}</p>".format(html.escape(empty_msg))
        use = df[columns].copy() if columns else df.copy()
        if limit: use = use.head(limit)
        return use.to_html(index=False, classes="table", border=0, escape=True)

    avoid_path = OUTPUT_DIR / "avoid_pairs.csv"
    avoid_df = pd.read_csv(avoid_path) if avoid_path.exists() else pd.DataFrame()
    avoid_sorted = avoid_df.sort_values(["pair_risk","co_applicable_n"], ascending=[False,False]) if not avoid_df.empty else avoid_df

    applied_html = df_to_html_table(applied, ["step","filter_id","name","eliminated_now","remaining"], "No safe reduction steps.", 60)
    skipped_html = df_to_html_table(skipped, ["filter_id","name","skipped_reason"], "No steps were skipped.", 60)
    avoid_html   = df_to_html_table(avoid_sorted, ["filter_id_1","filter_id_2","pair_risk","both_blocked_n","co_applicable_n"], "No high-risk pairs in this bucket.", 20)

    # Final Ordered (sequence-respecting)
    final_html = "<p class='muted mono'>Final Ordered SafeList not built (missing inputs).</p>"
    if final_safe_meta and final_safe_meta.get("outputs", {}).get("csv"):
        try:
            fs_df = pd.read_csv(final_safe_meta["outputs"]["csv"])
            final_html = df_to_html_table(fs_df, [], "Final Ordered SafeList is empty.", None if len(fs_df)<=400 else 400)
        except Exception:
            pass

    # Safest-first
    safest_html = "<p class='muted mono'>Safest-first SafeList not built (missing inputs).</p>"
    if final_safe_safest_meta and final_safe_safest_meta.get("outputs", {}).get("csv"):
        try:
            sf_df = pd.read_csv(final_safe_safest_meta["outputs"]["csv"])
            safest_html = df_to_html_table(sf_df, [], "Safest-first list is empty.", None if len(sf_df)<=400 else 400)
        except Exception:
            pass

    # Grouped-by-aggression
    grouped_html = "<p class='muted mono'>Grouped SafeList not built (missing inputs).</p>"
    if final_safe_grouped_meta and final_safe_grouped_meta.get("outputs", {}).get("html"):
        try:
            html_path = final_safe_grouped_meta["outputs"]["html"]
            grouped_html = Path(html_path).read_text(encoding="utf-8")
        except Exception:
            pass

    snap = (
        "<h1>DC5 Recommender — One-Pager</h1>"
        "<div class='card'><div><b>Seed:</b> {} &nbsp;·&nbsp; <b>Sum:</b> {} ({}) "
        "&nbsp;·&nbsp; <b>Structure:</b> {} &nbsp;·&nbsp; <b>Spread:</b> {} "
        "&nbsp;·&nbsp; <b>Parity:</b> {}</div>"
        "<div style='margin-top:8px'><span class='kpi'>Anti-affinity default: drop top {}%</span></div>"
        "<div style='margin-top:8px'>"
        "<span class='badge'>Applicable filters: {}</span>"
        "<span class='badge {}'>{}</span>"
        "<span class='badge'>Target: &lt; {}</span>"
        "<span class='badge'>Final size: {}</span>"
        "</div></div>"
    ).format(
        html.escape(str(base_env['seed'])),
        base_env['seed_sum'], html.escape(base_env['seed_sum_category']),
        html.escape(classify_structure(seed_list)),
        base_env['spread_seed'], parity_major,
        int(round((AFFINITY_EXCLUDE_TOP_PCT)*100)),
        len(applicable),
        "ok" if winner_kept else "bad", "Winner: " + ("KEPT" if winner_kept else "REMOVED"),
        TARGET_MAX+1,
        remaining if remaining is not None else "—"
    )

    html_doc = (
        HTML_CSS + snap +
        "<div class='card'><h2>Apply in this order (winner-preserving)</h2>{}</div>".format(applied_html) +
        "<div class='card'><h2>Skipped steps</h2>{}</div>".format(skipped_html) +
        "<div class='card'><h2>Avoid combining (today’s bucket)</h2>{}</div>".format(avoid_html) +
        ("<div class='card'><h2>Today — historically safe filter pairs (NoPool)</h2>{}</div>".format(
            df_to_html_table(
                nopool_df,
                ["Filter A ID","Filter B ID","Days both applicable (history)","Winner kept % (both applicable)","Either blocks winner %","Both block winner %","Parity note (only if meaningful)"],
                f"No historical pairs meet thresholds (min_days={NOPOOL_MIN_DAYS}, min_keep={NOPOOL_MIN_KEEP}%).",
                NOPOOL_MAX_ROWS
            )
        ) if include_nopool_panel else "") +
        "<div class='card'><h2>Final Ordered SafeList (sequence-respecting)</h2>{}</div>".format(final_html) +
        "<div class='card'><h2>Final SafeList — Safest-first (risk↑, support↓, eliminations↓)</h2>{}</div>".format(safest_html) +
        "<div class='card'><h2>Final SafeList — Grouped by Aggression (most ➜ least)</h2>{}</div>".format(grouped_html) +
        "</div></body></html>"
    )

    (OUTPUT_DIR / "one_pager.html").write_text(html_doc, encoding="utf-8")
    return seq_rows

__all__ = [
    "WINNERS_CSV","FILTERS_CSV","TODAY_POOL_CSV","OUTPUT_DIR",
    "TARGET_MAX","ALWAYS_KEEP_WINNER","MINIMIZE_BEYOND_TARGET",
    "AFFINITY_EXCLUDE_TOP_PCT",
    "main","combo_affinity","get_pool_for_seed","nopool_today_pairs",
    "build_final_ordered_safelist","build_final_safest_first_safelist",
    "build_final_grouped_safelist"
]

if __name__ == "__main__":
    main()

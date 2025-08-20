"""
DC5 Filter SAFETY Auditor — Winner-Block Rates by Seed Cases
------------------------------------------------------------
Goal
  • Tell you, with facts, which filters are SAFE (never block winners) and which FAIL (block real winners),
    overall and per seed “case” (structure, sum category, spread band, parity majority).
  • No minimization of eliminations; just truth tables and rates from your REAL chronological results.

Inputs (edit the two paths below):
  • WINNERS_CSV  — chronological winners; must include a 'Result' column (5-digit strings)
  • FILTERS_CSV  — filters with first 5 columns: id,name,enabled,applicable_if,expression

Outputs (CSV files):
  • filter_safety_overall.csv
      id, name, applicable_n, blocks_winner_n, failure_rate, safe_globally
  • filter_safety_by_seed_sumcat.csv
      id, seed_sum_category, applicable_n, blocks_winner_n, failure_rate, safe_in_bucket, support
  • filter_safety_by_seed_structure.csv
      id, seed_structure, applicable_n, blocks_winner_n, failure_rate, safe_in_bucket, support
  • filter_safety_by_seed_spread_band.csv
      id, seed_spread_band, applicable_n, blocks_winner_n, failure_rate, safe_in_bucket, support
  • filter_safety_by_seed_parity_majority.csv
      id, seed_parity_majority, applicable_n, blocks_winner_n, failure_rate, safe_in_bucket, support
  • always_safe_in_bucket.csv
      bucket_type, bucket_value, id, name, support

Assumptions (aligned with your app conventions):
  • Seed = previous draw (idx-1)
  • Hot/Cold = last 10 seeds; Due = absent in last 2 seeds (exposed if expressions need them)
  • Env exposes: prev_seed/prev_prev_seed, last2, hot_digits_10/cold_digits_10, due_digits_2,
                 plus back-compat keys hot_digits_20/cold_digits_20 mapped to last-10 sets.

Run:
  python safety_auditor.py

No sampling. No simulation. Every row = your REAL winner.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

# =========================
# === EDIT THESE PATHS ===
# =========================
WINNERS_CSV = "Structure_Breakdown_729_Entries_.csv"   # must contain column 'Result'
FILTERS_CSV = "lottery_filters_batch10 (21).csv"   # must contain columns: id,name,enabled,applicable_if,expression
OUTPUT_DIR  = Path(".")

# ===============
# Helper Mapping
# ===============
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
VTRAC = {0:0,5:0, 1:1,6:1, 2:2,7:2, 3:3,8:3, 4:4,9:4}
SUM_CAT = {
    "very_low": (0, 15),
    "low":      (16, 24),
    "mid":      (25, 33),
    "high":     (34, 45),
}

# ========================
# Data Loading & Sanity
# ========================

def load_winners(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Result' not in df.columns:
        raise ValueError("Winners CSV must have a 'Result' column.")
    df['Result'] = df['Result'].astype(str).str.replace("\D", "", regex=True).str.zfill(5)
    bad = df[~df['Result'].str.fullmatch(r"\d{5}")]
    if not bad.empty:
        raise ValueError(f"Non-5-digit rows found in 'Result':\n{bad}")
    return df.reset_index(drop=True)


def load_filters(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    req = ['id','name','enabled','applicable_if','expression']
    for r in req:
        if r not in df.columns:
            raise ValueError(f"Filters CSV missing required column: {r}")
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return False
        return str(x).strip().lower() in {"true","1","yes","y"}
    df['enabled'] = df['enabled'].map(to_bool)
    # clean quoting if present
    for col in ['applicable_if','expression']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('"""','"', regex=False).str.replace("'''","'", regex=False)
        df[col] = df[col].apply(lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"','\''} else s)
    return df

# ====================
# Context Computation
# ====================

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in s]


def sum_category(total: int) -> str:
    for name, (lo, hi) in SUM_CAT.items():
        if lo <= total <= hi:
            return name
    return "out_of_range"


def make_hot_cold_due(history: List[List[int]], k_hotcold: int = 10):
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
    from collections import Counter as C
    c = C(digs)
    counts = sorted(c.values(), reverse=True)
    if counts == [5]:
        return 'quint'
    if counts == [4,1]:
        return 'quad'
    if counts == [3,2]:
        return 'triple_double'
    if counts == [3,1,1]:
        return 'triple'
    if counts == [2,2,1]:
        return 'double_double'
    if counts == [2,1,1,1]:
        return 'double'
    return 'single'


def spread_band(spread: int) -> str:
    if spread <= 3:
        return '0-3'
    if spread <= 5:
        return '4-5'
    if spread <= 7:
        return '6-7'
    if spread <= 9:
        return '8-9'
    return '10+'


def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
    seed_str   = winners[idx-1]
    combo_str  = winners[idx]
    seed_list  = digits_of(seed_str)
    combo_list = digits_of(combo_str)

    seed_digits = set(seed_list)
    combo_digits = set(combo_list)

    seed_sum  = sum(seed_list)
    combo_sum = sum(combo_list)

    prev2_union = set(d for s in winners[max(0, idx-2):idx] for d in digits_of(s))

    # Hot/Cold/Due based on last 10; Due = absent in last 2
    history_digits = [digits_of(s) for s in winners[:idx]]
    hotk, coldk, due2 = make_hot_cold_due(history_digits, k_hotcold=10)

    prev_seed_str = winners[idx-1]
    prev_prev_seed_str = winners[idx-2] if idx >= 2 else None
    prev_seed_digits_list = digits_of(prev_seed_str)
    prev_prev_seed_digits_list = digits_of(prev_prev_seed_str) if prev_prev_seed_str is not None else []

    env = {
        'combo': combo_str,
        'combo_digits': combo_digits,
        'combo_digits_list': combo_list,
        'combo_sum': combo_sum,
        'combo_sum_category': sum_category(combo_sum),
        'seed': seed_str,
        'seed_digits': seed_digits,
        'seed_digits_list': seed_list,
        'seed_sum': seed_sum,
        'seed_sum_category': sum_category(seed_sum),
        'spread_seed': max(seed_list) - min(seed_list),
        'spread_combo': max(combo_list) - min(combo_list),
        'last2': prev2_union,
        'prev_seed': prev_seed_str,
        'prev_seed_digits': set(prev_seed_digits_list),
        'prev_seed_digits_list': prev_seed_digits_list,
        'prev_prev_seed': prev_prev_seed_str,
        'prev_prev_seed_digits': set(prev_prev_seed_digits_list),
        'prev_prev_seed_digits_list': prev_prev_seed_digits_list,
        'hot_digits_10': hotk,
        'cold_digits_10': coldk,
        'hot_digits_20': hotk,  # back-compat mapped to last-10
        'cold_digits_20': coldk, # back-compat mapped to last-10
        'due_digits_2': due2,
        'mirror': MIRROR,
        'vtrac': VTRAC,
        'any': any, 'all': all, 'len': len, 'sum': sum, 'max': max, 'min': min, 'set': set, 'sorted': sorted,
    }
    return env

# ========================
# Filter Evaluation Logic
# ========================

@dataclass(frozen=True)
class FilterDef:
    fid: str
    name: str
    enabled: bool
    applicable_if: str
    expression: str


def parse_filters(df: pd.DataFrame) -> List[FilterDef]:
    out = []
    for _, row in df.iterrows():
        out.append(FilterDef(
            str(row['id']).strip(),
            str(row['name']).strip(),
            bool(row['enabled']),
            str(row['applicable_if']).strip(),
            str(row['expression']).strip()
        ))
    return out


def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    if not expr:
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        # If a filter errors, treat as not applicable / not eliminating (conservative)
        return False

# ===================
# Main Safety Audit
# ===================

def main():
    winners_df = load_winners(WINNERS_CSV)
    filters_df = load_filters(FILTERS_CSV)
    winners = winners_df['Result'].tolist()
    if len(winners) < 2:
        raise SystemExit("Need at least 2 winners to compute seed/winner pairs.")
    filters = parse_filters(filters_df)

    # Overall tallies per filter
    applicable_n = Counter()
    blocks_winner_n = Counter()

    # Buckets
    # seed traits per draw
    per_draw_meta = []  # list of dicts for later bucketing

    for idx in range(1, len(winners)):
        env = build_env_for_draw(idx, winners)
        seed_list = env['seed_digits_list']
        sc = sum(env['seed_digits_list'])  # not used directly; seed_sum already
        seed_sum_cat = env['seed_sum_category']
        seed_spread = env['spread_seed']
        seed_spread_b = spread_band(seed_spread)
        seed_struct = classify_structure(seed_list)
        seed_even_majority = 'even>=3' if sum(1 for d in seed_list if d%2==0) >= 3 else 'even<=2'

        blocked_by = []
        for f in filters:
            if not f.enabled:
                continue
            if safe_eval(f.applicable_if, env):
                applicable_n[f.fid] += 1
                if safe_eval(f.expression, env):
                    blocks_winner_n[f.fid] += 1
                    blocked_by.append(f.fid)
        per_draw_meta.append({
            'idx': idx,
            'seed_sum_category': seed_sum_cat,
            'seed_structure': seed_struct,
            'seed_spread_band': seed_spread_b,
            'seed_parity_majority': seed_even_majority,
            'blocked_by': blocked_by,
        })

    # === Overall CSV ===
    rows_overall = []
    name_map = {str(r['id']).strip(): str(r['name']).strip() for _, r in filters_df.iterrows()}
    for fid in name_map.keys():
        app = applicable_n[fid]
        blk = blocks_winner_n[fid]
        failure = (blk / app) if app > 0 else 0.0
        rows_overall.append({
            'id': fid,
            'name': name_map.get(fid, ''),
            'applicable_n': app,
            'blocks_winner_n': blk,
            'failure_rate': round(failure, 6),
            'safe_globally': (blk == 0 and app > 0)
        })
    pd.DataFrame(rows_overall).sort_values(['safe_globally','failure_rate','applicable_n'], ascending=[False,True,False]).to_csv(OUTPUT_DIR / 'filter_safety_overall.csv', index=False)

    # Helper to aggregate by a bucket field
    def bucket_aggregate(field: str, outname: str):
        agg = []
        support_threshold = 5  # require at least 5 applicable cases to call it "safe in bucket"
        by_bucket = defaultdict(list)
        for md in per_draw_meta:
            by_bucket[md[field]].append(md['blocked_by'])
        for bucket_val, lists in by_bucket.items():
            # Reconstruct applicability by re-evaluating applicable_if per draw for this bucket
            # (We need env per draw again)
            indices = [md['idx'] for md in per_draw_meta if md[field] == bucket_val]
            app_c = Counter()
            blk_c = Counter()
            for idx in indices:
                env = build_env_for_draw(idx, winners)
                for f in filters:
                    if not f.enabled:
                        continue
                    if safe_eval(f.applicable_if, env):
                        app_c[f.fid] += 1
                        if safe_eval(f.expression, env):
                            blk_c[f.fid] += 1
            for fid in name_map.keys():
                app = app_c[fid]
                blk = blk_c[fid]
                if app == 0:
                    continue
                failure = (blk / app) if app > 0 else 0.0
                safe_bucket = (blk == 0 and app >= support_threshold)
                agg.append({
                    'id': fid,
                    field: bucket_val,
                    'applicable_n': app,
                    'blocks_winner_n': blk,
                    'failure_rate': round(failure, 6),
                    'safe_in_bucket': safe_bucket,
                    'support': app
                })
        pd.DataFrame(agg).sort_values(['safe_in_bucket','failure_rate','support'], ascending=[False,True,False]).to_csv(OUTPUT_DIR / outname, index=False)
        return agg

    agg_sumcat = bucket_aggregate('seed_sum_category', 'filter_safety_by_seed_sumcat.csv')
    agg_struct = bucket_aggregate('seed_structure', 'filter_safety_by_seed_structure.csv')
    agg_spread = bucket_aggregate('seed_spread_band', 'filter_safety_by_seed_spread_band.csv')
    agg_parity = bucket_aggregate('seed_parity_majority', 'filter_safety_by_seed_parity_majority.csv')

    # Always-safe in bucket (summary table)
    always_rows = []
    def harvest(label: str, agg):
        for row in agg:
            if row['safe_in_bucket']:
                always_rows.append({
                    'bucket_type': label,
                    'bucket_value': row[label if label.startswith('seed_') else label],
                    'id': row['id'],
                    'name': name_map.get(row['id'], ''),
                    'support': row['support']
                })
    # Because of field naming, pass explicit labels and pick values directly
    for row in agg_sumcat:
        if row['safe_in_bucket']:
            always_rows.append({'bucket_type':'seed_sum_category','bucket_value':row['seed_sum_category'],'id':row['id'],'name':name_map.get(row['id'],''),'support':row['support']})
    for row in agg_struct:
        if row['safe_in_bucket']:
            always_rows.append({'bucket_type':'seed_structure','bucket_value':row['seed_structure'],'id':row['id'],'name':name_map.get(row['id'],''),'support':row['support']})
    for row in agg_spread:
        if row['safe_in_bucket']:
            always_rows.append({'bucket_type':'seed_spread_band','bucket_value':row['seed_spread_band'],'id':row['id'],'name':name_map.get(row['id'],''),'support':row['support']})
    for row in agg_parity:
        if row['safe_in_bucket']:
            always_rows.append({'bucket_type':'seed_parity_majority','bucket_value':row['seed_parity_majority'],'id':row['id'],'name':name_map.get(row['id'],''),'support':row['support']})

    pd.DataFrame(always_rows).sort_values(['bucket_type','bucket_value','support'], ascending=[True,True,False]).to_csv(OUTPUT_DIR / 'always_safe_in_bucket.csv', index=False)

    print("Done. Wrote: filter_safety_overall.csv, filter_safety_by_* CSVs, and always_safe_in_bucket.csv")


if __name__ == '__main__':
    main()

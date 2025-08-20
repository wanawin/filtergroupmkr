"""
DC5 Filter Compatibility Auditor (Minimal Disable Sets ≤3)
---------------------------------------------------------
Purpose
  • Chronologically iterate your REAL winners (no sampling, no simulation)
  • For each draw, identify which filters would eliminate the actual winner
  • Find minimal disable sets of size 1–3 that would preserve the winner
  • Export per-draw logs + aggregate stats for singles/pairs/triples

Inputs (edit the two paths below):
  • WINNERS_CSV  — your chronological winners file (must include a 'Result' column)
  • FILTERS_CSV  — your filter definitions (first 5 columns: id,name,enabled,applicable_if,expression)

Assumptions aligned with your app (#121, #122, #123, #125):
  • The winners are chronological (seed = previous draw)
  • Filters use Python expressions referencing the same variable names as your app
  • Column for 5-digit combos is exactly 'Result'
  • Filter CSV uses lowercase 'id'
  • Dynamic context provided: seed_digits, combo_digits, seed_sum, combo_sum, mirror, v_tracs,
    last2 (digits from prev two seeds), hot_digits_20/cold_digits_20, due_digits_2, etc.

Outputs (files created next to the script):
  • per_draw_log.csv — per-draw culprit filters and all minimal disable sets (≤3)
  • singles_summary.csv — how often a single filter is the minimal fix
  • pairs_summary.csv   — how often a PAIR is the minimal fix
  • triples_summary.csv — how often a TRIPLE is the minimal fix

Run:
  python auditor.py

No sampling. No simulation. Every row is evaluated against the REAL winner.
"""

from __future__ import annotations
import csv
import itertools as it
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd

# =========================
# === EDIT THESE PATHS ===
# =========================
WINNERS_CSV = "winners.csv"   # must contain a 'Result' column with 5-digit strings (e.g., 27500)
FILTERS_CSV = "filters.csv"   # first 5 columns: id,name,enabled,applicable_if,expression
OUTPUT_DIR  = Path(".")

# ===============
# Helper Mapping
# ===============
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
# V-Trac groups (classic 5 buckets)
VTRAC = {
    0: 0, 5: 0,
    1: 1, 6: 1,
    2: 2, 7: 2,
    3: 3, 8: 3,
    4: 4, 9: 4,
}

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
    # Ensure 'Result' exists
    if 'Result' not in df.columns:
        raise ValueError("Winners CSV must have a 'Result' column.")
    # Normalize to 5-char zero-padded strings
    df['Result'] = df['Result'].astype(str).str.replace("\D", "", regex=True).str.zfill(5)
    # Verify all are 5 digits
    bad = df[~df['Result'].str.fullmatch(r"\d{5}")]
    if not bad.empty:
        raise ValueError(f"Non-5-digit rows found in 'Result':\n{bad}")
    return df.reset_index(drop=True)


def load_filters(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (we only require first 5 logical fields)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    required = ['id', 'name', 'enabled', 'applicable_if', 'expression']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Filters CSV is missing required column: {r}")

    # Coerce 'enabled' to bool
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return False
        xs = str(x).strip().lower()
        return xs in {"true", "1", "yes", "y"}

    df['enabled'] = df['enabled'].map(to_bool)
    # Strip wrapping quotes in expressions if present
    for col in ['applicable_if', 'expression']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('"""', '"', regex=False).str.replace("'''", "'", regex=False)
        # Remove leading/trailing quotes
        df[col] = df[col].apply(lambda s: s[1:-1] if len(s) >= 2 and ((s[0]==s[-1]=="\"") or (s[0]==s[-1]=="'")) else s)

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


def make_hot_cold_due(history: List[List[int]], k_hotcold: int = 20) -> Tuple[Set[int], Set[int], Set[int]]:
    """Return (hot_digits_20, cold_digits_20, due_digits_2).
    - hot/cold computed over last k_hotcold seeds (digits flattened)
    - due = digits absent in last 2 seeds
    """
    flat = [d for row in history[-k_hotcold:] for d in row]
    cnt = Counter(flat)
    if cnt:
        # Top 6 hot by frequency (ties naturally included by counts ordering)
        most = cnt.most_common()
        # Determine the 6th-place frequency to include ties
        topk = 6
        thresh = most[topk-1][1] if len(most) >= topk else (most[-1][1] if most else 0)
        hot = {d for d, c in most if c >= thresh}
        # Bottom 4 cold by frequency (include ties at the boundary)
        least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
        coldk = 4
        if least:
            cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
            cold = {d for d, c in least if c <= cth}
        else:
            cold = set()
    else:
        hot, cold = set(), set()

    # Due digits over last 2 seeds (absent in union)
    last2 = set(d for row in history[-2:] for d in row)
    due = set(range(10)) - last2
    return hot, cold, due


def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
    """Build the evaluation environment for draw idx.
    Assumes idx >= 1 (since seed = previous draw).
    """
    seed_str   = winners[idx-1]
    combo_str  = winners[idx]
    seed_digits_list  = digits_of(seed_str)
    combo_digits_list = digits_of(combo_str)

    seed_digits = set(seed_digits_list)
    combo_digits = set(combo_digits_list)

    seed_sum  = sum(seed_digits_list)
    combo_sum = sum(combo_digits_list)

    # Spread, parity, counts
    spread_seed  = max(seed_digits_list) - min(seed_digits_list)
    spread_combo = max(combo_digits_list) - min(combo_digits_list)

    even_count_combo = sum(1 for d in combo_digits_list if d % 2 == 0)
    odd_count_combo  = 5 - even_count_combo

    high_count_combo = sum(1 for d in combo_digits_list if d >= 5)
    low_count_combo  = 5 - high_count_combo

    # last2 digits set (prev two seeds)
    prev2_union = set(d for s in winners[max(0, idx-2):idx] for d in digits_of(s))

    # Also expose individual previous seeds
    prev_seed_str = winners[idx-1]
    prev_prev_seed_str = winners[idx-2] if idx >= 2 else None
    prev_seed_digits_list = digits_of(prev_seed_str)
    prev_seed_digits = set(prev_seed_digits_list)
    prev_prev_seed_digits_list = digits_of(prev_prev_seed_str) if prev_prev_seed_str is not None else []
    prev_prev_seed_digits = set(prev_prev_seed_digits_list)

    # Hot/Cold/Due
    history_digits = [digits_of(s) for s in winners[:idx]]  # up to seed
    hot20, cold20, due2 = make_hot_cold_due(history_digits, k_hotcold=10)

    # V-trac groups for combo
    v_tracs_combo = [VTRAC[d] for d in combo_digits_list]

    env = {
        # combo-specific
        'combo': combo_str,
        'combo_digits': combo_digits,
        'combo_digits_list': combo_digits_list,
        'combo_sum': combo_sum,
        'combo_sum_category': sum_category(combo_sum),
        'spread_combo': spread_combo,
        'even_count_combo': even_count_combo,
        'odd_count_combo': odd_count_combo,
        'high_count_combo': high_count_combo,
        'low_count_combo': low_count_combo,
        'v_tracs_combo': v_tracs_combo,

        # seed-specific
        'seed': seed_str,
        'seed_digits': seed_digits,
        'seed_digits_list': seed_digits_list,
        'seed_sum': seed_sum,
        'seed_sum_category': sum_category(seed_sum),
        'spread_seed': spread_seed,

        # history-derived
        'last2': prev2_union,
        'prev_seed': prev_seed_str,
        'prev_seed_digits': prev_seed_digits,
        'prev_seed_digits_list': prev_seed_digits_list,
        'prev_prev_seed': prev_prev_seed_str,
        'prev_prev_seed_digits': prev_prev_seed_digits,
        'prev_prev_seed_digits_list': prev_prev_seed_digits_list,
        'hot_digits_10': hot20,
        'cold_digits_10': cold20,
        'hot_digits_20': hot20,
        'cold_digits_20': cold20,
        'due_digits_2': due2,

        # mappings
        'mirror': MIRROR,
        'vtrac': VTRAC,

        # handy builtins for expressions
        'any': any,
        'all': all,
        'len': len,
        'sum': sum,
        'max': max,
        'min': min,
        'set': set,
        'sorted': sorted,
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
    filters: List[FilterDef] = []
    for _, row in df.iterrows():
        fid = str(row['id']).strip()
        name = str(row['name']).strip()
        enabled = bool(row['enabled'])
        applicable_if = str(row['applicable_if']).strip()
        expression = str(row['expression']).strip()
        filters.append(FilterDef(fid, name, enabled, applicable_if, expression))
    return filters


def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    if not expr:
        return True
    try:
        # Evaluate boolean python expression in restricted env
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        # If a filter expression errors, treat as not applicable / not eliminating
        return False


def winner_eliminated_by(filters: List[FilterDef], env: Dict[str, object]) -> List[FilterDef]:
    culprits = []
    for f in filters:
        if not f.enabled:
            continue
        if safe_eval(f.applicable_if, env):
            if safe_eval(f.expression, env):
                culprits.append(f)
    return culprits


def minimal_disable_sets(culprits: List[FilterDef], env: Dict[str, object]) -> List[Tuple[str, ...]]:
    """Return all minimal disable sets (by id) with size ≤ 3.
    Strategy: for k in 1..3, test all size-k subsets; return all that save the winner;
    stop at the first k that works.
    """
    if not culprits:
        return []

    culprit_ids = [f.fid for f in culprits]

    def saved_if_disable(disable_ids: Set[str]) -> bool:
        # Winner is saved if no remaining culprit survives
        for f in culprits:
            if f.fid in disable_ids:
                continue
            # Re-check as if this filter were the only gate (applicable_if already true),
            # but in practice, culprits were computed under the same env.
            return False  # if any active culprit remains, still eliminated
        return True

    # However, the above shortcut assumes culprits are fixed under env.
    # Since env doesn't change when disabling other filters for the WINNER test,
    # saving the winner == disabling ALL culprits still active.
    # Therefore, minimal sets are exactly minimal hitting sets of culprit_ids.

    # We still brute-force k=1..3 to be explicit and stable.
    cul_set = set(culprit_ids)
    for k in (1, 2, 3):
        found: List[Tuple[str, ...]] = []
        for subset in it.combinations(culprit_ids, k):
            if cul_set.issubset(set(subset)):
                # disabling this subset disables all culprits → winner saved
                found.append(tuple(sorted(subset)))
        if found:
            # Deduplicate identical tuples
            uniq = sorted(set(found))
            return uniq
    return []

# ===================
# Main Audit Routine
# ===================

def main():
    winners_df = load_winners(WINNERS_CSV)
    filters_df = load_filters(FILTERS_CSV)

    winners = winners_df['Result'].tolist()
    if len(winners) < 2:
        raise SystemExit("Not enough winners to run (need at least 2).")

    filters = parse_filters(filters_df)

    # Aggregates
    single_counter = Counter()        # filter_id -> count when {id} is a minimal fix
    pair_counter   = Counter()        # (id1,id2) sorted -> count
    triple_counter = Counter()        # (id1,id2,id3) sorted -> count

    rows = []  # per-draw log rows

    for idx in range(1, len(winners)):
        env = build_env_for_draw(idx, winners)
        culprits = winner_eliminated_by(filters, env)
        culprit_ids = [f.fid for f in culprits]

        if not culprits:
            rows.append({
                'draw_index': idx,
                'seed': winners[idx-1],
                'winner': winners[idx],
                'eliminated': False,
                'culprit_ids': '',
                'minimal_disable_sets': ''
            })
            continue

        # Find minimal disable sets ≤ 3
        msets = minimal_disable_sets(culprits, env)

        # Tally
        for s in msets:
            if len(s) == 1:
                single_counter[s[0]] += 1
            elif len(s) == 2:
                pair_counter[tuple(sorted(s))] += 1
            elif len(s) == 3:
                triple_counter[tuple(sorted(s))] += 1

        rows.append({
            'draw_index': idx,
            'seed': winners[idx-1],
            'winner': winners[idx],
            'eliminated': True,
            'culprit_ids': ",".join(culprit_ids),
            'minimal_disable_sets': ";".join(["+".join(s) for s in msets])
        })

    # Write per-draw log
    per_draw_path = OUTPUT_DIR / 'per_draw_log.csv'
    pd.DataFrame(rows).to_csv(per_draw_path, index=False)

    # Singles summary
    singles_rows = [{'filter_id': fid, 'minimal_single_fix_count': cnt} for fid, cnt in single_counter.items()]
    pd.DataFrame(singles_rows).sort_values('minimal_single_fix_count', ascending=False).to_csv(OUTPUT_DIR / 'singles_summary.csv', index=False)

    # Pairs summary
    pairs_rows = [{'filter_id_1': a, 'filter_id_2': b, 'minimal_pair_fix_count': cnt} for (a,b), cnt in pair_counter.items()]
    pd.DataFrame(pairs_rows).sort_values('minimal_pair_fix_count', ascending=False).to_csv(OUTPUT_DIR / 'pairs_summary.csv', index=False)

    # Triples summary
    triples_rows = [{'f1': a, 'f2': b, 'f3': c, 'minimal_triple_fix_count': cnt} for (a,b,c), cnt in triple_counter.items()]
    pd.DataFrame(triples_rows).sort_values('minimal_triple_fix_count', ascending=False).to_csv(OUTPUT_DIR / 'triples_summary.csv', index=False)

    print(f"Done. Wrote:\n- {per_draw_path}\n- {OUTPUT_DIR / 'singles_summary.csv'}\n- {OUTPUT_DIR / 'pairs_summary.csv'}\n- {OUTPUT_DIR / 'triples_summary.csv'}")


if __name__ == '__main__':
    main()

"""
DC5 Filter Recommender (App-Compatible, Target 25–40)
----------------------------------------------------
**Purpose**
  • Separate helper (does NOT modify your main app) that mirrors your app’s context and
    only uses filters that are *applicable* right now (same `applicable_if` logic).
  • Tells you **which filters to apply** and **which to avoid combining** so you end up with
    ~25–40 combos to play (when a pool file is provided) while minimizing winner risk based
    on historical failure rates by seed-case.

**Inputs (edit at the top):**
  • WINNERS_CSV  = "DC5_Midday_Full_Cleaned_Expanded.csv"
  • FILTERS_CSV  = "lottery_filters_batch_10.csv"
  • TODAY_POOL_CSV = None  # optional path to CSV containing your *generated* combos today
      - The file should have a 'combo' or 'Result' column with 5-digit combos.
  • TARGET_MIN = 25, TARGET_MAX = 40  # desired final pool size

**Outputs**
  • recommender_sequence.csv
      step, filter_id, name, est_risk, est_pair_conflicts, eliminated_now, remaining
  • avoid_pairs.csv
      filter_id_1, filter_id_2, co_applicable_n, both_blocked_n, pair_risk
  • pool_reduction_log.csv (only if TODAY_POOL_CSV is provided)
      step, applied_filter_id, eliminated, remaining

**Notes**
  • Hot/Cold = last 10; Due = absent in last 2; recomputed per draw.
  • Bucketing matches your app’s semantics: seed sum category (Very Low/Low/Mid/High), structure,
    spread band, parity majority. We score filter risk inside the current bucket.
  • Pair risk uses co-occurrence of *both blocking the winner* in historical draws within the bucket.

Run:  python recommender.py
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import itertools as it

# =========================
# === EDIT THESE PATHS ===
# =========================
WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
FILTERS_CSV = "lottery_filters_batch_10.csv"
TODAY_POOL_CSV: Optional[str] = None  # e.g., "pools/today.csv" (has 'combo' or 'Result' column)
TARGET_MAX = 44
MINIMIZE_BEYOND_TARGET = True
ALWAYS_KEEP_WINNER = True

OUTPUT_DIR = Path(".")

# ===============
# Helper Mapping
# ===============
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
# Your app uses V-TRAC groups labeled 1..5
VTRAC = {0:1,5:1, 1:2,6:2, 2:3,7:3, 3:4,8:4, 4:5,9:5}

# Sum categories (exact strings your app expects)
def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return 'Very Low'
    elif 16 <= total <= 24:
        return 'Low'
    elif 25 <= total <= 33:
        return 'Mid'
    else:
        return 'High'


def spread_band(spread: int) -> str:
    if spread <= 3: return '0-3'
    if spread <= 5: return '4-5'
    if spread <= 7: return '6-7'
    if spread <= 9: return '8-9'
    return '10+'

# ========================
# Data Loading & Sanity
# ========================

def load_winners(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = 'Result' if 'Result' in df.columns else None
    if not col:
        raise ValueError("Winners CSV must have a 'Result' column.")
    vals = df[col].astype(str).str.replace("\\D","", regex=True).str.zfill(5)
    bad = vals[~vals.str.fullmatch(r"\d{5}")]
    if len(bad) > 0:
        raise ValueError("Non-5-digit winners found; clean the CSV.")
    return vals.tolist()


def load_filters(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    req = ['id','name','enabled','applicable_if','expression']
    for r in req:
        if r not in df.columns:
            raise ValueError(f"Filters CSV is missing column: {r}")
    def to_bool(x):
        if isinstance(x, bool): return x
        if pd.isna(x): return False
        return str(x).strip().lower() in {"true","1","yes","y"}
    df['enabled'] = df['enabled'].map(to_bool)
    # Clean quoting leftovers
    for col in ['applicable_if','expression']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('"""','"', regex=False).str.replace("'''","'", regex=False)
        if len(df[col]) > 0:
            df[col] = df[col].apply(lambda s: s[1:-1] if len(s)>=2 and s[0]==s[-1] and s[0] in {'"','\''} else s)
    return df

# ====================
# Context Computation
# ====================

def digits_of(s: str) -> List[int]:
    return [int(ch) for ch in s]


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
    c = Counter(digs)
    counts = sorted(c.values(), reverse=True)
    if counts == [5]: return 'quint'
    if counts == [4,1]: return 'quad'
    if counts == [3,2]: return 'triple_double'
    if counts == [3,1,1]: return 'triple'
    if counts == [2,2,1]: return 'double_double'
    if counts == [2,1,1,1]: return 'double'
    return 'single'


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

    # history up to seed
    history_digits = [digits_of(s) for s in winners[:idx]]
    hot, cold, due = hot_cold_due(history_digits, k_hotcold=10)

    # build prev_pattern like your app
    def parity_label(digs):
        return 'Even' if sum(digs) % 2 == 0 else 'Odd'
    prev_pattern = []
    for digs in (prev_prev_seed_digits, prev_seed_digits, seed_list):
        prev_pattern.extend([sum_category(sum(digs)), parity_label(digs)])
    prev_pattern = tuple(prev_pattern)

    env = {
        'combo': combo,
        'combo_digits': set(combo_list),
        'combo_digits_list': combo_list,
        'combo_sum': combo_sum,
        'combo_sum_cat': sum_category(combo_sum),
        'combo_sum_category': sum_category(combo_sum),

        'seed': seed,
        'seed_digits': set(seed_list),
        'seed_digits_list': seed_list,
        'seed_sum': seed_sum,
        'prev_sum_cat': sum_category(seed_sum),
        'seed_sum_category': sum_category(seed_sum),

        'spread_seed': max(seed_list) - min(seed_list),
        'spread_combo': max(combo_list) - min(combo_list),
        'seed_vtracs': set(VTRAC[d] for d in seed_list),
        'combo_vtracs': set(VTRAC[d] for d in combo_list),
        'last2': set(seed_list) | set(prev_seed_digits),

        'prev_seed': prev_seed,
        'prev_seed_digits': prev_seed_digits,
        'prev_seed_digits_list': prev_seed_digits,
        'prev_prev_seed': prev_prev_seed,
        'prev_prev_seed_digits': prev_prev_seed_digits,
        'prev_prev_seed_digits_list': prev_prev_seed_digits,
        'prev_pattern': prev_pattern,

        'new_seed_digits': set(seed_list) - set(prev_seed_digits),
        'seed_counts': Counter(seed_list),
        'common_to_both': set(seed_list) & set(prev_seed_digits),

        'hot_digits_10': hot,
        'cold_digits_10': cold,
        'hot_digits_20': hot,  # back-compat mapped to last-10
        'cold_digits_20': cold, # back-compat mapped to last-10
        'hot_digits': sorted(hot),
        'cold_digits': sorted(cold),
        'due_digits': sorted(due),
        'due_digits_2': due,

        'mirror': MIRROR,
        'vtrac': VTRAC,
        'any': any, 'all': all, 'len': len, 'sum': sum, 'max': max, 'min': min, 'set': set, 'sorted': sorted,
        'Counter': Counter,
    }
    return env

# ========================
# Filter Evaluation
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
    for _, r in df.iterrows():
        out.append(FilterDef(
            str(r['id']).strip(),
            str(r['name']).strip(),
            bool(r['enabled']),
            str(r['applicable_if']).strip(),
            str(r['expression']).strip(),
        ))
    return out


def safe_eval(expr: str, env: Dict[str, object]) -> bool:
    if not expr:
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

# ========================
# Historical Risk Scoring
# ========================

def current_bucket(env: Dict[str, object]) -> Tuple[str,str,str,str]:
    seed_sum_cat = env['seed_sum_category']
    seed_list = env['seed_digits_list']
    struct = classify_structure(seed_list)
    sb = spread_band(env['spread_seed'])
    parity_major = 'even>=3' if sum(1 for d in seed_list if d%2==0) >= 3 else 'even<=2'
    return seed_sum_cat, struct, sb, parity_major


def historical_risk_for_applicable(filters: List[FilterDef], winners: List[str], idx_now: int) -> Tuple[Dict[str,float], Dict[Tuple[str,str], float]]:
    """Return (single_failure_rate, pair_risk) for the current bucket based on all past draws in same bucket."""
    env_now = build_env_for_draw(idx_now, winners)
    bucket = current_bucket(env_now)
    # Only consider historical indices that share bucket
    indices = []
    for i in range(1, idx_now):
        env_i = build_env_for_draw(i, winners)
        if current_bucket(env_i) == bucket:
            indices.append(i)
    # Tally per filter
    app_c = Counter()
    blk_c = Counter()
    # Pair co-occurrence (both block winner)
    coapp = Counter()  # pair co-applicable (both applicable) — optional
    both_block = Counter()  # pair where both block winner

    for i in indices:
        env_i = build_env_for_draw(i, winners)
        # For the true winner at i, a filter "blocks" if expression True.
        blocked = []
        applicable = []
        for f in filters:
            if not f.enabled: 
                continue
            if safe_eval(f.applicable_if, env_i):
                applicable.append(f.fid)
                app_c[f.fid] += 1
                if safe_eval(f.expression, env_i):
                    blk_c[f.fid] += 1
                    blocked.append(f.fid)
        # Pair stats
        for a, b in it.combinations(sorted(applicable), 2):
            coapp[(a,b)] += 1
        for a, b in it.combinations(sorted(blocked), 2):
            both_block[(a,b)] += 1

    single_failure_rate = {fid: (blk_c[fid]/app_c[fid]) if app_c[fid]>0 else 0.0 for fid in app_c}
    pair_risk = {}
    for (a,b), n in coapp.items():
        bb = both_block.get((a,b), 0)
        pair_risk[(a,b)] = bb / n if n>0 else 0.0
    return single_failure_rate, pair_risk

# ========================
# Today: Applicable Filters & Ranking
# ========================

def today_applicable_filters(filters: List[FilterDef], winners: List[str]) -> Tuple[int, Dict[str,FilterDef], Dict[str,object]]:
    """Return (idx_now, applicable_by_id, env_now). Uses last row as 'today'."""
    if len(winners) < 2:
        raise SystemExit("Need at least 2 winners.")
    idx_now = len(winners) - 1
    env_now = build_env_for_draw(idx_now, winners)
    app = {}
    for f in filters:
        if not f.enabled:
            continue
        if safe_eval(f.applicable_if, env_now):
            app[f.fid] = f
    return idx_now, app, env_now

# ========================
# Pool Reduction (optional)
# ========================

def load_pool(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = 'combo' if 'combo' in df.columns else ('Result' if 'Result' in df.columns else None)
    if not col:
        raise ValueError("Pool file must have 'combo' or 'Result' column.")
    vals = df[col].astype(str).str.replace("\\D","", regex=True).str.zfill(5)
    vals = vals[vals.str.fullmatch(r"\d{5}")]
    return vals.tolist()


def apply_filter_to_pool(f: FilterDef, base_env: Dict[str,object], pool: List[str]) -> Tuple[List[str], int]:
    keep = []
    elim = 0
    for s in pool:
        clist = sorted(digits_of(s))
        env = dict(base_env)
        env.update({
            'combo': s,
            'combo_digits': set(clist),
            'combo_digits_list': clist,
            'combo_sum': sum(clist),
            'combo_sum_cat': sum_category(sum(clist)),
            'combo_sum_category': sum_category(sum(clist)),
            'spread_combo': max(clist) - min(clist),
            'seed_vtracs': set(VTRAC[d] for d in base_env['seed_digits_list']),
            'combo_vtracs': set(VTRAC[d] for d in clist),
        })
        # Apply app & expression to this combo
        if safe_eval(f.applicable_if, env) and safe_eval(f.expression, env):
            elim += 1
        else:
            keep.append(s)
    return keep, elim

# ========================
# Main
# ========================

def main():
    winners = load_winners(WINNERS_CSV)
    fdf = load_filters(FILTERS_CSV)
    filters = parse_filters(fdf)

    idx_now, applicable, env_now = today_applicable_filters(filters, winners)

    # Historical risk in current bucket
    single_fail, pair_risk = historical_risk_for_applicable(list(applicable.values()), winners, idx_now)

    # Rank applicable filters by (risk asc, support desc)
    support = {fid: 0 for fid in applicable}
    # Recompute support in-bucket
    env_now_bucket = current_bucket(env_now)
    for i in range(1, idx_now):
        if current_bucket(build_env_for_draw(i, winners)) != env_now_bucket:
            continue
        for fid, f in applicable.items():
            if safe_eval(f.applicable_if, build_env_for_draw(i, winners)):
                support[fid] = support.get(fid, 0) + 1

    ranked = sorted(applicable.values(), key=lambda f: (single_fail.get(f.fid, 1.0), -support.get(f.fid,0), f.fid))

    # Pair conflicts (high-risk pairs among today's applicable)
    avoid_rows = []
    for a, b in it.combinations(sorted(applicable.keys()), 2):
        pr = pair_risk.get((a,b)) or pair_risk.get((b,a)) or 0.0
        # Only record if we have at least some evidence
        if pr > 0:
            # We don't have co_app counts here; recompute quickly
            co_app = 0
            both_blk = 0
            env_now_bucket = current_bucket(env_now)
            for i in range(1, idx_now):
                if current_bucket(build_env_for_draw(i, winners)) != env_now_bucket:
                    continue
                env_i = build_env_for_draw(i, winners)
                ai = safe_eval(applicable[a].applicable_if, env_i)
                bi = safe_eval(applicable[b].applicable_if, env_i)
                if ai and bi:
                    co_app += 1
                    ab = safe_eval(applicable[a].expression, env_i) and safe_eval(applicable[b].expression, env_i)
                    if ab:
                        both_blk += 1
            if co_app>0:
                avoid_rows.append({'filter_id_1':a,'filter_id_2':b,'co_applicable_n':co_app,'both_blocked_n':both_blk,'pair_risk':round(both_blk/max(co_app,1),6)})
    pd.DataFrame(avoid_rows).sort_values(['pair_risk','co_applicable_n'], ascending=[False,False]).to_csv(OUTPUT_DIR/'avoid_pairs.csv', index=False)

    # Build recommender sequence
    seq_rows = []
    base_env = env_now
    pool = load_pool(TODAY_POOL_CSV) if TODAY_POOL_CSV else None
    remaining = len(pool) if pool else None

    applied = []
    for step, f in enumerate(ranked, start=1):
        fid = f.fid
        est_risk = round(single_fail.get(fid, 1.0), 6)
        conflicts = []
        for other in applicable:
            if other == fid: 
                continue
            pr = pair_risk.get((fid,other)) or pair_risk.get((other,fid)) or 0.0
            if pr > 0:
                conflicts.append(other)
        conflicts_str = ",".join(sorted(conflicts))

        eliminated_now = None
        skipped_reason = None
        if pool is not None:
            new_pool, elim = apply_filter_to_pool(f, base_env, pool)
            # winner-preserving guard (historical runs)
            if ALWAYS_KEEP_WINNER and winner_today and (winner_today not in new_pool):
                skipped_reason = 'would_remove_winner'
                seq_rows.append({'step':step,'filter_id':fid,'name':f.name,'est_risk':est_risk,'est_pair_conflicts':conflicts_str,'eliminated_now':0,'remaining':remaining,'skipped_reason':skipped_reason})
                continue
            if elim == 0:
                seq_rows.append({'step':step,'filter_id':fid,'name':f.name,'est_risk':est_risk,'est_pair_conflicts':conflicts_str,'eliminated_now':0,'remaining':remaining,'skipped_reason':'no_reduction'})
                continue
            # accept
            pool = new_pool
            remaining = len(pool)
            eliminated_now = elim
            seq_rows.append({'step':step,'filter_id':fid,'name':f.name,'est_risk':est_risk,'est_pair_conflicts':conflicts_str,'eliminated_now':elim,'remaining':remaining,'skipped_reason':None})
            # stop early if we do not minimize beyond target
            if (remaining <= TARGET_MAX) and (not MINIMIZE_BEYOND_TARGET):
                break
        else:
            # No pool: record order only
            seq_rows.append({'step':step,'filter_id':fid,'name':f.name,'est_risk':est_risk,'est_pair_conflicts':conflicts_str,'eliminated_now':None,'remaining':None,'skipped_reason':None})

    pd.DataFrame(seq_rows).to_csv(OUTPUT_DIR/'recommender_sequence.csv', index=False)

    # Also dump a compact log for reduction if pool provided
    if TODAY_POOL_CSV:
        pd.DataFrame([r for r in seq_rows if r['eliminated_now'] is not None]).to_csv(OUTPUT_DIR/'pool_reduction_log.csv', index=False)

    # One‑pager (if pool provided)
    if TODAY_POOL_CSV:
        seed_list = base_env['seed_digits_list']
        parity_major = 'even>=3' if sum(1 for d in seed_list if d%2==0) >= 3 else 'even<=2'
        applied_steps = [r for r in seq_rows if r['eliminated_now'] not in (None,0)]
        skipped_steps = [r for r in seq_rows if r.get('skipped_reason')]
        final_remaining = remaining
        winner_kept = (winner_today in pool) if pool is not None else True

        # Load avoid pairs
        avoid_path = OUTPUT_DIR/'avoid_pairs.csv'
        avoid_df = pd.read_csv(avoid_path) if avoid_path.exists() else pd.DataFrame()

        # --- Build Markdown one‑pager ---
        md_lines = []
        md_lines.append("# DC5 Recommender — Today")
        md_lines.append("**Seed:** `{}`  |  **Sum:** {} ({})  |  **Structure:** {}  |  **Spread:** {}  |  **Parity:** {}\n".format(base_env['seed'], base_env['seed_sum'], base_env['seed_sum_category'], classify_structure(seed_list), base_env['spread_seed'], parity_major))
        md_lines.append("**Hot:** {}  |  **Cold:** {}  |  **Due:** {}\n".format(sorted(base_env['hot_digits']), sorted(base_env['cold_digits']), sorted(base_env['due_digits'])))
        md_lines.append("")
        md_lines.append("**Applicable filters now:** {}  |  **Target:** < {}  |  **Winner preserved:** {}\n".format(len(applicable), TARGET_MAX+1, '✅ YES' if winner_kept else '❌ NO'))
        md_lines.append("")
        md_lines.append("## Apply in this order (winner‑preserving)
")
        if applied_steps:
            for r in applied_steps:
                md_lines.append(f"- Step {r['step']}: **{r['filter_id']}** — {r['name']}  · eliminated **{r['eliminated_now']}** → remaining **{r['remaining']}**")
        else:
            md_lines.append("- No safe reduction steps available (either no pool or all steps would remove winner).")
        if skipped_steps:
            md_lines.append("
**Skipped:**")
            for r in skipped_steps:
                md_lines.append(f"- **{r['filter_id']}** — {r['name']}  ({r['skipped_reason']})")
        md_lines.append("
## Avoid combining (today’s bucket)
")
        if not avoid_df.empty:
            for _, row in avoid_df.head(12).iterrows():
                md_lines.append(f"- **{row['filter_id_1']} + {row['filter_id_2']}**  · pair_risk={row['pair_risk']}  (both_blocked/CoApp={row['both_blocked_n']}/{row['co_applicable_n']})")
        else:
            md_lines.append("- No high‑risk pairs observed in this bucket.")
        md_lines.append(f"
**Final pool size:** **{final_remaining}** (target < {TARGET_MAX+1})  |  **Winner present:** {'✅ YES' if winner_kept else '❌ NO'}
")
        (OUTPUT_DIR/'one_pager.md').write_text("
".join(md_lines), encoding='utf-8')

        # --- Build Styled HTML one‑pager (print‑ready) ---
        from datetime import datetime
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Prepare small HTML tables
        import html
        def df_to_html_table(df: pd.DataFrame, columns: list, empty_msg: str, limit: int = None):
            if df is None or df.empty:
                return f"<p class='muted'>{html.escape(empty_msg)}</p>"
            use = df[columns].copy()
            if limit:
                use = use.head(limit)
            return use.to_html(index=False, classes='table', border=0, escape=True)

        applied_df = pd.DataFrame(applied_steps)
        skipped_df = pd.DataFrame(skipped_steps)

        applied_html = df_to_html_table(
            applied_df,
            ['step','filter_id','name','eliminated_now','remaining'],
            'No safe reduction steps available.',
            limit=50,
        )
        skipped_html = df_to_html_table(
            skipped_df,
            ['filter_id','name','skipped_reason'],
            'No filters were skipped.',
            limit=50,
        )
        avoid_html = df_to_html_table(
            avoid_df.sort_values(['pair_risk','co_applicable_n'], ascending=[False,False]) if not avoid_df.empty else avoid_df,
            ['filter_id_1','filter_id_2','pair_risk','both_blocked_n','co_applicable_n'],
            'No high‑risk pairs observed in this bucket.',
            limit=12,
        )

        hot = ', '.join(str(x) for x in sorted(base_env['hot_digits']))
        cold = ', '.join(str(x) for x in sorted(base_env['cold_digits']))
        due = ', '.join(str(x) for x in sorted(base_env['due_digits']))

        html_doc = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8' />
<title>DC5 Recommender — One‑Pager</title>
<style>
  :root {{ --fg:#0f172a; --sub:#334155; --muted:#64748b; --accent:#2563eb; --ok:#16a34a; --bad:#dc2626; --bg:#ffffff; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin:0; padding:0; background:var(--bg); color:var(--fg); font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Helvetica, Arial, 'Apple Color Emoji','Segoe UI Emoji'; }}
  .wrap {{ max-width: 900px; margin: 28px auto 64px; padding: 0 20px; }}
  h1 {{ font-size: 24px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 24px 0 8px; }}
  .meta {{ color: var(--muted); margin-bottom: 18px; }}
  .badges span {{ display:inline-block; padding:6px 10px; border-radius: 999px; font-weight:600; margin-right:8px; background:#eef2ff; color:#3730a3; }}
  .badges .ok {{ background:#ecfdf5; color:#065f46; }}
  .badges .warn {{ background:#fff7ed; color:#9a3412; }}
  .badges .bad {{ background:#fef2f2; color:#991b1b; }}
  .grid {{ display:grid; grid-template-columns: 1fr; gap:16px; }}
  .card {{ border:1px solid #e2e8f0; border-radius:16px; padding:16px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
  .title {{ font-weight:700; margin-bottom:8px; }}
  .muted {{ color: var(--muted); }}
  .table {{ width:100%; border-collapse: collapse; }}
  .table th, .table td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e5e7eb; }}
  .table th {{ background:#f8fafc; font-weight:700; }}
  .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 0; }}
  .kpi {{ background:#f8fafc; padding:10px 12px; border-radius:12px; border:1px solid #e2e8f0; }}
  @media print {{
    .wrap {{ max-width: auto; margin: 0; padding: 0; }}
    .card {{ break-inside: avoid; page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <div class='wrap'>
    <h1>DC5 Recommender — One‑Pager</h1>
    <div class='meta'>Generated {generated_at}</div>

    <div class='card'>
      <div class='title'>Seed Snapshot</div>
      <div>Seed <strong>{html.escape(str(base_env['seed']))}</strong> · Sum <strong>{base_env['seed_sum']}</strong> ({html.escape(base_env['seed_sum_category'])}) · Structure <strong>{html.escape(classify_structure(seed_list))}</strong> · Spread <strong>{base_env['spread_seed']}</strong> · Parity <strong>{parity_major}</strong></div>
      <div class='kpis'>
        <div class='kpi'>Hot: {html.escape(hot)}</div>
        <div class='kpi'>Cold: {html.escape(cold)}</div>
        <div class='kpi'>Due: {html.escape(due)}</div>
      </div>
      <div class='badges' style='margin-top:8px;'>
        <span>Applicable: {len(applicable)}</span>
        <span class='{('ok' if winner_kept else 'bad')}'>Winner: {'KEPT' if winner_kept else 'REMOVED'}</span>
        <span>Target: &lt; {TARGET_MAX+1}</span>
        <span>Final size: {final_remaining if final_remaining is not None else '—'}</span>
      </div>
    </div>

    <div class='grid'>
      <div class='card'>
        <div class='title'>Apply in this order (winner‑preserving)</div>
        {applied_html}
      </div>

      <div class='card'>
        <div class='title'>Skipped filters</div>
        {skipped_html}
      </div>

      <div class='card'>
        <div class='title'>Avoid combining (today’s bucket)</div>
        {avoid_html}
      </div>
    </div>

    <div class='card' style='margin-top:16px;'>
      <div class='title'>Notes</div>
      <ul class='muted'>
        <li>Uses only filters your app marks as <em>applicable now</em> and mirrors its env naming.</li>
        <li>Historical risk computed within today’s seed bucket (sum category, structure, spread band, parity majority).</li>
        <li>Winner‑preserving mode: any step that removes the known winner is skipped.</li>
      </ul>
    </div>
  </div>
</body>
</html>"""
        (OUTPUT_DIR/'one_pager.html').write_text(html_doc, encoding='utf-8')

    print("Done. Wrote: recommender_sequence.csv, avoid_pairs.csv, one_pager.md / one_pager.html, and pool_reduction_log.csv (if pool provided).")


if __name__ == '__main__':
    main()

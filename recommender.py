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
    prev_prev_seed = winners[idx-2] if idx >= 2 else None
    prev_seed_digits = digits_of(prev_seed)
    prev_prev_seed_digits = digits_of(prev_prev_seed) if prev_prev_seed is not None else []

    # history up to seed
    history_digits = [digits_of(s) for s in winners[:idx]]
    hot, cold, due = hot_cold_due(history_digits, k_hotcold=10)

    # prev_pattern like your app (SumCat, Parity) for prev_prev, prev, seed
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
        "prev_seed_digits_list": prev_seed_digits,
        "prev_prev_seed": prev_prev_seed,
        "prev_prev_seed_digits": prev_prev_seed_digits,
        "prev_prev_seed_digits_list": prev_prev_seed_digits,
        "prev_pattern": prev_pattern,

        "new_seed_digits": set(seed_list) - set(prev_seed_digits),
        "seed_counts": Counter(seed_list),
        "common_to_both": set(seed_list) & set(prev_seed_digits),

        "hot_digits_10": hot,
        "cold_digits_10": cold,
        "hot_digits_20": hot,   # back-compat
        "cold_digits_20": cold, # back-compat
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
    if not expr:
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        # treat errors as not-applicable / not-eliminating
        return False

# =========================
# Risk using recent, similar context
# =========================
def historical_risk_for_applicable(
    filters: List[FilterDef],
    winners: List[str],
    idx_now: int,
    max_draws: Optional[int] = None,
    max_bucket_matches: Optional[int] = None,
    decay_half_life: Optional[int] = None,
):
    """
    max_draws: last N draws overall (0/None = all)
    max_bucket_matches: last K occurrences of this bucket (0/None = all)
    decay_half_life: exponential decay (in draws); None/0 = no weighting
    """
    env_now = build_env_for_draw(idx_now, winners)
    bucket = current_bucket(env_now)

    start_idx = max(1, (idx_now - (max_draws or (idx_now - 1))))
    candidate_indices = [
        i for i in range(start_idx, idx_now)
        if current_bucket(build_env_for_draw(i, winners)) == bucket
    ]
    if max_bucket_matches:
        candidate_indices = candidate_indices[-max_bucket_matches:]

    def w(i: int) -> float:
        if not decay_half_life:
            return 1.0
        age = (idx_now - i)  # draws ago
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

    # also return the list of indices used, so we can compute "support" consistently
    return single_failure_rate, pair_risk, candidate_indices

# =========================
# Core logic
# =========================
def today_applicable_filters(filters: List[FilterDef], winners: List[str]):
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

def load_pool(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = "combo" if "combo" in df.columns else ("Result" if "Result" in df.columns else None)
    if not col:
        raise ValueError("Pool file must have 'combo' or 'Result' column.")
    vals = df[col].astype(str).str.replace(r"\D","", regex=True).str.zfill(5)
    vals = vals[vals.str.fullmatch(r"\d{5}")]
    return vals.tolist()

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

# =========================
# Public entry
# =========================
def main(
    winners_csv: Optional[str] = None,
    filters_csv: Optional[str] = None,
    today_pool_csv: Optional[str] = None,
    target_max: int = TARGET_MAX,
    always_keep_winner: bool = ALWAYS_KEEP_WINNER,
    minimize_beyond_target: bool = MINIMIZE_BEYOND_TARGET,
    force_keep_combo: Optional[str] = None,       # NEW
    override_seed: Optional[str] = None,          # NEW
    override_prev: Optional[str] = None,          # NEW
    override_prevprev: Optional[str] = None,      # NEW
    max_draws: Optional[int] = None,              # NEW
    max_bucket_matches: Optional[int] = None,     # NEW
    decay_half_life: Optional[int] = None,        # NEW
    applicable_only: Optional[List[str]] = None,  # NEW - IDs pasted from main app
):
    # Resolve inputs (allow override)
    winners_path = winners_csv or WINNERS_CSV
    filters_path = filters_csv or FILTERS_CSV

    # Friendly existence check
    present = ", ".join(sorted(p.name for p in Path(".").glob("*.csv")))
    if not Path(winners_path).exists():
        raise FileNotFoundError(f"Missing {winners_path}. CSVs here: {present}")
    if not Path(filters_path).exists():
        raise FileNotFoundError(f"Missing {filters_path}. CSVs here: {present}")

    winners = load_winners(winners_path)
    filters = load_filters(filters_path)

    # ---- Optional: override today's seed/prevs to match what you typed in the main app
    if override_seed:
        seed = str(override_seed).strip().zfill(5)
        prev = (override_prev or winners[-1]).strip().zfill(5)
        prevprev = (override_prevprev or winners[-2]).strip().zfill(5)
        winners = winners[:-2] + [prevprev, prev, seed, "00000"]  # last is dummy "today"

    # Compute applicable for "today"
    idx_now, applicable, env_now = today_applicable_filters(filters, winners)

    # Restrict to "applicable only" list if provided
    if applicable_only:
        applicable_only = {fid.strip() for fid in applicable_only}
        applicable = {fid: f for fid, f in applicable.items() if fid in applicable_only}

    # Historical risks with recency controls
    single_fail, pair_risk, idx_used = historical_risk_for_applicable(
        list(applicable.values()), winners, idx_now,
        max_draws=max_draws, max_bucket_matches=max_bucket_matches,
        decay_half_life=decay_half_life
    )

    # Support counts computed over the same indices we used for risk
    support = {fid: 0 for fid in applicable}
    for i in idx_used:
        env_i = build_env_for_draw(i, winners)
        for fid, f in applicable.items():
            if safe_eval(f.applicable_if, env_i):
                support[fid] = support.get(fid, 0) + 1

    # Rank applicable by safety (lower risk, higher support)
    ranked = sorted(applicable.values(), key=lambda f: (single_fail.get(f.fid, 1.0), -support.get(f.fid,0), f.fid))

    # Pair conflicts for applicable (simple frequency) — SAFE FOR EMPTY
    avoid_rows = []
    for a, b in it.combinations(sorted(applicable.keys()), 2):
        pr = pair_risk.get((a,b)) or pair_risk.get((b,a)) or 0.0
        if pr > 0:
            co_app = 0
            both_blk = 0
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
                    "filter_id_1": a,
                    "filter_id_2": b,
                    "co_applicable_n": co_app,
                    "both_blocked_n": both_blk,
                    "pair_risk": round(both_blk / max(co_app, 1), 6)
                })
    avoid_df = pd.DataFrame(avoid_rows)
    if not avoid_df.empty:
        avoid_df = avoid_df.sort_values(["pair_risk","co_applicable_n"], ascending=[False,False])
    avoid_df.to_csv(OUTPUT_DIR / "avoid_pairs.csv", index=False)

    # ---- Build DO NOT APPLY / APPLY LATE / SAFE lists for today's applicable set
    rows = []
    for fid, f in applicable.items():
        risk = float(single_fail.get(fid, 0.0))
        sup  = int(support.get(fid, 0))
        if risk >= 0.20 and sup >= 6:
            tier = "DO_NOT_APPLY"
        elif risk >= 0.10:
            tier = "APPLY_LATE"
        else:
            tier = "SAFE"
        rows.append({"filter_id": fid, "name": f.name, "risk": round(risk, 4), "support": sup, "tier": tier})

    tier_order = {"DO_NOT_APPLY": 0, "APPLY_LATE": 1, "SAFE": 2}
    do_not_df = pd.DataFrame(rows)
    if not do_not_df.empty:
        do_not_df = do_not_df.sort_values(
            by=["tier","risk","support","filter_id"],
            ascending=[True, False, False, True],
            key=lambda s: s.map(tier_order) if s.name == "tier" else None
        )
        do_not_df.to_csv(OUTPUT_DIR / "do_not_apply.csv", index=False)

    # Reduction (optional if pool provided)
    seq_rows = []
    base_env = env_now
    pool = load_pool(today_pool_csv) if today_pool_csv else None
    remaining = len(pool) if pool else None

    # pick the "keep" combo: forced, else last real winner, else none
    winner_today = None
    if force_keep_combo:
        winner_today = str(force_keep_combo).strip().replace(" ", "")
    elif pool is not None:
        winner_today = (winners[idx_now] if winners[idx_now].isdigit() else None)

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

        if pool is not None:
            new_pool, elim = apply_filter_to_pool(f, base_env, pool)
            if always_keep_winner and winner_today and (winner_today not in new_pool):
                seq_rows.append({
                    "step": step, "filter_id": fid, "name": f.name,
                    "est_risk": est_risk, "est_support": support.get(fid, 0),
                    "est_pair_conflicts": conflicts_str,
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "would_remove_winner"
                })
                continue
            if elim == 0:
                seq_rows.append({
                    "step": step, "filter_id": fid, "name": f.name,
                    "est_risk": est_risk, "est_support": support.get(fid, 0),
                    "est_pair_conflicts": conflicts_str,
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "no_reduction"
                })
                continue
            # accept
            pool = new_pool
            remaining = len(pool)
            seq_rows.append({
                "step": step, "filter_id": fid, "name": f.name,
                "est_risk": est_risk, "est_support": support.get(fid, 0),
                "est_pair_conflicts": conflicts_str,
                "eliminated_now": elim, "remaining": remaining,
                "skipped_reason": None
            })
            if (remaining <= target_max) and (not minimize_beyond_target):
                break
        else:
            # record ordering only
            seq_rows.append({
                "step": step, "filter_id": fid, "name": f.name,
                "est_risk": est_risk, "est_support": support.get(fid, 0),
                "est_pair_conflicts": conflicts_str,
                "eliminated_now": None, "remaining": None,
                "skipped_reason": None
            })

    pd.DataFrame(seq_rows).to_csv(OUTPUT_DIR / "recommender_sequence.csv", index=False)
    if today_pool_csv:
        pd.DataFrame([r for r in seq_rows if r["eliminated_now"] not in (None,0)]).to_csv(
            OUTPUT_DIR / "pool_reduction_log.csv", index=False
        )

    # ======== One-pager (Markdown) ========
    if today_pool_csv:
        seed_list = base_env['seed_digits_list']
        parity_major = "even>=3" if sum(1 for d in seed_list if d%2==0) >= 3 else "even<=2"
        applied_steps = [r for r in seq_rows if r['eliminated_now'] not in (None,0)]
        skipped_steps = [r for r in seq_rows if r.get('skipped_reason')]
        final_remaining = remaining
        winner_kept = (winner_today in pool) if pool is not None else True

        md_lines: List[str] = []
        md_lines.append("# DC5 Recommender — Today")
        md_lines.append("**Seed:** `{}`  |  **Sum:** {} ({})  |  **Structure:** {}  |  **Spread:** {}  |  **Parity:** {}\n".format(
            base_env['seed'], base_env['seed_sum'], base_env['seed_sum_category'],
            classify_structure(seed_list), base_env['spread_seed'], parity_major
        ))
        md_lines.append("**Hot:** {}  |  **Cold:** {}  |  **Due:** {}\n".format(
            sorted(base_env['hot_digits']), sorted(base_env['cold_digits']), sorted(base_env['due_digits'])
        ))
        md_lines.append("")
        md_lines.append("**Applicable filters now:** {}  |  **Target:** < {}  |  **Winner preserved:** {}\n".format(
            len(applicable), target_max+1, '✅ YES' if winner_kept else '❌ NO'
        ))
        md_lines.append("")
        md_lines.append("## Apply in this order (winner-preserving)\n")
        if applied_steps:
            for r in applied_steps:
                md_lines.append("- Step {}: **{}** — {}  · eliminated **{}** → remaining **{}**".format(
                    r['step'], r['filter_id'], r['name'], r['eliminated_now'], r['remaining']
                ))
        else:
            md_lines.append("- No safe reduction steps available (either no pool or all steps would remove winner).")
        if skipped_steps:
            md_lines.append("\n**Skipped:**")
            for r in skipped_steps:
                md_lines.append("- **{}** — {}  ({})".format(r['filter_id'], r['name'], r['skipped_reason']))
        md_lines.append("\n## Avoid combining (today’s bucket)\n")
        avoid_path = OUTPUT_DIR / "avoid_pairs.csv"
        avoid_df2 = pd.read_csv(avoid_path) if avoid_path.exists() else pd.DataFrame()
        if not avoid_df2.empty:
            for _, row in avoid_df2.head(12).iterrows():
                md_lines.append("- **{} + {}**  · pair_risk={}  (both_blocked/CoApp={}/{})".format(
                    row['filter_id_1'], row['filter_id_2'], row['pair_risk'], row['both_blocked_n'], row['co_applicable_n']
                ))
        else:
            md_lines.append("- No high-risk pairs observed in this bucket.")
        md_lines.append("**Final pool size:** **{}** (target < {})  |  **Winner present:** {}\n".format(
            final_remaining, target_max+1, '✅ YES' if winner_kept else '❌ NO'
        ))
        (OUTPUT_DIR / "one_pager.md").write_text("\n".join(md_lines), encoding="utf-8")

        # ======== One-pager (HTML) ========
        HTML_CSS = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>DC5 Recommender — One-Pager</title>
<style>
:root { --fg:#0f172a; --muted:#64748b; --bg:#fff; }
body { margin:0; font:14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Helvetica, Arial; color:var(--fg); background:var(--bg);}
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
</style></head><body><div class='wrap'>"""
        seed_list = base_env['seed_digits_list']
        parity_major = "even>=3" if sum(1 for d in seed_list if d%2==0) >= 3 else "even<=2"
        winner_kept = (winner_today in pool) if pool is not None else True
        snap = (
            "<h1>DC5 Recommender — One-Pager</h1>"
            "<div class='card'><div><b>Seed:</b> {} &nbsp;·&nbsp; <b>Sum:</b> {} ({}) &nbsp;·&nbsp; "
            "<b>Structure:</b> {} &nbsp;·&nbsp; <b>Spread:</b> {} &nbsp;·&nbsp; <b>Parity:</b> {}</div>"
            "<div style='margin-top:8px'>"
            "<span class='kpi'>Hot: {}</span>"
            "<span class='kpi'>Cold: {}</span>"
            "<span class='kpi'>Due: {}</span>"
            "</div>"
            "<div style='margin-top:8px'>"
            "<span class='badge'>Applicable: {}</span>"
            "<span class='badge {}'>{}</span>"
            "<span class='badge'>Target: &lt; {}</span>"
            "<span class='badge'>Final size: {}</span>"
            "</div></div>"
        ).format(
            html.escape(str(base_env['seed'])),
            base_env['seed_sum'], html.escape(base_env['seed_sum_category']),
            html.escape(classify_structure(seed_list)),
            base_env['spread_seed'],
            parity_major,
            ", ".join(str(x) for x in sorted(base_env['hot_digits'])),
            ", ".join(str(x) for x in sorted(base_env['cold_digits'])),
            ", ".join(str(x) for x in sorted(base_env['due_digits'])),
            len(applicable),
            "ok" if winner_kept else "bad", "Winner: " + ("KEPT" if winner_kept else "REMOVED"),
            target_max+1,
            final_remaining if final_remaining is not None else "—"
        )

        def df_to_html_table(df: pd.DataFrame, columns: list, empty_msg: str, limit: int = None):
            if df is None or df.empty:
                return "<p class='muted'>{}</p>".format(html.escape(empty_msg))
            use = df[columns].copy()
            if limit:
                use = use.head(limit)
            return use.to_html(index=False, classes="table", border=0, escape=True)

        applied_steps = [r for r in seq_rows if r['eliminated_now'] not in (None,0)]
        skipped_steps = [r for r in seq_rows if r.get('skipped_reason')]
        avoid_path = OUTPUT_DIR / "avoid_pairs.csv"
        avoid_df2 = pd.read_csv(avoid_path) if avoid_path.exists() else pd.DataFrame()
        avoid_sorted = avoid_df2.sort_values(["pair_risk","co_applicable_n"], ascending=[False,False]) if not avoid_df2.empty else avoid_df2

        applied_html = df_to_html_table(pd.DataFrame(applied_steps),
                                        ["step","filter_id","name","eliminated_now","remaining"],
                                        "No safe reduction steps.", 50)
        skipped_html = df_to_html_table(pd.DataFrame(skipped_steps),
                                        ["filter_id","name","skipped_reason"],
                                        "No filters were skipped.", 50)
        avoid_html   = df_to_html_table(avoid_sorted,
                                        ["filter_id_1","filter_id_2","pair_risk","both_blocked_n","co_applicable_n"],
                                        "No high-risk pairs in this bucket.", 12)

        html_doc = HTML_CSS + snap + \
            "<div class='card'><h2>Apply in this order (winner-preserving)</h2>{}</div>".format(applied_html) + \
            "<div class='card'><h2>Skipped filters</h2>{}</div>".format(skipped_html) + \
            "<div class='card'><h2>Avoid combining (today’s bucket)</h2>{}</div>".format(avoid_html) + \
            "</div></body></html>"

        (OUTPUT_DIR / "one_pager.html").write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    main()

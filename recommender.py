# recommender.py (historical trainer + by-case recommender)
# Updated with case-by-case failure tracking, richer signatures, neighbor fallback, and aggressiveness ranking.

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Iterable, Union
from pathlib import Path
import itertools as it
import hashlib
import json

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

CASE_HISTORY = OUTPUT_DIR / "case_history.csv"
CASE_STATS = OUTPUT_DIR / "case_filter_stats.csv"
CASE_META = OUTPUT_DIR / "case_meta.json"

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
# Case signature
# =========================
def case_signature(seed: str, applicable: List[str], prev_aggr_rank: Optional[Dict[str,int]]=None) -> Dict[str,object]:
    digs = digits_of(seed)
    s = sum(digs)
    struct = tuple(Counter(digs).values())
    spread = max(digs)-min(digs)
    sig = {
        "sum_cat": sum_category(s),
        "struct": str(sorted(struct)),
        "spread_band": int(spread/3),
        "parity_major": "even" if sum(d%2 for d in digs)<=2 else "odd",
        "hot": set(d for d in digs if d in [8,9,2]),
        "cold": set(d for d in digs if d in [7,3,6]),
        "due": set(d for d in digs if d in [0,1,4,5]),
        "applicable": sorted(applicable),
        "prev_rank": prev_aggr_rank or {}
    }
    sig_key = f"{sig['sum_cat']}|{sig['struct']}|{sig['spread_band']}|{sig['parity_major']}|{','.join(sig['applicable'])}"
    sig["sig_key"] = sig_key
    return sig

# =========================
# Historical simulation
# =========================
def run_historical(winners_csv: str, filters_csv: str, last_n: Optional[int]=None):
    winners = load_winners(winners_csv)
    filters = load_filters(filters_csv)

    rows = []
    prev_ranks: Dict[str,int] = {}

    start = 1 if not last_n else max(1, len(winners)-last_n)
    for i in range(start, len(winners)):
        seed = winners[i-1]
        true_next = winners[i]
        applicable = [f for f in filters if f.enabled]
        sig = case_signature(seed,[f.fid for f in applicable],prev_ranks)
        eliminators = []
        for f in applicable:
            env = {"combo": true_next, "seed": seed}
            try:
                if not safe_eval(f.applicable_if, env):
                    continue
                if safe_eval(f.expression, env):
                    eliminators.append(f.fid)
            except Exception:
                continue
        rows.append({
            "draw_idx": i,
            "seed": seed,
            "winner": true_next,
            "sig_key": sig["sig_key"],
            "applicable": ",".join(sig["applicable"]),
            "eliminators": ",".join(eliminators)
        })
        # update prev ranks (aggressiveness = filter order by failure count)
        fail_ct = Counter(eliminators)
        prev_ranks = {fid:rank for rank,(fid,_) in enumerate(fail_ct.most_common(),1)}

    pd.DataFrame(rows).to_csv(CASE_HISTORY,index=False)

    # aggregate stats
    stat_map = defaultdict(lambda:{"n_app":0,"n_blk":0})
    for r in rows:
        for fid in r["applicable"].split(","):
            stat_map[(r["sig_key"],fid)]["n_app"]+=1
        for fid in r["eliminators"].split(",") if r["eliminators"] else []:
            stat_map[(r["sig_key"],fid)]["n_blk"]+=1
    stat_rows=[]
    for (sig_key,fid),vals in stat_map.items():
        n_app,n_blk=vals["n_app"],vals["n_blk"]
        stat_rows.append({"sig_key":sig_key,"filter_id":fid,"n_app":n_app,"n_blk":n_blk,"fail_rate":(n_blk/n_app if n_app else 0)})
    pd.DataFrame(stat_rows).to_csv(CASE_STATS,index=False)

    # meta
    meta={"hash":hashlib.md5(open(winners_csv,'rb').read()).hexdigest(),"n":len(winners)}
    json.dump(meta,open(CASE_META,"w"),indent=2)

# =========================
# Main
# =========================
def _load_case_tables(out_dir: Path = OUTPUT_DIR):
    stats_path = out_dir/"case_filter_stats.csv"
    hist_path  = out_dir/"case_history.csv"
    stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame(columns=["sig_key","filter_id","n_app","n_blk","fail_rate"]) 
    hist  = pd.read_csv(hist_path)  if hist_path.exists()  else pd.DataFrame(columns=["sig_key","applicable","eliminators"]) 
    return stats, hist


def _neighbor_fail_rates(sig_key: str, todays_app: List[str], stats: pd.DataFrame, hist: pd.DataFrame, k: int = 12, min_sim: float = 0.35):
    """Aggregate failure rates from nearest signatures by Jaccard over applicable sets."""
    if hist.empty or stats.empty:
        return {}
    # unique sig -> applicable set
    uniq = hist[["sig_key","applicable"]].drop_duplicates()
    uniq["app_set"] = uniq["applicable"].fillna("").apply(lambda s: set([t for t in str(s).split(",") if t]))
    A = set(todays_app)
    def jacc(s):
        B = s if isinstance(s,set) else set(s)
        if not A and not B: return 1.0
        if not A or not B: return 0.0
        return len(A & B) / max(1, len(A | B))
    uniq["sim"] = uniq["app_set"].apply(jacc)
    neigh = uniq.sort_values("sim", ascending=False).head(k)
    neigh = neigh[neigh["sim"] >= min_sim]
    if neigh.empty:
        return {}
    sel_keys = set(neigh["sig_key"].tolist())
    agg = stats[stats["sig_key"].isin(sel_keys)].groupby("filter_id")[['n_app','n_blk']].sum().reset_index()
    if agg.empty:
        return {}
    agg['fail_rate'] = agg.apply(lambda r: (r['n_blk']/r['n_app']) if r['n_app']>0 else 1.0, axis=1)
    return {row['filter_id']: float(row['fail_rate']) for _, row in agg.iterrows()}


def _pair_risk_by_case(todays_app: List[str], hist: pd.DataFrame, keys: List[str]):
    from collections import Counter
    if hist.empty or not keys:
        return {}
    rows = hist[hist['sig_key'].isin(keys)]
    if rows.empty:
        return {}
    Aset = set(todays_app)
    co = Counter(); both = Counter()
    for _, r in rows.iterrows():
        app = set([t for t in str(r.get('applicable','')).split(',') if t])
        elim = set([t for t in str(r.get('eliminators','')).split(',') if t])
        app_today = sorted(Aset & app)
        # co-app in this case window only
        for i in range(len(app_today)):
            for j in range(i+1, len(app_today)):
                a,b = app_today[i], app_today[j]
                co[(a,b)] += 1
                if a in elim and b in elim:
                    both[(a,b)] += 1
    out = {}
    for (a,b), n in co.items():
        bb = both.get((a,b), 0)
        out[(a,b)] = (bb / n) if n>0 else 0.0
    return out


def main(
    winners_csv: str = WINNERS_CSV,
    filters_csv: str = FILTERS_CSV,
    today_pool_csv: Optional[str] = TODAY_POOL_CSV,
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
    applicable_only: Optional[Union[str,Iterable[str]]] = None,
    train_mode: bool = False,
    incremental_last_n: int = 7,
    **_
):
    if train_mode:
        simulate_incremental(winners_csv, filters_csv, keep_last_n=incremental_last_n, out_dir=OUTPUT_DIR)
        return

    # --- Recommend path (case-aware) ---
    winners = load_winners(winners_csv)
    filters = load_filters(filters_csv)
    pool = load_pool(today_pool_csv) or []

    # Optional overrides to align with UI typing
    if override_seed:
        # Explicit seed triplet lets the app build the exact case for TODAY
        seed = str(override_seed).strip().zfill(5)
        prev = (override_prev or winners[-1]).strip().zfill(5)
        prevprev = (override_prevprev or winners[-2]).strip().zfill(5)
        winners = winners[:-2] + [prevprev, prev, seed, "00000"]

    if len(winners) < 2:
        raise SystemExit("Need at least 2 winners.")

    idx_now = len(winners) - 1
    env_now = build_env_for_draw(idx_now, winners)

    # Applicable today (optionally restricted by IDs pasted from UI)
    todays_app = {}
    ids_only = set(parse_applicable_only(applicable_only))
    for f in filters:
        if not f.enabled:
            continue
        if safe_eval(f.applicable_if, env_now):
            if ids_only and f.fid not in ids_only:
                continue
            todays_app[f.fid] = f

    applicable_ids_today = sorted(todays_app.keys())

    # Build today's signature
    sig = case_signature(env_now, applicable_ids_today)
    sig_key = signature_key(sig)

    # Load case tables
    stats, hist = _load_case_tables(OUTPUT_DIR)

    # Case failure rates (exact or neighbor)
    case_rates: Dict[str,float] = {}
    support_threshold = 6
    exact = stats[stats['sig_key'] == sig_key]
    if not exact.empty:
        exact_ok = exact[exact['n_app'] >= support_threshold]
        if not exact_ok.empty:
            case_rates = {r['filter_id']: float(r['fail_rate']) for _, r in exact_ok.iterrows()}
    if not case_rates:
        # neighbor fallback (more permissive to avoid defaulting)
        case_rates = _neighbor_fail_rates(sig_key, applicable_ids_today, stats, hist, k=25, min_sim=0.10)

    # Pair risk by case (exact or neighbor keys)
    neighbor_keys = []
    if not exact.empty:
        neighbor_keys = [sig_key]
    else:
        uniq = hist[['sig_key','applicable']].drop_duplicates()
        uniq['app_set'] = uniq['applicable'].fillna('').apply(lambda s: set([t for t in str(s).split(',') if t]))
        A = set(applicable_ids_today)
        uniq['sim'] = uniq['app_set'].apply(lambda B: (len(A & B)/max(1,len(A|B))) if (A or B) else 1.0)
        neighbor_keys = uniq.sort_values('sim', ascending=False).head(25)['sig_key'].tolist()
    pair_case = _pair_risk_by_case(applicable_ids_today, hist, neighbor_keys)

    # Winner-agnostic safety mode: if caller wants to preserve winner but we DON'T know it,
    # skip filters whose case-failure is high rather than risking removal of the unknown winner.
    winner_today = None
    if force_keep_combo:
        winner_today = str(force_keep_combo).strip().replace(" ", "")
    elif pool:
        wt = winners[idx_now]
        winner_today = wt if isinstance(wt,str) and wt.isdigit() else None

    agnostic_preserve = bool(always_keep_winner and not winner_today)
    risk_skip_threshold = 0.25  # skip filters with >=25% case fail rate when winner is unknown

    def risk_of(fid: str) -> float:
        return float(case_rates.get(fid, 0.5))

    # Rank by risk first; ties by ID (expected reduction will be captured during actual application)
    ranked = sorted(todays_app.values(), key=lambda f: (risk_of(f.fid), f.fid))

    base_env = env_now
    seq_rows = []
    remaining = len(pool)

    for step, f in enumerate(ranked, start=1):
        fid = f.fid
        est_risk = round(risk_of(fid), 6)
        conflicts = []
        for other in applicable_ids_today:
            if other == fid: continue
            pr = pair_case.get((min(fid,other), max(fid,other)), 0.0)
            if pr > 0:
                conflicts.append(other)
        conflicts_str = ",".join(sorted(conflicts))

        # Winner-agnostic guard: skip high-risk filters when we don't know the winner
        if agnostic_preserve and est_risk >= risk_skip_threshold:
            seq_rows.append({
                "step": step, "filter_id": fid, "name": f.name,
                "est_risk": est_risk, "est_pair_conflicts": conflicts_str,
                "eliminated_now": 0, "remaining": remaining,
                "skipped_reason": "risk_skip"
            })
            continue

        if pool:
            new_pool, elim = apply_filter_to_pool(f, base_env, pool)
            if always_keep_winner and winner_today and (winner_today not in new_pool):
                seq_rows.append({
                    "step": step, "filter_id": fid, "name": f.name,
                    "est_risk": est_risk, "est_pair_conflicts": conflicts_str,
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "would_remove_winner"
                })
                continue
            if elim == 0:
                seq_rows.append({
                    "step": step, "filter_id": fid, "name": f.name,
                    "est_risk": est_risk, "est_pair_conflicts": conflicts_str,
                    "eliminated_now": 0, "remaining": remaining,
                    "skipped_reason": "no_reduction"
                })
                continue
            pool = new_pool
            remaining = len(pool)
            seq_rows.append({
                "step": step, "filter_id": fid, "name": f.name,
                "est_risk": est_risk, "est_pair_conflicts": conflicts_str,
                "eliminated_now": elim, "remaining": remaining,
                "skipped_reason": None
            })
            if (remaining <= target_max) and (not minimize_beyond_target):
                break
        else:
            seq_rows.append({
                "step": step, "filter_id": fid, "name": f.name,
                "est_risk": est_risk, "est_pair_conflicts": conflicts_str,
                "eliminated_now": None, "remaining": None,
                "skipped_reason": None
            })

    pd.DataFrame(seq_rows).to_csv(OUTPUT_DIR/"recommender_sequence.csv", index=False)

    # Avoid-by-case export
    avoid_rows = []
    for (a,b), pr in sorted(pair_case.items(), key=lambda kv: (-kv[1], kv[0])):
        avoid_rows.append({"filter_id_1": a, "filter_id_2": b, "pair_risk": round(float(pr),6)})
    pd.DataFrame(avoid_rows).to_csv(OUTPUT_DIR/"avoid_by_case.csv", index=False)



if __name__ == "__main__":
    main()

    main()

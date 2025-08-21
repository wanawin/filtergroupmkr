"""
profiler.py — Historical Case Logger for DC5 Recommender

Purpose
-------
Builds a *deterministic* historical log of how your actual filters behave on
actual winners. For each draw i (i>=1):
  - seed = winners[i-1]
  - true_next = winners[i]
  - find all *applicable* filters under the seed's context
  - test each applicable filter against the true next winner
  - record which filters would have eliminated the winner

Outputs (written to current directory by default):
  - case_history.csv         — one row per draw with seed features, applicable IDs, eliminator IDs
  - case_filter_stats.csv    — per-signature (case) failure stats per filter (n_app, n_blk, fail_rate)
  - case_meta.json           — winners file hash + count (for incremental refresh)

This file is meant to be called from the Streamlit app before recommending.
It relies on helper functions from recommender.py if available.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Iterable, Tuple
import pandas as pd

# --- Try to use the shared helpers from recommender.py ---
try:
    from recommender import (
        WINNERS_CSV, FILTERS_CSV, OUTPUT_DIR,
        load_winners, load_filters, FilterDef,
        build_env_for_draw, safe_eval, digits_of, sum_category, VTRAC,
        case_signature, signature_key,
    )
except Exception:
    # Minimal local fallbacks (in case someone runs this standalone).
    WINNERS_CSV = "DC5_Midday_Full_Cleaned_Expanded.csv"
    FILTERS_CSV = "lottery_filters_batch_10.csv"
    OUTPUT_DIR = Path(".")

    from dataclasses import dataclass
    from collections import Counter

    MIRROR = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}
    VTRAC = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}

    def sum_category(total: int) -> str:
        if 0 <= total <= 15:  return "Very Low"
        if 16 <= total <= 24: return "Low"
        if 25 <= total <= 33: return "Mid"
        return "High"

    def digits_of(s: str) -> List[int]:
        return [int(ch) for ch in s if ch.isdigit()]

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
            raise ValueError("Winners CSV must have a 5-digit column (preferably 'Result').")
        vals = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
        return vals[vals.str.fullmatch(r"\d{5}")].tolist()

    def load_filters(path: str) -> List[FilterDef]:
        df = pd.read_csv(path)
        req = ["id","name","enabled","applicable_if","expression"]
        for r in req:
            if r not in df.columns:
                raise ValueError(f"Filters CSV missing column: {r}")
        def to_bool(x):
            if isinstance(x,bool): return x
            if pd.isna(x): return False
            return str(x).strip().lower() in {"true","1","yes","y"}
        df["enabled"] = df["enabled"].map(to_bool)
        out: List[FilterDef] = []
        for _, row in df.iterrows():
            out.append(FilterDef(
                str(row["id"]).strip(),
                str(row["name"]).strip(),
                bool(row["enabled"]),
                str(row.get("applicable_if","")),
                str(row.get("expression","")),
            ))
        return out

    def safe_eval(expr: str, env: Dict[str,object]) -> bool:
        if not expr: return True
        try:
            return bool(eval(expr,{"__builtins__":{}},env))
        except Exception:
            return False

    def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
        # Minimal env sufficient to evaluate most expressions
        seed   = winners[idx-1]
        combo  = winners[idx]
        seed_list  = digits_of(seed)
        combo_list = sorted(digits_of(combo))
        def classify_structure(digs: List[int]) -> str:
            from collections import Counter
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
        history_digits = [digits_of(s) for s in winners[:idx]]
        hot, cold, due = hot_cold_due(history_digits, k_hotcold=10)
        env = {
            "combo": combo,
            "combo_digits": set(combo_list),
            "combo_digits_list": combo_list,
            "combo_sum": sum(combo_list),
            "combo_sum_category": sum_category(sum(combo_list)),
            "seed": seed,
            "seed_digits": set(seed_list),
            "seed_digits_list": seed_list,
            "seed_sum": sum(seed_list),
            "seed_sum_category": sum_category(sum(seed_list)),
            "spread_seed": max(seed_list) - min(seed_list),
            "spread_combo": max(combo_list) - min(combo_list),
            "seed_vtracs": set(VTRAC[d] for d in seed_list),
            "combo_vtracs": set(VTRAC[d] for d in combo_list),
            "hot_digits": sorted(hot),
            "cold_digits": sorted(cold),
            "due_digits": sorted(due),
            # safe builtins
            "any": any, "all": all, "len": len, "sum": sum,
            "max": max, "min": min, "set": set, "sorted": sorted,
        }
        return env

    def case_signature(env: Dict[str,object], applicable_ids: Iterable[str]) -> Dict[str,object]:
        seed_list = env['seed_digits_list'] if 'seed_digits_list' in env else digits_of(env['seed'])
        parity_major = "even>=3" if sum(1 for d in seed_list if d%2==0) >= 3 else "even<=2"
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
        sig = {
            "sum_cat": env.get('seed_sum_category'),
            "struct": classify_structure(seed_list),
            "spread_band": spread_band(env.get('spread_seed', 0)),
            "parity_major": parity_major,
            "hot": tuple(sorted(env.get('hot_digits', []))),
            "cold": tuple(sorted(env.get('cold_digits', []))),
            "due": tuple(sorted(env.get('due_digits', []))),
            "applicable": tuple(sorted(set(applicable_ids)))
        }
        return sig

    import json as _json
    def signature_key(sig: Dict[str,object]) -> str:
        return _json.dumps(sig, sort_keys=True)

# ---------------- Utility ----------------

def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

# ---------------- Core logging ----------------

def build_case_history(winners_csv: str = WINNERS_CSV, filters_csv: str = FILTERS_CSV, out_dir: Path = OUTPUT_DIR) -> None:
    winners = load_winners(winners_csv)
    filters = load_filters(filters_csv)
    rows: List[Dict[str,object]] = []
    for idx in range(1, len(winners)):
        env = build_env_for_draw(idx, winners)
        applicable = [f for f in filters if f.enabled and safe_eval(f.applicable_if, env)]
        applicable_ids = [f.fid for f in applicable]
        sig = case_signature(env, applicable_ids)
        sig_key_str = signature_key(sig)
        # winner at this draw
        winner = winners[idx]
        # Test each applicable against the true next winner
        elim_ids: List[str] = []
        clist = sorted(digits_of(winner))
        base = dict(env)
        base.update({
            "combo": winner,
            "combo_digits": set(clist),
            "combo_digits_list": clist,
            "combo_sum": sum(clist),
            "combo_sum_category": sum_category(sum(clist)),
            "spread_combo": (max(clist) - min(clist)) if clist else 0,
            "combo_vtracs": set(VTRAC[d] for d in clist) if clist else set(),
        })
        for f in applicable:
            if safe_eval(f.applicable_if, base) and safe_eval(f.expression, base):
                elim_ids.append(f.fid)
        rows.append({
            "draw_idx": idx,
            "seed": winners[idx-1],
            "winner": winner,
            "sig_key": sig_key_str,
            "sum_cat": sig["sum_cat"],
            "struct": sig["struct"],
            "spread_band": sig["spread_band"],
            "parity_major": sig["parity_major"],
            "hot": ",".join(map(str, sig["hot"])),
            "cold": ",".join(map(str, sig["cold"])),
            "due": ",".join(map(str, sig["due"])),
            "applicable": ",".join(applicable_ids),
            "eliminators": ",".join(sorted(elim_ids)),
        })
    df = pd.DataFrame(rows)
    (out_dir/"case_history.csv").write_text("" if df.empty else df.to_csv(index=False), encoding="utf-8")
    _build_case_stats(out_dir)


def _build_case_stats(out_dir: Path = OUTPUT_DIR) -> None:
    hist_path = out_dir/"case_history.csv"
    if not hist_path.exists():
        return
    df = pd.read_csv(hist_path)
    if df.empty:
        (out_dir/"case_filter_stats.csv").write_text("", encoding="utf-8")
        return
    # explode applicable and eliminators to compute rates per (sig_key, filter)
    app = df.assign(applicable=df["applicable"].fillna("").str.split(",")).explode("applicable")
    app = app[app["applicable"].astype(str) != ""]
    elim = df.assign(eliminators=df["eliminators"].fillna("").str.split(",")).explode("eliminators")
    elim = elim[elim["eliminators"].astype(str) != ""]

    tot = app.groupby(["sig_key","applicable"]).size().rename("n_app").reset_index()
    blks = elim.groupby(["sig_key","eliminators"]).size().rename("n_blk").reset_index()

    merged = tot.merge(
        blks,
        left_on=["sig_key","applicable"],
        right_on=["sig_key","eliminators"],
        how="left"
    ).fillna({"n_blk": 0})

    merged["fail_rate"] = merged.apply(lambda r: (r["n_blk"]/r["n_app"]) if r["n_app"]>0 else 0.0, axis=1)
    merged = merged.rename(columns={"applicable":"filter_id"})[["sig_key","filter_id","n_app","n_blk","fail_rate"]]
    merged.to_csv(out_dir/"case_filter_stats.csv", index=False)


# ---------------- Incremental refresh ----------------

def refresh_incremental(
    winners_csv: str = WINNERS_CSV,
    filters_csv: str = FILTERS_CSV,
    keep_last_n: int = 7,
    out_dir: Path = OUTPUT_DIR,
) -> None:
    meta_path = out_dir/"case_meta.json"
    hist_path = out_dir/"case_history.csv"

    winners = load_winners(winners_csv)
    filters = load_filters(filters_csv)

    prev = {"hash":"","count":0}
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
        except Exception:
            prev = {"hash":"","count":0}

    # Start by trimming existing history
    prev_df = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    if not prev_df.empty:
        prev_df = prev_df.sort_values("draw_idx")
        # Drop oldest N rows
        prev_df = prev_df.iloc[keep_last_n:] if len(prev_df) > keep_last_n else pd.DataFrame()

    # Rebuild last N draws from winners
    start = max(1, len(winners) - keep_last_n)
    new_rows: List[Dict[str,object]] = []
    for idx in range(start, len(winners)):
        env = build_env_for_draw(idx, winners)
        applicable = [f for f in filters if f.enabled and safe_eval(f.applicable_if, env)]
        applicable_ids = [f.fid for f in applicable]
        sig = case_signature(env, applicable_ids)
        sig_key_str = signature_key(sig)
        winner = winners[idx]
        elim_ids: List[str] = []
        clist = sorted(digits_of(winner))
        base = dict(env)
        base.update({
            "combo": winner,
            "combo_digits": set(clist),
            "combo_digits_list": clist,
            "combo_sum": sum(clist),
            "combo_sum_category": sum_category(sum(clist)),
            "spread_combo": (max(clist) - min(clist)) if clist else 0,
            "combo_vtracs": set(VTRAC[d] for d in clist) if clist else set(),
        })
        for f in applicable:
            if safe_eval(f.applicable_if, base) and safe_eval(f.expression, base):
                elim_ids.append(f.fid)
        new_rows.append({
            "draw_idx": idx,
            "seed": winners[idx-1],
            "winner": winner,
            "sig_key": sig_key_str,
            "sum_cat": sig["sum_cat"],
            "struct": sig["struct"],
            "spread_band": sig["spread_band"],
            "parity_major": sig["parity_major"],
            "hot": ",".join(map(str, sig["hot"])),
            "cold": ",".join(map(str, sig["cold"])),
            "due": ",".join(map(str, sig["due"])),
            "applicable": ",".join(applicable_ids),
            "eliminators": ",".join(sorted(elim_ids)),
        })

    df_new = pd.DataFrame(new_rows)
    if prev_df.empty:
        df_all = df_new
    else:
        df_all = pd.concat([prev_df, df_new], ignore_index=True)

    df_all.to_csv(hist_path, index=False)
    _build_case_stats(out_dir)

    meta_path.write_text(json.dumps({"hash": _file_hash(winners_csv), "count": len(winners)}), encoding="utf-8")


if __name__ == "__main__":
    # Simple CLI for manual runs (optional)
    import argparse
    parser = argparse.ArgumentParser(description="Build/refresh historical case logs for DC5 recommender")
    parser.add_argument("--winners", default=WINNERS_CSV)
    parser.add_argument("--filters", default=FILTERS_CSV)
    parser.add_argument("--mode", choices=["full","incremental"], default="full")
    parser.add_argument("--keep_last_n", type=int, default=7)
    parser.add_argument("--out", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "full":
        build_case_history(args.winners, args.filters, out_dir)
    else:
        refresh_incremental(args.winners, args.filters, args.keep_last_n, out_dir)
    print("Done.")

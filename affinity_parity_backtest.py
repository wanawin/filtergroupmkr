#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Affinity Parity Backtest — patched to support **real archived pools**

Usage examples
--------------
# With real archived pools (recommended)
python affinity_parity_backtest.py \
  --winners DC5_Midday_Full_Cleaned_Expanded.csv \
  --n_days 240 \
  --top_pcts 0.2,0.3 \
  --use_archived_pools

# Original (synthetic) mode — keeps your prior behavior
python affinity_parity_backtest.py \
  --winners DC5_Midday_Full_Cleaned_Expanded.csv \
  --n_days 240 \
  --pool_mode 1digit --pool_size 1800 \
  --top_pcts 0.2,0.3

Outputs
-------
- <out_prefix>_summary.csv : accuracy per threshold (sum-parity & majority-parity)
- <out_prefix>_backtest.csv: per-day records (predictions & hits)
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Helpers (digits & parity)
# -----------------------------

def digits_of(s: str) -> List[int]:
    s = ''.join(ch for ch in str(s) if ch.isdigit())
    return [int(ch) for ch in s.zfill(5)[-5:]]


def sum_parity_label(s: str) -> str:
    return "even" if sum(digits_of(s)) % 2 == 0 else "odd"


def majority_parity_label(s: str) -> str:
    evens = sum(1 for d in digits_of(s) if d % 2 == 0)
    return "even" if evens >= 3 else "odd"


# -----------------------------
# Synthetic pool (legacy fallback)
# -----------------------------

def shares_k_digits(candidate: str, seed: str, k: int) -> bool:
    a = set(digits_of(candidate))
    b = set(digits_of(seed))
    return len(a.intersection(b)) >= k


def gen_pool_for_seed(seed: str, mode: str = "1digit", size: int = 1800) -> pd.DataFrame:
    """Legacy synthetic generator used in the original backtest.
    mode: '1digit' (share ≥1 digit) or '2digit' (share ≥2 digits).
    Returns DataFrame with a 'combo' column of 5-digit strings.
    """
    need_k = 1 if mode == "1digit" else 2
    seen = set()
    out: List[str] = []
    # Randomly sample until we fill the requested pool size
    while len(out) < int(size):
        x = f"{random.randint(0, 99999):05d}"
        if x in seen:
            continue
        if shares_k_digits(x, seed, need_k):
            out.append(x)
            seen.add(x)
    return pd.DataFrame({"combo": out})


# -----------------------------
# Affinity scoring
# -----------------------------

def score_affinity(df: pd.DataFrame, seed_value: str) -> pd.DataFrame:
    """Return df with columns: ['aff_score','aff_pct'] using recommender if available."""
    import importlib

    df = df.copy()
    if "combo" not in df.columns:
        if "Result" in df.columns:
            df = df.rename(columns={"Result": "combo"})
        else:
            raise RuntimeError("Expected 'combo' or 'Result' column in pool DataFrame.")
    df["combo"] = df["combo"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)

    # Try recommender APIs first
    try:
        rec = importlib.import_module("recommender")
        weights = getattr(rec, "DEFAULT_AFF_WEIGHTS", None)
        # Prefer vectorized scoring
        if hasattr(rec, "affinity_scores_and_pct"):
            out = rec.affinity_scores_and_pct(df, seed=seed_value, weights=weights)  # type: ignore[arg-type]
            if "aff_score" not in out.columns:
                raise RuntimeError("affinity_scores_and_pct did not produce 'aff_score'.")
            if "aff_pct" not in out.columns:
                # compute percentile if not provided
                vals = out["aff_score"].to_numpy(float)
                order = np.argsort(vals)
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.linspace(0.0, 1.0, len(vals))
                out["aff_pct"] = ranks
            return out
        # Fallback: per-combo apply
        if hasattr(rec, "combo_affinity"):
            df["aff_score"] = df["combo"].astype(str).map(lambda c: float(rec.combo_affinity(c, seed=seed_value, weights=weights)))  # type: ignore
            vals = df["aff_score"].to_numpy(float)
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.linspace(0.0, 1.0, len(vals))
            df["aff_pct"] = ranks
            return df
    except Exception:
        pass

    # If recommender isn't available, do a trivial placeholder (not recommended)
    # Here we just use negative Hamming distance to seed as a crude score
    seed_d = digits_of(seed_value)
    def crude(c: str) -> float:
        cd = digits_of(c)
        return -sum(int(cd[i] == seed_d[i]) for i in range(5))
    df["aff_score"] = df["combo"].map(crude)
    vals = df["aff_score"].to_numpy(float)
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(vals))
    df["aff_pct"] = ranks
    return df


# -----------------------------
# Backtest core
# -----------------------------

def run_backtest(
    winners_csv: str,
    n_days: int,
    pool_mode: str,
    pool_size: int,
    top_pcts: List[float],
    out_prefix: str,
    use_archived_pools: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    winners_df = pd.read_csv(winners_csv)

    # Identify columns
    res_col = None
    date_col = None
    for c in winners_df.columns:
        lc = str(c).lower()
        if lc in ("result", "combo"):
            res_col = c
        if lc in ("date", "drawdate", "draw_date"):
            date_col = c
    if res_col is None:
        raise RuntimeError("Winners CSV must contain a 'Result' or 'combo' column.")

    # Indices for testing (use last n_days); ensure we have seed and winner (t-1, t)
    end_idx = len(winners_df) - 1
    start_idx = max(1, end_idx - int(n_days) + 1)

    # For CLI UX: let pool_mode/size be ignored in archived mode
    if use_archived_pools:
        print("[info] Using archived real pools via recommender.get_pool_for_seed(...). 'pool_mode'/'pool_size' are ignored.")

    # Per-day rows
    rows = []

    for t in range(start_idx, end_idx + 1):
        seed = str(winners_df.loc[t - 1, res_col])
        winner = str(winners_df.loc[t, res_col])
        date_val = str(winners_df.loc[t, date_col]) if date_col else ""

        # Build/Load pool
        if use_archived_pools:
            import importlib
            rec = importlib.import_module("recommender")
            pool_df = rec.get_pool_for_seed(winners_df.iloc[t - 1])  # must return df with 'combo'
        else:
            pool_df = gen_pool_for_seed(seed, mode=pool_mode, size=int(pool_size))

        if pool_df is None or pool_df.empty:
            continue

        # Score affinity and percentiles (0..1 ascending)
        scored = score_affinity(pool_df, seed)

        # Compute predictions for each threshold
        for pct in top_pcts:
            k = float(pct)
            n_top = max(1, int(math.ceil(k * len(scored))))
            topk = scored.nlargest(n_top, "aff_score")  # highest affinity region

            # Top-k majority labels
            maj_sum = topk["combo"].map(sum_parity_label).mode()
            pred_sum = str(maj_sum.iloc[0]) if len(maj_sum) else None

            maj_majority = topk["combo"].map(majority_parity_label).mode()
            pred_majority = str(maj_majority.iloc[0]) if len(maj_majority) else None

            # Ground truth for the actual winner
            true_sum = sum_parity_label(winner)
            true_majority = majority_parity_label(winner)

            rows.append({
                "t": t,
                "date": date_val,
                "seed": seed,
                "winner": winner,
                "pool_size": int(len(scored)),
                "top_pct": k,
                "pred_sum_parity": pred_sum,
                "true_sum_parity": true_sum,
                "hit_sum": int(pred_sum == true_sum) if pred_sum else None,
                "pred_majority_parity": pred_majority,
                "true_majority_parity": true_majority,
                "hit_majority": int(pred_majority == true_majority) if pred_majority else None,
            })

    day_df = pd.DataFrame(rows)
    if day_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Summary by threshold
    def _acc(col_hit: str) -> pd.Series:
        sub = day_df.dropna(subset=[col_hit])
        if sub.empty:
            return pd.Series({"tests": 0, "hits": 0, "accuracy": np.nan})
        return pd.Series({
            "tests": int(sub.shape[0]),
            "hits": int(sub[col_hit].sum()),
            "accuracy": float(sub[col_hit].mean()),
        })

    summaries = []
    for k in sorted(set(day_df["top_pct"].tolist())):
        s_sum = _acc("hit_sum")
        s_maj = _acc("hit_majority")
        summaries.append({
            "top_pct": float(k),
            "tests_sum": s_sum["tests"],
            "hits_sum": s_sum["hits"],
            "accuracy_sum": s_sum["accuracy"],
            "tests_majority": s_maj["tests"],
            "hits_majority": s_maj["hits"],
            "accuracy_majority": s_maj["accuracy"],
        })
    sum_df = pd.DataFrame(summaries).sort_values("top_pct")
    return day_df, sum_df


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--winners", required=True, help="Path to winners CSV")
    p.add_argument("--n_days", type=int, default=240, help="Most-recent days to test")
    p.add_argument("--pool_mode", choices=["1digit", "2digit"], default="1digit")
    p.add_argument("--pool_size", type=int, default=1800)
    p.add_argument("--top_pcts", default="0.2,0.3", help="Comma list like '0.2,0.3'")
    p.add_argument("--out_prefix", default="parity_affinity")
    p.add_argument("--use_archived_pools", action="store_true", help="Load real pools via recommender.get_pool_for_seed")
    return p.parse_args()


def main():
    args = parse_args()
    top_pcts = [float(x.strip()) for x in str(args.top_pcts).split(",") if x.strip()]

    day_df, sum_df = run_backtest(
        winners_csv=args.winners,
        n_days=int(args.n_days),
        pool_mode=args.pool_mode,
        pool_size=int(args.pool_size),
        top_pcts=top_pcts,
        out_prefix=args.out_prefix,
        use_archived_pools=bool(args.use_archived_pools),
    )

    out_summary = f"{args.out_prefix}_summary.csv"
    out_days = f"{args.out_prefix}_backtest.csv"

    if not day_df.empty:
        day_df.to_csv(out_days, index=False)
        print(f"[ok] wrote {out_days} ({len(day_df)} rows)")
    else:
        print("[warn] no per-day rows produced (check pools & winners)")

    if not sum_df.empty:
        sum_df.to_csv(out_summary, index=False)
        print(f"[ok] wrote {out_summary} ({len(sum_df)} rows)")
    else:
        print("[warn] no summary produced")


if __name__ == "__main__":
    main()

import os, sys, math, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Affinity ▶ Winner Stats", page_icon="📈", layout="wide")
st.title("📈 Affinity vs Winner — Distribution & Buckets")

st.markdown(
    "This page checks, day-by-day, where the **actual winner** lands in the **affinity ranking** built from that day's seed.
"
    "Pool construction mirrors your production logic: **1-digit rule only**, **percentile filter vs full enumeration (optional)**, **deduped** canonical combos, then **remove quints & quads**.
"
)

# -----------------------------------------------------------------------------
# Imports from recommender (with a safe fallback for combo_affinity)
# -----------------------------------------------------------------------------
try:
    import recommender as rec
except Exception as e:
    st.error(f"Failed to import recommender.py: {e}")
    st.stop()

# parity-major label helper

def parity_major_label(digs):
    return "even>=3" if sum(1 for d in digs if d % 2 == 0) >= 3 else "even<=2"

# Use rec.combo_affinity if present; else define our local copy that mirrors the heuristic
if hasattr(rec, "combo_affinity"):
    combo_affinity = rec.combo_affinity  # type: ignore
else:
    def combo_affinity(env_now: dict, combo: str) -> float:
        cd = rec.digits_of(combo)
        seedd = env_now['seed_digits_list']
        combo_sum = sum(cd)
        seed_sum = env_now['seed_sum']
        sum_prox = math.exp(-abs(combo_sum - seed_sum) / 2.0)
        spread_c = max(cd) - min(cd)
        spread_s = env_now['spread_seed']
        spread_prox = math.exp(-abs(spread_c - spread_s) / 2.0)
        struct_match = 1.0 if rec.classify_structure(cd) == rec.classify_structure(seedd) else 0.0
        parity_match = 1.0 if parity_major_label(cd) == parity_major_label(seedd) else 0.0
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

# -----------------------------------------------------------------------------
# Deterministic pool generator (EXACT): 1-digit, percentile filter, dedup, remove quints/quads
# -----------------------------------------------------------------------------
from itertools import combinations_with_replacement

def full_canonical_multisets():
    """All canonical 5-digit multisets as strings (2002 total)."""
    return [''.join(str(d) for d in tup) for tup in combinations_with_replacement(range(10), 5)]

FULL_CANON = full_canonical_multisets()
FULL_SUMS = np.array([sum(rec.digits_of(s)) for s in FULL_CANON], dtype=float)

def sum_percentile_bounds(low_pct: float, high_pct: float):
    """Return (low, high) bounds on SUM using full enumeration percentiles."""
    low = np.percentile(FULL_SUMS, low_pct*100)
    high = np.percentile(FULL_SUMS, high_pct*100)
    return float(low), float(high)

def build_pool_1digit_exact(seed_digits, keep_sum_pct=(0.0, 1.0), remove_quints_quads=True):
    seed_set = set(seed_digits)
    low_p, high_p = keep_sum_pct
    low_sum, high_sum = sum_percentile_bounds(low_p, high_p)

    out = []
    for s in FULL_CANON:  # already canonical (deduped)
        digs = rec.digits_of(s)
        if len(set(digs) & seed_set) < 1:
            continue  # 1-digit rule
        ssum = sum(digs)
        if not (low_sum <= ssum <= high_sum):
            continue  # percentile filter vs full enumeration
        if remove_quints_quads:
            stc = rec.classify_structure(digs)
            if stc in {"quint", "quad"}:
                continue
        out.append(s)
    return out

# -----------------------------------------------------------------------------
# UI controls
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    winners_path = st.text_input("Winners CSV", "DC5_Midday_Full_Cleaned_Expanded.csv")
    uploaded = st.file_uploader("...or upload Winners CSV", type=["csv"], help="Upload overrides the path above.")
with col2:
    n_days = st.number_input("Days to test (most recent)", min_value=30, max_value=3000, value=240, step=10)
    sum_pct_keep = st.slider(
        "Keep SUM percentiles (vs full enumeration)",
        0.0, 1.0, (0.0, 1.0), 0.05,
        help="Example: (0.05, 0.95) keeps the middle 90% of sums. Default keeps all."
    )

col3, col4 = st.columns(2)
with col3:
    show_hist = st.checkbox("Show 5% percentile histogram", value=True)
with col4:
    run = st.button("Run Winner Affinity Stats", type="primary")

# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
if run:
    # Handle upload
    winners_to_use = winners_path
    if uploaded is not None:
        tmp_path = Path("uploaded_winners.csv")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        winners_to_use = str(tmp_path)

    # Load winners
    try:
        winners = rec.load_winners(winners_to_use)
    except Exception as e:
        st.error(f"Failed to read winners CSV: {e}")
        st.stop()

    N = len(winners)
    if N < 3:
        st.warning("Not enough rows in winners CSV.")
        st.stop()

    end_idx = N - 1
    start_idx = max(2, end_idx - int(n_days))
    # deterministic enumeration; no RNG needed

    results = []
    for t in range(start_idx, end_idx + 1):
        env = rec.build_env_for_draw(t, winners)
        seed_digits = env['seed_digits_list']
        true_win = winners[t]

        pool = build_pool_1digit_exact(seed_digits, keep_sum_pct=sum_pct_keep, remove_quints_quads=True)
        # Compute affinity for pool and winner
        aff_pool = np.fromiter((combo_affinity(env, c) for c in pool), dtype=float)
        aff_w = float(combo_affinity(env, true_win))
        # Percentile of winner within pool
        # rank definition: fraction of pool ≤ winner affinity
        pct = float((aff_pool <= aff_w).mean())  # 0..1
        results.append({
            "t": t,
            "seed": winners[t-1],
            "winner": true_win,
            "aff_winner": round(aff_w, 6),
            "pct": pct,
        })

    df = pd.DataFrame(results)

    # Bucketization
    def bucket(p):
        if p >= 0.95: return "Top 5%"
        if p >= 0.90: return "Top 10%"
        if p >= 0.80: return "High (80–100%)"
        if p < 0.20:  return "Low (0–20%)"
        return "Mid (20–80%)"

    df["bucket"] = df["pct"].map(bucket)

    # Summaries
    kept = df["seed"].count()  # same as total_days
    total_days = len(df)

    # Winner kept rate (winner in pool?)
    # Recompute with the same pool logic to test inclusion explicitly
    kept_flags = []
    pool_sizes = []
    for t in range(start_idx, end_idx + 1):
        env = rec.build_env_for_draw(t, winners)
        seed_digits = env['seed_digits_list']
        pool_t = build_pool_1digit_exact(seed_digits, keep_sum_pct=sum_pct_keep, remove_quints_quads=True)
        pool_sizes.append(len(pool_t))
        win_canon = ''.join(sorted(winners[t]))
        kept_flags.append(int(win_canon in set(pool_t)))
    kept_rate = (sum(kept_flags) / len(kept_flags) * 100) if kept_flags else 0.0

    by_bucket = (
        df.groupby("bucket").size().rename("days").reset_index().sort_values(
            by=["bucket"], key=lambda s: s.map({
                "Top 5%":5, "Top 10%":10, "High (80–100%)":80, "Mid (20–80%)":50, "Low (0–20%)":0
            })
        )
    )
    by_bucket["percent"] = (by_bucket["days"] / total_days * 100).round(2)

    st.subheader("Summary — Winner Affinity Percentile Buckets")
    st.dataframe(by_bucket, use_container_width=True)
    st.caption("High = winner in top 20% of affinity that day; Low = bottom 20%.")

    st.write(f"**Winner kept in pool:** {kept_rate:.1f}%  ·  **Avg pool size:** {np.mean(pool_sizes):.0f}  ·  **Median:** {np.median(pool_sizes):.0f}")

    if show_hist:
        st.subheader("Histogram — Winner Affinity Percentile (5% bins)")
        bins = np.linspace(0.0, 1.0, 21)  # 0.00,0.05,...,1.00
        labels = [f"{int(100*b)}–{int(100*bins[i+1])}%" for i, b in enumerate(bins[:-1])]
        df["pct_bin"] = pd.cut(df["pct"], bins=bins, labels=labels, include_lowest=True)
        hist = df.groupby("pct_bin").size().rename("days").reset_index()
        hist["percent"] = (hist["days"] / total_days * 100.0).round(2)
        st.dataframe(hist, use_container_width=True)

    # Download
    st.download_button("Download per-day winner affinity CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="winner_affinity_percentiles.csv",
        mime="text/csv")

    # Quick plain-English takeaway
    hi = float(by_bucket.loc[by_bucket["bucket"]=="High (80–100%)", "percent"].sum() or 0)
    lo = float(by_bucket.loc[by_bucket["bucket"]=="Low (0–20%)", "percent"].sum() or 0)
    st.info(f"Across the tested span, winners landed in the **top 20%** about **{hi:.1f}%** of days and in the **bottom 20%** about **{lo:.1f}%**.")

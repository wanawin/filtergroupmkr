import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Affinity Trim Backtest", page_icon="📉", layout="wide")
st.title("📉 Affinity Trim Backtest (blind & winner‑preserving)")

st.markdown(
    "Evaluate trimming the **top X% most look‑alike** combos by affinity on **real archived pools** over many days.\n\n"
    "**Blind:** always trim regardless of winner.\n\n"
    "**Winner‑preserving:** apply trim only on days where it doesn't remove the winner (skips otherwise)."
)

# --------------------------------------------------
# Inputs
# --------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    winners_csv = st.text_input("Winners CSV", "DC5_Midday_Full_Cleaned_Expanded.csv")
with col2:
    n_days = st.number_input("Days to test (most recent)", min_value=10, max_value=2000, value=180, step=10)
with col3:
    top_pct = st.slider("Trim OFF top X% (look‑alikes)", 0.05, 0.50, 0.25, 0.05)

use_archived = st.checkbox("Use archived real pools (required)", value=True,
    help="Loads pools/pool_YYYY-MM-DD.csv for each day via recommender.get_pool_for_seed().")
run = st.button("▶️ Run Trim Backtest", type="primary")

# --------------------------------------------------
# Work
# --------------------------------------------------
if run:
    import recommender as rec

    # Load winners (full df for date/row access + list for env helpers)
    try:
        winners_df = pd.read_csv(winners_csv)
        winners = rec.load_winners(winners_csv)
    except Exception as e:
        st.error(f"Failed to read winners CSV: {e}")
        st.stop()

    if len(winners) < 2:
        st.error("Need at least 2 rows in winners CSV.")
        st.stop()

    end_idx = len(winners) - 1
    start_idx = max(1, end_idx - int(n_days) + 1)

    rows = []
    miss_archives = 0

    for t in range(start_idx, end_idx + 1):
        seed = winners[t-1]
        winner = winners[t]

        # Load real pool for that seed's date
        try:
            seed_row = winners_df.iloc[t-1]
            pool_df = rec.get_pool_for_seed(seed_row)
        except Exception as e:
            miss_archives += 1
            rows.append({
                "t": t,
                "seed": seed,
                "winner": winner,
                "pool_before": np.nan,
                "pool_after_blind": np.nan,
                "removed_blind": np.nan,
                "removed_pct_blind": np.nan,
                "winner_kept_blind": np.nan,
                "applied_preserving": False,
                "pool_after_preserving": np.nan,
                "removed_preserving": np.nan,
                "removed_pct_preserving": np.nan,
            })
            continue

        if pool_df.empty or "combo" not in pool_df.columns:
            rows.append({
                "t": t,
                "seed": seed,
                "winner": winner,
                "pool_before": 0,
                "pool_after_blind": 0,
                "removed_blind": 0,
                "removed_pct_blind": 0.0,
                "winner_kept_blind": False,
                "applied_preserving": False,
                "pool_after_preserving": 0,
                "removed_preserving": 0,
                "removed_pct_preserving": 0.0,
            })
            continue

        scored = rec.affinity_scores_and_pct(pool_df, seed=seed)
        n0 = len(scored)
        keep_mask = scored["aff_pct"] < (1.0 - float(top_pct))
        trimmed = scored.loc[keep_mask]
        n1 = len(trimmed)
        removed = n0 - n1
        removed_pct = (removed / n0) if n0 else 0.0

        winner_kept = (trimmed["combo"].astype(str) == str(winner).zfill(5)).any()

        # Winner‑preserving variant: apply only if winner survives
        if winner_kept:
            applied_preserving = True
            pool_after_pres = n1
            removed_pres = removed
            removed_pct_pres = removed_pct
        else:
            applied_preserving = False
            pool_after_pres = n0  # skip trim
            removed_pres = 0
            removed_pct_pres = 0.0

        rows.append({
            "t": t,
            "seed": seed,
            "winner": winner,
            "pool_before": int(n0),
            "pool_after_blind": int(n1),
            "removed_blind": int(removed),
            "removed_pct_blind": round(removed_pct, 6),
            "winner_kept_blind": bool(winner_kept),
            "applied_preserving": bool(applied_preserving),
            "pool_after_preserving": int(pool_after_pres),
            "removed_preserving": int(removed_pres),
            "removed_pct_preserving": round(removed_pct_pres, 6),
        })

    df = pd.DataFrame(rows)

    if miss_archives:
        st.warning(f"{miss_archives} day(s) had no archived pool. Those rows are left blank.")

    st.subheader("Per‑day results")
    st.dataframe(df, use_container_width=True)

    # Summary
    have = df[~df["pool_before"].isna()].copy()
    if have.empty:
        st.error("No days with archived pools to evaluate.")
        st.stop()

    days = len(have)
    blind_keep_rate = float(have["winner_kept_blind"].mean())
    avg_removed_blind = float(have["removed_pct_blind"].mean())

    # Winner‑preserving coverage & reduction on days it applies
    applied = have[have["applied_preserving"]]
    coverage = len(applied) / days if days else 0.0
    avg_removed_pres = float(applied["removed_pct_preserving"].mean()) if len(applied) else 0.0

    st.subheader("Summary")
    summary = pd.DataFrame([
        {"metric": "days_evaluated", "value": days},
        {"metric": "blind_keep_rate (winner survives trim)", "value": round(blind_keep_rate, 4)},
        {"metric": "avg_removed_pct_blind", "value": round(avg_removed_blind, 4)},
        {"metric": "preserving_coverage (days applied)", "value": round(coverage, 4)},
        {"metric": "avg_removed_pct_preserving", "value": round(avg_removed_pres, 4)},
    ])
    st.dataframe(summary, use_container_width=True)

    # Downloads
    out_prefix = f"affinity_trim_top{int(round(top_pct*100))}"
    day_path = f"{out_prefix}_per_day.csv"
    sum_path = f"{out_prefix}_summary.csv"
    have.to_csv(day_path, index=False)
    summary.to_csv(sum_path, index=False)

    st.download_button("Download per‑day CSV", data=open(day_path, "rb").read(), file_name=day_path, mime="text/csv")
    st.download_button("Download summary CSV", data=open(sum_path, "rb").read(), file_name=sum_path, mime="text/csv")

st.caption("Use this to pick a default anti‑top trim (e.g., 25–30%) that keeps winners while shrinking the pool.")

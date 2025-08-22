# 03b_Affinity_Trim_Preview.py
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Affinity Trim Preview", page_icon="🧪", layout="wide")
st.title("🧪 Affinity Trim Preview (readout)")

st.markdown(
    "Preview how an **anti-top** affinity trim would change the pool on a given historical day, "
    "and whether the winner would have survived."
)

# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    winners_csv = st.text_input("Winners CSV", "DC5_Midday_Full_Cleaned_Expanded.csv")
with col2:
    top_pct = st.slider("Trim OFF top X% (look-alikes)", 0.05, 0.50, 0.25, 0.05)
with col3:
    use_archived = st.checkbox(
        "Use archived real pools",
        value=True,
        help="Loads pools/pool_YYYY-MM-DD.csv via recommender.get_pool_for_seed().",
    )

# 0 = most recent completed day; 1 = first possible
idx_choice = st.number_input(
    "Day index to preview (0 = most recent, 1 = first possible)",
    min_value=0, value=0, step=1,
    help="0 uses the most recent completed day; otherwise t uses seed at t-1 and winner at t."
)

run = st.button("▶️ Preview Trim", type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# Work
# ──────────────────────────────────────────────────────────────────────────────
if run:
    import numpy as np
    import recommender as rec

    # Load winners
    try:
        winners = rec.load_winners(winners_csv)
        winners_df = pd.read_csv(winners_csv)
    except Exception as e:
        st.error(f"Failed to load winners CSV: {e}")
        st.stop()

    if len(winners) < 2:
        st.error("Need at least 2 rows in winners CSV.")
        st.stop()

    t = (len(winners) - 1) if idx_choice == 0 else int(idx_choice)
    if not (1 <= t < len(winners)):
        st.error(f"Day index t must be between 1 and {len(winners)-1} (got {t}).")
        st.stop()

    # Pull the real pool for that day
    try:
        seed_row = winners_df.iloc[t-1]
        pool_df = rec.get_pool_for_seed(seed_row)
    except Exception as e:
        st.error(f"Failed to load pool for day t={t}: {e}")
        st.stop()

    if pool_df.empty or "combo" not in pool_df.columns:
        st.error("Pool DataFrame is empty or missing 'combo' column.")
        st.stop()

    seed = str(winners[t-1]).zfill(5)
    winner = str(winners[t]).zfill(5)

    # Score affinity + percentiles (inline, using rec.combo_affinity)
    def _digits_of(s: str):
        return [int(ch) for ch in str(s)]

    seed_list = _digits_of(seed)
    env_min = {
        "seed_digits_list": seed_list,
        "seed_sum": sum(seed_list),
        "spread_seed": (max(seed_list) - min(seed_list)),
    }

    scored = pool_df[["combo"]].copy()
    scored["combo"] = scored["combo"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    scored = scored[scored["combo"].str.fullmatch(r"\d{5}")]

    scored["aff"] = [rec.combo_affinity(env_min, c) for c in scored["combo"]]
    scored["aff_pct"] = scored["aff"].rank(pct=True, method="average")  # lowest→0, highest→1

    n0 = len(scored)
    keep_mask = scored["aff_pct"] < (1.0 - float(top_pct))
    trimmed = scored.loc[keep_mask].copy()
    n1 = len(trimmed)
    removed = n0 - n1

    winner_kept = (trimmed["combo"] == winner).any()

    # Readout
    icon = "✅" if winner_kept else "❌"
    st.success(f"Trimmed off top {int(round(top_pct*100))}% look-alikes. "
               f"Winner kept {icon}. Pool: {n0} → {n1} (−{removed}).")

    with st.expander("Details", expanded=False):
        st.write({
            "t": t, "seed": seed, "winner": winner,
            "pool_before": int(n0), "pool_after": int(n1),
            "removed": int(removed), "top_pct": float(top_pct),
            "winner_kept": bool(winner_kept),
        })
        st.dataframe(trimmed.head(30), use_container_width=True)

st.caption("Tip: Use the 'Affinity Trim Backtest' page to measure blind history and pick a default (e.g., 25–30%).")

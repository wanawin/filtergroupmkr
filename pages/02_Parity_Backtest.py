import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Parity Backtest", page_icon="📊", layout="wide")
st.title("📊 Parity Backtest (Affinity-Ordered Top% vs Winner)")

st.markdown(
    "This page runs a quick historical check: take each day, score combos by the same **affinity** your recommender uses, "
    "look at the **majority parity** in the top-k% (sum parity and majority parity), and compare to the actual winner."
)

# ---- Inputs ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Data")
    default_winners = "DC5_Midday_Full_Cleaned_Expanded.csv"
    winners_path = st.text_input("Winners CSV path", default_winners)
    uploaded = st.file_uploader("...or upload Winners CSV", type=["csv"], help="If you upload, it will override the path above for this run only.")

with col2:
    st.subheader("Backtest Settings")
    n_days = st.number_input("Days to test (most recent)", min_value=30, max_value=2000, value=240, step=10)
    pool_mode = st.selectbox("Pool mode", ["1digit", "2digit"], index=0, help="1digit: each combo shares ≥1 distinct digit with seed. 2digit: shares ≥2.")
    pool_size = st.number_input("Pool size per day", min_value=100, max_value=20000, value=1800, step=100)
    top_pcts = st.text_input("Top % thresholds (comma)", "0.2,0.3", help="e.g., 0.2 = top 20% by affinity")
        use_archived = st.checkbox("Use archived real pools (recommended)", value=True)


out_prefix = "parity_affinity"
run = st.button("▶️ Run Backtest", type="primary")

# ---- Run ------------------------------------------------------------------
if run:
    # If user uploaded a CSV, save it locally for the script
    winners_to_use = winners_path
    if uploaded is not None:
        tmp_path = Path("uploaded_winners.csv")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        winners_to_use = str(tmp_path)

    # Build command to run the standalone script
    cmd = [
        sys.executable,
        "affinity_parity_backtest.py",
        "--winners", winners_to_use,
        "--n_days", str(int(n_days)),
        "--pool_mode", pool_mode,
        "--pool_size", str(int(pool_size)),
        "--top_pcts", top_pcts,
        "--out_prefix", out_prefix,
    ]
    if use_archived:
        cmd.append("--use_archived_pools")


    st.info("Running backtest... this analyzes historical days and writes two CSVs below.")
    with st.spinner("Working..."):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        st.error("Backtest failed. See details below.")
        st.code(result.stderr or result.stdout or "No output", language="bash")
    else:
        st.success("Backtest complete!")
        if result.stdout:
            st.code(result.stdout, language="bash")

        # Load outputs if present
        sum_path = f"{out_prefix}_summary.csv"
        day_path = f"{out_prefix}_backtest.csv"

        if os.path.exists(sum_path):
            st.subheader("Summary (accuracy & lift)")
            df_sum = pd.read_csv(sum_path)
            st.dataframe(df_sum, use_container_width=True)
            st.download_button("Download summary CSV", data=open(sum_path, "rb").read(), file_name=sum_path, mime="text/csv")
        else:
            st.warning("Summary CSV not found.")

        if os.path.exists(day_path):
            st.subheader("Per-day results")
            df_day = pd.read_csv(day_path)
            st.dataframe(df_day, use_container_width=True)
            st.download_button("Download per-day CSV", data=open(day_path, "rb").read(), file_name=day_path, mime="text/csv")
        else:
            st.warning("Per-day CSV not found.")

st.caption("Tip: Use '1digit' & pool size ~1800 to mirror your normal pool. Try '2digit' & ~900 to see if stricter pools change the signal.")

# -----------------------------------------------------------------------------
# NEW: Probabilistic Parity (Bucket Prior)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧪 Probabilistic Parity (Bucket Prior)")

st.markdown(
    "This makes a parity call using a *history-based bucket prior* (no pool generation). "
    "It looks at past days with the **same bucket** as today's seed (SumCat × Structure × Spread × Parity-Major) "
    "and estimates P(EVEN) for the *next* draw with exponential recency weighting. "
    "Only predicts when confidence clears a threshold (abstains otherwise)."
)

colp1, colp2, colp3 = st.columns(3)
with colp1:
    winners_path_prior = st.text_input("Winners CSV (for prior)", winners_path)
with colp2:
    half_life = st.number_input("Half-life (draws)", min_value=10, max_value=1000, value=120, step=10)
with colp3:
    delta = st.slider("Confidence gate (|p-0.5|)", 0.0, 0.3, 0.10, 0.01)

n_days_prior = st.number_input("Days to test (most recent, prior)", 30, 2000, int(n_days), step=10)
run_prior = st.button("Run Probabilistic Prior")

if run_prior:
    import math
    import pandas as pd
    import recommender as rec

    # Load winners (support upload override from earlier UI)
    winners_to_use2 = winners_to_use if 'winners_to_use' in locals() else winners_path_prior
    try:
        winners_all = rec.load_winners(winners_to_use2)
    except Exception as e:
        st.error(f"Failed to read winners CSV: {e}")
        winners_all = []

    def bucket_key_from_env(env: dict):
        seed_sum_cat = env["seed_sum_category"]
        seed_list = env["seed_digits_list"]
        struct = rec.classify_structure(seed_list)
        sb = rec.spread_band(env["spread_seed"]) if hasattr(rec, 'spread_band') else (
            "0-3" if env["spread_seed"] <= 3 else
            "4-5" if env["spread_seed"] <= 5 else
            "6-7" if env["spread_seed"] <= 7 else
            "8-9" if env["spread_seed"] <= 9 else
            "10+"
        )
        parity_major = "even>=3" if sum(1 for d in seed_list if d % 2 == 0) >= 3 else "even<=2"
        return (seed_sum_cat, struct, sb, parity_major)

    def bucket_even_prior(winners: list[str], idx_now: int, half_life_draws: int = 120) -> float:
        if idx_now < 2:
            return 0.5
        env_now = rec.build_env_for_draw(idx_now, winners)
        b_now = bucket_key_from_env(env_now)
        def w(age):
            return 0.5 ** (age / float(half_life_draws)) if half_life_draws else 1.0
        num = 0.0
        den = 0.0
        for j in range(2, idx_now):
            env_j = rec.build_env_for_draw(j, winners)
            if bucket_key_from_env(env_j) != b_now:
                continue
            # Next winner at j is winners[j]
            even_next = int(sum(rec.digits_of(winners[j])) % 2 == 0)
            wt = w(idx_now - j)
            num += wt * even_next
            den += wt
        return (num / den) if den else 0.5

    if winners_all:
        # Trim to last N days for evaluation
        end_idx = len(winners_all) - 1
        start_idx = max(2, end_idx - int(n_days_prior))
        rows = []
        hits = 0
        calls = 0
        for t in range(start_idx, end_idx + 1):
            env_t = rec.build_env_for_draw(t, winners_all)
            p_even = bucket_even_prior(winners_all, t, half_life_draws=int(half_life))
            pred = None
            if abs(p_even - 0.5) >= float(delta):
                pred = 'even' if p_even >= 0.5 else 'odd'
                calls += 1
            true_even = (sum(rec.digits_of(winners_all[t])) % 2 == 0)
            hit = None
            if pred is not None:
                hit = int((pred == 'even') == bool(true_even))
                hits += hit
            rows.append({
                't': t,
                'seed': winners_all[t-1],
                'winner': winners_all[t],
                'p_even_prior': round(p_even, 4),
                'pred': pred if pred else 'abstain',
                'hit': hit,
            })
        dfp = pd.DataFrame(rows)
        st.dataframe(dfp, use_container_width=True)
        if calls:
            acc = hits / calls
            cov = calls / len(dfp)
            st.success(f"Prior-only predictor → accuracy={acc:.3f} on predicted days, coverage={cov:.3f}")
        else:
            st.warning("No predictions made at this confidence threshold (try lowering delta).")

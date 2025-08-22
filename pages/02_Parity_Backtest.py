import os, sys, subprocess
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Parity Backtest", page_icon="📊", layout="wide")
st.title("📊 Parity Backtest (Affinity-Ordered Top% vs Winner)")

st.markdown(
    "This runs a historical check: for each day, score combos by your **affinity**, "
    "look at the **majority parity** (sum parity and majority parity) in the top-k%, "
    "and compare to the actual winner. It writes two CSVs and shows them below."
)

# ---- Inputs ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Data")
    winners_path = st.text_input("Winners CSV path", "DC5_Midday_Full_Cleaned_Expanded.csv")
    uploaded = st.file_uploader("...or upload Winners CSV", type=["csv"],
                                help="Uploading overrides the path above for this run only.")

with col2:
    st.subheader("Backtest Settings")
    n_days = st.number_input("Days to test (most recent)", 30, 2000, 240, step=10)
    pool_mode = st.selectbox("Pool mode", ["1digit", "2digit"], index=0,
                             help="1digit: each combo shares ≥1 digit with seed. 2digit: shares ≥2.")
    pool_size = st.number_input("Pool size per day", 100, 20000, 1800, step=100)
    top_pcts = st.text_input("Top % thresholds (comma)", "0.2,0.3", help="e.g., 0.2 = top 20% by affinity")

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

    cmd = [
        sys.executable, "affinity_parity_backtest.py",
        "--winners", winners_to_use,
        "--n_days", str(int(n_days)),
        "--pool_mode", pool_mode,
        "--pool_size", str(int(pool_size)),
        "--top_pcts", top_pcts,
        "--out_prefix", out_prefix,
    ]

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

        sum_path = f"{out_prefix}_summary.csv"
        day_path = f"{out_prefix}_backtest.csv"

        if os.path.exists(sum_path):
            st.subheader("Summary (accuracy & lift)")
            df_sum = pd.read_csv(sum_path)
            st.dataframe(df_sum, use_container_width=True)
            st.download_button("Download summary CSV",
                               data=open(sum_path, "rb").read(),
                               file_name=sum_path, mime="text/csv")
        else:
            st.warning("Summary CSV not found.")

        if os.path.exists(day_path):
            st.subheader("Per-day results")
            df_day = pd.read_csv(day_path)
            st.dataframe(df_day, use_container_width=True)
            st.download_button("Download per-day CSV",
                               data=open(day_path, "rb").read(),
                               file_name=day_path, mime="text/csv")
        else:
            st.warning("Per-day CSV not found.")

st.caption("Tip: Use '1digit' & pool size ~1800 to mirror your normal pool. Try '2digit' & ~900 to see if stricter pools change the signal.")

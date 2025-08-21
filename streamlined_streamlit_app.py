import streamlit as st
import pandas as pd
import os
from pathlib import Path

# --- Title ---
st.title("DCS Recommender — Winner-Preserving")

# --- CSV Inputs ---
winners_csv = st.text_input("Winners CSV path (in repo)", "DC5_Midday_Full_Cleaned_Expanded.csv")
filters_csv = st.text_input("Filters CSV path (in repo)", "lottery_filters_batch_10.csv")

# --- Seed inputs with validation ---
seed = st.text_input("Override seed (5 digits, optional)", max_chars=5, placeholder="#####")
prev_seed = st.text_input("Override 1-back (optional)", max_chars=5, placeholder="#####")
prev_prev_seed = st.text_input("Override 2-back (optional)", max_chars=5, placeholder="#####")
keep_combo = st.text_input("Keep combo (5 digits)", max_chars=5, placeholder="#####")

applicable_ids = st.text_area("Paste Applicable-Only IDs (optional, comma or space separated)")

# --- Run button ---
if st.button("Run recommender now"):
    # Validation: ensure 5-digit numeric strings where filled
    invalids = []
    for field_name, value in [("Seed", seed), ("Prev seed", prev_seed), ("Prev-prev seed", prev_prev_seed), ("Keep combo", keep_combo)]:
        if value and (not value.isdigit() or len(value) != 5):
            invalids.append(f"{field_name}: must be exactly 5 digits")

    if invalids:
        st.error("Validation failed:\n" + "\n".join(invalids))
    else:
        kwargs = dict(
            winners_csv=winners_csv,
            filters_csv=filters_csv,
            seed=seed or None,
            prev_seed=prev_seed or None,
            prev_prev_seed=prev_prev_seed or None,
            keep_combo=keep_combo or None,
            applicable_ids=[x.strip() for x in applicable_ids.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]
        )

        try:
            from recommender import main as run_recommender
            result = run_recommender(**kwargs)
            if result is not None:
                st.success("✅ Recommender finished")
                st.dataframe(pd.DataFrame(result))
        except ImportError:
            st.error("❌ recommender.py is missing — please add it to the repo.")

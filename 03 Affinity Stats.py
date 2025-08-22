# -*- coding: utf-8 -*-
"""
03_Affinity_Stats — Streamlit page (patched)

What this does
--------------
Adds blue‑lined UI controls for the Affinity pre‑trim (keep bands, exclude top‑X%,
weights) and passes them into your existing pipeline in `recommender.py`.

How to use
----------
1) Make sure `recommender.py` (your updated file with affinity pre‑trim) and
   `affinity_ui_patch.py` (the sidebar UI helper) are in the same folder.
2) Replace your existing "03 Affinity Stats" Streamlit page with this file
   (or rename this to match your multipage filename convention).
3) Run Streamlit as usual.

This page tries, in order, to call the function you already have:
    • recommender.run_affinity_stats(...)
    • recommender.run_pipeline(...)
If neither exists, it will NOT break your app — it will show the collected
kwargs so you can quickly wire the call you prefer.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import streamlit as st

# --- Imports from your codebase ---
import importlib
recommender = importlib.import_module("recommender")
from affinity_ui_patch import render_affinity_sidebar

# --- Page config ---
st.set_page_config(page_title="03 • Affinity Stats (Patched)", layout="wide")
st.title("03 • Affinity Stats")
st.caption("Winner‑preserving affinity pre‑trim runs BEFORE CSV filters. Use the controls in the sidebar.")

# --- Collect UI options (winner‑preserving) ---
ui_opts = render_affinity_sidebar()  # {affinity_keep_bands, affinity_exclude_top_pct, affinity_weights}

# Optional: echo current + best defaults for quick revert
with st.expander("Current Affinity Settings", expanded=False):
    st.json(ui_opts)

# --- Call into your pipeline safely ---
# Try the most specific function name first, then fall back.
run_called = False
err: str | None = None

try:
    if hasattr(recommender, "run_affinity_stats"):
        # Signature should include the three affinity kwargs; pass through others via session/state in your module
        recommender.run_affinity_stats(**ui_opts)  # type: ignore[arg-type]
        run_called = True
    elif hasattr(recommender, "run_pipeline"):
        recommender.run_pipeline(**ui_opts)  # type: ignore[arg-type]
        run_called = True
except Exception as e:  # pragma: no cover
    err = f"Pipeline error: {e}"

# --- If nothing ran, provide a zero‑risk wiring hint without breaking the page ---
if not run_called:
    st.warning(
        "Couldn't find `run_affinity_stats` or `run_pipeline` in recommender.py. "
        "Below is a one‑liner you can drop into your function call to wire the UI options."
    )
    st.code(
        """
# Example inside your recommender.py (where you build the pool):
# def run_pipeline(..., affinity_keep_bands=None, affinity_exclude_top_pct=0.0, affinity_weights=None):
#     ...
#     pool = apply_affinity_pretrim(
#         pool,
#         keep_bands=affinity_keep_bands or [],
#         exclude_top_pct=float(affinity_exclude_top_pct or 0.0),
#         weights=affinity_weights or DEFAULT_AFF_WEIGHTS,
#         winner_preserving=True,
#     )
#     ...
        """,
        language="python",
    )
    st.subheader("Collected kwargs (ready to pass into your pipeline)")
    st.json(ui_opts)

if err:
    st.error(err)


# --- Archive today's pool for historical testing ---
from pathlib import Path

with st.expander("Archive today's pool for historical tests", expanded=False):
    # Use the same path your app uses; fallback to repo-root today_pool.csv
    pool_path = getattr(recommender, "TODAY_POOL_CSV", None) or "today_pool.csv"
    st.caption(f"Pool file: {pool_path}")

    if Path(pool_path).exists():
        if st.button("Archive today's pool now"):
            try:
                from recommender import archive_today_pool
                dest = archive_today_pool(pool_csv=pool_path)  # uses WINNERS_CSV to name by date
                st.success(f"Archived to {dest}")
            except Exception as e:
                st.error(f"Archive failed: {e}")
    else:
        st.info("No pool file found yet. Make sure today_pool.csv exists at repo root or set recommender.TODAY_POOL_CSV.")

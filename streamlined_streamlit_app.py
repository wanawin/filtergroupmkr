try:
    pass
except Exception as _boot_err:  # this will be replaced when the file executes
    import streamlit as st
    st.exception(_boot_err)
    st.stop()
# streamlined_streamlit_app.py
# One-screen Streamlit UI that:
#  1) Shows when the case tables were last refreshed (and basic meta)
#  2) Lets you refresh (Full or Incremental)
#  3) Runs the recommender using today's seed context
#  4) Shows and lets you download the outputs

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Import your modules
from profiler import build_case_history, refresh_incremental
from recommender import main as run_recommender
from recommender import (
    WINNERS_CSV, FILTERS_CSV, TODAY_POOL_CSV, OUTPUT_DIR
)

st.set_page_config(page_title="DC5 Recommender — Streamlined", layout="wide")
st.title("DC5 Recommender — Streamlined")

# Defaults (change here if your filenames differ)
DEFAULT_WINNERS = WINNERS_CSV
DEFAULT_FILTERS = FILTERS_CSV
DEFAULT_POOL    = TODAY_POOL_CSV if TODAY_POOL_CSV else "today_pool.csv"
OUT_DIR = Path(OUTPUT_DIR)

# ---------- Helpers ----------

def _fmt_dt(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except Exception:
        return "—"
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _load_meta() -> dict:
    meta_path = OUT_DIR / "case_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {}
    return {}

# ---------- Step 1: Refresh ----------

with st.expander("Step 1 · (Optional) Refresh Case Tables", expanded=False):
    st.write("Build or update the historical case tables used for case-aware recommendations.")

    winners_path = st.text_input("Winners CSV path", value=DEFAULT_WINNERS)
    filters_path = st.text_input("Filters CSV path", value=DEFAULT_FILTERS)

    # Status panel
    ch = OUT_DIR / "case_history.csv"
    cs = OUT_DIR / "case_filter_stats.csv"
    meta = _load_meta()

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("case_history.csv", "exists" if ch.exists() else "missing", _fmt_dt(ch))
    with colB:
        st.metric("case_filter_stats.csv", "exists" if cs.exists() else "missing", _fmt_dt(cs))
    with colC:
        last_count = str(meta.get("count", "—"))
        st.metric("Last winners count", last_count, None)

    st.caption("Choose a refresh type. Incremental drops the oldest N and adds the newest N.")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        keep_last_n = st.number_input("Incremental window (N draws)", min_value=1, max_value=500, value=7, step=1)
        if st.button("Incremental refresh now"):
            try:
                refresh_incremental(winners_csv=winners_path, filters_csv=filters_path, keep_last_n=int(keep_last_n))
                st.success("Incremental refresh completed.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")
    with col2:
        if st.button("Full rebuild now"):
            try:
                build_case_history(winners_csv=winners_path, filters_csv=filters_path)
                st.success("Full rebuild completed.")
            except Exception as e:
                st.error(f"Full rebuild failed: {e}")
    with col3:
        if st.button("Re-check status"):
            st.experimental_rerun()

st.markdown("---")

# ---------- Step 2: Recommender ----------

with st.expander("Step 2 · Run Recommender", expanded=True):
    st.write("Provide today's seed context (no winner needed). Leave Keep Combo blank if you don't know it.")

    # Use a form so the script doesn't re-run on every keystroke
    with st.form("run_form"):
        colA, colB = st.columns([1,1])
        with colA:
            pool_path = st.text_input("Today pool CSV (optional)", value=DEFAULT_POOL)
            target_max = st.number_input("Target pool size (stop when ≤ this)", min_value=10, max_value=200, value=44, step=1)
            always_keep = st.checkbox("Preserve winner (winner-agnostic safe mode if unknown)", value=True)
            minimize_beyond_target = st.checkbox("Keep minimizing even after target reached", value=True)
        with colB:
            st.caption("Seed triplet (optional, improves case matching)")
            override_seed = st.text_input("Seed (today's seed / yesterday's draw)", value="", max_chars=5, placeholder="#####")
            override_prev = st.text_input("Prev seed", value="", max_chars=5, placeholder="#####")
            override_prevprev = st.text_input("Prev-prev seed", value="", max_chars=5, placeholder="#####")

            st.caption("Keep Combo (optional)")
            force_keep_combo = st.text_input("Keep combo (5 digits)", value="", max_chars=5, placeholder="#####")

        st.caption("Paste Applicable-Only IDs (optional, comma or space separated)")
        applicable_only = st.text_area("Applicable IDs", value="", height=70)

        submitted = st.form_submit_button("Run recommender now")

    def _is_five(s: str) -> bool:
        s = (s or "").strip()
        return len(s) == 5 and s.isdigit()

    seeds_ok = True
    bad_fields = []
    for label, val in [("Seed", override_seed), ("Prev seed", override_prev), ("Prev-prev seed", override_prevprev), ("Keep combo", force_keep_combo)]:
        if val.strip() != "" and not _is_five(val):
            seeds_ok = False
            bad_fields.append(label)

    if submitted:
        if not seeds_ok:
            st.error(f"The following fields must be exactly 5 digits (or left blank): {', '.join(bad_fields)}")
        else:
            try:
                kwargs = dict(
                    winners_csv=DEFAULT_WINNERS,
                    filters_csv=DEFAULT_FILTERS,
                    today_pool_csv=pool_path if pool_path else None,
                    target_max=int(target_max),
                    always_keep_winner=bool(always_keep),
                    minimize_beyond_target=bool(minimize_beyond_target),
                    force_keep_combo=force_keep_combo.strip() or None,
                    override_seed=override_seed.strip() or None,
                    override_prev=override_prev.strip() or None,
                    override_prevprev=override_prevprev.strip() or None,
                    applicable_only=applicable_only.strip() or None,
                    train_mode=False,
                )
                from recommender import main as run_recommender
                run_recommender(**kwargs)
                st.success("Recommender finished.")
            except Exception as e:
                st.error(f"Recommender failed: {e}")

st.markdown("---")

# ---------- Step 3: Results ----------

with st.expander("Step 3 · Results", expanded=True):
    out_dir = OUT_DIR

    def try_show(csv_name: str, label: str, n: int = 50):
        path = out_dir / csv_name
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.subheader(label)
                st.dataframe(df.head(n))
                st.download_button(
                    f"Download {csv_name}",
                    data=path.read_bytes(),
                    file_name=csv_name,
                    mime="text/csv"
                )
            except Exception as e:
                st.warning(f"{csv_name}: could not load ({e})")
        else:
            st.info(f"{csv_name} not found yet.")

    try_show("recommender_sequence.csv", "Apply in this order (winner-preserving if enabled)")
    try_show("avoid_by_case.csv", "Avoid pairs — today’s case")
    try_show("avoid_pairs.csv", "Avoid pairs — global in bucket")
    try_show("do_not_apply.csv", "Do not apply / apply late — today’s case")
    try_show("case_filter_stats.csv", "(Trainer) Case filter stats")
    try_show("case_history.csv", "(Trainer) Case history")

st.caption("Tip: If results look unchanged day-to-day, use the Seed triplet to align the case, then refresh case tables.")

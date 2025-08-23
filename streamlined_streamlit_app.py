import streamlit as st
import pandas as pd
from pathlib import Path

# Import recommender entrypoint lazily later (so the app still loads if missing)
# from recommender import main as run_recommender

# ========================
# Paths (repo-root relative)
# ========================
DEFAULT_WINNERS = "DC5_Midday_Full_Cleaned_Expanded.csv"
DEFAULT_FILTERS = "lottery_filters_batch_10.csv"
DEFAULT_POOL    = "today_pool.csv"
CASE_HISTORY    = "case_history.csv"
CASE_STATS      = "case_filter_stats.csv"
OUT_DIR         = Path(".")

st.set_page_config(page_title="DCS Recommender — Streamlined", layout="wide")

# ---------- helpers ----------

def _fmt_dt(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except Exception:
        return "—"
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _exists_info(path_str: str):
    p = Path(path_str)
    return p.exists(), p

# ---------- Step 1: Build / refresh profiler tables ----------
with st.expander("Step 1 — Build or update case tables (Profiler)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        winners_path = st.text_input("Winners CSV path (in repo)", DEFAULT_WINNERS, key="winners_step1")
        filters_path = st.text_input("Filters CSV path (in repo)", DEFAULT_FILTERS, key="filters_step1")
    with c2:
        ch_ok, ch_p = _exists_info(CASE_HISTORY)
        cs_ok, cs_p = _exists_info(CASE_STATS)
        st.caption("Current profiler artifacts in repo root:")
        st.write(f"• **{CASE_HISTORY}**: {'✅' if ch_ok else '❌ missing'}  {('— '+_fmt_dt(ch_p)) if ch_ok else ''}")
        st.write(f"• **{CASE_STATS}**: {'✅' if cs_ok else '❌ missing'}  {('— '+_fmt_dt(cs_p)) if cs_ok else ''}")

    colA, colB, colC = st.columns([1,1,3])
    with colA:
        keep_last_n = st.number_input("Incremental window (N draws)", value=7, min_value=1, max_value=200, step=1)
        if st.button("Incremental refresh now", type="primary"):
            try:
                from profiler import refresh_incremental
            except Exception:
                st.error("Profiler module not found. Ensure **profiler.py** is in the repo.")
            else:
                try:
                    refresh_incremental(winners_csv=winners_path, filters_csv=filters_path, keep_last_n=int(keep_last_n))
                    st.success("Incremental refresh complete.")
                except Exception as e:
                    st.exception(e)
    with colB:
        if st.button("Full rebuild now"):
            try:
                from profiler import build_case_history
            except Exception:
                st.error("Profiler module not found. Ensure **profiler.py** is in the repo.")
            else:
                try:
                    build_case_history(winners_csv=winners_path, filters_csv=filters_path)
                    st.success("Full rebuild complete.")
                except Exception as e:
                    st.exception(e)
    with colC:
        if st.button("Re-check status"):
            ch_ok, ch_p = _exists_info(CASE_HISTORY)
            cs_ok, cs_p = _exists_info(CASE_STATS)
            st.info(f"{CASE_HISTORY}: {'present' if ch_ok else 'missing'} — {_fmt_dt(ch_p) if ch_ok else '—'}\n\n{CASE_STATS}: {'present' if cs_ok else 'missing'} — {_fmt_dt(cs_p) if cs_ok else '—'}")

# ---------- Step 2: Run recommender ----------
st.markdown("---")
st.subheader("Step 2 — Run recommender (winner-preserving)")

with st.form("run_form"):
    c1, c2 = st.columns(2)
    with c1:
        winners_csv = st.text_input("Winners CSV path (in repo)", DEFAULT_WINNERS, key="winners_step2")
        filters_csv = st.text_input("Filters CSV path (in repo)", DEFAULT_FILTERS, key="filters_step2")
        minimize_beyond_target = st.checkbox("Keep minimizing even after target reached", True)
    with c2:
        seed = st.text_input("Override seed (5 digits, optional)", max_chars=5, placeholder="#####")
        prev_seed = st.text_input("Override 1-back (optional)", max_chars=5, placeholder="#####")
        prev_prev_seed = st.text_input("Override 2-back (optional)", max_chars=5, placeholder="#####")
        keep_combo = st.text_input("Keep combo (5 digits)", max_chars=5, placeholder="#####")

    applicable_ids = st.text_area("Paste Applicable-Only IDs (optional, comma or space separated)")

    submitted = st.form_submit_button("Run recommender now", type="primary")

if submitted:
    # Validation: ensure 5-digit numeric strings where filled
    invalids = []
    for field_name, value in [("Seed", seed), ("Prev seed", prev_seed), ("Prev-prev seed", prev_prev_seed), ("Keep combo", keep_combo)]:
        if value and (not value.isdigit() or len(value) != 5):
            invalids.append(f"{field_name}: must be exactly 5 digits")
    if invalids:
        st.error("Validation failed:\n" + "\n".join(invalids))
    else:
        ids = [x.strip() for x in applicable_ids.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]
        kwargs = dict(
            winners_csv=winners_csv,
            filters_csv=filters_csv,
            today_pool_csv=DEFAULT_POOL if Path(DEFAULT_POOL).exists() else None,
            always_keep_winner=True,
            minimize_beyond_target=minimize_beyond_target,
            force_keep_combo=keep_combo or None,
            override_seed=seed or None,
            override_prev=prev_seed or None,
            override_prevprev=prev_prev_seed or None,
            applicable_only=ids or None,
        )

        # Import recommender safely
        try:
            from recommender import main as run_recommender
        except Exception:
            st.error("❌ recommender.py is missing — please add it to the repo.")
        else:
            try:
                result = run_recommender(**kwargs)
                st.success("✅ Recommender finished")

                # ---- Additional NoPool panel ----
                nopool_csv = Path("NoPool_today_pairs_TOP.csv")
                if nopool_csv.exists():
                    try:
                        df_np = pd.read_csv(nopool_csv)
                        if df_np.empty:
                            st.info("No additional NoPool recommendations today (min_days=60, min_keep=75%). See one_pager.html for details.")
                        else:
                            st.success(f"Additional NoPool recommendations: {len(df_np)} pair(s). See one_pager.html for details.")
                            st.dataframe(df_np.head(20), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not read {nopool_csv.name}: {e}")
                else:
                    st.info("NoPool summary file not found. (Feature may be off or no pairs qualified.)")

                # ---- Show outputs if present ----
                outputs = [
                    "recommender_sequence.csv",
                    "pool_reduction_log.csv",
                    "avoid_pairs.csv",
                    "do_not_apply.csv",
                    "one_pager.md",
                    "one_pager.html",
                ]
                for fname in outputs:
                    p = OUT_DIR / fname
                    if p.exists():
                        st.write(f"**{fname}** — updated {_fmt_dt(p)}")
                        if p.suffix == ".md":
                            st.markdown(p.read_text(encoding="utf-8"))
                        elif p.suffix == ".html":
                            st.download_button(
                                "Download one_pager.html",
                                data=p.read_bytes(),
                                file_name="one_pager.html",
                            )
                        else:
                            try:
                                st.dataframe(pd.read_csv(p), use_container_width=True)
                            except Exception:
                                st.code(p.read_text(encoding='utf-8')[:5000])

            except Exception as e:
                st.exception(e)

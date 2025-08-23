import streamlit as st
import pandas as pd
from pathlib import Path
import sys, traceback, importlib

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

def _latest(glob_pat: str) -> Path | None:
    files = list(Path(".").glob(glob_pat))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime: float):
    # mtime is used solely to bust cache when the file updates
    return pd.read_csv(path_str)

# Simple CSS for group headers
st.markdown("""
<style>
.group-header{
  padding:8px 12px; margin:18px 0 8px;
  border-left:8px solid #0ea5e9; background:#f0f9ff;
  border-radius:8px; font-weight:600;
}
.small-muted{color:#64748b; font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ---------- Step 1: Build / refresh profiler tables ----------
with st.expander("Step 1 — Build or update case tables (Profiler)", expanded=False):
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
st.subheader("Step 2 — Run recommender (winner prediction)")

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
            always_keep_winner=False,  # backtest-friendly
            minimize_beyond_target=minimize_beyond_target,
            force_keep_combo=keep_combo or None,
            override_seed=seed or None,
            override_prev=prev_seed or None,
            override_prevprev=prev_prev_seed or None,
            applicable_only=ids or None,
        )

        try:
            if "recommender" in sys.modules:
                del sys.modules["recommender"]
            recommender = importlib.import_module("recommender")
            run_recommender = recommender.main
        except Exception as e:
            st.error("❌ Could not import recommender.py. See diagnostics below.")
            with st.expander("Import diagnostics"):
                st.code("".join(traceback.format_exception_only(type(e), e)))
                st.write("sys.path:", sys.path)
                st.write("CWD:", str(Path.cwd()))
                st.write("Files in CWD:", ", ".join(sorted(p.name for p in Path('.').iterdir())))
        else:
            try:
                _ = run_recommender(**kwargs)
                st.success("✅ Recommender finished")
                st.session_state["last_run_ok"] = True
            except Exception as e:
                st.exception(e)

# ---- Always-visible outputs (persist across re-runs) ----
st.markdown("---")
st.subheader("Latest outputs on disk (auto-refresh)")

# Core files
outputs = [
    "recommender_sequence.csv",
    "pool_reduction_log.csv",
    "avoid_pairs.csv",
    "do_not_apply.csv",
    "one_pager.html",
]
for fname in outputs:
    p = OUT_DIR / fname
    if p.exists():
        st.write(f"**{fname}** — updated {_fmt_dt(p)}")
        if p.suffix == ".html":
            st.download_button("Download one_pager.html", data=p.read_bytes(), file_name="one_pager.html")
        else:
            try:
                df = _read_csv_cached(str(p), p.stat().st_mtime)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                st.code(p.read_text(encoding="utf-8")[:5000])

# ---- Final SafeLists section (persisted + group delineation) ----
st.markdown("---")
st.subheader("Final SafeLists")

tabs = st.tabs(["Sequence-biased", "Safest-first", "Grouped by aggressiveness"])

def _show_table_with_row_choice(df: pd.DataFrame, key_prefix: str):
    choice = st.radio("Rows to show", ["100", "300", "All"], horizontal=True, key=f"{key_prefix}_rows")
    if choice == "All":
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(900, 40 + 28 * len(df)))
    else:
        n = int(choice)
        st.dataframe(df.head(n), use_container_width=True, hide_index=True, height=700)

def _bucket_from_elims(c: float) -> str:
    try:
        c = float(c)
    except Exception:
        c = 0.0
    if c >= 701: return "701+"
    if c >= 501: return "501–700"
    if c >= 301: return "301–500"
    if c >= 101: return "101–300"
    if c >=  61: return "61–100"
    if c >=   1: return "1–60"
    return "0"

def _download_buttons(latest_csv: Path | None, latest_txt: Path | None, latest_html: Path | None, key_prefix: str):
    if not latest_csv:
        return
    fmt = st.radio("Download format", ["CSV", "TXT", "HTML"], horizontal=True, key=f"{key_prefix}_dlfmt")
    selected = {"CSV": latest_csv,
                "TXT": latest_txt if (latest_txt and latest_txt.exists()) else latest_csv,
                "HTML": latest_html if (latest_html and latest_html.exists()) else latest_csv}[fmt]
    mime = "text/csv" if fmt=="CSV" else "text/plain" if fmt=="TXT" else "text/html"
    with open(selected, "rb") as fh:
        st.download_button(f"Download ({fmt})", data=fh, file_name=selected.name, mime=mime, key=f"{key_prefix}_dlbtn")

# Tab 1: Sequence-biased
with tabs[0]:
    latest_csv  = _latest("final_safe_ordered_*.csv")
    latest_txt  = _latest("final_safe_ordered_*.txt")
    latest_html = _latest("final_safe_ordered_*.html")
    if latest_csv:
        st.caption(f"File: {latest_csv.name}")
        df = _read_csv_cached(str(latest_csv), latest_csv.stat().st_mtime)
        _show_table_with_row_choice(df, "seq")
        _download_buttons(latest_csv, latest_txt, latest_html, "seq")
    else:
        st.info("No sequence-biased list found yet.")

# Tab 2: Safest-first
with tabs[1]:
    latest_csv  = _latest("final_safe_safest_first_*.csv")
    latest_txt  = _latest("final_safe_safest_first_*.txt")
    latest_html = _latest("final_safe_safest_first_*.html")
    if latest_csv:
        st.caption(f"File: {latest_csv.name}")
        df = _read_csv_cached(str(latest_csv), latest_csv.stat().st_mtime)
        _show_table_with_row_choice(df, "safest")
        _download_buttons(latest_csv, latest_txt, latest_html, "safest")
    else:
        st.info("No safest-first list found yet.")

# Tab 3: Grouped by aggressiveness (with clear headers)
with tabs[2]:
    latest_csv  = _latest("final_safe_grouped_*.csv")
    latest_txt  = _latest("final_safe_grouped_*.txt")
    latest_html = _latest("final_safe_grouped_*.html")
    if latest_csv:
        st.caption(f"File: {latest_csv.name}")
        df = _read_csv_cached(str(latest_csv), latest_csv.stat().st_mtime)

        # ensure there is a 'group' column even if older file lacks it
        if "group" not in df.columns:
            # try to derive from elim_count if present
            if "elim_count" in df.columns:
                df["group"] = df["elim_count"].map(_bucket_from_elims)
            else:
                df["group"] = "unknown"

        order_buckets = ["701+","501–700","301–500","101–300","61–100","1–60","0","unknown"]
        for grp in order_buckets:
            block = df[df["group"] == grp]
            if block.empty:
                continue
            st.markdown(f"<div class='group-header'>{grp} — {len(block)} filters</div>", unsafe_allow_html=True)
            choice_key = f"group_{grp.replace('–','-').replace('+','plus')}"
            choice = st.radio("Rows to show", ["50", "All"], horizontal=True, key=choice_key)
            if choice == "All":
                st.dataframe(block, use_container_width=True, hide_index=True, height=min(800, 40 + 28 * len(block)))
            else:
                st.dataframe(block.head(50), use_container_width=True, hide_index=True, height=500)
            st.markdown("<div class='small-muted'>— end of group —</div>", unsafe_allow_html=True)

        _download_buttons(latest_csv, latest_txt, latest_html, "grouped")
    else:
        st.info("No grouped list found yet.")
        # ---------- Archetype reports (NEW) ----------
st.markdown("---")
st.header("Archetype reports")

def _load_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        df = pd.read_csv(p)
    except Exception:
        df = None
    return p, df

ar_files = [
    ("archetypes_history_summary.csv", "History summary"),
    ("archetypes_today_pool_mix.csv", "Today's pool mix"),
    ("archetypes_feature_stats.csv", "Single-feature stats"),
    ("archetypes_history_rows.csv", "History rows (full)"),
]

for fname, title in ar_files:
    p, df = _load_csv(fname)
    with st.expander(f"{title} — {fname}" + (f"  ·  updated {_fmt_dt(p)}" if p else ""), expanded=False):
        if p is None:
            st.info("Not found yet.")
            continue
        # Rows toggle
        if df is not None:
            show_all = st.radio(
                f"Rows to show for {fname}",
                options=["100", "300", "All"], horizontal=True, index=0
            )
            n = None
            if show_all == "100": n = 100
            elif show_all == "300": n = 300
            if n is not None:
                st.dataframe(df.head(n), use_container_width=True, hide_index=True)
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.code(p.read_text(encoding="utf-8")[:5000])

        # Download
        fmt = st.radio(f"Download format for {fname}", ["CSV", "TXT"], horizontal=True, index=0)
        with open(p, "rb") as fh:
            st.download_button(
                f"Download {fname}",
                data=fh,
                file_name=fname,
                mime="text/csv" if fmt == "CSV" else "text/plain",
                type="secondary",
            )


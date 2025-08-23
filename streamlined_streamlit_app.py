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

# --- NEW: helper to find the newest file matching a glob ---
def _latest(path_glob: str) -> Path | None:
    files = list(Path(".").glob(path_glob))
    if not files:
        ret

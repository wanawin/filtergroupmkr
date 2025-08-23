import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="DCS Recommender — Streamlined", layout="wide")

# ========================
# Paths (repo-root relative)
# ========================
DEFAULT_WINNERS = "DC5_Midday_Full_Cleaned_Expanded.csv"
DEFAULT_FILTERS = "lottery_filters_batch_10.csv"
DEFAULT_POOL    = "today_pool.csv"
CASE_HISTORY    = "case_history.csv"
CASE_STATS      = "case_filter_stats.csv"
OUT_DIR         = Path(".")

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

def _latest(path_glob: str) -> Path | None:
    files = list(Path(".").glob(path_glob))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def _resolve_recommender_main():
    """
    Try to get recommender.main with multiple strategies:
    1) Normal import
    2) Load from known file locations with importlib
    Returns (callable_main, debug_info_str)
    """
    debug = []
    try:
        import recommender  # type: ignore
        return recommender.main, "Loaded via normal import."
    except Exception as e:
        debug.append(f"Normal import failed: {e!r}")

    import importlib.util, sys, os

    # Candidate locations to try explicitly
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "recommender.py",
        Path.cwd() / "recommender.py",
        here.parent / "recommender.py",
        Path.cwd() / "app" / "recommender.py",
        Path.cwd() / "src" / "recommender.py",
    ]
    tried = []

"""
Drop-in sidebar controls to expose affinity pre-trim options.

Usage in your Streamlit app:

1. Save this file as `affinity_ui_patch.py` in the same folder as your main Streamlit app (`recommender.py`).
2. In your Streamlit page (where you run your pipeline), add at the top:

       from affinity_ui_patch import render_affinity_sidebar

3. Just before calling your pipeline (e.g., `results = run_pipeline(...)`), add:

       ui_opts = render_affinity_sidebar()
       results = run_pipeline(..., **ui_opts)

That’s it — the sidebar controls will appear automatically. Winner-preserving semantics remain enforced in recommender.py.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import streamlit as st

try:
    from recommender import DEFAULT_AFF_WEIGHTS  # type: ignore
except Exception:
    DEFAULT_AFF_WEIGHTS = {
        "sum": 1.0,
        "spread": 1.0,
        "structure": 1.0,
        "parity": 1.0,
        "hi8": 1.0,
        "overlap": 1.0,
    }

BEST_DEFAULTS = {
    "bands": [(0.05, 0.25), (0.60, 0.72)],
    "exclude_top": 0.20,
    "weights": {
        "sum": 1.2,
        "spread": 1.0,
        "structure": 1.0,
        "parity": 0.8,
        "hi8": 1.0,
        "overlap": 1.5,
    },
}

_HELP = {
    "bands": "Comma-separated percentile ranges in 0..1. Example: '0.05-0.25, 0.60-0.72'\nKeeps the UNION of these bands. Leave empty to disable.",
    "exclude": "Exclude top-X% highest-affinity combos *after* band keep (if any). Example: 0.20 drops top 20%. Set to 0 to disable.",
    "weights": "Weights for affinity components. Higher = more influence in score.",
}


def _parse_bands(text: str) -> List[Tuple[float, float]]:
    bands: List[Tuple[float, float]] = []
    if not text.strip():
        return bands
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            st.warning(f"Ignored band '{chunk}' — use 'lo-hi' format in 0..1.")
            continue
        lo_s, hi_s = chunk.split("-", 1)
        try:
            lo = float(lo_s.strip())
            hi = float(hi_s.strip())
        except ValueError:
            st.warning(f"Ignored band '{chunk}' — not numeric.")
            continue
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if hi <= lo:
            st.warning(f"Ignored band '{chunk}' — hi must be > lo.")
            continue
        bands.append((lo, hi))
    if not bands:
        return bands
    bands.sort()
    merged: List[Tuple[float, float]] = []
    cur_lo, cur_hi = bands[0]
    for lo, hi in bands[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    return merged


def render_affinity_sidebar() -> Dict[str, object]:
    """Render sidebar UI for affinity pre-trim.

    Returns a dict with keys:
      - affinity_keep_bands: List[Tuple[float, float]]
      - affinity_exclude_top_pct: float
      - affinity_weights: Dict[str, float]
    """
    with st.sidebar.expander("⚙️ Affinity Pre-Trim (winner-preserving)", expanded=False):
        st.caption("Runs BEFORE CSV filters. Winner is preserved — you'll see skip reasons in outputs.")

        st.markdown(
            f"**Best tested defaults:** bands={BEST_DEFAULTS['bands']}, "
            f"exclude_top={BEST_DEFAULTS['exclude_top']}, "
            f"weights={BEST_DEFAULTS['weights']}"
        )

        bands_text = st.text_input(
            "Keep bands (percentiles)",
            value="",
            help=_HELP["bands"],
            placeholder="e.g. 0.05-0.25, 0.60-0.72",
        )
        keep_bands = _parse_bands(bands_text)
        if keep_bands:
            st.write("Active bands:", keep_bands)

        exclude_top = st.slider(
            "Exclude top-X% by affinity (0..0.90)",
            min_value=0.0,
            max_value=0.90,
            value=0.0,
            step=0.01,
            help=_HELP["exclude"],
        )

        st.markdown("**Weights** (0 = ignore component)")
        w: Dict[str, float] = {}
        for key in ["sum", "spread", "structure", "parity", "hi8", "overlap"]:
            default = float(DEFAULT_AFF_WEIGHTS.get(key, 1.0))
            w[key] = st.slider(
                f"{key}", min_value=0.0, max_value=3.0, value=default, step=0.05,
                help=_HELP["weights"],
            )

        if st.button("🔄 Reset to Best Defaults"):
            st.session_state["affinity_keep_bands"] = BEST_DEFAULTS["bands"]
            st.session_state["affinity_exclude_top_pct"] = BEST_DEFAULTS["exclude_top"]
            st.session_state["affinity_weights"] = BEST_DEFAULTS["weights"]
            st.experimental_rerun()

        st.info(
            "Pre-trim preview — keep bands: "
            + (str(keep_bands) if keep_bands else "<none>")
            + f" | exclude top: {exclude_top:.2f} | weights: {w}"
        )

    return {
        "affinity_keep_bands": keep_bands,
        "affinity_exclude_top_pct": float(exclude_top),
        "affinity_weights": w,
    }

if __name__ == "__main__":
    st.set_page_config(page_title="Affinity UI Patch Test", layout="wide")
    st.title("Affinity Controls — Sandbox")
    opts = render_affinity_sidebar()
    st.write("Returned kwargs:")
    st.json(opts)

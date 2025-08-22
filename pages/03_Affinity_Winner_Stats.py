import math
from pathlib import Path
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Affinity Winner Stats", page_icon="📈", layout="wide")
st.title("Affinity vs Winner — Distribution & Buckets")

st.markdown(
    "This page evaluates where the **actual winner** lands in the **affinity ranking** "
    "built from that day's seed.\n\n"
    "Pool construction matches your production recipe:\n"
    "• 1-digit rule only (combo must share ≥1 distinct digit with the seed)\n"
    "• optional digit-sum percentile filter measured against the full 2002 canonical combos\n"
    "• canonical, de-duplicated combos by construction\n"
    "• remove quints and quads\n"
)

# -------------------------------
# Import recommender + helpers
# -------------------------------
try:
    import recommender as rec
except Exception as e:
    st.error(f"Failed to import recommender.py: {e}")
    st.stop()

def parity_major_label(digs):
    return "even>=3" if sum(1 for d in digs if d % 2 == 0) >= 3 else "even<=2"

# Use rec.combo_affinity if available; else fallback that mirrors prior heuristic
if hasattr(rec, "combo_affinity"):
    combo_affinity = rec.combo_affinity  # type: ignore
else:
    def combo_affinity(env_now: dict, combo: str) -> float:
        cd = rec.digits_of(combo)
        seedd = env_now["seed_digits_list"]
        combo_sum = sum(cd)
        seed_sum = env_now["seed_sum"]
        sum_prox = math.exp(-abs(combo_sum - seed_sum) / 2.0)
        spread_c = max(cd) - min(cd)
        spread_s = env_now["spread_seed"]
        spread_prox = math.exp(-abs(spread_c - spread_s) / 2.0)
        struct_match = 1.0 if rec.classify_structure(cd) == rec.classify_structure(seedd) else 0.0
        parity_match = 1.0 if parity_major_label(cd) == parity_major_label(seedd) else 0.0
        hi8_seed = sum(1 for d in seedd if d >= 8)
        hi8_combo = sum(1 for d in cd if d >= 8)
        if hi8_seed >= 3:
            hi8_score = math.exp(-max(0, hi8_combo - 1))
        else:
            hi8_score = math.exp(-abs(hi8_combo - hi8_seed) / 1.5)
        overlap = len(set(seedd) & set(cd))
        overlap_eq1 = 1.0 if overlap == 1 else 0.0
        W_SUM, W_SPD, W_STR, W_PAR, W_HI8, W_OV1 = 0.8, 0.3, 0.4, 0.15, 0.35, 0.25
        return (
            W_SUM * sum_prox + W_SPD * spread_prox + W_STR * struct_match +
            W_PAR * parity_match + W_HI8 * hi8_score + W_OV1 * overlap_eq1
        )

# -----------------------------------------------
# Full enumeration (canonical 5-digit multisets)
# -----------------------------------------------
def full_canonical_multisets():
    return ["".join(str(d) for d in tup) for tup in combinations_with_replacement(range(10), 5)]

FULL_CANON = full_canonical_multisets()
FULL_SUMS = np.array([sum(rec.digits_of(s)) for s in FULL_CANON], dtype=float)

def sum_percentile_bounds(low_pct: float, high_pct: float):
    low = np.percentile(FULL_SUMS, low_pct * 100.0)
    high = np.percentile(FULL_SUMS, high_pct * 100.0)
    return float(low), float(high)

def build_pool_1digit_exact(seed_digits, keep_sum_pct=(0.0, 1.0), remove_quints_quads=True):
    seed_set = set(seed_digits)
    low_p, high_p = keep_sum_pct
    low_sum, high_sum = sum_percentile_bounds(low_p, high_p)

    out = []
    for s in FULL_CANON:  # canonical and deduped
        digs = rec.digits_of(s)
        if len(set(digs) & seed_set) < 1:
            continue  # 1-digit rule
        ssum = sum(digs)
        if not (low_sum <= ssum <= high_sum):
            continue  # percentile filter vs full enumeration
        if remove_quints_quads:
            stc = rec.classify_structure(digs)
            if stc in {"quint", "quad"}:
                continue
        out.append(s)
    return out

# -------------------------------
# UI controls
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    winners_path = st.text_input("Winners CSV", "DC5_Midday_Full_Cleaned_Expanded.csv")
    uploaded = st.file_uploader("...or upload Winners CSV", type=["csv"], help="Upload overrides the path above.")
with col2:
    n_days = st.number_input("Days to test (most recent)", 30, 3000, 240, ste

import math
from pathlib import Path
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Affinity Winner Stats", page_icon="📈", layout="wide")
st.title("Affinity vs Winner — Distribution & Buckets")

st.markdown(
    """
This page evaluates where the **actual winner** lands in the **affinity ranking**
built from that day's seed.

Pipeline (matches your validated order of operations):
1) Enumerate on full **100k ordered space** (00000..99999)
2) **SUM percentile filter** measured on the **100k** distribution
3) **Canonicalize & de-dup** (sorted digits → one multiset per combo)
4) **Drop quints & quads**
5) **Intersect** with the 2,002 canonical multisets (safety / consistency)

Options:
- 1- or 2-digit enumeration rule
- Optional “winner-heavy” SUM zones on the 100k baseline
"""
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
            W_SUM * sum_prox + W_SPD * spread_prox + W_STR * struct_match
            + W_PAR * parity_match + W_HI8 * hi8_score + W_OV1 * overlap_eq1
        )


# -----------------------------------------------
# Canonical enumeration (2,002 multisets)
# -----------------------------------------------
def full_canonical_multisets():
    return ["".join(str(d) for d in tup) for tup in combinations_with_replacement(range(10), 5)]


FULL_CANON = full_canonical_multisets()
FULL_CANON_SET = set(FULL_CANON)  # for fast intersection


# -----------------------------------------------
# 100k ordered space (vectorized digits + sums)
# -----------------------------------------------
@st.cache_data
def _all100k_digits_and_sums():
    x = np.arange(100000, dtype=np.int32)
    d0 = (x // 10000) % 10
    d1 = (x // 1000) % 10
    d2 = (x // 100) % 10
    d3 = (x // 10) % 10
    d4 = x % 10
    D = np.stack([d0, d1, d2, d3, d4], axis=1).astype(np.int8)   # shape (100000, 5)
    S = D.sum(axis=1).astype(np.int16)                          # digit sums
    return D, S


ALLD, ALLS = _all100k_digits_and_sums()


def sum_pct_bounds_100k(low_pct: float, high_pct: float):
    """Percentile bounds for digit sums on the 100k ordered distribution."""
    return (
        float(np.percentile(ALLS, low_pct * 100.0)),
        float(np.percentile(ALLS, high_pct * 100.0)),
    )


def sum_pct_rank_100k(sum_val: int) -> float:
    """Percentile rank of a given digit-sum relative to the 100k ordered combos."""
    # Using CDF via mean of (S <= sum_val)
    return float((ALLS <= sum_val).mean())


# Winner-heavy zones (on 100k baseline) — optional gate
WINNER_HEAVY_ZONES = [
    (0.00, 0.26),
    (0.30, 0.35),
    (0.36, 0.43),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.80, 0.83),
    (0.93, 0.94),
]


def build_pool_ordered_then_dedup(
    seed_digits,
    digits_required: int = 1,                 # 1 or 2
    keep_sum_pct: tuple[float, float] = (0.0, 1.0),
    remove_quints_quads: bool = True,
    apply_winner_heavy: bool = False,
):
    """
    Enumerate on 100k ordered -> SUM percentile filter (100k baseline)
    -> canonicalize & de-dup -> drop quints/quads -> intersect with 2,002 canon.
    """
    seed_digits = list(seed_digits)
    seed_vals = np.array(seed_digits, dtype=np.int8)
    low_sum, high_sum = sum_pct_bounds_100k(*keep_sum_pct)

    # --- masks on the 100k ordered space ---
    m_sum = (ALLS >= low_sum) & (ALLS <= high_sum)

    # digits_required in {1,2}: count DISTINCT seed digits present per row
    present_count = np.zeros(ALLD.shape[0], dtype=np.int8)
    for d in seed_vals:
        present_count += (ALLD == d).any(axis=1)
    m_digits = present_count >= digits_required

    if apply_winner_heavy:
        # Precompute mask covering all winner-heavy ranges (convert pct → sum bounds)
        m_zone = np.zeros_like(m_sum, dtype=bool)
        for lo, hi in WINNER_HEAVY_ZONES:
            lo_s, hi_s = sum_pct_bounds_100k(lo, hi)
            m_zone |= (ALLS >= lo_s) & (ALLS <= hi_s)
        m = m_sum & m_digits & m_zone
    else:
        m = m_sum & m_digits

    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []

    # --- canonicalize & de-dup ---
    canon_set = set()
    for i in idx:
        digs = ALLD[i].tolist()
        canon = "".join(str(x) for x in sorted(digs))
        canon_set.add(canon)

    # --- drop quints/quads AFTER canonicalization ---
    if remove_quints_quads:
        canon_set = {
            s for s in canon_set
            if rec.classify_structure(rec.digits_of(s)) not in {"quint", "quad"}
        }

    # --- final safety: restrict to the 2,002 canonical universe ---
    canon_set &= FULL_CANON_SET

    return sorted(canon_set)


# -------------------------------
# UI controls
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    winners_path = st.text_input("Winners CSV", "DC5_Midday_Full_Cleaned_Expanded.csv")
    uploaded = st.file_uploader("...or upload Winners CSV", type=["csv"], help="Upload overrides the path above.")
with col2:
    n_days = st.number_input("Days to test (most recent)", min_value=30, max_value=3000, value=240, step=10)
    sum_pct_keep = st.slider(
        "Keep SUM percentiles (100k baseline)",
        0.0, 1.0, (0.0, 1.0), 0.05,
        help="Example: (0.05, 0.95) keeps the middle 90% of sums. Evaluated on the 100k ordered distribution."
    )

col3, col4 = st.columns(2)
with col3:
    digits_required = st.radio("Digit rule", options=[1, 2], horizontal=True, index=0,
                               help="Minimum distinct seed digits that must appear in a combo.")
with col4:
    use_winner_heavy = st.checkbox(
        "Apply winner-heavy SUM zones",
        value=False,
        help="Restricts SUM percentile to your empirically winner-heavy bands (on 100k baseline).",
    )

run = st.button("Run Winner Affinity Stats", type="primary")


# -------------------------------
# Main
# -------------------------------
if run:
    # Load winners
    winners_to_use = winners_path
    if uploaded is not None:
        tmp_path = Path("uploaded_winners.csv")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        winners_to_use = str(tmp_path)

    try:
        winners = rec.load_winners(winners_to_use)
    except Exception as e:
        st.error(f"Failed to read winners CSV: {e}")
        st.stop()

    N = len(winners)
    if N < 3:
        st.warning("Not enough rows in winners CSV.")
        st.stop()

    end_idx = N - 1
    start_idx = max(2, end_idx - int(n_days))

    rows = []
    kept_flags = []
    pool_sizes = []

    for t in range(start_idx, end_idx + 1):
        env = rec.build_env_for_draw(t, winners)
        seed_digits = env["seed_digits_list"]

        # Canonicalize the winner for membership checks (pool is canonical)
        true_win_raw = winners[t]
        true_win = "".join(sorted(true_win_raw))

        # Build pool with the validated order
        pool = build_pool_ordered_then_dedup(
            seed_digits=seed_digits,
            digits_required=int(digits_required),
            keep_sum_pct=tuple(float(x) for x in sum_pct_keep),
            remove_quints_quads=True,
            apply_winner_heavy=bool(use_winner_heavy),
        )
        pool_set = set(pool)
        pool_sizes.append(len(pool))
        kept_flags.append(int(true_win in pool_set))

        # Affinity percentile of the winner vs pool
        aff_pool = np.fromiter((combo_affinity(env, c) for c in pool), dtype=float)
        aff_w = float(combo_affinity(env, true_win))  # features are order-invariant; canonical is fine
        pct = float((aff_pool <= aff_w).mean())  # 0..1

        rows.append({
            "t": t,
            "seed": winners[t - 1],
            "winner": true_win_raw,
            "aff_winner": round(aff_w, 6),
            "pct": pct,
        })

    df = pd.DataFrame(rows)

    # Buckets
    def bucket(p):
        if p >= 0.95:
            return "Top 5%"
        if p >= 0.90:
            return "Top 10%"
        if p >= 0.80:
            return "High (80–100%)"
        if p < 0.20:
            return "Low (0–20%)"
        return "Mid (20–80%)"

    df["bucket"] = df["pct"].map(bucket)

    # Summary tables
    total_days = len(df)
    kept_rate = (sum(kept_flags) / len(kept_flags) * 100.0) if kept_flags else 0.0
    by_bucket = (
        df.groupby("bucket")
          .size()
          .rename("days")
          .reset_index()
    )
    by_bucket["percent"] = (by_bucket["days"] / total_days * 100.0).round(2)

    st.subheader("Summary — Winner Affinity Percentile Buckets")
    st.dataframe(by_bucket, use_container_width=True)
    st.write(
        f"Winner kept in pool: {kept_rate:.1f}%  |  "
        f"Avg pool size: {np.mean(pool_sizes):.0f}  |  "
        f"Median pool size: {np.median(pool_sizes):.0f}"
    )

    st.subheader("Histogram — Winner Affinity Percentile (5% bins)")
    bins = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00
    labels = [f"{int(100*b)}–{int(100*bins[i+1])}%" for i, b in enumerate(bins[:-1])]
    df["pct_bin"] = pd.cut(df["pct"], bins=bins, labels=labels, include_lowest=True)
    hist = df.groupby("pct_bin").size().rename("days").reset_index()
    hist["percent"] = (hist["days"] / total_days * 100.0).round(2)
    st.dataframe(hist, use_container_width=True)

    st.download_button(
        "Download per-day winner affinity CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="winner_affinity_percentiles.csv",
        mime="text/csv",
    )

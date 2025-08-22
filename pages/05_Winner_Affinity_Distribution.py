# 05_Winner_Affinity_Distribution.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Winner Affinity Distribution", page_icon="📈", layout="wide")
st.title("📈 Winner Affinity Distribution (by percentile)")

st.markdown(
    "For each historical day, compute the **winner's affinity percentile** (0..1) within that day's pool, "
    "using the same `combo_affinity` as your recommender. Requires **archived real pools**."
)

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    winners_csv = st.text_input("Winners CSV", "DC5_Midday_Full_Cleaned_Expanded.csv")
with col2:
    n_days = st.number_input("Days to test (most recent)", min_value=30, max_value=2000, value=180, step=10)
with col3:
    bins = st.slider("Histogram bins", min_value=10, max_value=100, value=30, step=5)

col4, col5 = st.columns(2)
with col4:
    top_marks = st.text_input("Top-X checks (comma, as fractions)", "0.10,0.20,0.25,0.30,0.40,0.50",
                              help="We’ll report how often the winner is in the top X% (aff_pct ≥ 1−X).")
with col5:
    keep_bands = st.text_input("Keep-band scan (ranges a-b, comma)", "0.60-1.00,0.70-1.00,0.40-1.00",
                               help="We’ll report winner coverage & avg pool kept for these bands of aff_pct.")

use_archived = st.checkbox("Use archived real pools (required)", value=True,
                           help="Loads pools/pool_YYYY-MM-DD.csv via recommender.get_pool_for_seed().")

run = st.button("▶️ Run", type="primary")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _affinity_table(pool_df: pd.DataFrame, seed: str, rec) -> pd.DataFrame:
    """Return df with columns: combo, aff, aff_pct (0..1; lowest->0, highest->1)."""
    seed = str(seed).zfill(5)
    digs = [int(ch) for ch in seed]
    env = {
        "seed_digits_list": digs,
        "seed_sum": sum(digs),
        "spread_seed": (max(digs) - min(digs)),
    }
    df = pool_df[["combo"]].copy()
    df["combo"] = df["combo"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    df = df[df["combo"].str.fullmatch(r"\d{5}")].copy()
    if df.empty:
        return df
    df["aff"] = [rec.combo_affinity(env, c) for c in df["combo"]]
    # Normalize rank to [0,1], lowest->0 highest->1 (robust to ties)
    r = pd.Series(df["aff"]).rank(method="average")  # 1..N
    N = max(len(df) - 1, 1)
    df["aff_pct"] = (r - 1) / N
    return df

def _parse_marks(s: str) -> list[float]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except:
            pass
    return [x for x in out if 0 < x < 1]

def _parse_bands(s: str) -> list[tuple[float,float]]:
    """Parse '0.6-1.0,0.7-1.0' etc."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part or "-" not in part:
            continue
        a, b = part.split("-", 1)
        try:
            lo = float(a); hi = float(b)
        except:
            continue
        lo, hi = max(0.0, lo), min(1.0, hi)
        if lo < hi:
            out.append((lo, hi))
    return out

# ── Work ──────────────────────────────────────────────────────────────────────
if run:
    import recommender as rec

    # Load winners (df for date row + list for convenience)
    try:
        winners_df = pd.read_csv(winners_csv)
        winners = rec.load_winners(winners_csv)
    except Exception as e:
        st.error(f"Failed to read winners CSV: {e}")
        st.stop()

    if len(winners) < 2:
        st.error("Need at least 2 winners.")
        st.stop()

    end_idx = len(winners) - 1
    start_idx = max(1, end_idx - int(n_days) + 1)

    rows = []
    skipped = 0

    for t in range(start_idx, end_idx + 1):
        seed = str(winners[t-1]).zfill(5)
        winner = str(winners[t]).zfill(5)

        # Load archived pool
        try:
            seed_row = winners_df.iloc[t-1]
            pool_df = rec.get_pool_for_seed(seed_row)
        except Exception:
            skipped += 1
            continue

        df = _affinity_table(pool_df, seed, rec)
        if df.empty:
            skipped += 1
            continue

        # Winner's percentile
        try:
            win_pct = float(df.loc[df["combo"] == winner, "aff_pct"].iloc[0])
        except IndexError:
            win_pct = np.nan  # winner not in pool

        rows.append({"t": t, "seed": seed, "winner": winner, "pool_size": len(df),
                     "winner_aff_pct": win_pct})

    if not rows:
        st.error("No days with archived pools to evaluate.")
        st.stop()

    per_day = pd.DataFrame(rows).dropna(subset=["winner_aff_pct"])
    st.subheader("Per-day winner affinity percentile")
    st.dataframe(per_day, use_container_width=True)

    # Histogram
    import matplotlib.pyplot as plt
    st.subheader("Distribution of winner_aff_pct")
    fig = plt.figure()
    plt.hist(per_day["winner_aff_pct"].values, bins=bins)
    plt.xlabel("winner_aff_pct (0=least similar … 1=most similar)")
    plt.ylabel("count of days")
    st.pyplot(fig)

    # Summary stats
    winp = per_day["winner_aff_pct"]
    q = winp.quantile([0.10, 0.25, 0.5, 0.75, 0.90]).round(4)
    summary = pd.DataFrame([
        {"metric": "days_evaluated", "value": len(per_day)},
        {"metric": "mean", "value": round(float(winp.mean()), 4)},
        {"metric": "median", "value": round(float(winp.median()), 4)},
        {"metric": "q10", "value": q.get(0.10, np.nan)},
        {"metric": "q25", "value": q.get(0.25, np.nan)},
        {"metric": "q50", "value": q.get(0.50, np.nan)},
        {"metric": "q75", "value": q.get(0.75, np.nan)},
        {"metric": "q90", "value": q.get(0.90, np.nan)},
    ])
    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)

    # Top-X checks
    marks = _parse_marks(top_marks)
    if marks:
        rows_top = []
        for x in marks:
            thr = 1.0 - x
            in_topx = (winp >= thr).mean()  # how often winner sits in top X%
            rows_top.append({
                "X": x,
                "topX_threshold (aff_pct ≥)": round(thr, 4),
                "winner_in_topX_rate": round(float(in_topx), 4),
                "winner_kept_if_drop_topX (blind_keep_rate)": round(float(1.0 - in_topx), 4),
            })
        st.subheader("Top-X% placement (how often winners live in the top band)")
        st.dataframe(pd.DataFrame(rows_top), use_container_width=True)

    # Keep-band scan: coverage & average kept fraction
    bands = _parse_bands(keep_bands)
    if bands:
        st.subheader("Keep-band scan (winner coverage & avg pool kept)")
        # For avg kept fraction we need per-day pool fractions; re-scan quickly
        results = []
        for t in per_day["t"].tolist():
            seed = str(winners[t-1]).zfill(5)
            winner = str(winners[t]).zfill(5)
            seed_row = winners_df.iloc[t-1]
            pool_df = rec.get_pool_for_seed(seed_row)
            df = _affinity_table(pool_df, seed, rec)
            # attach for this day
            df_day = df[["aff_pct"]].copy()
            df_day["t"] = t
            if "tmp" not in locals():
                tmp = df_day
            else:
                tmp = pd.concat([tmp, df_day], axis=0, ignore_index=True)

        scan_rows = []
        for (lo, hi) in bands:
            # winner coverage
            cover = ((per_day["winner_aff_pct"] >= lo) & (per_day["winner_aff_pct"] < hi)).mean()
            # avg kept fraction across days
            kept = tmp[(tmp["aff_pct"] >= lo) & (tmp["aff_pct"] < hi)]
            kept_frac = kept.groupby("t").size() / tmp.groupby("t").size()
            avg_kept = float(kept_frac.mean())
            scan_rows.append({
                "band": f"{lo:.2f}-{hi:.2f}",
                "winner_coverage": round(float(cover), 4),
                "avg_kept_pct": round(avg_kept, 4),
                "avg_removed_pct": round(1.0 - avg_kept, 4),
            })
        st.dataframe(pd.DataFrame(scan_rows), use_container_width=True)

    # Downloads
    per_day.to_csv("winner_affinity_per_day.csv", index=False)
    summary.to_csv("winner_affinity_summary.csv", index=False)
    st.download_button("Download per-day CSV", data=open("winner_affinity_per_day.csv", "rb").read(),
                       file_name="winner_affinity_per_day.csv", mime="text/csv")
    st.download_button("Download summary CSV", data=open("winner_affinity_summary.csv", "rb").read(),
                       file_name="winner_affinity_summary.csv", mime="text/csv")

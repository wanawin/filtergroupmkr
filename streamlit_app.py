# streamlit_app.py
import streamlit as st, pandas as pd, os
import recommender as rc  # core engine

st.set_page_config(page_title="DC5 Recommender", page_icon="🎯", layout="wide")
st.title("DC5 Recommender — Winner-Preserving (<45 target)")

with st.expander("CSV Inputs", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        winners_csv = st.text_input("Winners CSV path (in repo)", value=rc.WINNERS_CSV)
        filters_csv = st.text_input("Filters CSV path (in repo)", value=rc.FILTERS_CSV)
        override_seed = st.text_input("Override seed (5 digits, optional)", value="")
        override_prev = st.text_input("Override 1-back (optional)", value="")
        override_prevprev = st.text_input("Override 2-back (optional)", value="")
    with col2:
        target_max  = st.number_input("Target max pool size", 10, 100, value=44)
        keep_winner = st.checkbox("Always keep winner", True, help="Skip steps that would remove the keep combo (or last real winner if supplied).")
        minimize    = st.checkbox("Minimize below target if safe", True)
        max_draws = st.number_input("Use last N draws overall (0=all)", 0, 2000, value=365)
        max_bucket_matches = st.number_input("Use last K matches of this bucket (0=all)", 0, 500, value=80)
        decay_half_life = st.number_input("Decay half-life (draws, 0 = none)", 0, 720, value=90)

    st.markdown("Upload **today’s generated pool** (optional; CSV with `combo` or `Result`):")
    pool_file = st.file_uploader("Pool CSV", type=["csv"])
    pool_path = None
    if pool_file is not None:
        pool_path = "uploaded_pool.csv"
        open(pool_path, "wb").write(pool_file.read())

    keep_combo_text = st.text_input("Keep combo (optional, 5 digits)", value="")
    applicable_ids_txt = st.text_area("Paste today's *applicable* filter IDs (from your main app) — comma/space/line separated (optional)", height=90)
    applicable_ids = [x for x in [s.strip() for s in (applicable_ids_txt.replace(",", " ").split())] if x]

if st.button("Run recommender"):
    rc.WINNERS_CSV = winners_csv
    rc.FILTERS_CSV = filters_csv
    rc.TODAY_POOL_CSV = pool_path
    rc.TARGET_MAX = int(target_max)
    rc.ALWAYS_KEEP_WINNER = bool(keep_winner)
    rc.MINIMIZE_BEYOND_TARGET = bool(minimize)

    try:
        rc.main(
            winners_csv=winners_csv,
            filters_csv=filters_csv,
            today_pool_csv=pool_path,
            target_max=int(target_max),
            always_keep_winner=bool(keep_winner),
            minimize_beyond_target=bool(minimize),
            force_keep_combo=keep_combo_text.strip() or None,
            override_seed=override_seed.strip() or None,
            override_prev=override_prev.strip() or None,
            override_prevprev=override_prevprev.strip() or None,
            max_draws=(int(max_draws) or None),
            max_bucket_matches=(int(max_bucket_matches) or None),
            decay_half_life=(int(decay_half_life) or None),
            applicable_only=applicable_ids or None,
        )
        st.success("Done.")

        def try_read_csv(path):
            try: return pd.read_csv(path)
            except: return None

        seq   = try_read_csv("recommender_sequence.csv")
        avoid = try_read_csv("avoid_pairs.csv")
        red   = try_read_csv("pool_reduction_log.csv")
        dna   = try_read_csv("do_not_apply.csv")

        if seq is not None:
            st.subheader("Apply in this order")
            st.dataframe(seq, use_container_width=True)

        if dna is not None and not dna.empty:
            st.subheader("⛔ DO NOT APPLY (today)")
            st.dataframe(dna[dna["tier"] == "DO_NOT_APPLY"], use_container_width=True)
            st.subheader("⚠ Apply later with caution")
            st.dataframe(dna[dna["tier"] == "APPLY_LATE"], use_container_width=True)

        if avoid is not None and not avoid.empty:
            st.subheader("Avoid combining (today’s bucket)")
            st.dataframe(avoid, use_container_width=True)

        if red is not None and not red.empty:
            st.subheader("Pool reduction log")
            st.dataframe(red, use_container_width=True)

        # One-pagers
        if os.path.exists("one_pager.html"):
            st.subheader("One-pager (HTML)")
            st.components.v1.html(open("one_pager.html","r",encoding="utf-8").read(), height=1100, scrolling=True)
        if os.path.exists("one_pager.md"):
            st.subheader("One-pager (Markdown)")
            st.markdown(open("one_pager.md","r",encoding="utf-8").read())

    except Exception as e:
        st.error(str(e))
        import glob
        st.caption("CSVs present: " + ", ".join(sorted([os.path.basename(p) for p in glob.glob('*.csv')])))

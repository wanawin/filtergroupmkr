# streamlit_app.py
import streamlit as st, pandas as pd, os, io
import recommender as rc  # the core engine above

st.set_page_config(page_title="DC5 Recommender", page_icon="🎯", layout="wide")
st.title("DC5 Recommender — Winner-Preserving (<45 target)")

with st.expander("CSV Inputs", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        winners_csv = st.text_input("Winners CSV path (in repo)", value=rc.WINNERS_CSV)
        filters_csv = st.text_input("Filters CSV path (in repo)", value=rc.FILTERS_CSV)
    with col2:
        target_max  = st.number_input("Target max pool size", 10, 100, value=44)
        keep_winner = st.checkbox("Always keep winner", True)
        minimize    = st.checkbox("Minimize below target if safe", True)

    st.markdown("Upload **today’s generated pool** (optional; CSV with `combo` or `Result`):")
    pool_file = st.file_uploader("Pool CSV", type=["csv"])
    pool_path = None
    if pool_file is not None:
        pool_path = "uploaded_pool.csv"
        open(pool_path, "wb").write(pool_file.read())

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
        )
        st.success("Done.")

        def try_read_csv(path):
            try: return pd.read_csv(path)
            except: return None

        seq = try_read_csv("recommender_sequence.csv")
        avoid = try_read_csv("avoid_pairs.csv")
        red = try_read_csv("pool_reduction_log.csv")

        if seq is not None:
            st.subheader("Apply in this order")
            st.dataframe(seq, use_container_width=True)

        if avoid is not None:
            st.subheader("Avoid combining (today’s bucket)")
            st.dataframe(avoid, use_container_width=True)

        if red is not None:
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
        # Helpful: list CSVs present in container
        import glob
        st.caption("CSVs present: " + ", ".join(sorted([os.path.basename(p) for p in glob.glob('*.csv')])))


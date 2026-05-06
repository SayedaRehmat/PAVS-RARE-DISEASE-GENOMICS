import streamlit as st
import os

from src.analysis import run_pipeline_for_ui

st.set_page_config(layout="wide")

st.title("🧬 PAVS Rare Disease Genomics Platform")

if st.button("Run Full Analysis"):

    with st.spinner("Running full pipeline..."):
        results = run_pipeline_for_ui()

    st.success("Analysis Complete")

    # Show summary
    st.subheader("📊 Summary")
    st.write(results)

    st.markdown("---")

    # Show final dashboard (main output)
    final_fig = "outputs/fig00_FINAL_DASHBOARD.png"

    if os.path.exists(final_fig):
        st.image(final_fig, use_container_width=True)

    else:
        st.warning("Final dashboard not found")

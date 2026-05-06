import streamlit as st
import os
from src.analysis import run_pipeline_for_ui

st.set_page_config(layout="wide")

st.title("🧬 PAVS Rare Disease Genomics Platform")

# Sidebar
st.sidebar.title("Controls")

run_button = st.sidebar.button("Run Full Analysis")

data_path = "data/PAVS_cases.tsv"

# Run analysis
if run_button:
    with st.spinner("Running full genomic pipeline..."):
        output = run_pipeline_for_ui(data_path)

    st.success("Analysis Completed")

    # Summary
    st.header("📊 Summary Metrics")
    summary = output["summary"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", summary["total_cases"])
    col2.metric("Solved Cases", summary["solved_cases"])
    col3.metric("Neuro Cases", summary["neuro_cases"])

    col4, col5, col6 = st.columns(3)
    col4.metric("Founder Mutations", summary["founder_candidates"])
    col5.metric("VUS Candidates", summary["vus_candidates"])
    col6.metric("Treatable Unsolved", summary["unsolved_treatable"])

    st.markdown("---")

    # Model performance
    st.header("🤖 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("CV Accuracy", summary["gene_model_cv"])
    col2.metric("Top-3 Accuracy", summary["gene_model_top3"])
    col3.metric("Top-5 Accuracy", summary["gene_model_top5"])

    st.metric("Pathogenicity AUC", summary["path_auc"])

    st.markdown("---")

    # Figures
    st.header("📈 Analysis Visualizations")

    for fig_path in output["figures"]:
        if os.path.exists(fig_path):
            st.image(fig_path, use_container_width=True)

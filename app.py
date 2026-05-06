import streamlit as st
import pandas as pd

from analysis_core import *
from plots_plotly import *

st.set_page_config(layout="wide")
st.title("🧬 PAVS Clinical Genomics Dashboard")

# =========================
# LOAD
# =========================
@st.cache_data
def load():
    return load_data("data/PAVS_cases.tsv")

df, _ = load()

# =========================
# FILTERS
# =========================
st.sidebar.header("Filters")

sources = st.sidebar.multiselect(
    "Source",
    df["source"].dropna().unique(),
    default=df["source"].dropna().unique()
)

genes = st.sidebar.multiselect(
    "Gene",
    sorted(df["gene_symbol"].dropna().unique())
)

hgvs_query = st.sidebar.text_input("HGVS Variant")

filtered_df = df[df["source"].isin(sources)]

if genes:
    filtered_df = filtered_df[filtered_df["gene_symbol"].isin(genes)]

if hgvs_query:
    filtered_df = filtered_df[
        filtered_df["hgvs_c"].astype(str).str.contains(hgvs_query, na=False)
    ]

# =========================
# PATIENT SEARCH
# =========================
st.sidebar.subheader("Patient Lookup")

pid = st.sidebar.text_input("Patient ID")

if pid:
    res = filtered_df[filtered_df["patient_id"].astype(str) == pid]
    st.write(res)

# =========================
# NAV
# =========================
tab = st.sidebar.radio("Module", [
    "EDA", "Population", "Founders", "Treatable",
    "Neuro", "Similarity", "Gene Model",
    "Pathogenicity", "VUS", "HPO", "SHAP"
])

# =========================
# EDA
# =========================
if tab == "EDA":
    figs = plot_cohort_overview(filtered_df)
    for f in figs:
        st.plotly_chart(f, use_container_width=True)

# =========================
# POP
# =========================
elif tab == "Population":
    stats = get_population_stats(filtered_df)
    st.plotly_chart(plot_population(stats), use_container_width=True)

# =========================
# FOUNDERS
# =========================
elif tab == "Founders":
    founders = get_founders(filtered_df)
    st.plotly_chart(plot_founders(founders), use_container_width=True)

# =========================
# TREATABLE
# =========================
elif tab == "Treatable":
    treat = get_treatable_df(filtered_df)
    st.plotly_chart(plot_treatable(treat), use_container_width=True)

# =========================
# NEURO
# =========================
elif tab == "Neuro":
    neuro = get_neuro(filtered_df)
    st.plotly_chart(plot_neuro(neuro), use_container_width=True)

# =========================
# SIMILARITY
# =========================
elif tab == "Similarity":
    sim, names = get_hpo_similarity_advanced(filtered_df)
    st.plotly_chart(plot_disease_similarity(sim, names), use_container_width=True)

# =========================
# GENE MODEL
# =========================
elif tab == "Gene Model":
    if st.button("Run Model"):
        res = run_gene_model(filtered_df)
        st.plotly_chart(plot_gene_model(res), use_container_width=True)
        st.json(res)

# =========================
# PATHOGENICITY
# =========================
elif tab == "Pathogenicity":
    if st.button("Run"):
        p, r = run_pathogenicity_model(filtered_df)
        st.plotly_chart(plot_pathogenicity_curve(p, r), use_container_width=True)

# =========================
# VUS
# =========================
elif tab == "VUS":
    vus = get_vus(filtered_df)
    st.dataframe(vus.head(100))

# =========================
# HPO
# =========================
elif tab == "HPO":
    mat, terms = get_hpo_cooccurrence(filtered_df)
    st.plotly_chart(plot_hpo_cooccurrence(mat, terms), use_container_width=True)

# =========================
# SHAP
# =========================
elif tab == "SHAP":
    if st.button("Run SHAP"):
        shap_values = run_gene_model_with_shap(filtered_df)

        import shap
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, show=False)
        st.pyplot(fig)

# app.py

import streamlit as st
import pandas as pd

from analysis_core import *
from plots_plotly import *

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="PAVS Clinical Genomics Dashboard",
    layout="wide"
)

st.title("🧬 PAVS Rare Disease Genomics — Clinical Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load():
    df, hpo_labels = load_data("data/PAVS_cases.tsv")
    return df, hpo_labels

df, hpo_labels = load()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")

tabs = st.sidebar.radio(
    "Go to Module:",
    [
        "📊 Overview (EDA)",
        "🌍 Population Genomics",
        "🧬 Founder Mutations",
        "🧬 AR Architecture",
        "💊 Treatable Diseases",
        "🧠 Neuro Burden",
        "🔬 Disease Similarity",
        "🤖 Gene Model",
        "🧪 Pathogenicity Model",
        "⚠️ VUS Analysis",
        "🔗 HPO Co-occurrence",
        "🧬 ADAT3 Deep Dive",
        "📊 Final Clinical Dashboard"
    ]
)

# =========================
# MODULE 1 — EDA
# =========================
if tabs == "📊 Overview (EDA)":
    st.header("Cohort Overview")

    fig1, fig2, fig3, fig4 = plot_cohort_overview(df)

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig3, use_container_width=True)
    col2.plotly_chart(fig4, use_container_width=True)


# =========================
# MODULE 2 — POPULATION
# =========================
elif tabs == "🌍 Population Genomics":
    st.header("Population Stratification")

    stats = get_population_stats(df)
    fig = plot_population(stats)

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(stats)


# =========================
# MODULE 3 — FOUNDERS
# =========================
elif tabs == "🧬 Founder Mutations":
    st.header("Founder Mutation Discovery")

    saudi = df[df["source"] == "PAVS-Saudi"]
    founders = get_founders(saudi)

    fig = plot_founders(founders)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download Data")
    st.download_button(
        "Download Founder Table",
        founders.to_csv(index=False),
        "founders.csv"
    )


# =========================
# MODULE 4 — AR ARCHITECTURE
# =========================
elif tabs == "🧬 AR Architecture":
    st.header("Autosomal Recessive Architecture")

    fig = plot_ar_architecture(df)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# MODULE 5 — TREATABLE
# =========================
elif tabs == "💊 Treatable Diseases":
    st.header("Treatable Rare Diseases")

    treat_df = get_treatable_df(df)
    fig = plot_treatable(treat_df)

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(treat_df.head(100))


# =========================
# MODULE 6 — NEURO
# =========================
elif tabs == "🧠 Neuro Burden":
    st.header("Neurodevelopmental Disease Burden")

    neuro_df = get_neuro(df)
    fig = plot_neuro(neuro_df)

    st.plotly_chart(fig, use_container_width=True)


# =========================
# MODULE 7 — DISEASE SIMILARITY
# =========================
elif tabs == "🔬 Disease Similarity":
    st.header("Disease Phenotype Similarity")

    fig = plot_disease_similarity(df)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# MODULE 8 — GENE MODEL (LAZY LOAD)
# =========================
elif tabs == "🤖 Gene Model":
    st.header("Gene Prioritization Model")

    if st.button("Run Model"):
        with st.spinner("Training model..."):
            results = run_gene_model(df)

        fig = plot_gene_model(results)
        st.plotly_chart(fig, use_container_width=True)

        st.json(results)


# =========================
# MODULE 9 — PATHOGENICITY
# =========================
elif tabs == "🧪 Pathogenicity Model":
    st.header("ACMG Pathogenicity Classifier")

    if st.button("Run Pathogenicity Model"):
        with st.spinner("Running classifier..."):
            precision, recall = run_pathogenicity_model(df)

        fig = plot_pathogenicity_curve(precision, recall)
        st.plotly_chart(fig, use_container_width=True)


# =========================
# MODULE 10 — VUS
# =========================
elif tabs == "⚠️ VUS Analysis":
    st.header("VUS Reclassification Candidates")

    vus = get_vus(df)

    st.dataframe(vus.head(100))

    st.download_button(
        "Download VUS Candidates",
        vus.to_csv(index=False),
        "vus_candidates.csv"
    )


# =========================
# MODULE 11 — HPO CO-OCCURRENCE
# =========================
elif tabs == "🔗 HPO Co-occurrence":
    st.header("HPO Co-occurrence Matrix")

    fig = plot_hpo_cooccurrence(df)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# MODULE 12 — ADAT3
# =========================
elif tabs == "🧬 ADAT3 Deep Dive":
    st.header("ADAT3 Founder Mutation Analysis")

    fig = plot_adat3(df)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# MODULE 13 — FINAL DASHBOARD
# =========================
elif tabs == "📊 Final Clinical Dashboard":
    st.header("Clinical Summary Dashboard")

    stats = get_population_stats(df)
    founders = get_founders(df[df["source"] == "PAVS-Saudi"])
    neuro_df = get_neuro(df)
    vus = get_vus(df)
    treat_df = get_treatable_df(df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Cases", len(df))
    col2.metric("Unique Genes", df["gene_symbol"].nunique())
    col3.metric("Solved Cases", df["is_solved"].sum())

    st.divider()

    st.subheader("Founder Mutations")
    st.plotly_chart(plot_founders(founders), use_container_width=True)

    st.subheader("Neuro Burden")
    st.plotly_chart(plot_neuro(neuro_df), use_container_width=True)

    st.subheader("Treatable Diseases")
    st.plotly_chart(plot_treatable(treat_df), use_container_width=True)

    st.subheader("VUS Candidates")
    st.dataframe(vus.head(50))

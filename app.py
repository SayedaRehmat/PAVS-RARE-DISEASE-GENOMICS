import streamlit as st
import pandas as pd

from analysis_core import *
from plots_plotly import *

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Clinical Genomics Dashboard",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🧬 Clinical Genomics Research Dashboard")
st.markdown("""
Modern interactive platform for **rare disease cohort analysis**,  
combining **clinical genomics, phenotype modeling, and AI-driven insights**.
""")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load():
    return load_data("data/PAVS_cases.tsv")

df, _ = load()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📂 Modules")

tab = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🌍 Population",
    "🧬 Founder Mutations",
    "💊 Treatable Diseases",
    "🧠 Neuro Burden",
    "🔬 Disease Similarity",
    "🤖 Gene Model",
    "🧪 Pathogenicity",
    "⚠️ VUS Analysis",
    "🔗 HPO Network",
    "🧬 ADAT3 Deep Dive",
    "🔍 Patient Explorer",
    "📊 Clinical Dashboard"
])

# =========================
# OVERVIEW
# =========================
if tab == "📊 Overview":

    st.header("📊 Cohort Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", len(df))
    col2.metric("Genes", df["gene_symbol"].nunique())
    col3.metric("Solved", df["is_solved"].sum())

    st.markdown("### 📈 Phenotype Complexity")
    st.plotly_chart(plot_hist(df, "hpo_count", "HPO Distribution"), use_container_width=True)

    st.info("Higher HPO counts indicate more complex phenotypes typical of rare genetic disorders.")

# =========================
# POPULATION
# =========================
elif tab == "🌍 Population":

    st.header("🌍 Population Genomics")

    stats = get_population_stats(df).reset_index()

    st.markdown("### 📊 Cohort Stratification")
    st.plotly_chart(plot_bar(stats, "source", "n", "Cases per Source"), use_container_width=True)

    st.dataframe(stats)

    st.info("Population-level variation influences diagnostic yield and variant architecture.")

# =========================
# FOUNDERS
# =========================
elif tab == "🧬 Founder Mutations":

    st.header("🧬 Founder Variant Discovery")

    founders = get_founders(df)

    st.plotly_chart(
        plot_bar(founders.head(10), "gene_symbol", "count", "Top Founder Variants"),
        use_container_width=True
    )

    st.dataframe(founders.head(50))

    st.success("Recurrent variants may indicate founder effects in specific populations.")

# =========================
# TREATABLE
# =========================
elif tab == "💊 Treatable Diseases":

    st.header("💊 Treatable Genetic Disorders")

    treat = get_treatable(df)

    st.plotly_chart(
        plot_bar(safe_counts(treat["gene_symbol"], "gene"), "gene", "count", "Treatable Genes"),
        use_container_width=True
    )

    st.dataframe(treat.head(50))

    st.warning("These genes represent clinically actionable targets.")

# =========================
# NEURO
# =========================
elif tab == "🧠 Neuro Burden":

    st.header("🧠 Neurodevelopmental Burden")

    neuro = get_neuro(df)

    st.plotly_chart(
        plot_bar(safe_counts(neuro["gene_symbol"], "gene"), "gene", "count", "Neuro Genes"),
        use_container_width=True
    )

    st.info("Neuro-related phenotypes dominate rare disease cohorts.")

# =========================
# DISEASE SIMILARITY
# =========================
elif tab == "🔬 Disease Similarity":

    st.header("🔬 Phenotype-Based Disease Similarity")

    sim, names = get_disease_similarity(df)

    st.plotly_chart(
        plot_heatmap(sim, names, "Disease Similarity (Jaccard)"),
        use_container_width=True
    )

    st.info("Clusters indicate shared biological mechanisms.")

# =========================
# GENE MODEL
# =========================
elif tab == "🤖 Gene Model":

    st.header("🤖 Gene Prioritization Model")

    st.markdown("Predicts likely causal genes based on phenotype profiles.")

    if st.button("🚀 Run Gene Model"):
        with st.spinner("Training model..."):
            result = run_gene_model(df)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Model completed")

            st.metric("CV Accuracy", f"{result['cv_accuracy']:.2f}")

# =========================
# PATHOGENICITY
# =========================
elif tab == "🧪 Pathogenicity":

    st.header("🧪 Pathogenicity Classifier")

    if st.button("🚀 Run Classifier"):
        with st.spinner("Running model..."):
            p, r = run_pathogenicity_model(df)

        st.success("Completed")

        st.line_chart({"Precision": p, "Recall": r})

        st.info("Precision-Recall tradeoff for pathogenic variant prediction.")

# =========================
# VUS
# =========================
elif tab == "⚠️ VUS Analysis":

    st.header("⚠️ Variant of Uncertain Significance")

    vus = get_vus(df)

    st.dataframe(vus.head(100))

    st.download_button(
        "Download VUS",
        vus.to_csv(index=False),
        "vus.csv"
    )

# =========================
# HPO NETWORK
# =========================
elif tab == "🔗 HPO Network":

    st.header("🔗 Phenotype Co-occurrence")

    mat, terms = get_hpo_cooccurrence(df)

    st.plotly_chart(
        plot_heatmap(mat, terms, "HPO Co-occurrence"),
        use_container_width=True
    )

# =========================
# ADAT3
# =========================
elif tab == "🧬 ADAT3 Deep Dive":

    st.header("🧬 ADAT3 Analysis")

    sub, counts = get_adat3(df)

    st.plotly_chart(
        plot_bar(counts.reset_index(), "index", "hpo_list", "ADAT3 HPO"),
        use_container_width=True
    )

    st.dataframe(sub)

# =========================
# PATIENT EXPLORER
# =========================
elif tab == "🔍 Patient Explorer":

    st.header("🔍 Patient Search & Exploration")

    query = st.text_input("Search by gene, variant, phenotype")

    if query:
        results = search_patients(df, query)

        st.write(f"Found {len(results)} matches")
        st.dataframe(results)

        idx = st.number_input("Select patient index", 0, len(df)-1)

        if st.button("Find Similar Patients"):
            sim = patient_similarity(df, idx)
            st.dataframe(sim)

# =========================
# FINAL DASHBOARD
# =========================
elif tab == "📊 Clinical Dashboard":

    st.header("📊 Summary Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cases", len(df))
    col2.metric("Genes", df["gene_symbol"].nunique())
    col3.metric("Solved", df["is_solved"].sum())

    st.plotly_chart(plot_hist(df, "hpo_count", "Phenotype Load"), use_container_width=True)

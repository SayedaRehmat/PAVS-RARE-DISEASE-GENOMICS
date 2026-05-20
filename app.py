# =====================================================================================
# app.py
# PAVS Rare Disease Genomics Platform
# Professional Interactive Streamlit Dashboard
# Auto-Compatible with Light + Dark Themes
# =====================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# =====================================================================================
# PAGE CONFIG
# =====================================================================================

st.set_page_config(
    page_title="PAVS Rare Disease Genomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================================
# PATHS
# =====================================================================================

BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "data" / "PAVS_cases.tsv"
OUTPUT_DIR = BASE_DIR / "outputs"

# =====================================================================================
# LOAD DATA
# =====================================================================================

@st.cache_data
def load_main_data():
    df = pd.read_csv(DATA_PATH, sep="\t")

    df["is_solved"] = df["solved_status"] == "SOLVED"

    df["hpo_list"] = df["hpo_terms"].fillna("").apply(
        lambda x: [i.split("|")[0] for i in x.split(";") if i]
    )

    df["hpo_count"] = df["hpo_list"].apply(len)

    return df


@st.cache_data
def load_csv(name):
    path = OUTPUT_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


df = load_main_data()

# =====================================================================================
# GLOBAL METRICS
# =====================================================================================

TOTAL_CASES = len(df)
TOTAL_GENES = df["gene_symbol"].nunique()
TOTAL_DISEASES = df["disease_label"].nunique()
SOLVED_CASES = int(df["is_solved"].sum())
UNSOLVED = TOTAL_CASES - SOLVED_CASES
NEURO_CASES = int(df["hpo_terms"].str.contains("HP:0001249", na=False).sum())

# =====================================================================================
# THEME-SAFE CSS
# =====================================================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120,120,120,0.15);
}

/* Metric Cards */
.metric-card {
    border-radius: 22px;
    padding: 1.3rem;
    background: rgba(120,120,120,0.08);
    border: 1px solid rgba(120,120,120,0.15);
    backdrop-filter: blur(10px);
}

.metric-number {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.metric-label {
    font-size: 0.8rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Hero */
.hero {
    padding: 2.5rem;
    border-radius: 28px;
    background: linear-gradient(
        135deg,
        rgba(0,201,167,0.10),
        rgba(79,142,247,0.08)
    );
    border: 1px solid rgba(120,120,120,0.12);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.8rem;
}

.hero-sub {
    font-size: 1.05rem;
    opacity: 0.8;
    line-height: 1.7;
    max-width: 1000px;
}

/* Section title */
.section-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================================================================
# SIDEBAR
# =====================================================================================

st.sidebar.title("🧬 PAVS Platform")

section = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Cohort Analytics",
        "Population Genomics",
        "Founder Mutations",
        "Treatable Diseases",
        "Neurodevelopmental Burden",
        "Disease Similarity",
        "AI Gene Prioritization",
        "Pathogenicity Intelligence",
        "VUS Reclassification",
        "HPO Architecture",
        "ADAT3 Deep Dive",
        "Figure Explorer",
        "Research Summary"
    ]
)

st.sidebar.markdown("---")

st.sidebar.metric("Patient Cases", f"{TOTAL_CASES:,}")
st.sidebar.metric("Disease Genes", f"{TOTAL_GENES:,}")
st.sidebar.metric("Rare Diseases", f"{TOTAL_DISEASES:,}")
st.sidebar.metric("Solved Cases", f"{SOLVED_CASES:,}")

# =====================================================================================
# OVERVIEW
# =====================================================================================

if section == "Overview":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            Rare Disease Genomics<br>
            in the Arab World
        </div>

        <div class="hero-sub">
            A computational genomics intelligence platform analyzing 7,510 Saudi
            rare disease cases using founder mutation discovery, HPO-driven AI
            prioritization, phenotype architecture analysis, and translational
            precision medicine analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{TOTAL_CASES:,}</div>
            <div class="metric-label">Patient Cases</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{TOTAL_GENES:,}</div>
            <div class="metric-label">Disease Genes</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{TOTAL_DISEASES:,}</div>
            <div class="metric-label">Rare Diseases</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{SOLVED_CASES:,}</div>
            <div class="metric-label">Solved Cases</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{NEURO_CASES:,}</div>
            <div class="metric-label">Neuro Cases</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Cohort Landscape")

    solved_counts = df["solved_status"].value_counts().reset_index()
    solved_counts.columns = ["Status", "Cases"]

    fig = px.pie(
        solved_counts,
        names="Status",
        values="Cases",
        hole=0.5
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# COHORT ANALYTICS
# =====================================================================================

elif section == "Cohort Analytics":

    st.markdown('<div class="section-title">Cohort Analytics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Cases"]

        fig = px.bar(
            source_counts,
            x="Source",
            y="Cases",
            color="Source"
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        acmg = df["acmg_classification"].value_counts().reset_index()
        acmg.columns = ["Classification", "Cases"]

        fig = px.pie(
            acmg,
            names="Classification",
            values="Cases"
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### HPO Burden")

    fig = px.histogram(
        df,
        x="hpo_count",
        nbins=30
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# POPULATION GENOMICS
# =====================================================================================

elif section == "Population Genomics":

    st.markdown('<div class="section-title">Population Genomics</div>', unsafe_allow_html=True)

    fig_path = OUTPUT_DIR / "fig05_population_comparison.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

    st.markdown("""
    ### Key Interpretation

    The Saudi cohort demonstrates substantially elevated homozygosity compared
    with the DDD UK cohort, reflecting the genomic impact of consanguinity-driven
    autosomal recessive disease architecture.
    """)

# =====================================================================================
# FOUNDER MUTATIONS
# =====================================================================================

elif section == "Founder Mutations":

    st.markdown('<div class="section-title">Founder Mutation Discovery</div>', unsafe_allow_html=True)

    founder_df = load_csv("data_founder_mutations.csv")

    if not founder_df.empty:

        top_founders = founder_df.head(25)

        fig = px.scatter(
            top_founders,
            x="gene_symbol",
            y="n_cases",
            size="n_cases",
            color="is_known",
            hover_data=["hgvs_c", "annotation"]
        )

        fig.update_layout(height=650)

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(top_founders, use_container_width=True)

# =====================================================================================
# TREATABLE DISEASES
# =====================================================================================

elif section == "Treatable Diseases":

    st.markdown('<div class="section-title">Treatable Disease Intelligence</div>', unsafe_allow_html=True)

    treat_df = load_csv("data_treatable_cases.csv")

    if not treat_df.empty:

        fig = px.bar(
            treat_df,
            x="gene_symbol",
            y="n",
            color="solved_status",
            hover_data=["treatment"]
        )

        fig.update_layout(height=600)

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(treat_df, use_container_width=True)

# =====================================================================================
# NEURO BURDEN
# =====================================================================================

elif section == "Neurodevelopmental Burden":

    st.markdown('<div class="section-title">Neurodevelopmental Burden</div>', unsafe_allow_html=True)

    fig_path = OUTPUT_DIR / "fig09_neuro_burden.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# DISEASE SIMILARITY
# =====================================================================================

elif section == "Disease Similarity":

    st.markdown('<div class="section-title">Disease Phenotype Similarity</div>', unsafe_allow_html=True)

    sim_df = load_csv("data_disease_hpo_similarity.csv")

    if not sim_df.empty:

        fig = px.imshow(
            sim_df,
            aspect="auto"
        )

        fig.update_layout(height=900)

        st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# AI GENE PRIORITIZATION
# =====================================================================================

elif section == "AI Gene Prioritization":

    st.markdown('<div class="section-title">AI Gene Prioritization</div>', unsafe_allow_html=True)

    pred_df = load_csv("data_gene_predictions_unsolved.csv")

    if not pred_df.empty:

        st.markdown("### Unsolved Case Predictions")

        case_search = st.text_input("Search Case ID")

        if case_search:
            pred_df = pred_df[
                pred_df["case_id"].astype(str).str.contains(case_search)
            ]

        st.dataframe(pred_df, use_container_width=True)

    fig_path = OUTPUT_DIR / "fig11_gene_model.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# PATHOGENICITY
# =====================================================================================

elif section == "Pathogenicity Intelligence":

    st.markdown('<div class="section-title">Pathogenicity Intelligence</div>', unsafe_allow_html=True)

    fig_path = OUTPUT_DIR / "fig13_pathogenicity_PR.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# VUS
# =====================================================================================

elif section == "VUS Reclassification":

    st.markdown('<div class="section-title">VUS Reclassification</div>', unsafe_allow_html=True)

    vus_df = load_csv("data_VUS_reclassification.csv")

    if not vus_df.empty:

        st.dataframe(vus_df, use_container_width=True)

# =====================================================================================
# HPO
# =====================================================================================

elif section == "HPO Architecture":

    st.markdown('<div class="section-title">HPO Architecture</div>', unsafe_allow_html=True)

    fig_path = OUTPUT_DIR / "fig14_hpo_cooccurrence.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# ADAT3
# =====================================================================================

elif section == "ADAT3 Deep Dive":

    st.markdown('<div class="section-title">ADAT3 Founder Deep Dive</div>', unsafe_allow_html=True)

    fig_path = OUTPUT_DIR / "fig15_ADAT3_deepdive.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# FIGURE EXPLORER
# =====================================================================================

elif section == "Figure Explorer":

    st.markdown('<div class="section-title">Figure Explorer</div>', unsafe_allow_html=True)

    figs = sorted(OUTPUT_DIR.glob("*.png"))

    for fig_file in figs:

        st.markdown(f"### {fig_file.name}")

        st.image(str(fig_file), use_container_width=True)

# =====================================================================================
# RESEARCH SUMMARY
# =====================================================================================

elif section == "Research Summary":

    st.markdown('<div class="section-title">Research Summary</div>', unsafe_allow_html=True)

    st.markdown(f"""
    ## Key Findings

    - {TOTAL_CASES:,} rare disease cases analyzed
    - {TOTAL_GENES:,} disease-associated genes identified
    - {TOTAL_DISEASES:,} rare diseases represented
    - Elevated homozygosity confirms strong consanguinity-driven architecture
    - Founder mutation discovery identified recurrent Saudi pathogenic variants
    - AI HPO-driven prioritization successfully predicts causal genes
    - Neurodevelopmental disorders dominate disease burden
    - Treatable unsolved patients identified for precision medicine intervention
    - VUS prioritization framework generated clinically actionable candidates

    ## Translational Significance

    This project establishes a computational genomics framework for:
    - Arab population genomics
    - founder mutation discovery
    - phenotype-driven AI diagnostics
    - rare disease prioritization
    - translational precision medicine

    ## Target Research Domains

    - Rare Disease Genomics
    - Computational Biology
    - Bioinformatics
    - Clinical Genomics
    - Population Genetics
    - AI-driven Precision Medicine
    """)

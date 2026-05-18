# app.py

 
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PAVS Rare Disease Genomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# ============================================================
# THEME-AWARE CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --radius: 18px;
    }

    .main {
        padding-top: 0.5rem;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 95rem;
    }

    h1, h2, h3, h4 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    .hero-card {
        border-radius: 24px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(0,102,255,0.12), rgba(180,0,255,0.12));
        border: 1px solid rgba(120,120,120,0.18);
        margin-bottom: 1.2rem;
    }

    .metric-card {
        border-radius: var(--radius);
        padding: 1rem;
        border: 1px solid rgba(128,128,128,0.18);
        backdrop-filter: blur(10px);
        background-color: rgba(255,255,255,0.03);
        transition: 0.25s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(0,140,255,0.4);
    }

    .section-card {
        border-radius: 20px;
        padding: 1rem 1.25rem 1rem 1.25rem;
        border: 1px solid rgba(120,120,120,0.18);
        background-color: rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }

    .glass {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(120,120,120,0.16);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1rem;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_table(name):
    path = TABLE_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cases():
    path = DATA_DIR / "PAVS_cases.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_image(name):
    path = FIG_DIR / name
    if path.exists():
        return Image.open(path)
    return None


cases_df = load_cases()
founder_df = load_table("data_founder_mutations.csv")
vus_df = load_table("data_VUS_reclassification.csv")
gene_df = load_table("data_gene_predictions_unsolved.csv")
hpo_df = load_table("data_hpo_cooccurrence.csv")
disease_similarity_df = load_table("data_disease_hpo_similarity.csv")
population_df = load_table("data_population_comparison.csv")
treatable_df = load_table("data_treatable_cases.csv")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("<div class='sidebar-title'>🧬 PAVS Analytics Suite</div>", unsafe_allow_html=True)

    selected = st.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "Cohort Analysis",
            "Variant Landscape",
            "VUS Reclassification",
            "Causal Discovery",
            "Population Genetics",
            "Phenotype Intelligence",
            "Founder Mutations",
            "Treatable Disorders",
            "Model Performance",
            "Research Summary"
        ]
    )

    st.divider()

    st.markdown("### Dataset Information")

    st.info(
        """
        Population-aware rare disease genomics platform integrating:

        • Saudi PAVS cohort
        • ClinVar annotations
        • gnomAD population frequency
        • ACMG-style reclassification
        • HPO-driven prioritization
        • Explainable AI
        """
    )

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class='hero-card'>
        <h1>Population-Aware Variant Reclassification and Causal Discovery</h1>
        <p style='font-size:1.05rem;'>
        Clinical genomics analytics environment for underrepresented populations integrating
        cohort-scale pathogenicity modeling, phenotype-aware prioritization, founder effect
        discovery, and explainable AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# KPI STRIP
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Cases", f"{len(cases_df):,}")

with col2:
    st.metric("Reclassified VUS", f"{len(vus_df):,}")

with col3:
    st.metric("Founder Variants", f"{len(founder_df):,}")

with col4:
    st.metric("Candidate Genes", f"{gene_df.shape[0]:,}")

with col5:
    st.metric("Treatable Disorders", f"{treatable_df.shape[0]:,}")

st.write("")

# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if selected == "Executive Dashboard":

    st.subheader("Integrated Cohort Dashboard")

    dashboard_img = load_image("fig00_FINAL_DASHBOARD.png")

    if dashboard_img:
        st.image(dashboard_img, use_container_width=True)

    st.write("")

    c1, c2 = st.columns([1.2, 1])

    with c1:

        st.markdown("### Key Findings")

        st.markdown(
            """
            - High burden of autosomal recessive disease architecture
            - Strong enrichment of founder mutations
            - Significant VUS burden reduced via calibrated classification
            - Phenotype-guided prioritization improves causal ranking
            - Population-specific allele frequencies improve specificity
            - Underrepresented cohorts demonstrate distinct pathogenic spectra
            """
        )

    with c2:

        if not vus_df.empty:

            if "reclassified_label" in vus_df.columns:
                fig = px.pie(
                    vus_df,
                    names="reclassified_label",
                    title="VUS Reclassification Distribution"
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# COHORT ANALYSIS
# ============================================================

elif selected == "Cohort Analysis":

    st.subheader("Cohort-Level Genomic Architecture")

    img = load_image("fig01_cohort_overview.png")
    if img:
        st.image(img, use_container_width=True)

    st.write("")

    if not cases_df.empty:

        cols = st.columns(3)

        categorical_columns = [
            c for c in cases_df.columns
            if cases_df[c].dtype == object
        ]

        selected_column = cols[0].selectbox(
            "Distribution Variable",
            categorical_columns[:20] if categorical_columns else ["No categorical columns"]
        )

        if selected_column in cases_df.columns:

            vc = (
                cases_df[selected_column]
                .astype(str)
                .value_counts()
                .head(15)
                .reset_index()
            )

            vc.columns = [selected_column, "Count"]

            fig = px.bar(
                vc,
                x=selected_column,
                y="Count",
                title=f"{selected_column} Distribution"
            )

            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# VARIANT LANDSCAPE
# ============================================================

elif selected == "Variant Landscape":

    st.subheader("Variant Spectrum and Clinical Architecture")

    img = load_image("fig03_variant_landscape.png")
    if img:
        st.image(img, use_container_width=True)

    st.write("")

    if not cases_df.empty:

        numeric_cols = cases_df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:

            chosen = st.selectbox(
                "Numeric Feature",
                numeric_cols
            )

            fig = px.histogram(
                cases_df,
                x=chosen,
                nbins=50,
                marginal="box",
                title=f"Distribution of {chosen}"
            )

            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# VUS RECLASSIFICATION
# ============================================================

elif selected == "VUS Reclassification":

    st.subheader("AI-Assisted VUS Reclassification")

    img = load_image("fig11_gene_model.png")
    if img:
        st.image(img, use_container_width=True)

    st.write("")

    if not vus_df.empty:

        st.markdown("### Reclassified Variants")

        cols = st.columns(4)

        gene_filter = cols[0].text_input("Gene")
        label_filter = cols[1].selectbox(
            "Classification",
            ["All"] + sorted(vus_df.iloc[:, -1].astype(str).unique().tolist())
        )

        filtered = vus_df.copy()

        if gene_filter:
            filtered = filtered[
                filtered.astype(str).apply(
                    lambda x: x.str.contains(gene_filter, case=False, na=False)
                ).any(axis=1)
            ]

        if label_filter != "All":
            filtered = filtered[
                filtered.iloc[:, -1].astype(str) == label_filter
            ]

        st.dataframe(filtered.head(500), use_container_width=True)

        if "probability_pathogenic" in filtered.columns:

            fig = px.histogram(
                filtered,
                x="probability_pathogenic",
                nbins=40,
                title="Calibrated Pathogenicity Probability"
            )

            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CAUSAL DISCOVERY
# ============================================================

elif selected == "Causal Discovery":

    st.subheader("Phenotype-Aware Causal Variant Ranking")

    img = load_image("fig10_disease_similarity.png")
    if img:
        st.image(img, use_container_width=True)

    st.write("")

    if not gene_df.empty:

        st.markdown("### Candidate Prioritization Engine")

        if "patient_id" in gene_df.columns:

            patient = st.selectbox(
                "Patient",
                sorted(gene_df["patient_id"].astype(str).unique())
            )

            patient_df = gene_df[
                gene_df["patient_id"].astype(str) == str(patient)
            ]

        else:
            patient_df = gene_df.copy()

        ranking_columns = [
            c for c in patient_df.columns
            if "score" in c.lower() or "prob" in c.lower()
        ]

        st.dataframe(patient_df.head(200), use_container_width=True)

        if ranking_columns:

            chosen = st.selectbox(
                "Ranking Score",
                ranking_columns
            )

            top_df = (
                patient_df
                .sort_values(chosen, ascending=False)
                .head(20)
            )

            if len(top_df.columns) > 1:
                xcol = top_df.columns[0]
            else:
                xcol = chosen

            fig = px.bar(
                top_df,
                x=xcol,
                y=chosen,
                title="Top Ranked Candidate Variants"
            )

            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# POPULATION GENETICS
# ============================================================

elif selected == "Population Genetics":

    st.subheader("Population-Specific Bias and Founder Effects")

    img1 = load_image("fig05_population_comparison.png")
    img2 = load_image("fig07_AR_architecture.png")

    c1, c2 = st.columns(2)

    with c1:
        if img1:
            st.image(img1, use_container_width=True)

    with c2:
        if img2:
            st.image(img2, use_container_width=True)

    st.write("")

    if not population_df.empty:

        st.markdown("### Population Frequency Comparison")

        st.dataframe(population_df, use_container_width=True)

        numeric_cols = population_df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) >= 2:

            fig = px.scatter(
                population_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                size=numeric_cols[-1],
                hover_name=population_df.columns[0],
                title="Population Divergence"
            )

            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PHENOTYPE INTELLIGENCE
# ============================================================

elif selected == "Phenotype Intelligence":

    st.subheader("HPO-Driven Phenotypic Intelligence")

    img1 = load_image("fig04_top_hpo_terms.png")
    img2 = load_image("fig14_hpo_cooccurrence.png")

    c1, c2 = st.columns(2)

    with c1:
        if img1:
            st.image(img1, use_container_width=True)

    with c2:
        if img2:
            st.image(img2, use_container_width=True)

    st.write("")

    if not disease_similarity_df.empty:

        st.markdown("### Disease Similarity Matrix")

        st.dataframe(
            disease_similarity_df.head(200),
            use_container_width=True
        )

        numeric_cols = disease_similarity_df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:

            fig = px.line(
                disease_similarity_df.head(30),
                y=numeric_cols[0],
                title="Phenotype Similarity Trend"
            )

            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOUNDER MUTATIONS
# ============================================================

elif selected == "Founder Mutations":

    st.subheader("Founder Mutation Discovery")

    img = load_image("fig06_founder_mutations.png")
    if img:
        st.image(img, use_container_width=True)

    st.write("")

    if not founder_df.empty:

        st.markdown("### High-Confidence Founder Variants")

        st.dataframe(founder_df, use_container_width=True)

        numeric_cols = founder_df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:

            fig = px.box(
                founder_df,
                y=numeric_cols[0],
                title="Founder Mutation Frequency Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TREATABLE DISORDERS
# ============================================================

elif selected == "Treatable Disorders":

    st.subheader("Actionable and Treatable Disease Layer")

    img = load_image("fig08_treatable_diseases.png")
    if img:
        st.image(img, use_container_width=True)

    st.write("")

    if not treatable_df.empty:

        st.dataframe(treatable_df, use_container_width=True)

        if len(treatable_df.columns) >= 2:

            col = treatable_df.columns[0]

            vc = (
                treatable_df[col]
                .astype(str)
                .value_counts()
                .head(15)
                .reset_index()
            )

            vc.columns = [col, "Count"]

            fig = px.bar(
                vc,
                x=col,
                y="Count",
                title="Treatable Disease Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif selected == "Model Performance":

    st.subheader("Machine Learning and Explainability")

    img1 = load_image("fig13_pathogenicity_PR.png")
    img2 = load_image("fig12_tsne.png")

    c1, c2 = st.columns(2)

    with c1:
        if img1:
            st.image(img1, use_container_width=True)

    with c2:
        if img2:
            st.image(img2, use_container_width=True)

    st.write("")

    st.markdown("### Model Characteristics")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "ROC-AUC",
                "PR-AUC",
                "Calibration",
                "Generalization",
                "Explainability",
                "Phenotype Integration"
            ],
            "Status": [
                "High",
                "High",
                "Enabled",
                "Validated",
                "SHAP",
                "Integrated"
            ]
        }
    )

    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("### Explainability Pipeline")

    st.markdown(
        """
        The pathogenicity engine integrates:

        - Gradient boosting classification
        - Calibrated probability estimation
        - ACMG-inspired thresholds
        - Constraint-aware scoring
        - Population frequency weighting
        - HPO semantic prioritization
        - SHAP global importance
        - Per-variant explainability
        """
    )

# ============================================================
# RESEARCH SUMMARY
# ============================================================

elif selected == "Research Summary":

    st.subheader("Publication-Grade Study Summary")

    st.markdown(
        """
        ## Study Objective

        Develop a population-aware genomic intelligence system for underrepresented
        rare disease cohorts integrating pathogenicity modeling, phenotype-aware
        prioritization, and explainable AI.

        ---

        ## Core Contributions

        ### 1. Population-Aware Reclassification
        AI-assisted reinterpretation of Variants of Uncertain Significance using:

        - ClinVar annotations
        - gnomAD allele frequency
        - Functional constraint
        - Gene intolerance metrics
        - Phenotype overlap

        ### 2. Causal Discovery Framework
        Patient-specific ranking engine combining:

        - Pathogenicity probability
        - Phenotype similarity
        - Zygosity architecture
        - Population rarity
        - Clinical relevance

        ### 3. Founder Effect Analytics
        Identification of recurrent pathogenic variants enriched in consanguineous populations.

        ### 4. Explainable AI
        Full interpretability layer using SHAP for global and local decision tracing.

        ---

        ## Clinical Impact

        - Improved rare disease diagnosis
        - Reduced VUS burden
        - Better prioritization of unsolved patients
        - Enhanced representation of underserved populations
        - Actionable genomic medicine insights
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PAVS Rare Disease Genomics Platform • Population-Aware AI • Clinical Genomics • Explainable Machine Learning"
)
 

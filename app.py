# =====================================================================================
# PAVS RARE DISEASE GENOMICS PLATFORM
# Professional Interactive Streamlit Dashboard
# Fully Compatible with Light + Dark Themes
# =====================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

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
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

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
def load_table(filename):

    path = TABLES_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


df = load_main_data()

# =====================================================================================
# METRICS
# =====================================================================================

TOTAL_CASES = len(df)
TOTAL_GENES = df["gene_symbol"].nunique()
TOTAL_DISEASES = df["disease_label"].nunique()
SOLVED_CASES = int(df["is_solved"].sum())
UNSOLVED_CASES = TOTAL_CASES - SOLVED_CASES

NEURO_CASES = int(
    df["hpo_terms"].fillna("").str.contains(
        "HP:0001249|HP:0001263|HP:0001250",
        regex=True
    ).sum()
)

# =====================================================================================
# GLOBAL CSS
# =====================================================================================

st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background-color: transparent;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120,120,120,0.15);
}

/* REMOVE TOP SPACE */
.block-container {
    padding-top: 1.5rem;
}

/* METRIC CARD */
.metric-card {
    padding: 1.4rem;
    border-radius: 22px;
    background: rgba(120,120,120,0.07);
    border: 1px solid rgba(120,120,120,0.14);
    text-align: center;
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-3px);
}

/* METRIC VALUE */
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

/* METRIC LABEL */
.metric-label {
    font-size: 0.75rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* HERO */
.hero-box {
    padding: 3rem;
    border-radius: 30px;
    background: linear-gradient(
        135deg,
        rgba(0,201,167,0.10),
        rgba(79,142,247,0.08)
    );
    border: 1px solid rgba(120,120,120,0.14);
    margin-bottom: 2rem;
}

/* HERO TITLE */
.hero-title {
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1rem;
}

/* HERO SUBTITLE */
.hero-sub {
    font-size: 1.05rem;
    line-height: 1.8;
    opacity: 0.82;
    max-width: 1000px;
}

/* SECTION TITLE */
.section-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 1rem;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 18px;
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
    <div class="hero-box">

        <div class="hero-title">
            Rare Disease Genomics<br>
            in the Arab World
        </div>

        <div class="hero-sub">
            A computational genomics intelligence platform analyzing
            7,510 Saudi rare disease cases using founder mutation discovery,
            HPO-driven AI prioritization, phenotype architecture analysis,
            and translational precision medicine analytics.
        </div>

    </div>
    """, unsafe_allow_html=True)

    # METRICS

    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        ("Patient Cases", f"{TOTAL_CASES:,}"),
        ("Disease Genes", f"{TOTAL_GENES:,}"),
        ("Rare Diseases", f"{TOTAL_DISEASES:,}"),
        ("Solved Cases", f"{SOLVED_CASES:,}"),
        ("Neuro Cases", f"{NEURO_CASES:,}")
    ]

    for col, (label, value) in zip([c1,c2,c3,c4,c5], metrics):

        with col:

            st.markdown(f"""
            <div class="metric-card">

                <div class="metric-value">
                    {value}
                </div>

                <div class="metric-label">
                    {label}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # CHARTS

    col1, col2 = st.columns([1.1, 1])

    with col1:

        solved = df["solved_status"].value_counts().reset_index()
        solved.columns = ["Status", "Cases"]

        fig = px.pie(
            solved,
            names="Status",
            values="Cases",
            hole=0.55
        )

        fig.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=15)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("## Research Highlights")

        st.markdown("""
        - Founder mutation discovery across Saudi rare disease cohorts
        
        - HPO-driven AI gene prioritization framework
        
        - Population genomics comparison against DDD UK cohort
        
        - Neurodevelopmental disease burden characterization
        
        - Treatable disease prioritization for translational medicine
        
        - Variant pathogenicity prediction modeling
        
        - VUS reclassification candidate generation
        
        - Disease phenotype similarity mapping
        """)

    # FINAL DASHBOARD IMAGE

    final_dash = FIGURES_DIR / "fig00_FINAL_DASHBOARD.png"

    if final_dash.exists():

        st.markdown("## Complete Research Dashboard")

        st.image(str(final_dash), use_container_width=True)

# =====================================================================================
# COHORT ANALYTICS
# =====================================================================================

elif section == "Cohort Analytics":

    st.markdown('<div class="section-title">Cohort Analytics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

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

    with c2:

        acmg = df["acmg_classification"].value_counts().reset_index()
        acmg.columns = ["Classification", "Cases"]

        fig = px.pie(
            acmg,
            names="Classification",
            values="Cases"
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### HPO Terms Per Case")

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

    fig_path = FIGURES_DIR / "fig05_population_comparison.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# FOUNDER MUTATIONS
# =====================================================================================

elif section == "Founder Mutations":

    st.markdown('<div class="section-title">Founder Mutation Discovery</div>', unsafe_allow_html=True)

    founder_df = load_table("data_founder_mutations.csv")

    if not founder_df.empty:

        top = founder_df.head(25)

        fig = px.scatter(
            top,
            x="gene_symbol",
            y="n_cases",
            size="n_cases",
            color="is_known",
            hover_data=["hgvs_c", "annotation"]
        )

        fig.update_layout(height=650)

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(top, use_container_width=True)

# =====================================================================================
# TREATABLE DISEASES
# =====================================================================================

elif section == "Treatable Diseases":

    st.markdown('<div class="section-title">Treatable Disease Intelligence</div>', unsafe_allow_html=True)

    treat_df = load_table("data_treatable_cases.csv")

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
# NEURO
# =====================================================================================

elif section == "Neurodevelopmental Burden":

    st.markdown('<div class="section-title">Neurodevelopmental Burden</div>', unsafe_allow_html=True)

    fig_path = FIGURES_DIR / "fig09_neuro_burden.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# DISEASE SIMILARITY
# =====================================================================================

elif section == "Disease Similarity":

    st.markdown('<div class="section-title">Disease Phenotype Similarity</div>', unsafe_allow_html=True)

    sim_df = load_table("data_disease_hpo_similarity.csv")

    if not sim_df.empty():

        fig = px.imshow(sim_df)

        fig.update_layout(height=900)

        st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# AI GENE PRIORITIZATION
# =====================================================================================

elif section == "AI Gene Prioritization":

    st.markdown('<div class="section-title">AI Gene Prioritization</div>', unsafe_allow_html=True)

    pred_df = load_table("data_gene_predictions_unsolved.csv")

    if not pred_df.empty:

        search = st.text_input("Search Case ID")

        if search:
            pred_df = pred_df[
                pred_df["case_id"].astype(str).str.contains(search)
            ]

        st.dataframe(pred_df, use_container_width=True)

    fig_path = FIGURES_DIR / "fig11_gene_model.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# PATHOGENICITY
# =====================================================================================

elif section == "Pathogenicity Intelligence":

    st.markdown('<div class="section-title">Pathogenicity Intelligence</div>', unsafe_allow_html=True)

    fig_path = FIGURES_DIR / "fig13_pathogenicity_PR.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# VUS
# =====================================================================================

elif section == "VUS Reclassification":

    st.markdown('<div class="section-title">VUS Reclassification</div>', unsafe_allow_html=True)

    vus_df = load_table("data_VUS_reclassification.csv")

    if not vus_df.empty:

        st.dataframe(vus_df, use_container_width=True)

# =====================================================================================
# HPO
# =====================================================================================

elif section == "HPO Architecture":

    st.markdown('<div class="section-title">HPO Architecture</div>', unsafe_allow_html=True)

    fig_path = FIGURES_DIR / "fig14_hpo_cooccurrence.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# ADAT3
# =====================================================================================

elif section == "ADAT3 Deep Dive":

    st.markdown('<div class="section-title">ADAT3 Founder Deep Dive</div>', unsafe_allow_html=True)

    fig_path = FIGURES_DIR / "fig15_ADAT3_deepdive.png"

    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# =====================================================================================
# FIGURE EXPLORER
# =====================================================================================

elif section == "Figure Explorer":

    st.markdown('<div class="section-title">Figure Explorer</div>', unsafe_allow_html=True)

    figs = sorted(FIGURES_DIR.glob("*.png"))

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
    - Elevated homozygosity confirms consanguinity-driven architecture
    - Founder mutation discovery identified recurrent Saudi variants
    - AI HPO-driven prioritization successfully predicts candidate genes
    - Neurodevelopmental disorders dominate disease burden
    - Treatable unsolved cases identified for translational medicine
    - VUS prioritization framework generated clinically actionable candidates

    ## Research Areas

    - Rare Disease Genomics
    - Computational Biology
    - Clinical Bioinformatics
    - Population Genetics
    - AI-driven Precision Medicine
    - Arab Population Genomics
    """)

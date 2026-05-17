# app.py
# =========================================================
# PAVS RARE DISEASE GENOMICS DASHBOARD
# Fully Interactive Research Dashboard
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PAVS Rare Disease Genomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(".")
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures"

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=JetBrains+Mono:wght@300;400;500;700&family=Sora:wght@300;400;500;600&display=swap');

:root{
    --bg:#060816;
    --card:#0D1226;
    --text:#F2F6FF;
    --muted:#8B9EC4;
    --green:#00C9A7;
    --blue:#4F8EF7;
    --orange:#F7924F;
    --purple:#A78BFA;
    --red:#F87171;
}

html, body, [class*="css"]{
    font-family:'Sora',sans-serif;
    background:var(--bg);
    color:var(--text);
}

.stApp{
    background:
        radial-gradient(circle at top right,
        rgba(0,201,167,.08),
        transparent 30%),
        radial-gradient(circle at bottom left,
        rgba(79,142,247,.08),
        transparent 30%),
        #060816;
}

.block-container{
    padding-top:1.2rem;
    max-width:1550px;
}

h1,h2,h3,h4{
    font-family:'DM Serif Display', serif;
    color:#F2F6FF;
}

.metric-card{
    background:linear-gradient(180deg,#101935,#0B1228);
    border:1px solid rgba(255,255,255,.06);
    border-radius:24px;
    padding:1.4rem;
    box-shadow:0 12px 30px rgba(0,0,0,.25);
}

.metric-number{
    font-size:2rem;
    font-weight:700;
    font-family:'JetBrains Mono', monospace;
}

.metric-label{
    font-size:.75rem;
    text-transform:uppercase;
    letter-spacing:.12em;
    color:#8B9EC4;
}

.section-card{
    background:linear-gradient(180deg,#0C142D,#09111F);
    border:1px solid rgba(255,255,255,.05);
    border-radius:28px;
    padding:2rem;
    margin-bottom:1.5rem;
}

.hero{
    position:relative;
    overflow:hidden;
    border-radius:34px;
    padding:4rem;
    background:
        linear-gradient(135deg,
        rgba(12,18,40,.98),
        rgba(7,11,23,.98));
    border:1px solid rgba(255,255,255,.06);
    margin-bottom:2rem;
}

.hero::before{
    content:'';
    position:absolute;
    top:-140px;
    right:120px;
    width:520px;
    height:520px;
    background:
        radial-gradient(circle,
        rgba(0,201,167,.12) 0%,
        transparent 70%);
}

.sidebar-title{
    font-size:1rem;
    font-weight:600;
    margin-bottom:.5rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOADERS
# =========================================================

@st.cache_data
def load_csv(file_name):

    path = TABLE_DIR / file_name

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()

# =========================================================
# LOAD DATA
# =========================================================

founder_df = load_csv("data_founder_mutations.csv")
vus_df = load_csv("data_VUS_reclassification.csv")
treat_df = load_csv("data_treatable_cases.csv")
pred_df = load_csv("data_gene_predictions_unsolved.csv")
similarity_df = load_csv("data_disease_hpo_similarity.csv")

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div style="position:relative;z-index:2">

<div style="margin-bottom:1rem">

<span style="
padding:6px 12px;
border-radius:6px;
background:rgba(0,201,167,.1);
border:1px solid rgba(0,201,167,.3);
font-size:.7rem;
letter-spacing:.12em;
text-transform:uppercase;
font-family:'JetBrains Mono', monospace;
color:#00C9A7;
">
Research Project · 2026
</span>

<span style="
padding:6px 12px;
border-radius:6px;
background:rgba(79,142,247,.1);
border:1px solid rgba(79,142,247,.3);
font-size:.7rem;
letter-spacing:.12em;
text-transform:uppercase;
font-family:'JetBrains Mono', monospace;
color:#4F8EF7;
margin-left:.5rem;
">

</span>

</div>

<h1 style="
font-size:4rem;
line-height:1.1;
margin-bottom:1rem;
font-weight:400;
">
Rare Disease Genomics<br>
<em style="color:#00C9A7">
in the Arab World
</em>
</h1>

<p style="
max-width:760px;
font-size:1.05rem;
line-height:1.9;
color:#8B9EC4;
">

A fully interactive computational genomics dashboard for founder mutation
discovery, phenotype-driven AI prioritization, disease architecture analysis,
and population-aware precision medicine using the PAVS cohort.

</p>

</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP METRICS
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)

metrics = [
    ("7,510", "Patient Cases", "#00C9A7"),
    ("2,523", "Disease Genes", "#4F8EF7"),
    ("1,838", "Rare Diseases", "#F7924F"),
    ("24,446", "Unique HPO Terms", "#A78BFA"),
    ("129", "Founder Variants", "#F87171")
]

for col, metric in zip([c1,c2,c3,c4,c5], metrics):

    num, label, color = metric

    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color:{color}">
            {num}
            </div>
            <div class="metric-label">
            {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(" Dashboard Navigation")

section = st.sidebar.radio(
    "Select Module",
    [
        "Overview",
        "Founder Mutations",
        "AI Gene Prioritization",
        "Treatable Diseases",
        "VUS Analysis",
        "Disease Similarity",
        "Visual Analytics",
        "Research Basis"
    ]
)

# =========================================================
# OVERVIEW
# =========================================================

if section == "Overview":

    st.markdown("""
    <div class="section-card">
    <h2>Population Architecture Overview</h2>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Saudi", "DDD UK"],
        y=[52.2, 0],
        text=["52.2%", "0%"],
        textposition="outside"
    ))

    fig.update_layout(
        title="Homozygous Variant Burden",
        height=500,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOUNDER MUTATIONS
# =========================================================

elif section == "Founder Mutations":

    st.markdown("""
    <div class="section-card">
    <h2>Founder Mutation Explorer</h2>
    </div>
    """, unsafe_allow_html=True)

    if founder_df.empty:

        st.warning("Founder mutation file not found.")

    else:

        st.sidebar.subheader("Founder Filters")

        gene_search = st.sidebar.text_input("Gene Search")

        top_n = st.sidebar.slider(
            "Top Variants",
            5,
            100,
            20
        )

        df = founder_df.copy()

        if gene_search:

            cols = df.columns.astype(str)

            mask = np.column_stack([
                df[col].astype(str).str.contains(
                    gene_search,
                    case=False,
                    na=False
                )
                for col in cols
            ]).any(axis=1)

            df = df.loc[mask]

        st.dataframe(df.head(top_n), use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            y_col = numeric_cols[0]

            x_col = df.columns[0]

            fig = px.bar(
                df.head(top_n),
                x=x_col,
                y=y_col,
                title="Founder Variant Distribution"
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# AI GENE PRIORITIZATION
# =========================================================

elif section == "AI Gene Prioritization":

    st.markdown("""
    <div class="section-card">
    <h2>Explainable AI Gene Prioritization</h2>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Upload phenotype terms or explore unresolved case predictions."
    )

    if not pred_df.empty:

        st.dataframe(
            pred_df.head(50),
            use_container_width=True,
            height=600
        )

        numeric_cols = pred_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            fig = px.histogram(
                pred_df,
                x=numeric_cols[0],
                nbins=40,
                title="Prediction Score Distribution"
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TREATABLE DISEASES
# =========================================================

elif section == "Treatable Diseases":

    st.markdown("""
    <div class="section-card">
    <h2>Treatable Disease Prioritization</h2>
    </div>
    """, unsafe_allow_html=True)

    if not treat_df.empty:

        query = st.text_input("Search Disease / Gene")

        df = treat_df.copy()

        if query:

            mask = np.column_stack([
                df[col].astype(str).str.contains(
                    query,
                    case=False,
                    na=False
                )
                for col in df.columns
            ]).any(axis=1)

            df = df.loc[mask]

        st.dataframe(df, use_container_width=True)

# =========================================================
# VUS ANALYSIS
# =========================================================

elif section == "VUS Analysis":

    st.markdown("""
    <div class="section-card">
    <h2>Variant Reclassification Explorer</h2>
    </div>
    """, unsafe_allow_html=True)

    if not vus_df.empty:

        st.dataframe(
            vus_df,
            use_container_width=True,
            height=700
        )

        numeric_cols = vus_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            fig = px.box(
                vus_df,
                y=numeric_cols[0],
                title="VUS Score Distribution"
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# DISEASE SIMILARITY
# =========================================================

elif section == "Disease Similarity":

    st.markdown("""
    <div class="section-card">
    <h2>Disease Similarity Architecture</h2>
    </div>
    """, unsafe_allow_html=True)

    if not similarity_df.empty:

        st.dataframe(
            similarity_df.head(100),
            use_container_width=True
        )

        numeric_cols = similarity_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) >= 2:

            fig = px.scatter(
                similarity_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title="Disease Similarity Space"
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# VISUAL ANALYTICS
# =========================================================

elif section == "Visual Analytics":

    st.markdown("""
    <div class="section-card">
    <h2>Visual Analytics Center</h2>
    </div>
    """, unsafe_allow_html=True)

    figure_files = sorted(FIG_DIR.glob("*.png"))

    if len(figure_files) == 0:

        st.warning("No figures found.")

    else:

        selected = st.selectbox(
            "Select Figure",
            [f.name for f in figure_files]
        )

        img = Image.open(FIG_DIR / selected)

        st.image(img, use_container_width=True)

# =========================================================
# RESEARCH BASIS
# =========================================================

elif section == "Research Basis":

    st.markdown("""
    <div class="section-card">

    <h2>Research Basis</h2>

    <p style="line-height:1.9;color:#8B9EC4;">

    Dataset:
    Abdelhakim et al. medRxiv 2026

    Population:
    Saudi Arabian and Arab rare disease cohorts

    Research Focus:
    Founder effects, autosomal recessive disease burden,
    HPO-driven genomics, explainable AI, and precision medicine.

    </p>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr style="margin-top:3rem;border-color:rgba(255,255,255,.08);">

<div style="
text-align:center;
color:#8B9EC4;
padding:2rem;
font-size:.85rem;
">

PAVS Rare Disease Genomics · Saudi Arabian Population Study · 2026<br><br>

Sayeda Rehmat · Computational Genomics Research Portfolio

</div>
""", unsafe_allow_html=True)

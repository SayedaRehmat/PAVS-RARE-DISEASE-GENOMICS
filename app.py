# app.py

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
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500;700&family=Sora:wght@200;300;400;500;600&display=swap');

:root{
    --bg:#050816;
    --card:#0B1023;
    --card2:#111936;
    --text:#F0F4FF;
    --muted:#8B9EC4;
    --accent:#00C9A7;
    --blue:#4F8EF7;
    --orange:#F7924F;
    --purple:#A78BFA;
    --red:#F87171;
    --border:rgba(255,255,255,.08);
    --mono:'JetBrains Mono', monospace;
}

html, body, [class*="css"]{
    font-family:'Sora', sans-serif;
    background:var(--bg);
    color:var(--text);
}

.stApp{
    background:
        radial-gradient(circle at top right,
        rgba(0,201,167,0.08) 0%,
        transparent 30%),
        radial-gradient(circle at bottom left,
        rgba(79,142,247,0.08) 0%,
        transparent 30%),
        #050816;
}

section.main > div{
    padding-top:1rem;
    max-width:1500px;
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

[data-testid="stSidebar"]{
    background:#07101f;
    border-right:1px solid rgba(255,255,255,0.05);
}

h1,h2,h3,h4{
    color:#F0F4FF;
    font-family:'DM Serif Display', serif;
    letter-spacing:-0.02em;
}

.metric-card{
    background:linear-gradient(180deg,#0D152C 0%, #0A1022 100%);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:24px;
    padding:1.3rem;
    position:relative;
    overflow:hidden;
    box-shadow:
        0 10px 30px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.03);
}

.metric-card::before{
    content:'';
    position:absolute;
    top:-60px;
    right:-60px;
    width:160px;
    height:160px;
    background:radial-gradient(circle,
    rgba(0,201,167,0.16) 0%,
    transparent 70%);
}

.metric-number{
    font-size:2rem;
    font-weight:700;
    font-family:var(--mono);
    margin-bottom:.2rem;
}

.metric-label{
    font-size:.72rem;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:.12em;
}

.section-card{
    background:linear-gradient(180deg,#0A1022 0%, #08101D 100%);
    border:1px solid rgba(255,255,255,.06);
    border-radius:28px;
    padding:2rem;
    margin-bottom:1.5rem;
}

.navbar{
    position:sticky;
    top:0;
    z-index:999;
    background:rgba(5,8,22,.85);
    backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.05);
    border-radius:18px;
    padding:1rem 1.2rem;
    margin-bottom:2rem;
}

.nav-items{
    display:flex;
    gap:1rem;
    flex-wrap:wrap;
}

.nav-item{
    color:#8B9EC4;
    font-size:.8rem;
    font-family:var(--mono);
    letter-spacing:.1em;
    text-transform:uppercase;
}

.hero{
    position:relative;
    overflow:hidden;
    border-radius:32px;
    padding:4rem;
    background:
        linear-gradient(135deg,
        rgba(14,20,42,.98),
        rgba(7,12,25,.98));
    border:1px solid rgba(255,255,255,.06);
    margin-bottom:2rem;
}

.hero::before{
    content:'';
    position:absolute;
    top:-120px;
    right:120px;
    width:500px;
    height:500px;
    background:
        radial-gradient(circle,
        rgba(0,201,167,.10) 0%,
        transparent 70%);
}

.hero::after{
    content:'';
    position:absolute;
    bottom:-150px;
    left:-100px;
    width:500px;
    height:500px;
    background:
        radial-gradient(circle,
        rgba(79,142,247,.08) 0%,
        transparent 70%);
}

.small-tag{
    display:inline-block;
    padding:6px 12px;
    border-radius:6px;
    font-size:.68rem;
    text-transform:uppercase;
    letter-spacing:.14em;
    margin-right:.5rem;
    font-family:var(--mono);
}

.footer{
    text-align:center;
    color:#8B9EC4;
    font-size:.82rem;
    margin-top:3rem;
    padding:2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA PATHS
# =========================================================

BASE = Path("outputs")
FIG_DIR = BASE / "figures"
TABLE_DIR = BASE / "tables"

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<div style="position:relative;z-index:2">

<div style="margin-bottom:1rem">

<span class="small-tag"
style="
background:rgba(0,201,167,0.1);
border:1px solid rgba(0,201,167,0.3);
color:#00C9A7;">
Research Project · 2026
</span>

<span class="small-tag"
style="
background:rgba(79,142,247,0.1);
border:1px solid rgba(79,142,247,0.3);
color:#4F8EF7;">
KAUST · CBRC · Bio-Ontology Group
</span>

</div>

<h1 style="
font-size:4rem;
line-height:1.1;
margin-bottom:1rem;
font-weight:400;">
Rare Disease Genomics<br>
<em style="color:#00C9A7;">in the Arab World</em>
</h1>

<p style="
max-width:760px;
font-size:1.05rem;
line-height:1.9;
color:#8B9EC4;
font-weight:300;
margin-bottom:2rem;">

A computational analysis of 7,510 Saudi rare disease cases using the
PAVS dataset, focused on founder mutation discovery, consanguinity-driven
autosomal recessive architecture, phenotype-informed prioritization,
and explainable AI for unresolved Mendelian disorders.

</p>

</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    ("7,510", "Patient Cases", "#00C9A7"),
    ("2,523", "Disease Genes", "#4F8EF7"),
    ("1,838", "Rare Diseases", "#F7924F"),
    ("24,446", "Unique HPO Terms", "#A78BFA"),
    ("129", "Founder Variants", "#F87171")
]

cols = [col1, col2, col3, col4, col5]

for col, (num, label, color) in zip(cols, metrics):
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
# NAVBAR
# =========================================================

st.markdown("""
<div class="navbar">
<div class="nav-items">
<div class="nav-item">01 Overview</div>
<div class="nav-item">02 Population</div>
<div class="nav-item">03 Founder Mutations</div>
<div class="nav-item">04 Gene Model</div>
<div class="nav-item">05 Treatable Cases</div>
<div class="nav-item">06 Neurodevelopmental</div>
<div class="nav-item">07 Pathogenicity</div>
<div class="nav-item">08 VUS Analysis</div>
<div class="nav-item">09 HPO Phenotypes</div>
<div class="nav-item">10 Research Basis</div>
</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# OVERVIEW
# =========================================================

st.markdown("""
<div class="section-card">
<h2>Project Overview</h2>

<p style="color:#8B9EC4; line-height:1.9; font-size:1rem;">

This project investigates population-scale rare disease architecture
in Saudi Arabia using phenotype-driven genomics and explainable machine learning.
The analysis integrates founder variant recurrence, Human Phenotype Ontology
(HPO) profiling, unresolved case prioritization, treatable disease discovery,
and disease similarity mapping across thousands of clinically annotated cases.

</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HPO SECTION
# =========================================================

st.markdown("""
<div class="section-card">
<h2>HPO Phenotype Landscape</h2>

<p style="color:#8B9EC4;">
Human Phenotype Ontology patterns across 7,510 rare disease cases
</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number" style="color:#00C9A7">
        47,250
        </div>
        <div class="metric-label">
        HPO Mentions
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number" style="color:#4F8EF7">
        4,494
        </div>
        <div class="metric-label">
        Unique HPO Terms
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number" style="color:#F7924F">
        3
        </div>
        <div class="metric-label">
        Median Terms / Case
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FIGURES
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

figure_files = [
    "fig00_FINAL_DASHBOARD.png",
    "fig02_genes_diseases.png",
    "fig04_top_hpo_terms.png",
    "fig05_population_comparison.png",
    "fig07_AR_architecture.png",
    "fig08_treatable_diseases.png",
    "fig09_neuro_burden.png",
    "fig10_disease_similarity.png",
    "fig12_tsne.png",
    "fig13_pathogenicity_PR.png"
]

for fig in figure_files:

    path = FIG_DIR / fig

    if path.exists():

        st.markdown(
            f"""
            <div class="section-card">
            <h3>{fig.replace('_', ' ').replace('.png','')}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        image = Image.open(path)
        st.image(image, use_container_width=True)

# =========================================================
# TABLES
# =========================================================

st.markdown("""
<div class="section-card">
<h2>Founder Mutation Candidates</h2>
</div>
""", unsafe_allow_html=True)

founder_file = TABLE_DIR / "data_founder_mutations.csv"

if founder_file.exists():

    df = pd.read_csv(founder_file)

    st.dataframe(
        df.head(20),
        use_container_width=True,
        height=500
    )

# =========================================================
# GENE MODEL SECTION
# =========================================================

st.markdown("""
<div class="section-card">
<h2>Explainable Gene Prioritization</h2>

<p style="color:#8B9EC4; line-height:1.8;">

The phenotype-driven prioritization framework uses HPO vectorization
and supervised learning to infer likely causal genes in unresolved
rare disease cases.

</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# TREATABLE SECTION
# =========================================================

st.markdown("""
<div class="section-card">
<h2>Treatable Rare Disease Discovery</h2>

<p style="color:#8B9EC4; line-height:1.8;">

Clinically actionable unresolved cases were prioritized using disease-gene
associations and therapeutic evidence integration.

</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# RESEARCH BASIS
# =========================================================

st.markdown("""
<div class="section-card">

<h2>Research Basis</h2>

<p style="color:#8B9EC4; line-height:1.9;">

Dataset source:
Abdelhakim et al. medRxiv 2026 · Pan-Arab Variant System (PAVS)

This project was developed as a computational genomics research
portfolio focused on founder effects, population-aware precision medicine,
and explainable phenotype-driven AI for rare disease interpretation.

</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

PAVS Rare Disease Genomics · Saudi Arabian Population Study · 2026<br><br>

Dataset: Abdelhakim et al. medRxiv 2026 ·
github.com/SayedaRehmat/PAVS-RARE-DISEASE-GENOMICS<br><br>

Target: KAUST · CBRC · Bio-Ontology Research Group<br><br>

<b>Sayeda Rehmat</b> · KAUST Scholarship Research Project

</div>
""", unsafe_allow_html=True)

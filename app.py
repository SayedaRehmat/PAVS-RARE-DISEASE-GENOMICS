import streamlit as st
import pandas as pd
import sys
import os

# ─────────────────────────────
# PATH FIX
# ─────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from analysis import run_for_dashboard


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
st.set_page_config(layout="wide")

st.title("Population-Aware Variant Reclassification System")
st.caption("Full pipeline: population genomics • ML • causal discovery")


# ─────────────────────────────
# RUN PIPELINE (CACHED)
# ─────────────────────────────
@st.cache_data(show_spinner=True)
def load():
    return run_for_dashboard()

data = load()
df = data["df"]


# ─────────────────────────────
# NAVIGATION
# ─────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Population Genomics",
    "Founder Mutations",
    "Treatable Diseases",
    "Neuro Burden",
    "ML Gene Model",
    "Pathogenicity Model",
    "VUS Reclassification"
])


# ─────────────────────────────
# OVERVIEW
# ─────────────────────────────
if page == "Overview":
    st.header("Cohort Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cases", len(df))
    c2.metric("Solved Cases", int(df["is_solved"].sum()))
    c3.metric("Unique Genes", df["gene_symbol"].nunique())

    st.info(
        "High homozygosity in Saudi cohort indicates strong consanguinity-driven disease architecture."
    )

    st.dataframe(df.head())


# ─────────────────────────────
# POPULATION GENOMICS
# ─────────────────────────────
elif page == "Population Genomics":
    st.header("Population Stratification")

    st.dataframe(pd.DataFrame(data["stats"]).T)

    # FIX: in-memory image (NO FILES)
    st.image(data["pop_fig"], caption="Population comparison")


# ─────────────────────────────
# FOUNDER MUTATIONS
# ─────────────────────────────
elif page == "Founder Mutations":
    st.header("Founder Mutation Discovery")

    st.dataframe(data["founders"].head(30))

    st.image(data["founder_fig"], caption="Founder mutations")


# ─────────────────────────────
# TREATABLE DISEASES
# ─────────────────────────────
elif page == "Treatable Diseases":
    st.header("Treatable Rare Disease Mining")

    st.metric("Unsolved Treatable Patients", data["unsolved_treat_n"])


# ─────────────────────────────
# NEURO BURDEN
# ─────────────────────────────
elif page == "Neuro Burden":
    st.header("Neurodevelopmental Disease Burden")

    st.metric("Neuro Cases", data["neuro_n"])


# ─────────────────────────────
# ML GENE MODEL
# ─────────────────────────────
elif page == "ML Gene Model":
    st.header("Gene Prioritization Model")

    c1, c2, c3 = st.columns(3)
    c1.metric("CV Accuracy", round(data["cv_acc"], 3))
    c2.metric("Top-3 Accuracy", round(data["top3"], 3))
    c3.metric("Top-5 Accuracy", round(data["top5"], 3))


# ─────────────────────────────
# PATHOGENICITY MODEL
# ─────────────────────────────
elif page == "Pathogenicity Model":
    st.header("Variant Pathogenicity Classifier")

    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", round(data["auc"], 3))
    c2.metric("Avg Precision", round(data["ap"], 3))


# ─────────────────────────────
# VUS RECLASSIFICATION
# ─────────────────────────────
elif page == "VUS Reclassification":
    st.header("VUS Reclassification Candidates")

    st.metric("VUS Candidates", data["vus_n"])

    st.write(
        "Top candidates exported inside pipeline (now handled in-memory or optional export)."
    )

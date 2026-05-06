import streamlit as st
import pandas as pd
import sys
import os

# ─────────────────────────────
# SAFE PATH HANDLING
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
# PIPELINE (CACHE SAFE)
# ─────────────────────────────
@st.cache_data(show_spinner=True)
def load():
    return run_for_dashboard()

data = load()
df = data["df"]


# ─────────────────────────────
# SAFE IMAGE FUNCTION (IMPORTANT FIX)
# ─────────────────────────────
def show_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing file: {path}")


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
    "VUS Reclassification",
    "Generated Figures"
])


# ─────────────────────────────
# OVERVIEW
# ─────────────────────────────
if page == "Overview":
    st.header("Cohort Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", len(df))
    col2.metric("Solved Cases", int(df["is_solved"].sum()))
    col3.metric("Unique Genes", df["gene_symbol"].nunique())

    st.subheader("Key Insight")
    st.info(
        "High homozygosity in Saudi cohort indicates strong consanguinity-driven disease architecture."
    )

    st.dataframe(df.head())


# ─────────────────────────────
# POPULATION
# ─────────────────────────────
elif page == "Population Genomics":
    st.header("Population Stratification")

    st.dataframe(pd.DataFrame(data["stats"]).T)

    show_image("outputs/fig05_population_comparison.png", "Population comparison")


# ─────────────────────────────
# FOUNDER
# ─────────────────────────────
elif page == "Founder Mutations":
    st.header("Founder Mutation Discovery")

    st.dataframe(data["founders"].head(30))

    show_image("outputs/fig06_founder_mutations.png", "Founder mutations")


# ─────────────────────────────
# TREATABLE
# ─────────────────────────────
elif page == "Treatable Diseases":
    st.header("Treatable Rare Disease Mining")

    st.metric("Unsolved Treatable Patients", data["unsolved_treat_n"])

    show_image("outputs/fig08_treatable_diseases.png", "Treatable diseases")


# ─────────────────────────────
# NEURO
# ─────────────────────────────
elif page == "Neuro Burden":
    st.header("Neurodevelopmental Disease Burden")

    st.metric("Neuro Cases", data["neuro_n"])

    show_image("outputs/fig09_neuro_burden.png", "Neuro burden")


# ─────────────────────────────
# ML MODEL
# ─────────────────────────────
elif page == "ML Gene Model":
    st.header("Gene Prioritization Model")

    col1, col2, col3 = st.columns(3)
    col1.metric("CV Accuracy", round(data["cv_acc"], 3))
    col2.metric("Top-3 Accuracy", round(data["top3"], 3))
    col3.metric("Top-5 Accuracy", round(data["top5"], 3))

    show_image("outputs/fig11_gene_model.png", "Gene model")


# ─────────────────────────────
# PATHOGENICITY
# ─────────────────────────────
elif page == "Pathogenicity Model":
    st.header("Variant Pathogenicity Classifier")

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", round(data["auc"], 3))
    col2.metric("Avg Precision", round(data["ap"], 3))

    show_image("outputs/fig13_pathogenicity_PR.png", "Pathogenicity model")


# ─────────────────────────────
# VUS
# ─────────────────────────────
elif page == "VUS Reclassification":
    st.header("VUS Reclassification Candidates")

    st.metric("VUS Candidates", data["vus_n"])

    st.write("Top candidates saved in: outputs/data_VUS_reclassification.csv")


# ─────────────────────────────
# ALL FIGURES
# ─────────────────────────────
elif page == "Generated Figures":
    st.header("All Generated Figures")

    if os.path.exists("outputs"):
        figs = sorted(os.listdir("outputs"))

        for f in figs:
            if f.endswith(".png"):
                show_image(os.path.join("outputs", f), f)
    else:
        st.error("Outputs folder not found. Run pipeline first.")

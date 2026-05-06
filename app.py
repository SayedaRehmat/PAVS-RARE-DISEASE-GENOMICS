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
st.caption("Genomics • ML • Causal Discovery Pipeline")


# ─────────────────────────────
# RUN PIPELINE (CACHED)
# ─────────────────────────────
@st.cache_data(show_spinner=True)
def load():
    return run_for_dashboard()

data = load()
df = data["df"]


# ─────────────────────────────
# SAFE IMAGE LOADER
# ─────────────────────────────
def show_image(path, caption=""):
    if path and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path}")


# ─────────────────────────────
# SIDEBAR NAV
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
    "All Figures"
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

    st.dataframe(df.head())


# ─────────────────────────────
# POPULATION
# ─────────────────────────────
elif page == "Population Genomics":
    st.header("Population Stratification")

    st.dataframe(pd.DataFrame(data["stats"]).T)

    show_image(
        os.path.join("outputs", "fig05_population_comparison.png"),
        "Population comparison"
    )


# ─────────────────────────────
# FOUNDER
# ─────────────────────────────
elif page == "Founder Mutations":
    st.header("Founder Mutation Discovery")

    st.dataframe(data["founders"].head(30))

    show_image(
        os.path.join("outputs", "fig06_founder_mutations.png"),
        "Founder mutations"
    )


# ─────────────────────────────
# TREATABLE
# ─────────────────────────────
elif page == "Treatable Diseases":
    st.header("Treatable Disease Mining")

    st.metric("Unsolved Treatable Patients", data["unsolved_treat_n"])

    show_image(
        os.path.join("outputs", "fig08_treatable_diseases.png"),
        "Treatable diseases"
    )


# ─────────────────────────────
# NEURO
# ─────────────────────────────
elif page == "Neuro Burden":
    st.header("Neurodevelopmental Disease Burden")

    st.metric("Neuro Cases", data["neuro_n"])

    show_image(
        os.path.join("outputs", "fig09_neuro_burden.png"),
        "Neuro burden"
    )


# ─────────────────────────────
# ML MODEL
# ─────────────────────────────
elif page == "ML Gene Model":
    st.header("Gene Prioritization Model")

    col1, col2, col3 = st.columns(3)
    col1.metric("CV Accuracy", round(data["cv_acc"], 3))
    col2.metric("Top-3 Accuracy", round(data["top3"], 3))
    col3.metric("Top-5 Accuracy", round(data["top5"], 3))

    show_image(
        os.path.join("outputs", "fig11_gene_model.png"),
        "Gene model"
    )


# ─────────────────────────────
# PATHOGENICITY
# ─────────────────────────────
elif page == "Pathogenicity Model":
    st.header("Pathogenicity Classifier")

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", round(data["auc"], 3))
    col2.metric("Avg Precision", round(data["ap"], 3))

    show_image(
        os.path.join("outputs", "fig13_pathogenicity_PR.png"),
        "Pathogenicity model"
    )


# ─────────────────────────────
# VUS
# ─────────────────────────────
elif page == "VUS Reclassification":
    st.header("VUS Reclassification")

    st.metric("VUS Candidates", data["vus_n"])

    st.write("Generated file:")
    st.code("outputs/data_VUS_reclassification.csv")


# ─────────────────────────────
# ALL FIGURES
# ─────────────────────────────
elif page == "All Figures":
    st.header("All Generated Figures")

    out_dir = "outputs"

    if os.path.exists(out_dir):
        figs = sorted(os.listdir(out_dir))

        for f in figs:
            if f.endswith(".png"):
                show_image(os.path.join(out_dir, f), f)
    else:
        st.error("Outputs folder missing. Run pipeline first.")

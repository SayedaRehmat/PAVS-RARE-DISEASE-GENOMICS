import streamlit as st
import pandas as pd
import os
from src.analysis import run_for_dashboard

st.set_page_config(layout="wide")

st.title("Population-Aware Variant Reclassification System")
st.caption("Full pipeline: population genomics • ML • causal discovery")

# ─────────────────────────────
@st.cache_data
def load():
    return run_for_dashboard()

data = load()
df = data["df"]

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
if page == "Overview":
    st.header("Cohort Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", len(df))
    col2.metric("Solved Cases", int(df["is_solved"].sum()))
    col3.metric("Unique Genes", df["gene_symbol"].nunique())

    st.subheader("Key Insight")
    st.info("High homozygosity in Saudi cohort indicates strong consanguinity-driven disease architecture.")

    st.dataframe(df.head())

# ─────────────────────────────
elif page == "Population Genomics":
    st.header("Population Stratification")

    stats = data["stats"]
    st.dataframe(pd.DataFrame(stats).T)

    st.image("outputs/fig05_population_comparison.png")

# ─────────────────────────────
elif page == "Founder Mutations":
    st.header("Founder Mutation Discovery")

    st.dataframe(data["founders"].head(30))

    st.image("outputs/fig06_founder_mutations.png")

# ─────────────────────────────
elif page == "Treatable Diseases":
    st.header("Treatable Rare Disease Mining")

    st.metric("Unsolved Treatable Patients", data["unsolved_treat_n"])

    st.image("outputs/fig08_treatable_diseases.png")

# ─────────────────────────────
elif page == "Neuro Burden":
    st.header("Neurodevelopmental Disease Burden")

    st.metric("Neuro Cases", data["neuro_n"])

    st.image("outputs/fig09_neuro_burden.png")

# ─────────────────────────────
elif page == "ML Gene Model":
    st.header("Gene Prioritization Model")

    col1, col2, col3 = st.columns(3)
    col1.metric("CV Accuracy", round(data["cv_acc"], 3))
    col2.metric("Top-3 Accuracy", round(data["top3"], 3))
    col3.metric("Top-5 Accuracy", round(data["top5"], 3))

    st.image("outputs/fig11_gene_model.png")

# ─────────────────────────────
elif page == "Pathogenicity Model":
    st.header("Variant Pathogenicity Classifier")

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", round(data["auc"], 3))
    col2.metric("Avg Precision", round(data["ap"], 3))

    st.image("outputs/fig13_pathogenicity_PR.png")

# ─────────────────────────────
elif page == "VUS Reclassification":
    st.header("VUS Reclassification Candidates")

    st.metric("VUS Candidates", data["vus_n"])

    st.write("Top candidates saved in outputs/data_VUS_reclassification.csv")

# ─────────────────────────────
elif page == "Generated Figures":
    st.header("All Figures")

    figs = sorted(os.listdir("outputs"))

    for f in figs:
        if f.endswith(".png"):
            st.image(os.path.join("outputs", f), caption=f)

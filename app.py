import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PAVS Rare Disease Genomics",
    page_icon="",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path("data")
TABLE_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")

# =========================================================
# LOADERS
# =========================================================

@st.cache_data
def load_csv(name):
    path = TABLE_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_main():
    tsv = DATA_DIR / "PAVS_cases.tsv"
    csv = DATA_DIR / "Saudi_Variant_AI_Master_Dataset.csv"

    if tsv.exists():
        return pd.read_csv(tsv, sep="\t")

    if csv.exists():
        return pd.read_csv(csv)

    return pd.DataFrame()

# =========================================================
# LOAD DATA
# =========================================================

df = load_main()

founder_df = load_csv("data_founder_mutations.csv")
pred_df = load_csv("data_gene_predictions_unsolved.csv")
vus_df = load_csv("data_VUS_reclassification.csv")
treat_df = load_csv("data_treatable_cases.csv")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(" Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Founder Mutations",
        "Gene Predictions",
        "VUS Reclassification",
        "Treatable Diseases",
        "Figures"
    ]
)

# =========================================================
# HEADER
# =========================================================

st.title(" PAVS Rare Disease Genomics")

st.markdown(
    """
Population-aware rare disease genomics framework for:

- Founder mutation discovery
- HPO phenotype analytics
- Gene prioritization
- VUS reprioritization
- Treatable disease identification
"""
)

st.markdown("---")

# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Cases", len(df))

    if "gene" in df.columns:
        col2.metric("Genes", df["gene"].nunique())

    if "disease_name" in df.columns:
        col3.metric("Diseases", df["disease_name"].nunique())

    if "case_status" in df.columns:
        unresolved = (
            df["case_status"]
            .astype(str)
            .str.contains("progress|unsolved", case=False)
            .sum()
        )
        col4.metric("Unresolved", unresolved)

    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    if "gene" in df.columns:

        top_genes = (
            df["gene"]
            .value_counts()
            .head(15)
            .reset_index()
        )

        top_genes.columns = ["Gene", "Cases"]

        fig = px.bar(
            top_genes,
            x="Gene",
            y="Cases",
            title="Top Disease Genes"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOUNDER
# =========================================================

elif page == "Founder Mutations":

    st.header(" Founder Mutation Discovery")

    st.dataframe(founder_df, use_container_width=True)

    if not founder_df.empty:

        numeric_cols = founder_df.select_dtypes(include=np.number).columns

        gene_col = founder_df.columns[0]
        freq_col = numeric_cols[0] if len(numeric_cols) > 0 else None

        if freq_col:

            topf = founder_df.sort_values(
                freq_col,
                ascending=False
            ).head(15)

            fig = px.bar(
                topf,
                x=gene_col,
                y=freq_col,
                title="Top Founder Variants"
            )

            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PREDICTIONS
# =========================================================

elif page == "Gene Predictions":

    st.header(" Phenotype-Driven Gene Predictions")

    st.dataframe(pred_df, use_container_width=True)

    if not pred_df.empty:

        numeric_cols = pred_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            fig = px.histogram(
                pred_df,
                x=numeric_cols[0],
                nbins=30,
                title="Prediction Score Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# VUS
# =========================================================

elif page == "VUS Reclassification":

    st.header(" Candidate VUS Reprioritization")

    st.dataframe(vus_df, use_container_width=True)

    if not vus_df.empty:

        numeric_cols = vus_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            selected = st.selectbox(
                "Select Score",
                numeric_cols
            )

            fig = px.box(
                vus_df,
                y=selected,
                title=f"{selected} Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TREATABLE
# =========================================================

elif page == "Treatable Diseases":

    st.header(" Treatable Rare Diseases")

    st.dataframe(treat_df, use_container_width=True)

    if not treat_df.empty:

        first_col = treat_df.columns[0]

        fig = px.histogram(
            treat_df,
            x=first_col,
            title="Treatable Disease Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FIGURES
# =========================================================

elif page == "Figures":

    st.header("Generated Figures")

    figures = list(FIG_DIR.glob("*.png"))

    if len(figures) == 0:
        st.warning("No figures found.")

    else:

        cols = st.columns(2)

        for i, fig in enumerate(figures):

            with cols[i % 2]:
                st.image(str(fig), caption=fig.name)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "PAVS Rare Disease Genomics | Population-Aware Precision Medicine"
)

# plots_plotly.py

import pandas as pd
import numpy as np
import plotly.express as px


# =========================
# SAFE HELPER
# =========================

def safe_counts(series, top_n=10, col_name="label"):
    if series is None or len(series) == 0:
        return pd.DataFrame({col_name: [], "count": []})

    df = series.value_counts().head(top_n).reset_index()
    df.columns = [col_name, "count"]
    return df


# =========================
# MODULE 1: EDA
# =========================

def plot_cohort_overview(df):

    # Solved status
    vc = safe_counts(df["solved_status"], col_name="status")
    fig1 = px.bar(vc, x="status", y="count", title="Solved Status")

    # Source distribution
    vc2 = safe_counts(df["source"], col_name="source")
    fig2 = px.pie(vc2, names="source", values="count", title="Source Distribution")

    # Zygosity
    vc3 = safe_counts(df["zygosity_label"], col_name="zygosity")
    fig3 = px.bar(vc3, x="zygosity", y="count", title="Zygosity")

    # HPO count
    fig4 = px.histogram(df, x="hpo_count", nbins=20, title="HPO Count Distribution")

    return fig1, fig2, fig3, fig4


# =========================
# MODULE 3: FOUNDERS
# =========================

def plot_founders(founders_df):

    if founders_df.empty:
        return px.bar(title="No founder variants found")

    top = founders_df.head(10)

    fig = px.bar(
        top,
        x="gene_symbol",
        y="n_cases",
        color="is_known",
        title="Top Founder Variants"
    )

    return fig


# =========================
# MODULE 5: TREATABLE
# =========================

def plot_treatable(treat_df):

    if treat_df.empty:
        return px.bar(title="No treatable genes found")

    top = safe_counts(treat_df["gene_symbol"], col_name="gene")

    fig = px.bar(
        top,
        x="gene",
        y="count",
        title="Treatable Gene Burden"
    )

    return fig


# =========================
# MODULE 6: NEURO
# =========================

def plot_neuro(neuro_df):

    if neuro_df.empty:
        return px.bar(title="No neuro cases")

    top = safe_counts(neuro_df["gene_symbol"], col_name="gene")

    fig = px.bar(
        top,
        x="gene",
        y="count",
        title="Top Neuro Genes"
    )

    return fig


# =========================
# MODULE 7: DISEASE SIMILARITY
# =========================

def plot_disease_similarity(sim, names):

    if len(names) == 0:
        return px.imshow([[0]], title="No similarity data")

    df = pd.DataFrame(sim, index=names, columns=names)

    fig = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        title="Disease Similarity (Jaccard)"
    )

    return fig


# =========================
# MODULE 11: HPO CO-OCCURRENCE
# =========================

def plot_hpo_cooccurrence(mat, terms):

    if len(terms) == 0:
        return px.imshow([[0]], title="No HPO data")

    df = pd.DataFrame(mat, index=terms, columns=terms)

    fig = px.imshow(
        df,
        aspect="auto",
        title="HPO Co-occurrence"
    )

    return fig


# =========================
# MODULE 12: ADAT3
# =========================

def plot_adat3(df):

    adat3 = df[df["gene_symbol"] == "ADAT3"]

    if adat3.empty:
        return px.bar(title="No ADAT3 cases")

    counts = adat3["hpo_list"].explode()

    if counts.empty:
        return px.bar(title="No HPO terms")

    top = safe_counts(counts, top_n=15, col_name="hpo")

    fig = px.bar(
        top,
        x="hpo",
        y="count",
        title="ADAT3 HPO Profile"
    )

    return fig
# =========================
# MODULE 2: POPULATION
# =========================
def plot_population(stats_df):

    if stats_df.empty:
        return px.bar(title="No population data")

    df = stats_df.reset_index().rename(columns={"index": "source"})

    fig = px.bar(
        df,
        x="source",
        y="n",
        color="solved_pct",
        title="Population Cohort Size & Solved Rate"
    )

    return fig


# =========================
# MODULE 4: AR ARCHITECTURE
# =========================
def plot_ar_architecture(df):

    vc = safe_counts(df["zygosity_label"], col_name="zygosity")

    fig = px.pie(
        vc,
        names="zygosity",
        values="count",
        title="Autosomal Recessive Architecture"
    )

    return fig


# =========================
# MODULE 8: GENE MODEL
# =========================
def plot_gene_model(results):

    df = pd.DataFrame({
        "metric": ["CV Accuracy", "Top-3", "Top-5"],
        "value": [results["cv"], results["top3"], results["top5"]]
    })

    fig = px.bar(
        df,
        x="metric",
        y="value",
        title="Gene Model Performance"
    )

    return fig


# =========================
# MODULE 9: PATHOGENICITY
# =========================
def plot_pathogenicity_curve(precision, recall):

    df = pd.DataFrame({
        "precision": precision,
        "recall": recall
    })

    fig = px.line(
        df,
        x="recall",
        y="precision",
        title="Precision-Recall Curve"
    )

    return fig

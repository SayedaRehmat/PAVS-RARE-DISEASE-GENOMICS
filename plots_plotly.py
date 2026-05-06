# plots_plotly.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from analysis_core import (
    get_population_stats,
    get_disease_similarity,
    get_hpo_cooccurrence,
    get_adat3
)

# =========================
# MODULE 1: EDA
# =========================
def plot_cohort_overview(df):
    fig1 = px.bar(
        df["solved_status"].value_counts().reset_index(),
        x="index", y="solved_status",
        title="Solved Status"
    )

    fig2 = px.pie(
        df, names="source",
        title="Source Distribution"
    )

    fig3 = px.histogram(
        df, x="hpo_count",
        nbins=30,
        title="HPO Terms per Case"
    )

    fig4 = px.bar(
        df["acmg_classification"].value_counts().reset_index(),
        x="index", y="acmg_classification",
        title="ACMG Classification"
    )

    return fig1, fig2, fig3, fig4


# =========================
# MODULE 2: POPULATION
# =========================
def plot_population(stats_df):
    df = stats_df.reset_index()

    fig = px.bar(
        df,
        x="index",
        y=["solved_pct", "hom_pct"],
        barmode="group",
        title="Population Comparison"
    )
    return fig


# =========================
# MODULE 3: FOUNDERS
# =========================
def plot_founders(founders):
    top = founders.head(20)

    fig = px.bar(
        top,
        x="n_cases",
        y="gene_symbol",
        orientation="h",
        title="Top Founder Mutations",
        hover_data=["hgvs_c"]
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


# =========================
# MODULE 4: AR ARCHITECTURE
# =========================
def plot_ar_architecture(df):
    zyg = df["zygosity_label"].value_counts(normalize=True) * 100

    fig = px.bar(
        x=zyg.index,
        y=zyg.values,
        labels={"x": "Zygosity", "y": "%"},
        title="Zygosity Distribution"
    )
    return fig


# =========================
# MODULE 5: TREATABLE
# =========================
def plot_treatable(treat_df):
    top = treat_df["gene_symbol"].value_counts().head(15).reset_index()

    fig = px.bar(
        top,
        x="index",
        y="gene_symbol",
        title="Treatable Genes",
        labels={"index": "Gene", "gene_symbol": "Cases"}
    )
    return fig


# =========================
# MODULE 6: NEURO
# =========================
def plot_neuro(neuro_df):
    top = neuro_df["gene_symbol"].value_counts().head(15).reset_index()

    fig = px.bar(
        top,
        x="gene_symbol",
        y="index",
        orientation="h",
        title="Top Neuro Genes"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


# =========================
# MODULE 7: DISEASE SIMILARITY
# =========================
def plot_disease_similarity(df):
    sim, names = get_disease_similarity(df)

    fig = go.Figure(
        data=go.Heatmap(
            z=sim,
            x=names,
            y=names,
            colorscale="Reds"
        )
    )

    fig.update_layout(
        title="Disease Similarity (HPO)",
        xaxis_tickangle=45
    )
    return fig


# =========================
# MODULE 8: GENE MODEL
# =========================
def plot_gene_model(results):
    fig = px.bar(
        x=["CV", "Top-3", "Top-5"],
        y=[
            results["cv"] * 100,
            results["top3"] * 100,
            results["top5"] * 100
        ],
        labels={"x": "Metric", "y": "%"},
        title="Gene Model Performance"
    )
    return fig


# =========================
# MODULE 9: PATHOGENICITY
# =========================
def plot_pathogenicity_curve(precision, recall):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name="PR Curve"
    ))

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    return fig


# =========================
# MODULE 11: HPO CO-OCCURRENCE
# =========================
def plot_hpo_cooccurrence(df):
    mat, labels = get_hpo_cooccurrence(df)

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale="Blues"
        )
    )

    fig.update_layout(
        title="HPO Co-occurrence",
        xaxis_tickangle=45
    )
    return fig


# =========================
# MODULE 12: ADAT3
# =========================
def plot_adat3(df):
    _, counts = get_adat3(df)

    top = counts.head(15).reset_index()

    fig = px.bar(
        top,
        x="count",
        y="index",
        orientation="h",
        title="ADAT3 HPO Profile"
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig

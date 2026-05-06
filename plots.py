# plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from analysis_core import (
    get_population_stats,
    get_founders,
    get_treatable,
    get_neuro,
    get_disease_similarity,
    get_hpo_cooccurrence,
    get_adat3
)

sns.set_style("whitegrid")


# =========================
# MODULE 1: EDA
# =========================
def plot_cohort_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Solved status
    df["solved_status"].value_counts().plot(
        kind="bar", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Solved Status")

    # Source
    df["source"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=axes[0, 1]
    )
    axes[0, 1].set_title("Source")

    # ACMG
    df["acmg_classification"].value_counts().plot(
        kind="barh", ax=axes[1, 0]
    )
    axes[1, 0].set_title("ACMG Classification")

    # HPO count
    df["hpo_count"].plot(kind="hist", bins=20, ax=axes[1, 1])
    axes[1, 1].set_title("HPO Terms per Case")

    plt.tight_layout()
    return fig


# =========================
# MODULE 2: POPULATION
# =========================
def plot_population(stats_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    stats_df[["solved_pct", "hom_pct"]].plot(
        kind="bar", ax=ax
    )
    ax.set_ylabel("%")
    ax.set_title("Population Comparison")
    return fig


# =========================
# MODULE 3: FOUNDERS
# =========================
def plot_founders(founders):
    top = founders.head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["gene_symbol"], top["n_cases"])
    ax.invert_yaxis()
    ax.set_title("Top Founder Mutations")
    ax.set_xlabel("Cases")
    return fig


# =========================
# MODULE 4: AR ARCHITECTURE
# =========================
def plot_ar_architecture(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    zyg = df["zygosity_label"].value_counts(normalize=True) * 100
    zyg.plot(kind="bar", ax=ax)

    ax.set_ylabel("%")
    ax.set_title("Zygosity Distribution")
    return fig


# =========================
# MODULE 5: TREATABLE
# =========================
def plot_treatable(treat_df):
    fig, ax = plt.subplots(figsize=(10, 5))

    treat_df["gene_symbol"].value_counts().head(15).plot(
        kind="bar", ax=ax
    )

    ax.set_title("Treatable Genes")
    return fig


# =========================
# MODULE 6: NEURO
# =========================
def plot_neuro(neuro_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    neuro_df["gene_symbol"].value_counts().head(15).plot(
        kind="barh", ax=ax
    )

    ax.invert_yaxis()
    ax.set_title("Top Neuro Genes")
    return fig


# =========================
# MODULE 7: DISEASE SIMILARITY
# =========================
def plot_disease_similarity(df):
    sim, names = get_disease_similarity(df)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        sim,
        xticklabels=names,
        yticklabels=names,
        cmap="Reds",
        ax=ax
    )

    plt.xticks(rotation=45, ha="right")
    ax.set_title("Disease Similarity (HPO)")
    return fig


# =========================
# MODULE 8: GENE MODEL
# =========================
def plot_gene_model(results):
    fig, ax = plt.subplots(figsize=(6, 4))

    vals = [
        results["cv"] * 100,
        results["top3"] * 100,
        results["top5"] * 100
    ]

    labels = ["CV", "Top-3", "Top-5"]

    ax.bar(labels, vals)
    ax.set_ylabel("%")
    ax.set_title("Gene Model Performance")

    return fig


# =========================
# MODULE 9: PATHOGENICITY
# =========================
def plot_pathogenicity_curve(precision, recall):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")

    return fig


# =========================
# MODULE 11: HPO CO-OCCURRENCE
# =========================
def plot_hpo_cooccurrence(df):
    mat, labels = get_hpo_cooccurrence(df)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        mat,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax
    )

    plt.xticks(rotation=45, ha="right")
    ax.set_title("HPO Co-occurrence")

    return fig


# =========================
# MODULE 12: ADAT3
# =========================
def plot_adat3(adat3_df):
    _, counts = get_adat3(adat3_df)

    fig, ax = plt.subplots(figsize=(8, 6))

    counts.head(15).plot(kind="barh", ax=ax)
    ax.invert_yaxis()

    ax.set_title("ADAT3 HPO Profile")
    return fig

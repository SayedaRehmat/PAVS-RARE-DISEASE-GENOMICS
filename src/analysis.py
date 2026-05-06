# analysis_core.py

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================
# CONSTANTS (unchanged)
# =========================

TREATABLE_GENES = {
    "SLC19A3": "Biotin+Thiamine",
    "GAA": "ERT",
    "ATP7B": "Chelation",
    "GBA": "ERT/SRT",
    "PAH": "BH4",
    "BTD": "Biotin",
    "CBS": "Pyridoxine",
}

KNOWN_FOUNDERS = {
    "ELAC2": "Saudi founder",
    "ADAT3": "Pan-Arab founder",
}

NEURO_HPO = {
    "HP:0001263": "Developmental delay",
    "HP:0001249": "Intellectual disability",
    "HP:0001250": "Seizure",
}

# =========================
# LOAD DATA
# =========================

def load_pavs_data(path):
    df = pd.read_csv(path, sep="\t")

    df["is_solved"] = df["solved_status"] == "SOLVED"
    df["hpo_list"] = df["hpo_terms"].apply(
        lambda s: [x.split("|")[0].strip() for x in s.split(";")]
        if pd.notna(s) else []
    )
    df["hpo_count"] = df["hpo_list"].apply(len)
    df["is_neuro"] = df["hpo_list"].apply(
        lambda lst: any(h in NEURO_HPO for h in lst)
    )

    return df, {}

# =========================
# MODULE 2: POPULATION
# =========================

def get_population_stats(df):
    stats = {}

    for src in df["source"].unique():
        sub = df[df["source"] == src]
        stats[src] = {
            "n": len(sub),
            "solved_pct": sub["is_solved"].mean() * 100,
            "hom_pct": (sub["zygosity_label"] == "homozygous").mean() * 100,
            "median_hpo": sub["hpo_count"].median(),
        }

    return pd.DataFrame(stats).T

# =========================
# MODULE 3: FOUNDERS
# =========================

def get_founders(df):
    sub = df[df["gene_symbol"].notna() & df["hgvs_c"].notna()]

    gv = (
        sub.groupby(["gene_symbol", "hgvs_c"])
        .size()
        .reset_index(name="n_cases")
        .sort_values("n_cases", ascending=False)
    )

    founders = gv[gv["n_cases"] >= 3].copy()
    founders["is_known"] = founders["gene_symbol"].isin(KNOWN_FOUNDERS)

    return founders

# =========================
# MODULE 5: TREATABLE
# =========================

def get_treatable(df):
    treat = df[df["gene_symbol"].isin(TREATABLE_GENES)].copy()
    unsolved = treat[treat["solved_status"] == "IN_PROGRESS"]
    return treat, unsolved

# =========================
# MODULE 6: NEURO
# =========================

def get_neuro(df):
    return df[df["is_neuro"]].copy()

# =========================
# MODULE 7: DISEASE SIMILARITY
# =========================

def get_disease_similarity(df):
    diseases = df["disease_label"].dropna().unique()[:20]

    dis_sets = {}
    for d in diseases:
        sub = df[df["disease_label"] == d]
        terms = set(h for lst in sub["hpo_list"] for h in lst)
        if terms:
            dis_sets[d] = terms

    names = list(dis_sets.keys())
    n = len(names)

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = dis_sets[names[i]], dis_sets[names[j]]
            sim[i, j] = len(a & b) / len(a | b) if (a | b) else 0

    return sim, names

# =========================
# MODULE 8: GENE MODEL
# =========================

def run_gene_model(df):
    sub = df[
        df["is_solved"]
        & df["gene_symbol"].notna()
        & (df["hpo_list"].apply(len) > 0)
    ]

    gene_freq = sub["gene_symbol"].value_counts()
    valid = gene_freq[gene_freq >= 5].index

    ml_df = sub[sub["gene_symbol"].isin(valid)]

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(ml_df["hpo_list"])

    le = LabelEncoder()
    y = le.fit_transform(ml_df["gene_symbol"])

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(rf, X, y, cv=cv).mean()

    rf.fit(X, y)
    proba = rf.predict_proba(X)

    top3 = np.mean([
        y[i] in np.argsort(proba[i])[::-1][:3]
        for i in range(len(y))
    ])

    top5 = np.mean([
        y[i] in np.argsort(proba[i])[::-1][:5]
        for i in range(len(y))
    ])

    return {
        "cv": cv_acc,
        "top3": top3,
        "top5": top5
    }

# =========================
# MODULE 9: PATHOGENICITY
# =========================

def run_pathogenicity(df):
    sub = df[df["acmg_classification"].notna()].copy()

    sub["y"] = sub["acmg_classification"].isin(
        ["PATHOGENIC", "LIKELY_PATHOGENIC"]
    ).astype(int)

    sub["hpo_count"] = sub["hpo_count"].fillna(0)

    X = sub[["hpo_count"]].values
    y = sub["y"].values

    clf = RandomForestClassifier(n_estimators=100)

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()

    clf.fit(X, y)
    prob = clf.predict_proba(X)[:, 1]

    ap = average_precision_score(y, prob)

    return auc, ap

# =========================
# MODULE 10: VUS
# =========================

def get_vus(df):
    vus = df[df["acmg_classification"] == "UNCERTAIN_SIGNIFICANCE"].copy()

    gene_counts = df[df["is_solved"]]["gene_symbol"].value_counts()

    vus["gene_solved"] = vus["gene_symbol"].map(gene_counts).fillna(0)

    vus["priority"] = vus["gene_solved"] * 0.6 + vus["hpo_count"]

    return vus.sort_values("priority", ascending=False)

# =========================
# MODULE 11: HPO CO-OCCURRENCE
# =========================

def get_hpo_cooccurrence(df):
    all_terms = df["hpo_list"].explode().dropna()
    top = all_terms.value_counts().head(20).index.tolist()

    mat = np.zeros((20, 20))

    for terms in df["hpo_list"]:
        for i, h1 in enumerate(top):
            for j, h2 in enumerate(top):
                if h1 in terms and h2 in terms:
                    mat[i, j] += 1

    return mat, top

# =========================
# MODULE 12: ADAT3
# =========================

def get_adat3(df):
    adat3 = df[df["gene_symbol"] == "ADAT3"].copy()

    counts = adat3["hpo_list"].explode().value_counts()

    return adat3, counts

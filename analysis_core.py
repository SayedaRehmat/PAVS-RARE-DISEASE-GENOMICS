import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_curve

# =========================
# CONSTANTS
# =========================
TREATABLE_GENES = ["SLC19A3", "GAA", "ATP7B", "GBA", "PAH", "BTD", "CBS"]

NEURO_HPO = {"HP:0001263", "HP:0001249", "HP:0001250"}

# =========================
# LOAD DATA
# =========================
def load_data(path):
    df = pd.read_csv(path, sep="\t")

    df["is_solved"] = df["solved_status"] == "SOLVED"

    df["hpo_list"] = df["hpo_terms"].fillna("").apply(
        lambda s: [x.split("|")[0].strip() for x in s.split(";") if x]
    )

    df["hpo_count"] = df["hpo_list"].apply(len)

    df["is_neuro"] = df["hpo_list"].apply(
        lambda x: any(h in NEURO_HPO for h in x)
    )

    return df, {}

# =========================
# MODULE 2: POPULATION
# =========================
def get_population_stats(df):
    stats = df.groupby("source").agg(
        n=("source", "count"),
        solved_pct=("is_solved", "mean"),
        median_hpo=("hpo_count", "median")
    ).reset_index()

    stats["solved_pct"] = stats["solved_pct"] * 100
    return stats

# =========================
# MODULE 3: FOUNDERS
# =========================
def get_founders(df):
    sub = df[df["gene_symbol"].notna() & df["hgvs_c"].notna()]

    g = (
        sub.groupby(["gene_symbol", "hgvs_c"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return g

# =========================
# MODULE 4: AR ARCHITECTURE
# =========================
def get_ar_architecture(df):
    ar = df[df["zygosity_label"] == "homozygous"]
    return ar["gene_symbol"].value_counts().reset_index().rename(
        columns={"index": "gene", "gene_symbol": "count"}
    )

# =========================
# MODULE 5: TREATABLE
# =========================
def get_treatable(df):
    return df[df["gene_symbol"].isin(TREATABLE_GENES)].copy()

# =========================
# MODULE 6: NEURO
# =========================
def get_neuro(df):
    return df[df["is_neuro"]].copy()

# =========================
# MODULE 7: DISEASE SIMILARITY
# =========================
def get_disease_similarity(df):
    diseases = df["disease_label"].dropna().unique()[:15]

    dis_sets = {
        d: set(sum(df[df["disease_label"] == d]["hpo_list"], []))
        for d in diseases
    }

    names = list(dis_sets.keys())
    n = len(names)

    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a, b = dis_sets[names[i]], dis_sets[names[j]]
            mat[i, j] = len(a & b) / len(a | b) if a | b else 0

    return mat, names

# =========================
# MODULE 8: GENE MODEL
# =========================
def run_gene_model(df):
    sub = df[df["gene_symbol"].notna() & (df["hpo_count"] > 0)]

    if len(sub) < 30:
        return {"error": "Dataset too small for modeling"}

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(sub["hpo_list"])

    le = LabelEncoder()
    y = le.fit_transform(sub["gene_symbol"])

    if len(set(y)) < 3:
        return {"error": "Not enough gene classes"}

    model = RandomForestClassifier(n_estimators=100)

    try:
        cv = StratifiedKFold(n_splits=3)
        acc = cross_val_score(model, X, y, cv=cv).mean()
    except:
        acc = 0

    return {"cv_accuracy": float(acc)}

# =========================
# MODULE 9: PATHOGENICITY
# =========================
def run_pathogenicity_model(df):
    sub = df[df["acmg_classification"].notna()].copy()

    if len(sub) < 20:
        return [], []

    sub["y"] = sub["acmg_classification"].isin(
        ["PATHOGENIC", "LIKELY_PATHOGENIC"]
    ).astype(int)

    X = sub[["hpo_count"]].values
    y = sub["y"].values

    model = RandomForestClassifier()
    model.fit(X, y)

    prob = model.predict_proba(X)[:, 1]

    precision, recall, _ = precision_recall_curve(y, prob)

    return precision.tolist(), recall.tolist()

# =========================
# MODULE 10: VUS
# =========================
def get_vus(df):
    vus = df[df["acmg_classification"] == "UNCERTAIN_SIGNIFICANCE"].copy()
    vus["priority"] = vus["hpo_count"]
    return vus.sort_values("priority", ascending=False)

# =========================
# MODULE 11: HPO CO-OCCURRENCE
# =========================
def get_hpo_cooccurrence(df):
    terms = df["hpo_list"].explode().value_counts().head(20).index.tolist()

    mat = np.zeros((20, 20))

    for lst in df["hpo_list"]:
        for i, a in enumerate(terms):
            for j, b in enumerate(terms):
                if a in lst and b in lst:
                    mat[i, j] += 1

    return mat, terms

# =========================
# MODULE 12: ADAT3
# =========================
def get_adat3(df):
    sub = df[df["gene_symbol"] == "ADAT3"]
    counts = sub["hpo_list"].explode().value_counts()
    return sub, counts

# =========================
# MODULE 13: FINAL METRICS
# =========================
def get_summary_metrics(df):
    return {
        "cases": len(df),
        "genes": df["gene_symbol"].nunique(),
        "solved": int(df["is_solved"].sum())
    }

# =========================
# ADVANCED FEATURES
# =========================

# 🔍 Patient Search
def search_patients(df, query):
    mask = df.astype(str).apply(
        lambda col: col.str.contains(query, case=False, na=False)
    )
    return df[mask.any(axis=1)]

# 🧬 Patient Similarity
def patient_similarity(df, idx):
    if idx >= len(df):
        return pd.DataFrame()

    target = set(df.iloc[idx]["hpo_list"])

    scores = []
    for _, row in df.iterrows():
        s = set(row["hpo_list"])
        sim = len(target & s) / len(target | s) if target | s else 0
        scores.append(sim)

    df_copy = df.copy()
    df_copy["similarity"] = scores

    return df_copy.sort_values("similarity", ascending=False).head(10)

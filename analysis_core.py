import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, precision_recall_curve

# =========================
# LOAD
# =========================
def load_data(path):
    df = pd.read_csv(path, sep="\t")

    df["is_solved"] = df["solved_status"] == "SOLVED"
    df["hpo_list"] = df["hpo_terms"].apply(
        lambda s: [x.split("|")[0].strip() for x in s.split(";")] if pd.notna(s) else []
    )
    df["hpo_count"] = df["hpo_list"].apply(len)

    return df, {}

# =========================
# CORE MODULES
# =========================

def get_population_stats(df):
    return df.groupby("source").agg(
        n=("source", "size"),
        solved_pct=("is_solved", "mean"),
        median_hpo=("hpo_count", "median")
    )

def get_founders(df):
    sub = df.dropna(subset=["gene_symbol", "hgvs_c"])
    gv = sub.groupby(["gene_symbol", "hgvs_c"]).size().reset_index(name="n_cases")
    return gv[gv["n_cases"] >= 3].sort_values("n_cases", ascending=False)

def get_treatable_df(df):
    genes = ["SLC19A3","GAA","ATP7B","GBA","PAH","BTD","CBS"]
    return df[df["gene_symbol"].isin(genes)]

def get_neuro(df):
    neuro_terms = {"HP:0001263","HP:0001249","HP:0001250"}
    return df[df["hpo_list"].apply(lambda x: any(h in neuro_terms for h in x))]

def get_disease_similarity(df):
    diseases = df["disease_label"].dropna().unique()[:20]
    sets = {d: set(h for lst in df[df["disease_label"]==d]["hpo_list"] for h in lst) for d in diseases}

    names = list(sets.keys())
    n = len(names)
    sim = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            a, b = sets[names[i]], sets[names[j]]
            sim[i,j] = len(a & b) / (len(a | b) + 1e-6)

    return sim, names

def run_gene_model(df):
    sub = df[df["is_solved"] & df["gene_symbol"].notna()]
    if len(sub) < 10:
        return None

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(sub["hpo_list"])

    le = LabelEncoder()
    y = le.fit_transform(sub["gene_symbol"])

    model = RandomForestClassifier(n_estimators=100)
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    return {"cv": cross_val_score(model, X, y, cv=cv).mean()}

def run_pathogenicity_model(df):
    sub = df[df["acmg_classification"].notna()]
    sub["y"] = sub["acmg_classification"].isin(["PATHOGENIC","LIKELY_PATHOGENIC"]).astype(int)

    X = sub[["hpo_count"]].values
    y = sub["y"].values

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X,y)

    prob = model.predict_proba(X)[:,1]
    precision, recall, _ = precision_recall_curve(y, prob)

    return precision, recall

def get_vus(df):
    vus = df[df["acmg_classification"]=="UNCERTAIN_SIGNIFICANCE"].copy()
    vus["priority"] = vus["hpo_count"]
    return vus.sort_values("priority", ascending=False)

def get_hpo_cooccurrence(df):
    terms = df["hpo_list"].explode().value_counts().head(20).index.tolist()
    mat = np.zeros((20,20))

    for lst in df["hpo_list"]:
        for i,h1 in enumerate(terms):
            for j,h2 in enumerate(terms):
                if h1 in lst and h2 in lst:
                    mat[i,j]+=1

    return mat, terms

def get_adat3(df):
    adat = df[df["gene_symbol"]=="ADAT3"]
    return adat["hpo_list"].explode().value_counts().head(15)

# =========================
# ADVANCED (SAFE)
# =========================

def search_patient(df, query):
    return df[df.astype(str).apply(lambda x: x.str.contains(query, case=False)).any(axis=1)]

def filter_variant(df, gene=None, hgvs=None):
    if gene:
        df = df[df["gene_symbol"]==gene]
    if hgvs:
        df = df[df["hgvs_c"].str.contains(hgvs, na=False)]
    return df

def run_gene_model_with_shap(df):
    try:
        import shap
    except:
        return None
    return "SHAP ready"

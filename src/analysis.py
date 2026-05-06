import pandas as pd
import numpy as np

# Treatable genes dictionary
TREATABLE_GENES = {
    "SLC19A3": "Biotin + Thiamine",
    "GAA": "Enzyme replacement",
    "ATP7B": "Copper chelation",
    "GBA": "ERT",
    "PAH": "BH4 therapy",
    "BTD": "Biotin",
    "CBS": "Vitamin B6 / Betaine",
    "GAMT": "Creatine",
    "ACADVL": "Avoid fasting",
    "ACADM": "Avoid fasting",
}

# ─────────────────────────────────────
def load_data():
    df = pd.read_csv("data/PAVS_cases.tsv", sep="\t")

    df["is_solved"] = df["solved_status"] == "SOLVED"
    df["is_hom"] = df["zygosity_label"] == "homozygous"

    # HPO processing
    df["hpo_list"] = df["hpo_terms"].apply(
        lambda x: [i.split("|")[0] for i in str(x).split(";")]
        if pd.notna(x) else []
    )

    df["hpo_count"] = df["hpo_list"].apply(len)

    return df

# ─────────────────────────────────────
def compute_overview(df):
    return {
        "total_cases": len(df),
        "solved_cases": int(df["is_solved"].sum()),
        "homozygous_pct": float(df["is_hom"].mean())
    }

# ─────────────────────────────────────
def compute_founders(df):
    founders = (
        df.groupby(["gene_symbol", "hgvs_c"])
        .size()
        .reset_index(name="cases")
        .sort_values("cases", ascending=False)
    )
    return founders

# ─────────────────────────────────────
def compute_treatable(df):
    treat = df[df["gene_symbol"].isin(TREATABLE_GENES)].copy()
    treat["treatment"] = treat["gene_symbol"].map(TREATABLE_GENES)

    unsolved = treat[treat["solved_status"] == "IN_PROGRESS"]

    return treat, unsolved

# ─────────────────────────────────────
def compute_vus(df):
    vus = df[df["acmg_classification"] == "UNCERTAIN_SIGNIFICANCE"].copy()

    vus["score"] = (
        vus["gnomad_pli"].fillna(0) * 2 +
        vus["hpo_count"] * 0.5 +
        (vus["vep_impact"] == "HIGH").astype(int) * 3
    )

    vus = vus.sort_values("score", ascending=False)

    return vus

# ─────────────────────────────────────
def build_system():
    df = load_data()

    overview = compute_overview(df)
    founders = compute_founders(df)
    treat, treat_unsolved = compute_treatable(df)
    vus = compute_vus(df)

    return {
        "df": df,
        "overview": overview,
        "founders": founders,
        "treatable": treat,
        "treatable_unsolved": treat_unsolved,
        "vus": vus
    }

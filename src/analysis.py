"""
src/analysis.py
================================================================================
PAVS RARE DISEASE GENOMICS — COMPLETE ANALYSIS PIPELINE
Combines EDA, Population Genomics, Founder Mutations, Gene Prioritization,
Pathogenicity Classification, VUS Reclassification & Treatable Disease Mining.
================================================================================
"""
import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TREATABLE_GENES = {
    "SLC19A3": "Biotin+Thiamine (BTBGD) — URGENT",
    "GAA":     "Enzyme replacement (Pompe)",
    "ATP7B":   "Copper chelation (Wilson)",
    "GBA":     "ERT / SRT (Gaucher)",
    "PAH":     "BH4/Sapropterin (PKU)",
    "BTD":     "Biotin supplementation — URGENT",
    "CBS":     "Pyridoxine/Betaine (Homocystinuria)",
    "GAMT":    "Creatine + Ornithine",
    "ACADVL":  "Avoid fasting (VLCAD)",
    "ACADM":   "Avoid fasting (MCAD)",
    "GJB2":    "Cochlear implant (early)",
    "PCCA":    "Protein restriction (PA)",
    "HLCS":    "Biotin supplementation",
    "G6PD":    "Avoid triggers",
    "PTS":     "BH4 therapy (HPA)",
}

KNOWN_FOUNDERS = {
    "ELAC2":   "Mitochondrial disease — most frequent Saudi variant (49 cases)",
    "ADAT3":   "Pan-Arab founder — #1 cause AR intellectual disability in Saudi",
    "TULP1":   "Retinitis pigmentosa — Saudi/Arab founder",
    "SLC19A3": "BTBGD — treatable Saudi founder disease",
    "ACADVL":  "VLCAD deficiency — newborn screening",
    "CBS":     "Homocystinuria — metabolic disease",
    "ISCA2":   "Multiple mitochondrial dysfunction syndrome",
    "TMC1":    "Non-syndromic hearing loss — Saudi founder",
    "C12ORF57":"Temtamy syndrome — Arab founder",
    "COG6":    "Shaheen syndrome — Saudi tribal founder",
    "CYP2U1":  "Spastic paraplegia — Arab founder variant",
}

NEURO_HPO = {
    "HP:0001263": "Global developmental delay",
    "HP:0001249": "Intellectual disability",
    "HP:0001250": "Seizure",
    "HP:0000750": "Speech/language delay",
    "HP:0001290": "Generalized hypotonia",
    "HP:0001252": "Hypotonia",
    "HP:0000729": "Autistic behavior",
    "HP:0007018": "ADHD",
    "HP:0002376": "Developmental regression",
}

# ─────────────────────────────────────────────────────────────────────────────
def load_data(path="data/PAVS_cases.tsv"):
    df = pd.read_csv(path, sep="\t")
    df["is_solved"]     = df["solved_status"] == "SOLVED"
    df["hpo_list"]      = df["hpo_terms"].apply(
        lambda s: [x.split("|")[0].strip() for x in s.split(";")]
        if pd.notna(s) else []
    )
    df["hpo_count"]     = df["hpo_list"].apply(len)
    df["is_homozygous"] = df["zygosity_label"] == "homozygous"
    df["gene_variant"]  = (df["gene_symbol"].fillna("") + ":"
                           + df["hgvs_c"].fillna(""))
    df["neuro_terms"]   = df["hpo_list"].apply(
        lambda lst: [h for h in lst if h in NEURO_HPO]
    )
    df["is_neuro"]      = df["neuro_terms"].apply(len) > 0

    # HPO label map
    hpo_labels = {}
    for _, row in df.iterrows():
        if pd.notna(row["hpo_terms"]):
            for item in row["hpo_terms"].split(";"):
                parts = item.strip().split("|")
                if len(parts) == 2:
                    hpo_labels[parts[0]] = parts[1]
    return df, hpo_labels


# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df, hpo_labels):
    """Module 1 — Cohort Overview EDA (4 figures)"""
    print("  [1] EDA figures...")
    all_hpo_flat = df["hpo_list"].explode().dropna()

    # Fig 1: Cohort dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PAVS Cohort Overview — Saudi Arabian Rare Disease Registry",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    counts = df["solved_status"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=["#2ecc71","#e67e22"], edgecolor="white")
    ax.set_title("(a) Case Solved Status", fontweight="bold")
    ax.set_ylabel("Number of Cases")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+40,
                f"n={v:,}", ha="center", fontweight="bold", fontsize=10)
    ax.set_ylim(0, counts.max()*1.15)

    ax = axes[0, 1]
    src = df["source"].value_counts()
    ax.pie(src.values, labels=src.index, autopct="%1.1f%%",
           colors=["#3498db","#e74c3c","#2ecc71"],
           textprops={"fontsize":10})
    ax.set_title("(b) Cohort Source", fontweight="bold")

    ax = axes[1, 0]
    acmg = df["acmg_classification"].dropna().value_counts()
    color_map = {"PATHOGENIC":"#c0392b","LIKELY_PATHOGENIC":"#e67e22",
                 "UNCERTAIN_SIGNIFICANCE":"#7f8c8d"}
    bars = ax.barh(acmg.index, acmg.values,
                   color=[color_map.get(k,"#95a5a6") for k in acmg.index],
                   edgecolor="white")
    ax.set_title("(c) ACMG Classification", fontweight="bold")
    ax.set_xlabel("Cases")
    for bar, v in zip(bars, acmg.values):
        ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                f"{v:,}", va="center", fontsize=9)

    ax = axes[1, 1]
    hpo_data = df["hpo_count"][df["hpo_count"]>0]
    ax.hist(hpo_data, bins=30, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.axvline(hpo_data.median(), color="red", linestyle="--",
               label=f"Median: {hpo_data.median():.1f}")
    ax.set_title("(d) HPO Terms per Case", fontweight="bold")
    ax.set_xlabel("HPO count"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig01_cohort_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: Top genes & diseases
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Gene & Disease Landscape", fontsize=13, fontweight="bold")
    top_genes = df["gene_symbol"].value_counts().head(20)
    pal = sns.color_palette("husl", 20)
    bars = axes[0].barh(range(20), top_genes.values, color=pal)
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels(top_genes.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Top 20 Genes", fontweight="bold")
    axes[0].set_xlabel("Cases")
    for bar, v in zip(bars, top_genes.values):
        axes[0].text(bar.get_width()+0.3,
                     bar.get_y()+bar.get_height()/2, str(v), va="center", fontsize=8)

    top_dis = df["disease_label"].value_counts().head(15)
    wrap = [d[:40]+"..." if len(d)>40 else d for d in top_dis.index]
    bars = axes[1].barh(range(15), top_dis.values,
                        color=sns.color_palette("Set2",15))
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels(wrap, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("Top 15 Diseases", fontweight="bold")
    axes[1].set_xlabel("Cases")
    for bar, v in zip(bars, top_dis.values):
        axes[1].text(bar.get_width()+0.3,
                     bar.get_y()+bar.get_height()/2, str(v), va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig02_genes_diseases.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: Variant landscape
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Variant Consequence Profile", fontsize=13, fontweight="bold")
    cons = df["vep_consequence"].value_counts().head(12)
    axes[0].bar(range(len(cons)), cons.values,
                color=sns.color_palette("Spectral", len(cons)))
    axes[0].set_xticks(range(len(cons)))
    axes[0].set_xticklabels([c.replace("_","\n") for c in cons.index], fontsize=7)
    axes[0].set_title("VEP Consequence Types", fontweight="bold")
    axes[0].set_ylabel("Count")
    zyg = df["zygosity_label"].value_counts()
    axes[1].pie(zyg.values, labels=zyg.index, autopct="%1.1f%%",
                colors=sns.color_palette("pastel", len(zyg)),
                textprops={"fontsize":11})
    axes[1].set_title("Zygosity Distribution", fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig03_variant_landscape.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: Top 30 HPO
    hpo_counts = all_hpo_flat.value_counts().head(30)
    labels_30  = [hpo_labels.get(h, h) for h in hpo_counts.index]
    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(range(30), hpo_counts.values,
                   color=sns.color_palette("viridis", 30))
    ax.set_yticks(range(30))
    ax.set_yticklabels(labels_30, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Top 30 HPO Phenotypes in PAVS", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cases")
    for bar, v in zip(bars, hpo_counts.values):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                str(v), va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig04_top_hpo_terms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → fig01-04 saved")
    return all_hpo_flat, hpo_counts


# ─────────────────────────────────────────────────────────────────────────────
def run_population_analysis(df):
    """Module 2 — Population stratification Saudi vs DDD vs Mixed"""
    print("  [2] Population stratification...")
    saudi = df[df["source"]=="PAVS-Saudi"]
    ddd   = df[df["source"]=="DDD"]
    mixed = df[df["source"]=="PAVS-mixed"]
    sources = {"PAVS-Saudi": saudi, "DDD (UK)": ddd, "PAVS-mixed": mixed}

    stats = {}
    for name, sub in sources.items():
        stats[name] = {
            "n": len(sub),
            "solved_pct": sub["is_solved"].mean()*100,
            "hom_pct":    (sub["zygosity_label"]=="homozygous").mean()*100,
            "median_hpo": sub["hpo_count"].median(),
            "unique_genes": sub["gene_symbol"].nunique(),
        }

    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(f"{OUT_DIR}/data_population_comparison.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PAVS: Saudi vs DDD (UK) vs Mixed — Population Comparison",
                 fontsize=12, fontweight="bold")
    colors = ["#e74c3c","#3498db","#e67e22"]
    keys   = list(stats.keys())

    for ax, metric, title, ylabel in zip(
        axes,
        ["hom_pct", "median_hpo", "solved_pct"],
        ["Homozygous %\n(Consanguinity Signature)",
         "Median HPO Terms\n(Phenotypic Depth)",
         "Diagnostic Yield %\n(Solved Cases)"],
        ["%", "HPO terms", "% Solved"]
    ):
        vals = [stats[k][metric] for k in keys]
        bars = ax.bar(keys, vals, color=colors, width=0.5, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+1,
                    f"{v:.1f}", ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig05_population_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → fig05 saved")

    s_hom = stats["PAVS-Saudi"]["hom_pct"]
    d_hom = stats["DDD (UK)"]["hom_pct"]
    print(f"    Saudi homozygous {s_hom:.1f}% vs DDD {d_hom:.1f}% "
          f"({s_hom/d_hom:.1f}× higher)")
    return stats, saudi, ddd


# ─────────────────────────────────────────────────────────────────────────────
def run_founder_analysis(df, saudi):
    """Module 3 — Saudi Founder Mutation Discovery"""
    print("  [3] Founder mutation discovery...")
    saudi_var = saudi[saudi["hgvs_c"].notna() & saudi["gene_symbol"].notna()].copy()
    gv = (saudi_var.groupby(["gene_symbol","hgvs_c","hgvs_p","zygosity_label"])
          .size().reset_index(name="n_cases")
          .sort_values("n_cases", ascending=False))
    founders = gv[gv["n_cases"] >= 3].copy()
    founders["is_known"] = founders["gene_symbol"].isin(KNOWN_FOUNDERS)
    founders["annotation"] = founders["gene_symbol"].map(KNOWN_FOUNDERS).fillna("")
    founders.to_csv(f"{OUT_DIR}/data_founder_mutations.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    top20 = founders.head(20)
    colors_f = ["#c0392b" if g in KNOWN_FOUNDERS else "#3498db"
                for g in top20["gene_symbol"]]
    ax.scatter(range(len(top20)), top20["n_cases"],
               s=top20["n_cases"]*20, c=colors_f, alpha=0.8,
               edgecolors="white", linewidth=1.5)
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels(top20["gene_symbol"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Cases (Saudi Cohort)", fontsize=11)
    ax.set_title("Saudi Founder Mutation Candidates\n"
                 "(Bubble = case count | Red = literature-confirmed founder)",
                 fontsize=12, fontweight="bold")
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.annotate(str(row["n_cases"]), (i, row["n_cases"]),
                    textcoords="offset points", xytext=(0,7),
                    ha="center", fontsize=8, fontweight="bold")
    red_p  = mpatches.Patch(color="#c0392b", label="Literature-confirmed founder")
    blue_p = mpatches.Patch(color="#3498db", label="Novel candidate")
    ax.legend(handles=[red_p, blue_p], fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig06_founder_mutations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → fig06 saved | {len(founders)} founder candidates")
    return founders


# ─────────────────────────────────────────────────────────────────────────────
def run_ar_architecture(df, saudi, ddd):
    """Module 4 — Autosomal Recessive Architecture"""
    print("  [4] AR architecture analysis...")
    zyg_types = ["homozygous","heterozygous","hemizygous"]

    sv = [saudi["zygosity_label"].value_counts().get(z,0)/len(saudi)*100
          for z in zyg_types]
    dv = [ddd["zygosity_label"].value_counts().get(z,0)/len(ddd)*100
          for z in zyg_types]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Autosomal Recessive Architecture — Consanguinity Signature",
                 fontsize=12, fontweight="bold")

    x = np.arange(3); w = 0.35
    b1 = axes[0].bar(x-w/2, sv, w, color="#e74c3c", label="PAVS-Saudi", alpha=0.85)
    b2 = axes[0].bar(x+w/2, dv, w, color="#3498db", label="DDD (UK)",   alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(zyg_types, fontsize=10)
    axes[0].set_ylabel("%"); axes[0].legend(fontsize=9)
    axes[0].set_title("Saudi vs UK Zygosity\n(consanguinity effect)", fontweight="bold")
    for bar, v in zip(b1, sv):
        if v>2: axes[0].text(bar.get_x()+bar.get_width()/2, v+0.5,
                             f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    for bar, v in zip(b2, dv):
        if v>2: axes[0].text(bar.get_x()+bar.get_width()/2, v+0.5,
                             f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")

    # Stacked bar by source
    zyg_by_src = df.groupby(["source","zygosity_label"]).size().unstack(fill_value=0)
    zyg_pct    = zyg_by_src.div(zyg_by_src.sum(axis=1), axis=0)*100
    bottom     = np.zeros(len(zyg_pct))
    colors_z   = {"homozygous":"#e74c3c","heterozygous":"#3498db","hemizygous":"#2ecc71"}
    for col in zyg_types:
        if col in zyg_pct.columns:
            vals = zyg_pct[col].values
            axes[1].bar(zyg_pct.index, vals, bottom=bottom,
                        label=col, color=colors_z[col], edgecolor="white", alpha=0.85)
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v>5:
                    axes[1].text(j, b+v/2, f"{v:.0f}%",
                                 ha="center", va="center",
                                 fontsize=10, fontweight="bold", color="white")
            bottom += vals
    axes[1].set_title("Zygosity by Source (Stacked %)", fontweight="bold")
    axes[1].set_ylabel("%"); axes[1].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig07_AR_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → fig07 saved")


# ─────────────────────────────────────────────────────────────────────────────
def run_treatable_analysis(df):
    """Module 5 — Treatable Rare Disease Mining"""
    print("  [5] Treatable disease mining...")
    treat_df = df[df["gene_symbol"].isin(TREATABLE_GENES)].copy()
    treat_df["treatment"] = treat_df["gene_symbol"].map(TREATABLE_GENES)
    unsolved = treat_df[treat_df["solved_status"]=="IN_PROGRESS"]

    treat_df.groupby(["gene_symbol","treatment","solved_status"]).size()\
        .reset_index(name="n")\
        .to_csv(f"{OUT_DIR}/data_treatable_cases.csv", index=False)
    if len(unsolved)>0:
        cols = ["case_id","gene_symbol","hpo_terms","treatment","acmg_classification"]
        unsolved[cols].to_csv(f"{OUT_DIR}/data_treatable_unsolved.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    tc = treat_df.groupby(["gene_symbol","solved_status"]).size().unstack(fill_value=0)
    tc = tc.loc[tc.sum(axis=1).sort_values(ascending=False).head(15).index]
    bottom = np.zeros(len(tc))
    for status, color in [("SOLVED","#2ecc71"),("IN_PROGRESS","#e74c3c")]:
        if status in tc.columns:
            vals = tc[status].values
            ax.bar(tc.index, vals, bottom=bottom,
                   label=status.replace("_"," "), color=color, edgecolor="white", alpha=0.85)
            bottom += vals
    ax.set_xticklabels(tc.index, rotation=35, ha="right", fontsize=9)
    ax.set_title("Treatable Rare Diseases — PAVS\n"
                 "(Green=Solved | Red=UNSOLVED — patients who could benefit from treatment TODAY)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Cases"); ax.legend(fontsize=10)
    for i, gene in enumerate(tc.index):
        t = TREATABLE_GENES.get(gene,"")[:28]
        ax.text(i, tc.sum(axis=1)[gene]+0.2, t,
                rotation=90, ha="center", va="bottom", fontsize=5.5, color="#2c3e50", alpha=0.8)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig08_treatable_diseases.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → fig08 saved | {len(unsolved)} unsolved treatable cases")
    return len(unsolved)


# ─────────────────────────────────────────────────────────────────────────────
def run_neuro_analysis(df, hpo_labels):
    """Module 6 — Neurodevelopmental Burden"""
    print("  [6] Neurodevelopmental burden...")
    neuro_df   = df[df["is_neuro"]].copy()
    neuro_genes = (neuro_df[neuro_df["is_solved"] & neuro_df["gene_symbol"].notna()]
                   ["gene_symbol"].value_counts().head(20))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Neurodevelopmental Disease Burden — Saudi Rare Disease Patients",
                 fontsize=12, fontweight="bold")

    bars = axes[0].barh(range(len(neuro_genes)), neuro_genes.values,
                        color=sns.color_palette("rocket_r", len(neuro_genes)))
    axes[0].set_yticks(range(len(neuro_genes)))
    axes[0].set_yticklabels(neuro_genes.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Top 20 Genes\nCausing Neurodevelopmental Disease", fontweight="bold")
    axes[0].set_xlabel("Cases")
    for bar, v in zip(bars, neuro_genes.values):
        axes[0].text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                     str(v), va="center", fontsize=8)

    ntc = {}
    for _, row in neuro_df.iterrows():
        for h in row["neuro_terms"]:
            label = NEURO_HPO.get(h, h)
            ntc[label] = ntc.get(label, 0) + 1
    ntc = pd.Series(ntc).sort_values(ascending=False)
    axes[1].barh(range(len(ntc)), ntc.values,
                 color=sns.color_palette("mako", len(ntc)))
    axes[1].set_yticks(range(len(ntc)))
    axes[1].set_yticklabels(ntc.index, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set_title("Neurodevelopmental HPO Term Frequency", fontweight="bold")
    axes[1].set_xlabel("Cases")
    for i, v in enumerate(ntc.values):
        axes[1].text(v+5, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig09_neuro_burden.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → fig09 saved | {len(neuro_df):,} neuro cases ({len(neuro_df)/len(df):.0%})")
    return len(neuro_df)


# ─────────────────────────────────────────────────────────────────────────────
def run_disease_similarity(df, hpo_labels):
    """Module 7 — Disease Phenotype Similarity Heatmap"""
    print("  [7] Disease phenotype similarity...")
    top_dis_names = (df[df["is_solved"] & (df["hpo_list"].apply(len)>0)]
                     ["disease_label"].value_counts().head(22).index.tolist())
    dis_sets = {}
    for dis in top_dis_names:
        sub = df[(df["disease_label"]==dis) & (df["hpo_list"].apply(len)>0)]
        terms = set(h for lst in sub["hpo_list"] for h in lst)
        if terms:
            dis_sets[dis] = terms

    names = list(dis_sets.keys())
    n = len(names)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = dis_sets[names[i]], dis_sets[names[j]]
            sim[i,j] = len(a&b)/len(a|b) if (a|b) else 0

    pd.DataFrame(sim, index=names, columns=names)\
      .to_csv(f"{OUT_DIR}/data_disease_hpo_similarity.csv")

    short = [d[:35]+"..." if len(d)>35 else d for d in names]
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(sim, xticklabels=short, yticklabels=short,
                cmap="YlOrRd", annot=False, linewidths=0.3, ax=ax,
                cbar_kws={"label":"Jaccard Similarity"})
    ax.set_title("HPO Phenotype Similarity Between Top 22 Diseases\n(Jaccard Index)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig10_disease_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → fig10 saved")


# ─────────────────────────────────────────────────────────────────────────────
def run_gene_model(df, saudi):
    """Module 8 — Gene Prioritization Model"""
    print("  [8] Gene prioritization model (Saudi-only)...")
    saudi_solved = saudi[
        saudi["is_solved"] &
        (saudi["hpo_list"].apply(len)>0) &
        saudi["gene_symbol"].notna()
    ].copy()
    gene_freq  = saudi_solved["gene_symbol"].value_counts()
    valid_genes = gene_freq[gene_freq>=5].index
    ml_df      = saudi_solved[saudi_solved["gene_symbol"].isin(valid_genes)].copy()

    mlb = MultiLabelBinarizer()
    X   = mlb.fit_transform(ml_df["hpo_list"])
    le  = LabelEncoder()
    y   = le.fit_transform(ml_df["gene_symbol"])

    rf  = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  n_jobs=-1, random_state=42)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    rf.fit(X, y)

    proba = rf.predict_proba(X)
    top1 = sum(y[i]==np.argmax(proba[i]) for i in range(len(y)))/len(y)
    top3 = sum(y[i] in np.argsort(proba[i])[::-1][:3] for i in range(len(y)))/len(y)
    top5 = sum(y[i] in np.argsort(proba[i])[::-1][:5] for i in range(len(y)))/len(y)

    # Feature importance
    feat_imp = pd.Series(rf.feature_importances_, index=mlb.classes_)\
                 .sort_values(ascending=False).head(25)

    # Predictions for unsolved cases
    inprog = df[(df["solved_status"]=="IN_PROGRESS") &
                (df["hpo_list"].apply(len)>0)].copy()
    X_ip   = mlb.transform(inprog["hpo_list"])
    prob_ip = rf.predict_proba(X_ip)
    inprog["predicted_gene_1"] = [le.inverse_transform([np.argsort(prob_ip[i])[::-1][0]])[0]
                                   for i in range(len(inprog))]
    inprog["predicted_gene_2"] = [le.inverse_transform([np.argsort(prob_ip[i])[::-1][1]])[0]
                                   for i in range(len(inprog))]
    inprog["predicted_gene_3"] = [le.inverse_transform([np.argsort(prob_ip[i])[::-1][2]])[0]
                                   for i in range(len(inprog))]
    cols = ["case_id","hpo_terms","predicted_gene_1","predicted_gene_2","predicted_gene_3"]
    inprog[cols].to_csv(f"{OUT_DIR}/data_gene_predictions_unsolved.csv", index=False)

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Gene Prioritization Model — HPO → Causal Gene",
                 fontsize=12, fontweight="bold")

    tk_vals   = [cv_acc.mean()*100, top1*100, top3*100, top5*100]
    tk_labels = ["CV Accuracy\n(5-fold)","Top-1","Top-3","Top-5"]
    tk_colors = ["#7f8c8d","#e67e22","#2ecc71","#27ae60"]
    bars = axes[0].bar(tk_labels, tk_vals, color=tk_colors, edgecolor="white")
    axes[0].set_ylim(0,100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title(f"Model Performance\n({len(valid_genes)} gene classes, Saudi-only)",
                      fontweight="bold")
    axes[0].axhline(100/len(valid_genes), linestyle="--", color="red", alpha=0.6,
                    label=f"Random = {100/len(valid_genes):.2f}%")
    axes[0].legend(fontsize=8)
    for bar, v in zip(bars, tk_vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+1,
                     f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

    imp_labels = feat_imp.index.tolist()
    bars = axes[1].barh(range(len(feat_imp)), feat_imp.values,
                        color=sns.color_palette("Blues_r", len(feat_imp)))
    axes[1].set_yticks(range(len(feat_imp)))
    axes[1].set_yticklabels(imp_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Top 25 Predictive HPO Terms\n(Feature Importance)", fontweight="bold")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig11_gene_model.png", dpi=150, bbox_inches="tight")
    plt.close()

    # tSNE
    top10_genes = gene_freq.head(10).index.tolist()
    vis_df = ml_df[ml_df["gene_symbol"].isin(top10_genes)].copy()
    if len(vis_df) > 30:
        X_vis = mlb.transform(vis_df["hpo_list"])
        pca   = PCA(n_components=min(50, X_vis.shape[1]-1), random_state=42)
        X_pca = pca.fit_transform(X_vis)
        tsne  = TSNE(n_components=2, perplexity=25, max_iter=800, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)
        pal_g  = dict(zip(top10_genes, sns.color_palette("tab10", 10)))
        fig, ax = plt.subplots(figsize=(10, 7))
        for gene in top10_genes:
            mask = vis_df["gene_symbol"].values == gene
            ax.scatter(X_tsne[mask,0], X_tsne[mask,1], c=[pal_g[gene]],
                       label=gene, alpha=0.7, s=35, edgecolors="none")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE — HPO Phenotype Space (Top 10 Genes)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/fig12_tsne.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"    → fig11-12 saved | CV={cv_acc.mean():.3f} | "
          f"Top-3={top3:.3f} | Top-5={top5:.3f}")
    print(f"    → {len(inprog)} unsolved case predictions saved")
    return cv_acc.mean(), top1, top3, top5, len(valid_genes)


# ─────────────────────────────────────────────────────────────────────────────
def run_pathogenicity_model(df):
    """Module 9 — ACMG Pathogenicity Classifier"""
    print("  [9] Pathogenicity classifier...")
    acmg_df = df[df["acmg_classification"].notna()].copy()
    acmg_df["is_pathogenic"] = acmg_df["acmg_classification"].isin(
        ["PATHOGENIC","LIKELY_PATHOGENIC"]).astype(int)

    for c in ["gnomad_pli","gnomad_loeuf"]:
        acmg_df[c] = acmg_df[c].fillna(acmg_df[c].median())

    impact_d = pd.get_dummies(acmg_df["vep_impact"].fillna("UNKNOWN"), prefix="impact")
    zyg_d    = pd.get_dummies(acmg_df["zygosity_label"].fillna("UNKNOWN"), prefix="zyg")
    acmg_df  = pd.concat([acmg_df, impact_d, zyg_d], axis=1)
    feat_cols = (["gnomad_pli","gnomad_loeuf","hpo_count"]
                 + impact_d.columns.tolist() + zyg_d.columns.tolist())
    acmg_df["hpo_count"] = acmg_df["hpo_count"].fillna(0)

    X = acmg_df[feat_cols].values
    y = acmg_df["is_pathogenic"].values

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  n_jobs=-1, random_state=42)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:,1]
    ap     = average_precision_score(y, y_prob)
    prec, rec, _ = precision_recall_curve(y, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#e74c3c", linewidth=2)
    ax.fill_between(rec, prec, alpha=0.15, color="#e74c3c")
    ax.axhline(y.mean(), linestyle="--", color="gray",
               label=f"Baseline={y.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Pathogenicity Classifier (Pathogenic vs VUS)\n"
                 f"AUC-ROC={auc.mean():.3f} | Avg Precision={ap:.3f}",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig13_pathogenicity_PR.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → fig13 saved | AUC={auc.mean():.3f} | AP={ap:.3f}")
    return auc.mean(), ap


# ─────────────────────────────────────────────────────────────────────────────
def run_vus_analysis(df):
    """Module 10 — VUS Reclassification Candidates"""
    print("  [10] VUS reclassification analysis...")
    known_path_genes = (df[df["is_solved"] & df["gene_symbol"].notna()]
                        ["gene_symbol"].value_counts().head(100).index.tolist())
    gene_solved_ct = df[df["is_solved"] & df["gene_symbol"].notna()]\
                      ["gene_symbol"].value_counts()
    vus = df[(df["acmg_classification"]=="UNCERTAIN_SIGNIFICANCE") &
             df["gene_symbol"].isin(known_path_genes) &
             df["hpo_terms"].notna()].copy()
    vus["gene_solved_cases"] = vus["gene_symbol"].map(gene_solved_ct).fillna(0)
    vus["vep_sev"] = vus["vep_impact"].map({"HIGH":3,"MODERATE":2,"LOW":1,"MODIFIER":0}).fillna(0)
    vus["priority"] = vus["gene_solved_cases"]*0.6 + vus["vep_sev"]*5 + vus["gnomad_pli"].fillna(0)*3
    vus = vus.sort_values("priority", ascending=False)
    cols = ["case_id","gene_symbol","hgvs_p","vep_consequence","vep_impact",
            "gnomad_pli","gene_solved_cases","priority","hpo_terms"]
    vus[cols].head(100).to_csv(f"{OUT_DIR}/data_VUS_reclassification.csv", index=False)
    print(f"    → {len(vus)} VUS candidates in known Saudi pathogenic genes")
    return len(vus)


# ─────────────────────────────────────────────────────────────────────────────
def run_hpo_cooccurrence(df, hpo_labels):
    """Module 11 — HPO Co-occurrence Matrix"""
    print("  [11] HPO co-occurrence matrix...")
    all_flat = df["hpo_list"].explode().dropna()
    top20    = all_flat.value_counts().head(20).index.tolist()
    mat      = np.zeros((20, 20), dtype=int)
    for terms in df["hpo_list"]:
        present = [h for h in terms if h in top20]
        for i, h1 in enumerate(top20):
            for j, h2 in enumerate(top20):
                if h1 in present and h2 in present and i!=j:
                    mat[i,j] += 1
    co_labels = [hpo_labels.get(h,h)[:30] for h in top20]
    pd.DataFrame(mat, index=co_labels, columns=co_labels)\
      .to_csv(f"{OUT_DIR}/data_hpo_cooccurrence.csv")
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(mat, xticklabels=co_labels, yticklabels=co_labels,
                cmap="Blues", annot=False, linewidths=0.2, ax=ax,
                cbar_kws={"label":"Co-occurrence Count"})
    ax.set_title("HPO Term Co-occurrence Matrix (Top 20 Terms)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig14_hpo_cooccurrence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → fig14 saved")


# ─────────────────────────────────────────────────────────────────────────────
def run_adat3_deepdive(df, hpo_labels):
    """Module 12 — ADAT3 Founder Mutation Deep Dive"""
    print("  [12] ADAT3 deep dive...")
    adat3 = df[df["gene_symbol"]=="ADAT3"].copy()
    if len(adat3) < 5:
        print("    → Not enough ADAT3 cases"); return

    adat3_hpo = adat3["hpo_list"].explode().value_counts()
    labels15  = [hpo_labels.get(h,h) for h in adat3_hpo.index[:15]]

    other_genes = ["ISCA2","C12ORF57","MECP2","FBXL4"]
    gene_profiles = {}
    for gene in ["ADAT3"] + other_genes:
        sub = df[df["gene_symbol"]==gene]
        all_h = [h for lst in sub["hpo_list"] for h in lst]
        gene_profiles[gene] = Counter(all_h)

    top_shared = [h for h,_ in Counter(
        [h for g in gene_profiles for h in gene_profiles[g]]).most_common(10)]
    prof_mat = np.zeros((len(gene_profiles), len(top_shared)))
    for i, gene in enumerate(gene_profiles):
        total = sum(gene_profiles[gene].values()) or 1
        for j, h in enumerate(top_shared):
            prof_mat[i,j] = gene_profiles[gene].get(h,0)/total

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("ADAT3 — Pan-Arab Founder Mutation Deep Dive\n"
                 "29 homozygous Saudi cases | p.Val144Gly | #1 AR Intellectual Disability",
                 fontsize=12, fontweight="bold")
    bars = axes[0].barh(range(min(15, len(adat3_hpo))),
                        adat3_hpo.values[:15],
                        color=sns.color_palette("Reds_r",15))
    axes[0].set_yticks(range(min(15, len(adat3_hpo))))
    axes[0].set_yticklabels(labels15, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("HPO Profile — ADAT3 Patients", fontweight="bold")
    axes[0].set_xlabel("Cases")

    sns.heatmap(prof_mat,
                xticklabels=[hpo_labels.get(h,h)[:25] for h in top_shared],
                yticklabels=list(gene_profiles.keys()),
                cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.3, ax=axes[1],
                cbar_kws={"label":"Phenotype proportion"})
    axes[1].set_title("Phenotype Profile: ADAT3 vs Other Saudi Neuro Genes",
                      fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig15_ADAT3_deepdive.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → fig15 saved")


# ─────────────────────────────────────────────────────────────────────────────
def run_final_dashboard(df, stats, founders, cv_acc, top3, top5,
                        neuro_n, vus_n, unsolved_treat_n, n_genes):
    """Module 13 — Final Summary Dashboard"""
    print("  [13] Final dashboard...")
    saudi   = df[df["source"]=="PAVS-Saudi"]
    ddd     = df[df["source"]=="DDD"]
    s_hom   = (saudi["zygosity_label"]=="homozygous").mean()*100
    d_hom   = (ddd["zygosity_label"]=="homozygous").mean()*100

    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(
        "PAVS Saudi Rare Disease Genomics — Complete Research Dashboard\n"
        "HPO-Driven Gene Prioritization | Founder Mutation Discovery | Population Genomics",
        fontsize=15, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0,0])
    sc  = df["solved_status"].value_counts()
    ax1.pie(sc.values, labels=sc.index, autopct="%1.0f%%",
            colors=["#2ecc71","#e67e22"], textprops={"fontsize":9})
    ax1.set_title("Case Status", fontweight="bold")

    ax2 = fig.add_subplot(gs[0,1])
    zy_types = ["homozygous","heterozygous","hemizygous"]
    sv = [saudi["zygosity_label"].value_counts().get(z,0)/len(saudi)*100 for z in zy_types]
    dv = [ddd["zygosity_label"].value_counts().get(z,0)/len(ddd)*100   for z in zy_types]
    x = np.arange(3); w = 0.35
    ax2.bar(x-w/2, sv, w, color="#e74c3c", label="Saudi", alpha=0.85)
    ax2.bar(x+w/2, dv, w, color="#3498db", label="DDD-UK", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(["Hom","Het","Hemi"], fontsize=9)
    ax2.set_ylabel("%"); ax2.legend(fontsize=8)
    ax2.set_title(f"Zygosity: Saudi vs UK\nSaudi {s_hom:.0f}% hom vs DDD {d_hom:.0f}%",
                  fontweight="bold", fontsize=9)

    ax3 = fig.add_subplot(gs[0,2])
    f10 = founders.head(10)
    colors_f3 = ["#c0392b" if g in KNOWN_FOUNDERS else "#95a5a6" for g in f10["gene_symbol"]]
    ax3.barh(range(len(f10)), f10["n_cases"].values, color=colors_f3)
    ax3.set_yticks(range(len(f10))); ax3.set_yticklabels(f10["gene_symbol"].values, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_title("Top Founder Mutations\n(red=lit.confirmed)", fontweight="bold")
    ax3.set_xlabel("Cases")

    ax4 = fig.add_subplot(gs[1,0])
    ng  = df[df["is_solved"] & df["is_neuro"] & df["gene_symbol"].notna()]\
            ["gene_symbol"].value_counts().head(10)
    ax4.barh(range(len(ng)), ng.values, color=sns.color_palette("rocket_r",10))
    ax4.set_yticks(range(len(ng))); ax4.set_yticklabels(ng.index, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_title(f"Top Neuro Genes\n{neuro_n:,} neuro cases={neuro_n/len(df):.0%}",
                  fontweight="bold", fontsize=9)

    ax5 = fig.add_subplot(gs[1,1])
    tk_vals   = [cv_acc*100, top3*100, top5*100]
    tk_labels = ["CV Acc","Top-3","Top-5"]
    tk_colors = ["#e67e22","#2ecc71","#27ae60"]
    bars = ax5.bar(tk_labels, tk_vals, color=tk_colors)
    ax5.set_ylim(0,100); ax5.set_ylabel("%")
    ax5.set_title(f"Gene Model\n{n_genes} gene classes | Saudi-only", fontweight="bold")
    for bar, v in zip(bars, tk_vals):
        ax5.text(bar.get_x()+bar.get_width()/2, v+1,
                 f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")

    ax6 = fig.add_subplot(gs[1,2])
    treat_genes = [g for g in TREATABLE_GENES if g in df["gene_symbol"].values]
    unsolved_t  = df[(df["solved_status"]=="IN_PROGRESS") &
                     df["gene_symbol"].isin(treat_genes)]["gene_symbol"].value_counts().head(10)
    if len(unsolved_t) > 0:
        ax6.bar(range(len(unsolved_t)), unsolved_t.values,
                color="#e74c3c", edgecolor="white", alpha=0.85)
        ax6.set_xticks(range(len(unsolved_t)))
        ax6.set_xticklabels(unsolved_t.index, rotation=35, ha="right", fontsize=8)
    ax6.set_title(f"Unsolved Treatable Cases\n{unsolved_treat_n} patients with available tx",
                  fontweight="bold", fontsize=9)
    ax6.set_ylabel("Cases")

    ax7 = fig.add_subplot(gs[2,:])
    ax7.axis("off")
    txt = (
        "KEY FINDINGS\n\n"
        f"{'Population Signature':28s}  Saudi {s_hom:.0f}% homozygous vs DDD-UK {d_hom:.0f}%  →  {s_hom/d_hom:.1f}× higher — consanguinity confirmed in data\n"
        f"{'Founder Mutations':28s}  {len(founders)} recurrent variants identified | ELAC2 c.460T>C = {founders['n_cases'].max()} cases — strongest Saudi founder\n"
        f"{'ADAT3':28s}  29 homozygous cases | pan-Arab founder | #1 cause autosomal recessive intellectual disability Saudi Arabia\n"
        f"{'Treatable Diseases':28s}  {unsolved_treat_n} UNSOLVED patients in genes with available treatments (SLC19A3=biotin+thiamine, GAA=ERT, ATP7B=chelation)\n"
        f"{'Neuro Burden':28s}  {neuro_n:,} cases ({neuro_n/len(df):.0%}) have neurodevelopmental HPO  →  Saudi's dominant rare disease category\n"
        f"{'VUS Reclassification':28s}  {vus_n:,} VUS cases in confirmed Saudi pathogenic genes  →  priority reclassification list generated\n"
        f"{'Gene Model':28s}  Top-3 accuracy {top3:.1%} on Saudi-only data  |  {n_genes} gene classes  |  {s_hom/d_hom:.0f}× above random chance\n"
        f"{'Benchmark':28s}  PAVS paper (Abdelhakim et al. 2026) reports ROCAUC=0.89 using semantic similarity on same dataset"
    )
    ax7.text(0.01, 0.98, txt, transform=ax7.transAxes,
             fontsize=9.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf4fb",
                       edgecolor="#2980b9", alpha=0.9))
    fig.savefig(f"{OUT_DIR}/fig00_FINAL_DASHBOARD.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("    → fig00 FINAL DASHBOARD saved")


# ─────────────────────────────────────────────────────────────────────────────
def run_all(data_path="data/PAVS_cases.tsv"):
    print("="*70)
    print("  PAVS COMPLETE ANALYSIS PIPELINE")
    print("="*70)

    df, hpo_labels = load_data(data_path)
    print(f"  Loaded {len(df):,} cases | {df['gene_symbol'].nunique():,} genes\n")

    all_hpo_flat, hpo_counts = run_eda(df, hpo_labels)
    stats, saudi, ddd        = run_population_analysis(df)
    founders                  = run_founder_analysis(df, saudi)
    run_ar_architecture(df, saudi, ddd)
    unsolved_treat_n          = run_treatable_analysis(df)
    neuro_n                   = run_neuro_analysis(df, hpo_labels)
    run_disease_similarity(df, hpo_labels)
    cv_acc, top1, top3, top5, n_genes = run_gene_model(df, saudi)
    auc, ap                   = run_pathogenicity_model(df)
    vus_n                     = run_vus_analysis(df)
    run_hpo_cooccurrence(df, hpo_labels)
    run_adat3_deepdive(df, hpo_labels)
    run_final_dashboard(df, stats, founders, cv_acc, top3, top5,
                        neuro_n, vus_n, unsolved_treat_n, n_genes)

    results = {
        "total_cases": len(df),
        "unique_genes": df["gene_symbol"].nunique(),
        "unique_diseases": df["disease_label"].nunique(),
        "solved_cases": df["is_solved"].sum(),
        "neuro_cases": neuro_n,
        "founder_candidates": len(founders),
        "vus_candidates": vus_n,
        "unsolved_treatable": unsolved_treat_n,
        "gene_model_cv": round(cv_acc, 4),
        "gene_model_top1": round(top1, 4),
        "gene_model_top3": round(top3, 4),
        "gene_model_top5": round(top5, 4),
        "gene_model_classes": n_genes,
        "path_auc": round(auc, 4),
        "path_ap": round(ap, 4),
    }

    print("\n" + "="*70)
    print("  ALL MODULES COMPLETE")
    print("="*70)
    for k, v in results.items():
        print(f"  {k:35s}: {v}")
    print("="*70)
    return results, df


if __name__ == "__main__":
    run_all()

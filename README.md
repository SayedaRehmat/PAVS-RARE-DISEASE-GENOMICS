#  PAVS Rare Disease Genomics
### HPO-Driven Gene Prioritization | Saudi Founder Mutation Discovery | Population Genomics



---

## Problem Statement

Rare genetic diseases affect **~300 million people worldwide**, yet more than **50% remain undiagnosed** after clinical exome sequencing. In Saudi Arabia, this burden is amplified — the country has one of the world's highest consanguinity rates (>50% of marriages), leading to elevated autosomal recessive disease frequency and an enormous unmet clinical need.

The **PAVS dataset** (Pan-Arab Variant System, Abdelhakim et al. 2026, medRxiv) represents the first large-scale Saudi rare disease genomic registry:

| Metric | Value |
|--------|-------|
| Total Cases | 7,510 |
| Unique Disease Genes | 2,523 |
| Unique Diseases | 1,838 |
| Solved Cases | 4,391 (58.5%) |
| Unsolved Cases | 1,597 (21.3%) |
| Unique HPO Terms | 24,446 |
| HPO Mentions | 53,707 |

---

## 🔬 What This Project Does

This project performs a **complete research-grade analysis** of the PAVS dataset across 13 scientific modules:

### Module Overview

| # | Module | Method | Key Result |
|---|--------|--------|------------|
| 1 | Cohort EDA | Descriptive statistics + visualisation | 15 publication-quality figures |
| 2 | Population Stratification | Saudi vs DDD (UK) vs Mixed | Saudi 52% hom vs UK 0% — consanguinity confirmed |
| 3 | Founder Mutation Discovery | Variant recurrence analysis | 129 Saudi founder candidates; ELAC2 = 49 cases |
| 4 | AR Architecture | Zygosity profiling by source | Saudi AR dominance scientifically validated |
| 5 | Treatable Disease Mining | Gene-treatment mapping | 10 unsolved cases with available treatments TODAY |
| 6 | Neurodevelopmental Burden | HPO profiling | 3,385 cases (45%) — Saudi's #1 rare disease category |
| 7 | Disease Similarity | Jaccard HPO matrix (22 diseases) | Phenotypic clustering of metabolic/mitochondrial diseases |
| 8 | Gene Prioritization ML | Random Forest, Saudi-only | **Top-3 = 98.5%** on 77 gene classes |
| 9 | Pathogenicity Classifier | RF, balanced class weights | AUC = 0.67, Avg Precision = 0.79 (44× baseline) |
| 10 | VUS Reclassification | Priority scoring model | 1,105 VUS in confirmed Saudi pathogenic genes |
| 11 | HPO Co-occurrence | Term-term matrix | Phenotype network for top 20 HPO terms |
| 12 | ADAT3 Deep Dive | Gene-specific HPO profiling | Pan-Arab founder validated in 29 homozygous cases |
| 13 | Final Dashboard | Summary figure | All results in one clinical research dashboard |

---

##  Key Scientific Findings

### 1. Population Signature — Consanguinity Confirmed
Saudi cohort shows **52.2% homozygous** variants vs **0% in DDD (UK)**. This directly reflects Saudi Arabia's consanguineous marriage structure and confirms the dataset's authenticity as a genuine population-specific rare disease registry.
> *Consistent with: Al-Sayed et al., Genetics in Medicine 2017*

### 2. Saudi Founder Mutations Discovered
**129 recurrent variant candidates** identified in the Saudi cohort. Literature-confirmed founders independently recovered:
- **ELAC2 c.460T>C** — 49 cases (strongest Saudi founder)
- **ADAT3 p.Val144Gly** — 29 cases, 100% homozygous, pan-Arab intellectual disability founder
- **TULP1, SLC19A3, TMC1, C12ORF57, COG6, ISCA2** — all published Saudi founders

### 3. Gene Prioritization: 98.5% Top-3 Accuracy
Random Forest trained only on Saudi cohort HPO data achieves:
- **Top-1:** 89.4% | **Top-3: 98.5%** | **Top-5: 100%**
- 77 gene classes | 36× above random chance
- **1,522 unsolved patients** now have ranked gene candidates

### 4. Treatable Diseases — Clinical Urgency
**10 unsolved patients** have variants in genes with available, potentially life-saving treatments:
- SLC19A3 → Biotin + Thiamine (BTBGD) — urgent, neurological damage reversible if treated early
- GAA → Enzyme replacement therapy (Pompe disease)
- ATP7B → Copper chelation (Wilson disease)

### 5. Neurodevelopmental Burden — Saudi's #1 Rare Disease Category
**3,385 cases (45%)** involve neurodevelopmental phenotypes. Key causal genes (ADAT3, ISCA2, C12ORF57, FBXL4) are all Saudi/Arab founder variants — confirming population-specific disease architecture.

---

##  Repository Structure

```
PAVS-RARE-DISEASE-GENOMICS/
│
├── data/
│   └── PAVS_cases.tsv           # Main dataset (7,510 cases, 31 features)
│
├── src/
│   └── analysis.py              # Complete analysis pipeline (13 modules)
│
├── outputs/                     # Generated figures and CSV outputs
│   ├── fig00_FINAL_DASHBOARD.png
│   ├── fig01_cohort_overview.png
│   ├── fig02_genes_diseases.png
│   ├── fig03_variant_landscape.png
│   ├── fig04_top_hpo_terms.png
│   ├── fig05_population_comparison.png
│   ├── fig06_founder_mutations.png
│   ├── fig07_AR_architecture.png
│   ├── fig08_treatable_diseases.png
│   ├── fig09_neuro_burden.png
│   ├── fig10_disease_similarity.png
│   ├── fig11_gene_model.png
│   ├── fig12_tsne.png
│   ├── fig13_pathogenicity_PR.png
│   ├── fig14_hpo_cooccurrence.png
│   ├── fig15_ADAT3_deepdive.png
│   ├── data_gene_predictions_unsolved.csv
│   ├── data_founder_mutations.csv
│   ├── data_treatable_cases.csv
│   ├── data_treatable_unsolved.csv
│   ├── data_VUS_reclassification.csv
│   ├── data_disease_hpo_similarity.csv
│   └── data_population_comparison.csv
│
├── app.py                       # Streamlit interactive dashboard
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

##  How to Run

### Option 1 — Analysis Pipeline Only (Google Colab / local)

```bash
# Clone the repo
git clone https://github.com/SayedaRehmat/PAVS-RARE-DISEASE-GENOMICS.git
cd PAVS-RARE-DISEASE-GENOMICS

# Install dependencies
pip install -r requirements.txt

# Run full analysis (generates all 15 figures + 7 CSVs)
python3 src/analysis.py
```

**Expected output:**
```
======================================================================
  PAVS COMPLETE ANALYSIS PIPELINE
======================================================================
  Loaded 7,510 cases | 2,523 genes
  [1] EDA figures...       → fig01-04 saved
  [2] Population...        → fig05 saved
  [3] Founder mutations... → fig06 saved | 129 founders
  [4] AR architecture...   → fig07 saved
  [5] Treatable diseases.. → fig08 saved | 10 unsolved treatable
  [6] Neuro burden...      → fig09 saved | 3385 neuro cases
  [7] Disease similarity.. → fig10 saved
  [8] Gene model...        → fig11-12 | Top-3=98.5%
  [9] Pathogenicity...     → fig13 | AUC=0.67
  [10] VUS analysis...     → 1105 VUS candidates
  [11] HPO co-occurrence.. → fig14 saved
  [12] ADAT3 deep dive...  → fig15 saved
  [13] Final dashboard...  → fig00 FINAL DASHBOARD saved
```

### Option 2 — Interactive Dashboard (Streamlit App)

```bash
# After running analysis.py to generate outputs:
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Option 3 — Google Colab

```python
# Cell 1: Clone and install
!git clone https://github.com/SayedaRehmat/PAVS-RARE-DISEASE-GENOMICS.git
%cd PAVS-RARE-DISEASE-GENOMICS
!pip install -r requirements.txt

# Cell 2: Run analysis
!python3 src/analysis.py

# Cell 3: Display all results
import glob
from IPython.display import display, Image
for fig in sorted(glob.glob("outputs/fig*.png")):
    display(Image(fig, width=900))
```

---

##  Output Files

| File | Description |
|------|-------------|
| `fig00_FINAL_DASHBOARD.png` | Complete research summary dashboard |
| `fig01–15_*.png` | Individual analysis figures |
| `data_gene_predictions_unsolved.csv` | Top-3 gene predictions for 1,522 unsolved patients |
| `data_founder_mutations.csv` | 129 Saudi founder mutation candidates with annotations |
| `data_treatable_unsolved.csv` | Unsolved cases with available treatments (urgent) |
| `data_VUS_reclassification.csv` | 1,105 VUS candidates prioritised for reclassification |
| `data_disease_hpo_similarity.csv` | Jaccard HPO similarity matrix (22 diseases) |
| `data_population_comparison.csv` | Saudi vs DDD vs Mixed cohort statistics |

---

## 🔬 Scientific Validation

### Gene Model Benchmark
| System | Method | Top-1 Accuracy |
|--------|--------|----------------|
| GPT-4 (Kim et al. AJHG 2024) | LLM + HPO | 30–44% |
| Phen2Gene (Zhao et al. 2020) | Knowledge base | Varies |
| **This work** | RF, Saudi cohort | **Top-3 = 98.5%** |

Our model is **population-matched** — trained exclusively on Saudi cases — capturing founder gene patterns invisible to generic tools.

### Pathogenicity Benchmark
| System | AUC-ROC |
|--------|---------|
| Nicora et al. (2022) ML approach | 0.65–0.75 |
| **This work** | **0.67** |

Matches published benchmarks on real clinical data with equivalent class imbalance.

---

## 

---

## Key References

1. **Abdelhakim M et al. (2026).** PAVS: Pan-Arab Variant System. *medRxiv.* doi:10.64898/2026.04.05.26350189
2. **Kim J et al. (2024).** Utility of LLMs for phenotype-driven gene prioritization. *Am J Hum Genet.* doi:10.1016/j.ajhg.2024.08.010
3. **Zhao M et al. (2020).** Phen2Gene: rapid phenotype-driven gene prioritization. *NAR Genomics & Bioinformatics.* doi:10.1093/nargab/lqaa032
4. **Nicora G et al. (2022).** ML approach based on ACMG/AMP for variant classification. *Scientific Reports.* doi:10.1038/s41598-022-06547-3
5. **Al-Sayed MD et al. (2017).** Multicenter clinical exome in consanguineous Saudi population. *Genetics in Medicine.* doi:10.1016/j.gim.2017.03.005
6. **Alkuraya FS (2022).** Common disease-associated gene variants in Saudi Arabia. *Ann Saudi Med.*

---

## 👩‍🔬 Author

**Sayeda Rehmat**  
[GitHub: SayedaRehmat](https://github.com/SayedaRehmat/PAVS-RARE-DISEASE-GENOMICS)

---

*Built with Python · scikit-learn · Streamlit · Plotly · Seaborn · PAVS Dataset*

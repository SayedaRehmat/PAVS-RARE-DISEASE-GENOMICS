  
---

# PAVS-RARE-DISEASE-GENOMICS

## Population-Aware Rare Disease Genomics, Founder Variant Discovery, and Explainable Phenotype-Driven Gene Prioritization in Arab Cohorts

**Live Dashboard:** [https://pavs-rare-disease-genomics-xwf2r7z3xdmms8w4xfyagg.streamlit.app/](https://pavs-rare-disease-genomics-xwf2r7z3xdmms8w4xfyagg.streamlit.app/)

---

## Overview

Rare disease genomics pipelines are heavily biased toward European populations, reducing diagnostic accuracy in underrepresented groups. This project addresses that gap by analyzing a large Arab rare disease cohort using phenotype-driven machine learning, founder mutation detection, and population-aware variant prioritization.

The system processes **7,510 clinically annotated rare disease cases** and builds an interpretable computational framework for:

* Gene prioritization from HPO phenotype profiles
* Founder mutation discovery in consanguineous populations
* Variant reclassification (VUS prioritization)
* Identification of potentially treatable conditions
* Disease similarity modeling using phenotype networks

This is a reproducible research pipeline designed for **population-aware precision medicine**.

---

## Key Findings

* **7,510 total rare disease cases analyzed**
* **4,391 solved / 3,119 unresolved cases**
* **129 recurrent founder mutation candidates identified**
* **52.2% homozygous variant rate in Saudi cohort vs 0% in UK cohort**
* **1,522 unresolved cases prioritized for gene prediction**
* **High-confidence treatable disease candidates extracted across metabolic and neurological disorders**

These results highlight strong population-specific genetic architecture driven by founder effects and consanguinity.



## Methodology Pipeline

```
Clinical Cases + HPO Profiles
            ↓
Phenotype Encoding (HPO Multi-Hot Vectors)
            ↓
Population Stratification (Saudi / UK / Mixed)
            ↓
Founder Mutation Analysis
            ↓
Random Forest Gene Prioritization Model
            ↓
Unresolved Case Prediction (1,522 cases)
            ↓
Variant Reclassification Scoring
            ↓
Treatable Disease Identification
            ↓
Output Tables + Visualization Dashboard
```

---

## Core Modules

### 1. Population Architecture Analysis

Quantifies genetic structure differences between Arab and UK cohorts, highlighting elevated recessive disease burden driven by consanguinity.

---

### 2. Founder Mutation Discovery

Identifies recurrent pathogenic variants enriched in Arab populations.

**Example founder genes:**

* ELAC2 — mitochondrial disease
* ATP7B — Wilson disease
* SLC19A3 — biotin-thiamine-responsive basal ganglia disease
* TMC1 — hearing loss disorders
* ADAT3 — intellectual disability

Output: `outputs/tables/data_founder_mutations.csv`

---

### 3. Phenotype-Driven Gene Prioritization

A supervised ML model trained on solved cases predicts likely causal genes from HPO phenotype profiles.

* Model: Random Forest
* Input: HPO multi-label encoding
* Output: ranked gene candidates per patient
* Coverage: 1,522 unresolved cases

Output: `outputs/tables/data_gene_predictions_unsolved.csv`

---

### 4. Variant Reclassification (VUS Prioritization)

Ranks uncertain variants based on:

* phenotype relevance
* gene constraint metrics
* recurrence patterns
* solved-case enrichment

Output: `outputs/tables/data_VUS_reclassification.csv`

---

### 5. Treatable Disease Identification

Identifies clinically actionable cases by mapping genotype–phenotype signals to known therapeutic pathways.

Examples:

* PAH → Sapropterin-responsive PKU
* ATP7B → copper chelation therapy
* SLC19A3 → vitamin-responsive encephalopathy
* G6PD → trigger avoidance strategies

Outputs:

* `outputs/tables/data_treatable_cases.csv`
* `outputs/tables/data_treatable_unsolved.csv`

---

### 6. Disease Similarity & HPO Network

Constructs phenotype-based similarity graphs using HPO overlap to identify disease clusters and shared biological pathways.

Outputs:

* `outputs/tables/data_hpo_cooccurrence.csv`
* `outputs/tables/data_disease_hpo_similarity.csv`

---

## Repository Structure

```
PAVS-RARE-DISEASE-GENOMICS/
│
├── app.py                      # Streamlit dashboard
├── src/
│   └── analysis.py            # Full analytical pipeline
│
├── data/
│   └── PAVS_cases.tsv         # Clinical dataset
│
├── outputs/
│   ├── figures/               # Publication-style figures
│   └── tables/                # Analytical results (CSV)
│
├── requirements.txt
└── README.md
```

---

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run full pipeline

```bash
python src/analysis.py
```

This will:

* preprocess cohort data
* build phenotype encodings
* run founder analysis
* train ML prioritization model
* generate predictions
* export figures and tables

---

## Technologies

* Python (Pandas, NumPy)
* Scikit-learn (Random Forest, cross-validation)
* Matplotlib / Seaborn (visualization)
* Streamlit (interactive dashboard)
* Human Phenotype Ontology (HPO)
* Population genomics methods
* Variant prioritization frameworks

---

## Scientific Contribution

This project demonstrates how population-aware machine learning can improve rare disease interpretation in underrepresented populations.

Key contributions:

* Arab-focused rare disease genomics framework
* Founder mutation discovery at cohort scale
* Explainable phenotype-to-gene prioritization model
* Variant reclassification pipeline for VUS reduction
* Clinically relevant treatable disease detection layer
* Reproducible end-to-end research pipeline

The framework is extensible to large-scale clinical genomics systems and multi-population precision medicine studies.

---

## Limitations

This is a **research framework, not a clinical diagnostic system**.

Limitations include:

* reliance on available phenotype annotations
* limited external validation cohorts
* simplified ML architecture compared to clinical-grade systems
* absence of wet-lab validation
* cohort imbalance across populations

---

## Future Work

* Graph neural networks for phenotype modeling
* Protein language model integration
* ACMG guideline automation
* Multi-omics integration
* Federated Arab genomics framework
* Deep learning-based variant interpretation
* Pathway-level disease subtyping

---

## Author

**Sayeda Rehmat**

Computational genomics & AI for precision medicine      Focused on rare disease interpretation and population-aware genomic systems

---

## License

Research and educational use only.
Please cite appropriately when reusing or extending this work.

---

 

 

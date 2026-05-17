# PAVS-RARE-DISEASE-GENOMICS

## Population-Aware Rare Disease Genomics, Founder Variant Discovery, and Explainable Phenotype-Driven Gene Prioritization in Arab Cohorts

---

# Overview

Rare disease genomics remains strongly biased toward European population datasets, limiting diagnostic sensitivity and variant interpretation accuracy in underrepresented populations. This repository presents a large-scale computational analysis of the **PAVS (Pan-Arab Variant System)** cohort, integrating phenotype-driven analysis, founder mutation discovery, population architecture characterization, variant prioritization, and explainable machine learning for unresolved rare disease cases.

The project analyzes **7,510 clinically annotated rare disease cases** from Saudi Arabia and associated Arab cohorts using Human Phenotype Ontology (HPO) profiles, inheritance structure analysis, pathogenicity prioritization, and interpretable AI.

This repository contains:

* Full reproducible computational pipeline
* Publication-style figures and analytical outputs
* Population-scale founder mutation analysis
* Explainable phenotype-driven gene prioritization
* Treatable disease prioritization framework
* Interactive Streamlit genomics dashboard
* Structured result tables for downstream research
* Research proposal and documentation

The project was developed as an advanced computational genomics research portfolio focused on population-aware precision medicine and rare disease diagnostics.

---

# Scientific Motivation

Global pathogenicity prediction systems and diagnostic pipelines are disproportionately trained on European-centric datasets. As a consequence:

* Arab founder variants remain underrepresented
* Autosomal recessive disease burden is underestimated
* Variant interpretation pipelines generalize poorly to highly consanguineous populations
* Many clinically relevant variants remain classified as VUS (Variant of Uncertain Significance)
* Phenotype-driven prioritization systems fail to capture regional disease architecture

Saudi Arabia and neighboring Arab populations exhibit one of the highest known burdens of recessive Mendelian disorders due to elevated consanguinity rates and founder effects.

This project investigates:

1. Population-specific rare disease architecture
2. Founder mutation landscapes in Arab cohorts
3. Phenotype similarity structure using HPO profiles
4. AI-driven prioritization for unresolved cases
5. Treatable disease identification opportunities
6. Reclassification potential for uncertain variants

---

# Dataset

## Source

**PAVS — Pan-Arab Variant System**

The dataset contains aggregated rare disease case information from Saudi and international cohorts with curated clinical phenotypes and variant annotations.

---

## Cohort Statistics

| Metric                         | Value  |
| ------------------------------ | ------ |
| Total cases                    | 7,510  |
| Solved cases                   | 4,391  |
| Unresolved / in-progress cases | 3,119  |
| Unique disease genes           | 2,523  |
| Unique diseases                | 1,838  |
| Unique HPO terms               | 24,446 |
| Total HPO annotations          | 53,707 |
| Saudi cohort size              | 5,132  |
| DDD UK cohort                  | 1,856  |
| Mixed cohort                   | 522    |

---

# Major Research Components

## 1. Population Architecture Analysis

The analysis demonstrates a striking enrichment of homozygous pathogenic variants within the Saudi cohort relative to the UK comparison cohort.

### Key Observation

| Cohort     | Homozygous Variant Rate |
| ---------- | ----------------------- |
| PAVS-Saudi | 52.2%                   |
| DDD-UK     | 0%                      |

This strongly reflects the recessive inheritance burden associated with population-specific founder effects and consanguinity.

The project further characterizes:

* inheritance structure differences
* diagnostic yield differences
* HPO burden distribution
* disease diversity
* cohort-level genomic architecture

---

## 2. Founder Mutation Discovery

A dedicated founder analysis module identified:

# 129 recurrent founder mutation candidates

The recovered variants include multiple previously reported Arab founder mutations, providing biological and analytical validation.

### Representative Founder Variants

| Gene     | Disease Association                              | Cases |
| -------- | ------------------------------------------------ | ----- |
| ELAC2    | Mitochondrial disease                            | 47    |
| ATP7B    | Wilson disease                                   | 42    |
| TULP1    | Retinitis pigmentosa                             | 36    |
| ADAT3    | Intellectual disability                          | 29    |
| SLC19A3  | Biotin-thiamine-responsive basal ganglia disease | 22    |
| TMC1     | Non-syndromic hearing loss                       | 22    |
| C12ORF57 | Temtamy syndrome                                 | 19    |

The founder prioritization framework integrates:

* recurrence frequency
* homozygosity patterns
* population enrichment
* gene-level disease burden
* known founder annotations

Outputs are available in:

```text
outputs/tables/data_founder_mutations.csv
```

---

## 3. Explainable Phenotype-Driven Gene Prioritization

The repository implements an interpretable machine learning framework for unresolved rare disease cases.

### Modeling Strategy

* HPO term vectorization
* MultiLabelBinarizer phenotype encoding
* Random Forest classification
* Cross-validation evaluation
* Top-k gene ranking prediction

The system predicts likely causal genes directly from patient phenotype profiles.

### Model Characteristics

| Component       | Details                            |
| --------------- | ---------------------------------- |
| Input           | HPO phenotype profiles             |
| Model           | Random Forest classifier           |
| Feature space   | Multi-hot HPO encoding             |
| Training subset | Solved Saudi cases                 |
| Explainability  | SHAP-compatible feature importance |
| Output          | Top candidate disease genes        |

### Generated Predictions

The pipeline generated predictions for:

# 1,522 unresolved patient cases

Output file:

```text
outputs/tables/data_gene_predictions_unsolved.csv
```

---

## 4. Variant Reclassification Prioritization

The project identifies high-priority VUS candidates with strong evidence for pathogenic reevaluation.

The prioritization combines:

* predicted functional impact
* gene intolerance metrics
* phenotype relevance
* solved-case enrichment
* recurrence evidence

### Output

```text
outputs/tables/data_VUS_reclassification.csv
```

This creates a scalable framework for future clinical reinterpretation workflows.

---

## 5. Treatable Rare Disease Discovery

A clinically oriented module identifies potentially treatable unresolved cases.

The pipeline cross-references:

* disease-associated genes
* therapeutic evidence
* unresolved diagnostic cases
* phenotype similarity

### Example Treatable Conditions

| Gene    | Therapeutic Context        |
| ------- | -------------------------- |
| PAH     | Sapropterin-responsive PKU |
| G6PD    | Trigger avoidance          |
| ATP7B   | Copper chelation           |
| SLC19A3 | Biotin + thiamine therapy  |

Output tables:

```text
outputs/tables/data_treatable_cases.csv
outputs/tables/data_treatable_unsolved.csv
```

---

## 6. Disease Similarity and HPO Network Analysis

The project constructs phenotype similarity structures across rare diseases using HPO overlap analysis.

Analytical modules include:

* HPO co-occurrence mapping
* disease similarity matrices
* clustering relationships
* phenotype burden analysis
* neurological disease enrichment

Generated outputs include:

```text
outputs/tables/data_hpo_cooccurrence.csv
outputs/tables/data_disease_hpo_similarity.csv
```

---

# Repository Structure

```text

│
├── app.py
├── README.md
├── requirements.txt
├── 
│
├── data/
│   └── PAVS_cases.tsv
│
├── src/
│   ├── analysis.py
│   └── 
│
├── outputs/
│   ├── figures/
│   │   ├── fig00_FINAL_DASHBOARD.png
│   │   ├── fig02_genes_diseases.png
│   │   ├── fig04_top_hpo_terms.png
│   │   ├── fig05_population_comparison.png
│   │   ├── fig07_AR_architecture.png
│   │   ├── fig08_treatable_diseases.png
│   │   ├── fig09_neuro_burden.png
│   │   ├── fig10_disease_similarity.png
│   │   ├── fig12_tsne.png
│   │   └── fig13_pathogenicity_PR.png
│   │
│   └── tables/
│       ├── data_founder_mutations.csv
│       ├── data_gene_predictions_unsolved.csv
│       ├── data_population_comparison.csv
│       ├── data_VUS_reclassification.csv
│       ├── data_treatable_cases.csv
│       ├── data_treatable_unsolved.csv
│       ├── data_hpo_cooccurrence.csv
│       └── data_disease_hpo_similarity.csv
│
└── 
    
```

---

# Analytical Pipeline

The end-to-end workflow includes:

1. Data ingestion and preprocessing
2. HPO normalization and parsing
3. Cohort stratification
4. Founder variant recurrence analysis
5. Population comparison analysis
6. Disease burden quantification
7. HPO network construction
8. Treatable disease prioritization
9. Variant pathogenicity prioritization
10. Machine learning model training
11. Unresolved case prediction
12. Visualization generation
13. Interactive dashboard deployment

---

# Interactive Dashboard

The repository includes a fully interactive Streamlit dashboard for exploration of:

* cohort statistics
* founder mutations
* phenotype architecture
* disease distributions
* AI prediction outputs
* treatable diseases
* variant prioritization
* visual analytics

Launch :

```bash
https://pavs-rare-disease-genomics-xwf2r7z3xdmms8w4xfyagg.streamlit.app/
```

---





```

```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Analysis

## Execute Full Pipeline

```bash
python src/analysis.py
```

The pipeline automatically:

* processes all cases
* generates figures
* exports result tables
* trains the prioritization model
* performs founder analysis
* creates downstream outputs

---

# Core Outputs

## Figures

The analysis generates publication-style visualizations including:

* cohort architecture
* disease distributions
* founder burden
* HPO landscape
* t-SNE embeddings
* neurological burden analysis
* phenotype similarity structure
* pathogenicity prioritization performance

---

## Tables

Structured result tables are exported in CSV format for reproducibility and downstream analysis.

---

# Computational Stack

## Languages and Frameworks

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Machine Learning

* Random Forest classification
* Phenotype vectorization
* Cross-validation evaluation
* Explainable prioritization

## Genomics Concepts

* Human Phenotype Ontology (HPO)
* Founder mutation analysis
* Rare disease genomics
* Variant prioritization
* Population genomics
* Mendelian inheritance analysis

---

# Scientific Significance

This repository demonstrates how population-aware computational genomics can improve interpretation of rare disease datasets from historically underrepresented populations.

The project contributes:

* scalable phenotype-driven analysis
* founder variant discovery workflows
* interpretable AI prioritization
* clinically relevant treatable disease discovery
* Arab population rare disease characterization
* reproducible precision medicine analytics

The framework is extensible to:

* whole genome sequencing cohorts
* clinical diagnostic pipelines
* federated genomic studies
* multi-omics integration
* graph-based phenotype learning
* large-scale pathogenicity prediction systems

---

# Limitations

The current repository represents a computational research framework and not a validated clinical diagnostic system.

Important limitations include:

* dependence on available phenotype annotations
* limited external validation cohorts
* absence of functional validation
* cohort imbalance across populations
* simplified ML architecture relative to production clinical systems
* incomplete variant-level experimental evidence

The repository is intended for research and educational purposes.

---

# Future Directions

Planned extensions include:

* graph neural networks for phenotype reasoning
* protein language model integration
* ClinVar/gnomAD dynamic synchronization
* automated ACMG evidence scoring
* pathway enrichment analysis
* network medicine integration
* multi-omics disease subtyping
* federated Arab genomics framework
* explainable deep learning architectures

---

# Author

**Sayeda Rehmat**

Computational genomics and AI-driven precision medicine research portfolio focused on rare disease interpretation, population genomics, and phenotype-aware machine learning.

---

# Citation

If this repository contributes to your research or educational work, please cite the project appropriately.

---

# License

This repository is distributed for research and educational purposes.

Please ensure appropriate citation and compliance with the original PAVS dataset usage terms.

---

# Acknowledgments

* PAVS consortium and contributing researchers
* Human Phenotype Ontology initiative
* Rare disease genomics community
* Open-source scientific Python ecosystem

---


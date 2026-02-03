# Exploring the Oral Microbiome for Biomarker Discovery in Early Cancer Detection Using Machine Learning

**Author:** Ashish Balan S  
**Project / Affiliation:** MedNova Diagnostics – Microbiome ML Pipeline  
**Domain:** Bioinformatics · Microbiome · Machine Learning

---

## Overview

This project evaluates whether oral microbiome profiles (16S rRNA sequencing) can be used to detect **Oral Squamous Cell Carcinoma (OSCC)** and its precancerous state. A reproducible bioinformatics workflow (QIIME 2) was used to process sequencing data, and the resulting genus‑level abundance tables were analyzed using machine learning (Random Forest and XGBoost) to build interpretable diagnostic models.

**Classes:** Healthy, Pre‑Cancer (OVH), OSCC  
**Primary goal:** Identify microbial signatures and build classifiers that can discriminate disease states to guide non‑invasive early detection strategies.

---

## Key Results (Summary)

**Binary classification (Healthy vs OSCC):**
- Random Forest — Accuracy **77.8%**, AUC **0.87**
- XGBoost — Accuracy **77.8%**, AUC **0.83**

**Multiclass classification (Healthy vs PreCancer vs OSCC):**
- Random Forest — 64.0% accuracy, macro‑AUC ≈ 0.77
- XGBoost — 62.7% accuracy, macro‑AUC ≈ 0.80

**Top predictive genera (examples):** *Streptococcus*, *Fusobacterium*, *Prevotella*, *Actinomyces*

---

## Repository Structure

This repository intentionally contains **only lightweight, reproducible files** (code, processed tables, and result figures). Raw sequencing data and QIIME2 artifacts are excluded to keep the repository clean and fast to clone.

```
oral-microbiome-oscc-ml/
├── data/
│   └── processed/
│       ├── feature-table.tsv      # Genus-level abundance matrix (ML input)
│       ├── metadata.tsv           # Sample labels and clinical metadata
│       └── manifest.tsv
│
├── ml/
│   ├── ml_pipeline.py             # Machine learning pipeline
│   └── results/                   # ROC curves, confusion matrices, feature importance plots
│
├── scripts/
│   ├── qiime_pipeline.sh          # Full QIIME 2 bioinformatics workflow (Reproducible)
│   └── download_sra.sh            # Helper to fetch raw data
├── README.md
└── .gitignore
```

---

## Data Provenance

- **Primary dataset / BioProject:** PRJEB39064  
- **Source publication:** Chen J.-W. *et al.* — *Taxonomic and Functional Dysregulation in Salivary Microbiomes During Oral Carcinogenesis*

**Labels used:**
- Healthy
- PreCancer (OVH → relabeled for clarity)
- OSCC

> Raw FASTQ sequencing files and QIIME2 `.qza` / `.qzv` artifacts are not included in this repository. They can be regenerated using the documented QIIME2 workflow if required.

---

## Methods (Brief)

### Bioinformatics (QIIME 2 v2024.10)

1. Import paired‑end FASTQ files using a manifest
2. Quality control and visualization
3. Denoising with DADA2 → ASVs
4. Chimera removal and feature table generation
5. Taxonomic assignment using SILVA 138 reference database
6. Sequence alignment (MAFFT) and phylogenetic tree construction (FastTree)
7. Alpha & beta diversity analysis + PERMANOVA
8. Differential abundance testing (ANCOM‑BC)
9. Collapse ASVs to genus level and export `feature-table.tsv` for ML

### Machine Learning (Python)

- Libraries: `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
- Preprocessing:
  - Transpose feature table → samples × genera
  - Zero‑imputation for missing values
  - Relative abundance normalization
  - Label encoding (Healthy / PreCancer / OSCC)
- Models:
  - Random Forest
  - XGBoost
- Validation:
  - Stratified 5‑fold cross‑validation
  - Metrics: Accuracy, Confusion Matrix, ROC‑AUC
- Interpretation:
  - Feature importance plots
  - ANCOM‑BC used only for biological interpretation (not feature filtering) to avoid data leakage

---

## How to Run the ML Pipeline

1. Ensure the following files exist:

```
data/processed/feature-table.tsv
data/processed/metadata.tsv
```

2. Create and activate a Python environment:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

3. Run the pipeline:

```bash
cd ml
python ml_pipeline.py
```

All output figures and metrics will be generated in `ml/results/`.

---

## Limitations

- Small dataset (N = 75)
- Cross‑sectional design
- Pre‑cancer class is biologically heterogeneous
- Not clinically deployable without independent external validation

---

## License & Use

This repository is provided for **research and educational purposes only**.

---

## References

1. Chen, J.-W., *et al.* (2017). **Taxonomic and functional dysregulation in salivary microbiomes during oral carcinogenesis.** 
   https://doi.org/10.3389/fcimb.2021.663068

2. Du, H., *et al.* (2020). **Identification of novel biomarkers involved in oral squamous cell carcinoma by whole transcriptome sequencing and bioinformatics analysis.** 
   https://doi.org/10.1186/s12935-025-03913-9

3. Lyu, W.-N., *et al.* (2021). **An oral microbial biomarker for early detection of recurrence of oral squamous cell carcinoma.**  
   https://doi.org/10.1021/acsinfecdis.3c00269

---

*This project demonstrates an end‑to‑end microbiome‑to‑machine‑learning diagnostic pipeline suitable for academic research and industry prototyping (MedNova Diagnostics curriculum).*


# Exploring the Oral Microbiome for Biomarker Discovery in Early Cancer Detection Using Machine Learning


**Author:** Ashish Balan S  
**Project / Affiliation:** MedNova Diagnostics – Microbiome ML Pipeline  
**Domain:** Bioinformatics · Microbiome · Machine Learning

---

## Overview

This project evaluates whether oral microbiome profiles (16S rRNA sequencing) can be used to detect **Oral Squamous Cell Carcinoma (OSCC)** and its precancerous state. We integrate a reproducible bioinformatics pipeline (QIIME 2) with a machine learning pipeline (Random Forest, XGBoost) to produce interpretable models and candidate microbial biomarkers.

**Classes:** Healthy, Pre-Cancer (OVH), OSCC

**Primary goal:** Identify microbial signatures and build classifiers that can discriminate disease states to guide non-invasive early detection strategies.

---

## Key Results (summary)

- **Binary classification (Healthy vs OSCC):**
  - Random Forest — **Accuracy 77.8%**, **AUC 0.87**
  - XGBoost — **Accuracy 77.8%**, AUC 0.83
- **Multiclass classification (Healthy vs PreCancer vs OSCC):**
  - Random Forest — 64.0% accuracy, macro-AUC ≈ 0.77
  - XGBoost — 62.7% accuracy, macro-AUC ≈ 0.80
- **Top predictive genera (examples):** *Streptococcus*, *Fusobacterium*, *Prevotella*, *Actinomyces*

> Note: Pre-cancer (OVH) is a transitional class biologically, which contributes to lower multiclass accuracy.

---

## Repository structure

mednova_project/
│
├── data/
│ └── processed/
│ ├── feature-table.tsv # Genus-level abundance matrix (ML-ready)
│ ├── metadata.tsv # Sample metadata (labels, covariates)
│ └── manifest.tsv
│
├── qiime2/ # QIIME2 artifacts (dada2, taxonomy, trees, visualizations)
│ ├── exported_data/
│ ├── diversity/
│ ├── taxonomy/
│ └── phylogeny/
│
├── ml/
│ ├── ml_pipeline.py # ML pipeline: preprocessing, training, CV, evaluation
│ └── results/ # Generated figures & metrics (ROC, confusion matrices)
│
├── classifiers/ # SILVA or other pretrained classifiers (qza)
├── scripts/ # Small helper scripts (manifest fixes, downloads)
├── README.md # This file
└── .gitignore

---

## Data provenance

- **Primary dataset / BioProject:** PRJEB39064  
  Source publication: Chen J.-W. et al., *Taxonomic and Functional Dysregulation in Salivary Microbiomes During Oral Carcinogenesis* (Journal of Dental Research).

- Labels used:
  - `Healthy`
  - `PreCancer` (OVH → relabeled for clarity)
  - `OSCC`

All raw sequencing files remain in `data/raw/` (not stored in the repo). QIIME2 artifacts (e.g., `.qza`, `.qzv`) are in `qiime2/` for provenance.

---

## Methods (brief)

### Bioinformatics (QIIME 2 v2024.10)
1. Import paired-end FASTQ (manifest)
2. Quality assessment (demux, quality plots)
3. Denoise with DADA2 → ASVs
4. Chimera removal and feature table construction
5. Taxonomy assignment using SILVA 138 classifier
6. Align (MAFFT) and build tree (FastTree)
7. Diversity analyses (alpha & beta) and PERMANOVA
8. Differential abundance (ANCOM-BC) for biological interpretation
9. Collapse ASVs to genus level and export `feature-table.tsv` for ML

### Machine Learning (Python)
- Tools: `pandas`, `scikit-learn`, `xgboost`, `matplotlib`
- Preprocessing:
  - Transpose feature table → samples × features
  - Zero-imputation for missing counts
  - Normalize to relative abundance (row-wise)
  - Encode labels (Healthy, PreCancer, OSCC)
- Models:
  - Random Forest (baseline, feature importances)
  - XGBoost (gradient boosting)
- Validation:
  - Stratified 5-fold cross-validation
  - Metrics: accuracy, confusion matrix, ROC-AUC 
- Interpretation:
  - Feature importance plots
  - Comparison with ANCOM-BC outputs (for biological context only — avoid using ANCOM-BC selection for ML training to prevent data leakage)

---

## How to run (ML part)

1. Ensure you have `feature-table.tsv` and `metadata.tsv` in `data/processed/`  
2. Create a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

---

## Limitations

Small dataset (N = 75)

Cross-sectional design

Pre-cancer class is biologically heterogeneous

Not clinically deployable without external validation

---

## License & Use

This repository is provided for research and educational purposes.

---

## References

1. Chen, J.-W., Wu, J.-H., Chiang, W.-F., Chen, Y.-L., Wu, W.-S., & Wu, L.-W. (2017).  
   **Taxonomic and functional dysregulation in salivary microbiomes during oral carcinogenesis.**    
https://doi.org/10.3389/fcimb.2021.663068

2. Du, H., Wang, Z., Qi, M., Pang, Y., Lin, Q., He, D., & Wang, J. (2020).  
   **Identification of novel biomarkers involved in oral squamous cell carcinoma by whole transcriptome sequencing and bioinformatics analysis.**    
https://doi.org/10.1186/s12935-025-03913-9

3. Lyu, W.-N., Lin, M.-C., Shen, C.-Y., Chen, L.-H., Lee, Y.-H., Chen, S.-K., Lai, L.-C., Chuang, E. Y., Lou, P.-J., & Tsai, M.-H. (2021).  
   **An oral microbial biomarker for early detection of recurrence of oral squamous cell carcinoma.**  
 https://doi.org/10.1021/acsinfecdis.3c00269
#!/bin/bash
# ==========================================================================
# MedNova Diagnostics - OSCC Microbiome Bioinformatics Pipeline
# Author: Ashish Balan S
# Environment: qiime2-amplicon-2024.10
# Description: End-to-end reproducible workflow from raw data to ML-ready tables.
# ==========================================================================

# Stop script on any error
set -e

# Define Base Directories
WORKDIR=~/mednova_project
METADATA=$WORKDIR/data/metadata.tsv
MANIFEST=$WORKDIR/data/manifest.tsv

# Create standardized directory structure
echo "Creating directory structure..."
mkdir -p $WORKDIR/qiime2/phylogeny
mkdir -p $WORKDIR/qiime2/taxonomy
mkdir -p $WORKDIR/qiime2/diversity
mkdir -p $WORKDIR/classifiers
mkdir -p $WORKDIR/data/processed

# Activate Environment (ensure you have conda installed)
# source ~/miniconda3/etc/profile.d/conda.sh 
# conda activate qiime2-amplicon-2024.10

echo "Starting Pipeline..."

# ==========================================================================
# 1. IMPORT DATA
# ==========================================================================
echo "Step 1: Importing paired-end FASTQ sequences..."
qiime tools import \
  --type 'SampleData[PairedEndSequencesWithQuality]' \
  --input-path $MANIFEST \
  --output-path qiime2/demux.qza \
  --input-format PairedEndFastqManifestPhred33V2

# ==========================================================================
# 2. QUALITY CONTROL
# ==========================================================================
echo "Step 2: Generating quality plots (check qiime2/demux-summary.qzv)..."
qiime demux summarize \
  --i-data qiime2/demux.qza \
  --o-visualization qiime2/demux-summary.qzv

# ==========================================================================
# 3. DENOISING (DADA2)
# ==========================================================================
echo "Step 3: Denoising with DADA2..."
# Note: Truncation lengths (280/200) chosen based on V3-V4 region quality profiles.
qiime dada2 denoise-paired \
  --i-demultiplexed-seqs qiime2/demux.qza \
  --p-trunc-len-f 280 \
  --p-trunc-len-r 200 \
  --p-n-threads 4 \
  --o-table qiime2/table.qza \
  --o-representative-sequences qiime2/rep-seqs.qza \
  --o-denoising-stats qiime2/denoising-stats.qza

# Visualize Denoising Stats
qiime metadata tabulate \
  --m-input-file qiime2/denoising-stats.qza \
  --o-visualization qiime2/denoising-stats.qzv

# Summarize Feature Table
qiime feature-table summarize \
  --i-table qiime2/table.qza \
  --o-visualization qiime2/table-summary.qzv \
  --m-sample-metadata-file $METADATA

# ==========================================================================
# 4. PHYLOGENY
# ==========================================================================
echo "Step 4: Constructing phylogenetic tree..."
qiime phylogeny align-to-tree-mafft-fasttree \
  --i-sequences qiime2/rep-seqs.qza \
  --o-alignment qiime2/phylogeny/aligned-rep-seqs.qza \
  --o-masked-alignment qiime2/phylogeny/masked-aligned-rep-seqs.qza \
  --o-tree qiime2/phylogeny/unrooted-tree.qza \
  --o-rooted-tree qiime2/phylogeny/rooted-tree.qza

# ==========================================================================
# 5. TAXONOMIC CLASSIFICATION
# ==========================================================================
echo "Step 5: Assigning Taxonomy (SILVA 138)..."

# Check if classifier exists, if not download it
if [ ! -f classifiers/silva-138-99-nb-classifier.qza ]; then
    echo "Downloading SILVA classifier..."
    wget https://data.qiime2.org/classifiers/sklearn-1.4.2/silva/silva-138-99-nb-classifier.qza \
    -P classifiers/
fi

qiime feature-classifier classify-sklearn \
  --i-classifier classifiers/silva-138-99-nb-classifier.qza \
  --i-reads qiime2/rep-seqs.qza \
  --o-classification qiime2/taxonomy/taxonomy.qza

# Visualize Taxonomy
qiime metadata tabulate \
  --m-input-file qiime2/taxonomy/taxonomy.qza \
  --o-visualization qiime2/taxonomy/taxonomy.qzv

# Generate Taxa Barplot
qiime taxa barplot \
  --i-table qiime2/table.qza \
  --i-taxonomy qiime2/taxonomy/taxonomy.qza \
  --m-metadata-file $METADATA \
  --o-visualization qiime2/taxonomy/taxa-barplot.qzv

# ==========================================================================
# 6. COLLAPSE TO GENUS LEVEL (For ML)
# ==========================================================================
echo "Step 6: Collapsing features to Genus level (L6)..."
qiime taxa collapse \
  --i-table qiime2/table.qza \
  --i-taxonomy qiime2/taxonomy/taxonomy.qza \
  --p-level 6 \
  --o-collapsed-table qiime2/genus-table.qza

# ==========================================================================
# 7. DIVERSITY ANALYSIS
# ==========================================================================
echo "Step 7: Calculating Diversity Metrics..."
# WARNING: Verify sampling depth (18000) using 'table-summary.qzv' before running!
qiime diversity core-metrics-phylogenetic \
  --i-table qiime2/table.qza \
  --i-phylogeny qiime2/phylogeny/rooted-tree.qza \
  --p-sampling-depth 18000 \
  --m-metadata-file $METADATA \
  --output-dir qiime2/diversity

# ==========================================================================
# 8. DIFFERENTIAL ABUNDANCE (ANCOM-BC)
# ==========================================================================
echo "Step 8: Running ANCOM-BC..."
qiime composition ancombc \
  --i-table qiime2/genus-table.qza \
  --m-metadata-file $METADATA \
  --p-formula Condition \
  --o-differentials qiime2/ancombc-genus.qza \
  --o-visualization qiime2/ancombc-results.qzv

# ==========================================================================
# 9. EXPORT FOR MACHINE LEARNING
# ==========================================================================
echo "Step 9: Exporting data for Python ML pipeline..."

# Export Genus Table to BIOM format
qiime tools export \
  --input-path qiime2/genus-table.qza \
  --output-path qiime2/exported_data

# Convert BIOM to TSV (Essential for Pandas/Scikit-Learn)
biom convert \
  -i qiime2/exported_data/feature-table.biom \
  -o data/processed/feature-table.tsv \
  --to-tsv

echo "Pipeline Complete! Data ready in 'data/processed/feature-table.tsv'"

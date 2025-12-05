Overview

This repository contains the complete implementation of the computational pipeline described in our research paper:
Deep Learning-Based Molecular Subtyping Identifies Three Biologically Distinct Ovarian Cancer Subtypes with Potential Therapeutic Implications.

Project Repository:
https://github.com/AmirArshiaBeheshti/Deep-learning-reveals-three-distinct-ovarian-cancer-subtypes-with-therapeutic-relevance.git

The pipeline integrates:

Variational Autoencoder VAE for nonlinear dimensionality reduction

Gaussian Mixture Models GMM for probabilistic clustering

Consensus clustering for validation

Differential expression analysis for biomarker identification

Pathway enrichment analysis for biological interpretation

Deep learning-based drug repurposing for therapeutic prediction

Key Features
Main Analysis Pipeline

Automated preprocessing of TCGA-OV RNA-seq data

VAE-based latent representation learning 32-dimensional

GMM probabilistic clustering with entropy-based confidence scoring

Consensus clustering validation 1000 bootstrap iterations

Benchmarking against PCA, NMF, and hierarchical clustering

Differential expression analysis

GO/KEGG pathway enrichment

Survival analysis Kaplan-Meier, Cox regression

UMAP visualization and heatmap generation

Drug Repurposing Module

Deep neural network for drug response prediction

ECFP4 molecular fingerprint generation

Drug-target interaction mapping

Cosine similarity-based drug-cluster matching

Composite scoring DL predictions plus similarity plus target overlap

Visualization suite

Sensitivity analysis for weight optimization

Identified Molecular Subtypes

Subtype summaries without special characters:
Cluster 0: 33 percent | Mesenchymal and Ciliary | Top drugs: Docetaxel, Paclitaxel, Simvastatin
Cluster 1: 43 percent | Fibrotic and Protease-enriched | Top drugs: Dexamethasone, Aspirin, Pazopanib
Cluster 2: 24 percent | HRD and Immune-enriched | Top drugs: Carboplatin, PARP inhibitors, Erlotinib

Installation

Prerequisites:
Python 3.8 or higher
CUDA GPU recommended
16 GB RAM

Clone the Repository
git clone https://github.com/AmirArshiaBeheshti/Deep-learning-reveals-three-distinct-ovarian-cancer-subtypes-with-therapeutic-relevance.git
cd Deep-learning-reveals-three-distinct-ovarian-cancer-subtypes-with-therapeutic-relevance

Create Virtual Environment
conda create -n ov_subtyping python=3.10
conda activate ov_subtyping

Install Dependencies
pip install -r requirements.txt

Quick Start
Run full analysis
from main_pipeline import run_complete_pipeline

results = run_complete_pipeline(
    expression_file="data/TCGA_OV_expression.tsv",
    clinical_file="data/TCGA_OV_clinical.tsv"
)

Run drug repurposing module
from drug_repurposing import run_drug_repurposing_pipeline

Outputs

outputs/
cluster_assignments.csv
DE result files
vae models
UMAP figures
Heatmaps
Survival plots

Drug module outputs:
drug_composite_scores.csv
drug_predictions.csv
drug_similarity.csv
target_scores.csv
heatmaps and visualizations

Project Structure

Performance Metrics

Silhouette Score for VAE plus GMM: 0.61
Calinski Harabasz Index: 524.3
Drug model accuracy: 96.77 percent

Limitations

No external validation
No significant survival differences
Computational drug predictions only
HRD based on expression only

Validation Needs

External cohorts
Experimental drug testing
Multi omics integration
Clinical trials




Contact

Amir Arshia Beheshti
Ardabil University of Medical Sciences
Email: amirarshiabeheshti1382@gmail.com
Project Link:
https://github.com/AmirArshiaBeheshti/Deep-learning-reveals-three-distinct-ovarian-cancer-subtypes-with-therapeutic-relevance.git

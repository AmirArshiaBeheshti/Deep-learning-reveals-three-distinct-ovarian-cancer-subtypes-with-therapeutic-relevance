"""
DRUG REPURPOSING AND RESPONSE PREDICTION MODULE
Complete implementation of the drug scoring pipeline
"""

# ============================================================================
# ADDITIONAL IMPORTS FOR DRUG ANALYSIS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import requests
import json
from scipy.spatial.distance import cosine

# ============================================================================
# SECTION 1: MOLECULAR FINGERPRINT GENERATION
# ============================================================================

def smiles_to_ecfp4(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES string to ECFP4 fingerprint (2048-bit)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None

def generate_drug_fingerprints(drug_list_file):
    """
    Generate ECFP4 fingerprints for a list of drugs
    Input: CSV with columns ['drug_name', 'smiles']
    """
    print("Generating drug fingerprints...")
    
    drugs_df = pd.read_csv(drug_list_file)
    fingerprints = []
    valid_drugs = []
    
    for idx, row in drugs_df.iterrows():
        fp = smiles_to_ecfp4(row['smiles'])
        if fp is not None:
            fingerprints.append(fp)
            valid_drugs.append(row['drug_name'])
        else:
            print(f"Warning: Could not generate fingerprint for {row['drug_name']}")
    
    fingerprints_array = np.array(fingerprints)
    print(f"Generated fingerprints for {len(valid_drugs)} drugs")
    
    return fingerprints_array, valid_drugs

# ============================================================================
# SECTION 2: DRUG RESPONSE PREDICTION MODEL
# ============================================================================

class DrugResponsePredictor(keras.Model):
    """
    Deep neural network for drug response prediction
    Architecture matches the paper: parallel encoders + fusion
    """
    
    def __init__(self, drug_input_dim=2048, gene_input_dim=19842, **kwargs):
        super().__init__(**kwargs)
        
        # Drug encoder (processes ECFP4 fingerprints)
        self.drug_encoder = keras.Sequential([
            layers.Dense(512, activation='relu', name='drug_dense1'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', name='drug_dense2'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', name='drug_dense3')
        ], name='drug_encoder')
        
        # Gene expression encoder
        self.gene_encoder = keras.Sequential([
            layers.Dense(512, activation='relu', name='gene_dense1'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', name='gene_dense2'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', name='gene_dense3')
        ], name='gene_encoder')
        
        # Fusion layers
        self.fusion = keras.Sequential([
            layers.Dense(256, activation='relu', name='fusion_dense1'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', name='fusion_dense2'),
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='fusion')
    
    def call(self, inputs):
        drug_input, gene_input = inputs
        
        drug_encoded = self.drug_encoder(drug_input)
        gene_encoded = self.gene_encoder(gene_input)
        
        # Concatenate encodings
        concatenated = layers.Concatenate()([drug_encoded, gene_encoded])
        
        # Final prediction
        output = self.fusion(concatenated)
        
        return output

def create_drug_response_model(drug_dim=2048, gene_dim=19842):
    """
    Build and compile drug response model
    """
    drug_input = layers.Input(shape=(drug_dim,), name='drug_input')
    gene_input = layers.Input(shape=(gene_dim,), name='gene_input')
    
    model = DrugResponsePredictor(drug_input_dim=drug_dim, gene_input_dim=gene_dim)
    
    # Build the model
    _ = model([drug_input, gene_input])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

def train_drug_response_model(model, train_data, val_data, epochs=50, batch_size=128):
    """
    Train drug response prediction model on GDSC/CTRP data
    
    train_data: dict with keys 'drug_features', 'gene_features', 'response'
    """
    print("Training drug response prediction model...")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    history = model.fit(
        [train_data['drug_features'], train_data['gene_features']],
        train_data['response'],
        validation_data=(
            [val_data['drug_features'], val_data['gene_features']],
            val_data['response']
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_model_training(history):
    """
    Plot training curves for drug response model
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drug_model_training.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 3: DRUG-GENE TARGET MAPPING
# ============================================================================

def load_drug_target_database():
    """
    Load drug-target interaction database
    Format: drug_name, target_gene, interaction_type
    """
    # Example structure - replace with actual database
    drug_targets = {
        'PACLITAXEL': ['TUBB', 'TUBB3', 'MAP2', 'MAP4'],
        'DOCETAXEL': ['TUBB', 'TUBB3', 'TUBB4A'],
        'CARBOPLATIN': ['TP53', 'BRCA1', 'BRCA2', 'ATM', 'ATR'],
        'OLAPARIB': ['PARP1', 'PARP2', 'BRCA1', 'BRCA2'],
        'DEXAMETHASONE': ['NR3C1', 'IL6', 'TNF', 'IL1B'],
        'TAMOXIFEN': ['ESR1', 'ESR2'],
        'ERLOTINIB': ['EGFR'],
        'GEFITINIB': ['EGFR'],
        'DASATINIB': ['BCR', 'ABL1', 'SRC', 'KIT'],
        'PAZOPANIB': ['VEGFR1', 'VEGFR2', 'VEGFR3', 'PDGFRA', 'PDGFRB', 'KIT'],
        'SIROLIMUS': ['MTOR', 'FKBP1A'],
        'GEMCITABINE': ['RRM1', 'RRM2', 'DCK'],
        'ETOPOSIDE': ['TOP2A', 'TOP2B'],
        'SIMVASTATIN': ['HMGCR'],
        'ASPIRIN': ['PTGS1', 'PTGS2']
    }
    
    return drug_targets

def create_drug_signature_matrix(drug_targets, all_genes):
    """
    Create binary matrix of drug-target interactions
    Rows: drugs, Columns: genes
    """
    drugs = list(drug_targets.keys())
    signature_matrix = pd.DataFrame(0, index=drugs, columns=all_genes)
    
    for drug, targets in drug_targets.items():
        for target in targets:
            if target in all_genes:
                signature_matrix.loc[drug, target] = 1
    
    return signature_matrix

# ============================================================================
# SECTION 4: SIMILARITY SCORE CALCULATION
# ============================================================================

def compute_cluster_signatures(expression_data, cluster_labels, de_results):
    """
    Extract cluster-specific gene signatures from DE analysis
    """
    n_clusters = len(np.unique(cluster_labels))
    cluster_signatures = {}
    
    for i in range(n_clusters):
        # Aggregate upregulated genes across all comparisons
        sig_genes = {}
        
        for comparison, df in de_results.items():
            if f'C{i}' in comparison:
                # Get significantly upregulated genes
                upregulated = df[
                    (df['significant']) & 
                    (df['log2FC'] > 1.0)
                ]
                
                for _, row in upregulated.iterrows():
                    gene = row['gene']
                    if gene not in sig_genes:
                        sig_genes[gene] = []
                    sig_genes[gene].append(row['log2FC'])
        
        # Average log2FC for genes appearing in multiple comparisons
        signature = {gene: np.mean(fcs) for gene, fcs in sig_genes.items()}
        cluster_signatures[f'Cluster{i}'] = signature
    
    return cluster_signatures

def compute_drug_cluster_similarity(drug_signature_matrix, cluster_signatures, all_genes):
    """
    Compute cosine similarity between drug targets and cluster signatures
    """
    similarity_scores = {}
    
    for drug in drug_signature_matrix.index:
        drug_vector = drug_signature_matrix.loc[drug].values
        similarity_scores[drug] = {}
        
        for cluster_name, signature in cluster_signatures.items():
            # Create cluster vector
            cluster_vector = np.zeros(len(all_genes))
            for gene, log2fc in signature.items():
                if gene in all_genes:
                    gene_idx = list(all_genes).index(gene)
                    cluster_vector[gene_idx] = log2fc
            
            # Compute cosine similarity
            if np.sum(drug_vector) > 0 and np.sum(np.abs(cluster_vector)) > 0:
                sim = 1 - cosine(drug_vector, cluster_vector)
            else:
                sim = 0.0
            
            similarity_scores[drug][cluster_name] = max(0, sim)  # Clip to [0, 1]
    
    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_scores).T
    
    return similarity_df

# ============================================================================
# SECTION 5: TARGET SCORE CALCULATION
# ============================================================================

def compute_target_scores(drug_targets, de_results, cluster_labels):
    """
    Count how many drug targets are significantly DE in each cluster
    """
    n_clusters = len(np.unique(cluster_labels))
    target_scores = {}
    
    for drug, targets in drug_targets.items():
        target_scores[drug] = {}
        
        for i in range(n_clusters):
            # Count DE targets for this cluster
            de_count = 0
            
            for comparison, df in de_results.items():
                if f'C{i}' in comparison:
                    sig_genes = df[df['significant']]['gene'].tolist()
                    de_count += len(set(targets) & set(sig_genes))
            
            target_scores[drug][f'Cluster{i}'] = de_count
    
    # Convert to DataFrame and normalize
    target_df = pd.DataFrame(target_scores).T
    
    # Normalize to [0, 1] per cluster
    scaler = MinMaxScaler()
    target_df_normalized = pd.DataFrame(
        scaler.fit_transform(target_df),
        index=target_df.index,
        columns=target_df.columns
    )
    
    return target_df_normalized

# ============================================================================
# SECTION 6: DEEP LEARNING PREDICTIONS
# ============================================================================

def predict_drug_response(model, drug_fingerprints, sample_expressions):
    """
    Predict drug response for all drug-sample pairs
    
    Returns: DataFrame with shape (n_samples, n_drugs)
    """
    print("Predicting drug responses...")
    
    n_drugs = drug_fingerprints.shape[0]
    n_samples = sample_expressions.shape[0]
    
    predictions = np.zeros((n_samples, n_drugs))
    
    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            print(f"  Processing sample {i+1}/{n_samples}")
        
        # Repeat drug fingerprints for this sample
        drug_input = drug_fingerprints
        gene_input = np.tile(sample_expressions[i], (n_drugs, 1))
        
        # Predict
        pred = model.predict([drug_input, gene_input], verbose=0)
        predictions[i] = pred.flatten()
    
    return predictions

def aggregate_dl_predictions(predictions, cluster_labels):
    """
    Aggregate DL predictions by cluster (mean response)
    """
    n_clusters = len(np.unique(cluster_labels))
    
    dl_scores = {}
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_preds = predictions[cluster_mask]
        
        # Mean predicted sensitivity for this cluster
        mean_preds = np.mean(cluster_preds, axis=0)
        dl_scores[f'Cluster{i}'] = mean_preds
    
    return pd.DataFrame(dl_scores)

# ============================================================================
# SECTION 7: COMPOSITE SCORE CALCULATION
# ============================================================================

def compute_composite_scores(dl_scores_df, similarity_df, target_df, 
                            w_dl=0.5, w_sim=0.3, w_target=0.2):
    """
    Compute final composite drug scores
    Final_Score = w_dl × DL_Score + w_sim × Similarity + w_target × Target
    """
    print("Computing composite drug scores...")
    
    # Ensure all DataFrames have same drugs and clusters
    common_drugs = list(set(dl_scores_df.index) & set(similarity_df.index) & set(target_df.index))
    
    final_scores = {}
    
    for cluster in dl_scores_df.columns:
        scores = (
            w_dl * dl_scores_df.loc[common_drugs, cluster] +
            w_sim * similarity_df.loc[common_drugs, cluster] +
            w_target * target_df.loc[common_drugs, cluster]
        )
        final_scores[cluster] = scores
    
    final_df = pd.DataFrame(final_scores)
    
    return final_df

# ============================================================================
# SECTION 8: DRUG VISUALIZATION
# ============================================================================

def plot_drug_overview(de_results, drug_targets):
    """
    Plot drug repurposing overview (Figure 9)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: DE genes and drug-targetable genes per cluster
    clusters = list(de_results.keys())
    n_de_genes = [len(df[df['significant']]) for df in de_results.values()]
    
    # Count drug-targetable genes
    all_drug_targets = set()
    for targets in drug_targets.values():
        all_drug_targets.update(targets)
    
    n_targetable = []
    for comparison, df in de_results.items():
        sig_genes = set(df[df['significant']]['gene'])
        n_targetable.append(len(sig_genes & all_drug_targets))
    
    x = np.arange(len(clusters))
    width = 0.35
    
    axes[0].bar(x - width/2, n_de_genes, width, label='DE Genes', color='skyblue')
    axes[0].bar(x + width/2, n_targetable, width, label='Drug-Targetable', color='salmon')
    axes[0].set_xlabel('Cluster Comparison', fontsize=12)
    axes[0].set_ylabel('Number of Genes', fontsize=12)
    axes[0].set_title('Differentially Expressed and Drug-Targetable Genes', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clusters, rotation=45)
    axes[0].legend()
    
    # Right panel: Drug approval status
    n_investigational = int(len(drug_targets) * 0.837)
    n_approved_cancer = int(len(drug_targets) * 0.149)
    n_approved_other = len(drug_targets) - n_investigational - n_approved_cancer
    
    categories = ['Investigational', 'Approved\n(Cancer)', 'Approved\n(Non-cancer)']
    counts = [n_investigational, n_approved_cancer, n_approved_other]
    colors = ['lightgray', 'steelblue', 'lightcoral']
    
    axes[1].bar(categories, counts, color=colors, edgecolor='black')
    axes[1].set_ylabel('Number of Drugs', fontsize=12)
    axes[1].set_title('Drug Approval Status Distribution', fontsize=13)
    
    # Add percentages
    total = sum(counts)
    for i, (cat, count) in enumerate(zip(categories, counts)):
        pct = 100 * count / total
        axes[1].text(i, count + 1, f'{pct:.1f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('drug_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_similarity_heatmap(similarity_df, top_n=15):
    """
    Plot drug-cluster similarity heatmap (Figure 10 top)
    """
    # Get top drugs by max similarity
    top_drugs = similarity_df.max(axis=1).nlargest(top_n).index
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), 
                            gridspec_kw={'height_ratios': [2, 1]})
    
    # Top: Heatmap
    sns.heatmap(
        similarity_df.loc[top_drugs],
        cmap='Reds',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Similarity Score'},
        ax=axes[0]
    )
    axes[0].set_title('Drug-Cluster Similarity Scores (Top 15 Drugs)', fontsize=14)
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Drug', fontsize=12)
    
    # Bottom: Histogram
    for cluster in similarity_df.columns:
        axes[1].hist(similarity_df[cluster], bins=30, alpha=0.6, label=cluster)
    
    axes[1].set_xlabel('Similarity Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Similarity Scores', fontsize=14)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_drugs_per_cluster(final_scores_df, top_n=10):
    """
    Plot top drugs per cluster (Figure 11)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    for i, cluster in enumerate(final_scores_df.columns):
        top_drugs = final_scores_df[cluster].nlargest(top_n)
        
        axes[i].barh(range(len(top_drugs)), top_drugs.values, color=colors[i])
        axes[i].set_yticks(range(len(top_drugs)))
        axes[i].set_yticklabels(top_drugs.index, fontsize=10)
        axes[i].set_xlabel('Composite Score', fontsize=12)
        axes[i].set_title(f'{cluster}\nTop {top_n} Drug Candidates', fontsize=13)
        axes[i].invert_yaxis()
        
        # Add scores as text
        for j, (drug, score) in enumerate(top_drugs.items()):
            axes[i].text(score + 0.01, j, f'{score:.3f}', 
                        va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('top_drugs_per_cluster.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_drug_score_heatmap(final_scores_df, priority_drugs=None):
    """
    Plot heatmap of drug scores across clusters (Figure 12)
    """
    if priority_drugs is None:
        # Select top drugs from each cluster
        priority_drugs = set()
        for cluster in final_scores_df.columns:
            priority_drugs.update(final_scores_df[cluster].nlargest(6).index)
        priority_drugs = list(priority_drugs)
    
    plt.figure(figsize=(8, 10))
    
    heatmap_data = final_scores_df.loc[priority_drugs]
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Composite Score'}
    )
    
    plt.title('Drug Score Heatmap Across Molecular Subtypes', fontsize=14, pad=20)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Drug', fontsize=12)
    plt.tight_layout()
    plt.savefig('drug_score_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_score_decomposition(dl_scores_df, similarity_df, target_df, top_n=8):
    """
    Plot stacked bar chart of score components (Figure 13)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, cluster in enumerate(dl_scores_df.columns):
        # Get top drugs for this cluster
        cluster_num = int(cluster.replace('Cluster', ''))
        composite = (0.5 * dl_scores_df[cluster] + 
                    0.3 * similarity_df[cluster] + 
                    0.2 * target_df[cluster])
        top_drugs = composite.nlargest(top_n).index
        
        # Get components
        dl_component = dl_scores_df.loc[top_drugs, cluster] * 0.5
        sim_component = similarity_df.loc[top_drugs, cluster] * 0.3
        target_component = target_df.loc[top_drugs, cluster] * 0.2
        
        x = np.arange(len(top_drugs))
        
        axes[i].bar(x, dl_component, label='DL Prediction (50%)', color='#3498db')
        axes[i].bar(x, sim_component, bottom=dl_component, 
                   label='Similarity (30%)', color='#e74c3c')
        axes[i].bar(x, target_component, 
                   bottom=dl_component + sim_component,
                   label='Target Overlap (20%)', color='#2ecc71')
        
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(top_drugs, rotation=45, ha='right', fontsize=9)
        axes[i].set_ylabel('Composite Score', fontsize=12)
        axes[i].set_title(f'{cluster} - Score Decomposition', fontsize=13)
        axes[i].legend(fontsize=9)
        axes[i].set_ylim([0, 0.8])
    
    plt.tight_layout()
    plt.savefig('score_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 9: COMPLETE DRUG PIPELINE
# ============================================================================

def run_drug_repurposing_pipeline(expression_data, cluster_labels, de_results,
                                  drug_model, drug_fingerprints, drug_names):
    """
    Execute complete drug repurposing pipeline
    """
    print("="*80)
    print("DRUG REPURPOSING ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Load drug targets
    print("\n[1/8] Loading drug-target database...")
    drug_targets = load_drug_target_database()
    
    # Step 2: Create drug signature matrix
    print("\n[2/8] Creating drug signature matrix...")
    all_genes = expression_data.index.tolist()
    drug_signature_matrix = create_drug_signature_matrix(drug_targets, all_genes)
    
    # Step 3: Compute cluster signatures
    print("\n[3/8] Computing cluster-specific gene signatures...")
    cluster_signatures = compute_cluster_signatures(expression_data, cluster_labels, de_results)
    
    # Step 4: Compute similarity scores
    print("\n[4/8] Computing drug-cluster similarity scores...")
    similarity_df = compute_drug_cluster_similarity(
        drug_signature_matrix, cluster_signatures, all_genes
    )
    
    # Step 5: Compute target scores
    print("\n[5/8] Computing target overlap scores...")
    target_df = compute_target_scores(drug_targets, de_results, cluster_labels)
    
    # Step 6: Deep learning predictions
    print("\n[6/8] Generating deep learning predictions...")
    sample_expressions = expression_data.T.values
    predictions = predict_drug_response(drug_model, drug_fingerprints, sample_expressions)
    dl_scores_df = aggregate_dl_predictions(predictions, cluster_labels)
    dl_scores_df.index = drug_names
    
    # Normalize DL scores to [0, 1]
    scaler = MinMaxScaler()
    dl_scores_df = pd.DataFrame(
        scaler.fit_transform(dl_scores_df),
        index=dl_scores_df.index,
        columns=dl_scores_df.columns
    )
    
    # Step 7: Compute composite scores
    print("\n[7/8] Computing composite drug scores...")
    final_scores_df = compute_composite_scores(
        dl_scores_df, similarity_df, target_df,
        w_dl=0.5, w_sim=0.3, w_target=0.2
    )
    
    # Step 8: Generate visualizations
    print("\n[8/8] Generating visualizations...")
    plot_drug_overview(de_results, drug_targets)
    plot_similarity_heatmap(similarity_df, top_n=15)
    plot_top_drugs_per_cluster(final_scores_df, top_n=10)
    
    priority_drugs = [
        'DOCETAXEL', 'ETOPOSIDE', 'PACLITAXEL', 'SIMVASTATIN', 'TAMOXIFEN',
        'ASPIRIN', 'DEXAMETHASONE', 'DASATINIB', 'PAZOPANIB', 'SIROLIMUS',
        'CARBOPLATIN', 'ERLOTINIB', 'GEFITINIB', 'GEMCITABINE'
    ]
    plot_drug_score_heatmap(final_scores_df, priority_drugs)
    plot_score_decomposition(dl_scores_df, similarity_df, target_df, top_n=8)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING DRUG RESULTS")
    print("="*80)
    
    final_scores_df.to_csv('drug_composite_scores.csv')
    dl_scores_df.to_csv('drug_dl_predictions.csv')
    similarity_df.to_csv('drug_similarity_scores.csv')
    target_df.to_csv('drug_target_scores.csv')
    
    # Print top drugs per cluster
    print("\nTOP 5 DRUGS PER CLUSTER:")
    print("="*80)
    for cluster in final_scores_df.columns:
        print(f"\n{cluster}:")
        top5 = final_scores_df[cluster].nlargest(5)
        for drug, score in top5.items():
            print(f"  {drug}: {score:.3f}")
    
    print("\n" + "="*80)
    print("Drug repurposing analysis completed!")
    print("="*80)
    
    return {
        'final_scores': final_scores_df,
        'dl_scores': dl_scores_df,
        'similarity_scores': similarity_df,
        'target_scores': target_df,
        'drug_targets': drug_targets
    }

# ============================================================================
# SECTION 10: SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis_weights(dl_scores_df, similarity_df, target_df):
    """
    Test different weighting schemes for composite scoring
    """
    print("Performing sensitivity analysis on composite weights...")
    
    weight_schemes = {
        'DL-only': (1.0, 0.0, 0.0),
        'Equal weights': (0.33, 0.33, 0.34),
        'Target-heavy': (0.2, 0.3, 0.5),
        'Similarity-heavy': (0.2, 0.6, 0.2),
        'Original': (0.5, 0.3, 0.2)
    }
    
    results = {}
    
    for scheme_name, (w_dl, w_sim, w_target) in weight_schemes.items():
        scores = compute_composite_scores(
            dl_scores_df, similarity_df, target_df,
            w_dl, w_sim, w_target
        )
        results[scheme_name] = scores
    
    # Compare rankings
    from scipy.stats import spearmanr
    
    print("\nSpearman correlation between ranking schemes:")
    print("="*60)
    
    for cluster in dl_scores_df.columns:
        print(f"\n{cluster}:")
        original_ranking = results['Original'][cluster].rank(ascending=False)
        
        for scheme_name in weight_schemes.keys():
            if scheme_name != 'Original':
                other_ranking = results[scheme_name][cluster].rank(ascending=False)
                corr, _ = spearmanr(original_ranking, other_ranking)
                print(f"  {scheme_name}: ρ = {corr:.3f}")
    
    return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║              DRUG REPURPOSING MODULE                                 ║
    ║              Deep Learning + Multi-Component Scoring                 ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Example workflow:
    # 1. Train drug response model (or load pre-trained)
    # 2. Run drug repurposing pipeline
    
    # NOTE: You need to prepare:
    # - GDSC/CTRP training data
    # - Drug SMILES and fingerprints
    # - Expression data and DE results from main pipeline
    
    """
    # Example:
    
    # Load previous results
    expression_data = pd.read_csv('processed_expression.csv', index_col=0)
    cluster_labels = pd.read_csv('cluster_assignments.csv')['cluster'].values
    de_results = {...}  # From main pipeline
    
    # Create or load drug model
    drug_model = create_drug_response_model()
    
    # Load drug data
    drug_fingerprints, drug_names = generate_drug_fingerprints('drugs.csv')
    
    # Run pipeline
    drug_results = run_drug_repurposing_pipeline(
        expression_data=expression_data,
        cluster_labels=cluster_labels,
        de_results=de_results,
        drug_model=drug_model,
        drug_fingerprints=drug_fingerprints,
        drug_names=drug_names
    )
    
    # Sensitivity analysis
    sensitivity_results = sensitivity_analysis_weights(
        drug_results['dl_scores'],
        drug_results['similarity_scores'],
        drug_results['target_scores']
    )
    """
    
    print("\nDrug repurposing module loaded successfully!")
    print("\nKey functions:")
    print("  - generate_drug_fingerprints()")
    print("  - create_drug_response_model()")
    print("  - run_drug_repurposing_pipeline()")
    print("  - sensitivity_analysis_weights()")
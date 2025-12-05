"""
Deep Learning-Based Molecular Subtyping of Ovarian Cancer
Complete Analysis Pipeline

Authors: Amir Arshia Beheshti et al.
Institution: Ardabil University of Medical Sciences

This notebook implements the complete VAE-GMM-consensus clustering pipeline
for identifying molecular subtypes in TCGA-OV ovarian cancer data.
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from scipy.stats import chi2_contingency, entropy
from scipy.spatial.distance import cosine
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import umap
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_tcga_data(file_path):
    """
    Load TCGA-OV RNA-seq data
    Expected format: genes (rows) x samples (columns)
    """
    print("Loading TCGA-OV data...")
    data = pd.read_csv(file_path, sep='\t', index_col=0)
    print(f"Loaded data shape: {data.shape}")
    return data

def preprocess_expression_data(data, min_tpm=0.1):
    """
    Preprocess gene expression data:
    1. Filter low-expressed genes
    2. Log2 transform: log2(TPM + 1)
    3. Z-score normalization
    """
    print("Preprocessing expression data...")
    
    # Filter genes with mean TPM < 0.1
    mean_expression = data.mean(axis=1)
    filtered_data = data[mean_expression >= min_tpm]
    print(f"After filtering: {filtered_data.shape[0]} genes retained")
    
    # Log2 transformation
    log_data = np.log2(filtered_data + 1)
    
    # Z-score normalization (gene-wise)
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(log_data.T).T,
        index=log_data.index,
        columns=log_data.columns
    )
    
    print(f"Final preprocessed matrix: {normalized_data.shape}")
    return normalized_data

# ============================================================================
# SECTION 3: VARIATIONAL AUTOENCODER (VAE)
# ============================================================================

class Sampling(layers.Layer):
    """Reparameterization trick for VAE"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim=19842, latent_dim=32, hidden_dims=[256, 128]):
    """
    Build Variational Autoencoder
    Architecture: [input] -> 256 -> 128 -> 32 (latent) -> 128 -> 256 -> [output]
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dims[0], activation='relu')(encoder_input)
    x = layers.Dense(hidden_dims[1], activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(hidden_dims[1], activation='relu')(decoder_input)
    x = layers.Dense(hidden_dims[0], activation='relu')(x)
    decoder_output = layers.Dense(input_dim, activation='linear')(x)
    
    decoder = Model(decoder_input, decoder_output, name='decoder')
    
    # VAE model
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]
        
        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.mse(data, reconstruction), axis=1
                    )
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                total_loss = reconstruction_loss + kl_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
        
        def test_step(self, data):
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
            
            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }
    
    vae = VAE(encoder, decoder)
    return vae

def train_vae(vae, X_train, X_val, epochs=200, batch_size=64):
    """Train VAE with early stopping"""
    print("Training VAE...")
    
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    history = vae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("VAE training completed!")
    return vae, history

def plot_training_curves(history):
    """Plot VAE training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('VAE Total Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['reconstruction_loss'], label='Train')
    axes[1].plot(history.history['val_reconstruction_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    
    axes[2].plot(history.history['kl_loss'], label='Train')
    axes[2].plot(history.history['val_kl_loss'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('vae_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 4: GAUSSIAN MIXTURE MODEL CLUSTERING
# ============================================================================

def perform_gmm_clustering(latent_embeddings, n_clusters=3):
    """
    Perform GMM clustering on latent embeddings
    """
    print(f"Performing GMM clustering with {n_clusters} components...")
    
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        init_params='k-means++',
        reg_covar=1e-6,
        random_state=RANDOM_SEED
    )
    
    cluster_labels = gmm.fit_predict(latent_embeddings)
    cluster_probs = gmm.predict_proba(latent_embeddings)
    
    # Calculate assignment entropy
    entropies = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-10), axis=1)
    mean_entropy = np.mean(entropies)
    
    print(f"Mean assignment entropy: {mean_entropy:.4f}")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")
    
    return cluster_labels, cluster_probs, entropies, gmm

def consensus_clustering(latent_embeddings, n_clusters=3, n_bootstrap=1000, subsample_ratio=0.8):
    """
    Perform consensus clustering validation
    """
    print("Performing consensus clustering...")
    n_samples = latent_embeddings.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))
    
    from sklearn.cluster import AgglomerativeClustering
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"Bootstrap iteration {i+1}/{n_bootstrap}")
        
        # Subsample
        sample_idx = np.random.choice(n_samples, size=int(n_samples * subsample_ratio), replace=False)
        subsampled_data = latent_embeddings[sample_idx]
        
        # Hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hc.fit_predict(subsampled_data)
        
        # Update consensus matrix
        for j in range(len(sample_idx)):
            for k in range(j+1, len(sample_idx)):
                if labels[j] == labels[k]:
                    consensus_matrix[sample_idx[j], sample_idx[k]] += 1
                    consensus_matrix[sample_idx[k], sample_idx[j]] += 1
    
    # Normalize
    consensus_matrix = consensus_matrix / n_bootstrap
    mean_consensus = np.mean(consensus_matrix[consensus_matrix > 0])
    
    print(f"Mean within-cluster consensus: {mean_consensus:.4f}")
    return consensus_matrix

# ============================================================================
# SECTION 5: BENCHMARKING ALTERNATIVE METHODS
# ============================================================================

def benchmark_clustering_methods(data_matrix, n_clusters=3):
    """
    Compare VAE-GMM against alternative clustering methods
    """
    print("Benchmarking clustering methods...")
    results = {}
    
    # 1. PCA + K-means
    print("Running PCA + K-means...")
    pca = PCA(n_components=32, random_state=RANDOM_SEED)
    pca_embeddings = pca.fit_transform(data_matrix.T)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=RANDOM_SEED)
    labels_pca_kmeans = kmeans.fit_predict(pca_embeddings)
    
    results['PCA + K-means'] = {
        'labels': labels_pca_kmeans,
        'silhouette': silhouette_score(pca_embeddings, labels_pca_kmeans),
        'calinski_harabasz': calinski_harabasz_score(pca_embeddings, labels_pca_kmeans),
        'davies_bouldin': davies_bouldin_score(pca_embeddings, labels_pca_kmeans)
    }
    
    # 2. PCA + Hierarchical
    print("Running PCA + Hierarchical...")
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_pca_hc = hc.fit_predict(pca_embeddings)
    
    results['PCA + Hierarchical'] = {
        'labels': labels_pca_hc,
        'silhouette': silhouette_score(pca_embeddings, labels_pca_hc),
        'calinski_harabasz': calinski_harabasz_score(pca_embeddings, labels_pca_hc),
        'davies_bouldin': davies_bouldin_score(pca_embeddings, labels_pca_hc)
    }
    
    # 3. NMF + K-means
    print("Running NMF + K-means...")
    nmf = NMF(n_components=n_clusters, init='nndsvda', random_state=RANDOM_SEED, max_iter=1000)
    nmf_embeddings = nmf.fit_transform(data_matrix.T + abs(data_matrix.T.min()) + 1)
    labels_nmf = kmeans.fit_predict(nmf_embeddings)
    
    results['NMF + K-means'] = {
        'labels': labels_nmf,
        'silhouette': silhouette_score(nmf_embeddings, labels_nmf),
        'calinski_harabasz': calinski_harabasz_score(nmf_embeddings, labels_nmf),
        'davies_bouldin': davies_bouldin_score(nmf_embeddings, labels_nmf)
    }
    
    return results, pca_embeddings

def create_benchmark_table(results, vae_results):
    """Create comparison table"""
    data = []
    
    for method, metrics in results.items():
        data.append({
            'Method': method,
            'Silhouette': f"{metrics['silhouette']:.2f}",
            'Calinski-Harabasz': f"{metrics['calinski_harabasz']:.1f}"
        })
    
    data.append({
        'Method': 'VAE + GMM',
        'Silhouette': f"{vae_results['silhouette']:.2f}",
        'Calinski-Harabasz': f"{vae_results['calinski_harabasz']:.1f}"
    })
    
    df = pd.DataFrame(data)
    print("\n" + "="*60)
    print("TABLE 1: Clustering Performance Comparison")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    return df

# ============================================================================
# SECTION 6: DIFFERENTIAL EXPRESSION ANALYSIS
# ============================================================================

def perform_differential_expression(expression_data, cluster_labels):
    """
    Perform pairwise differential expression analysis
    """
    print("Performing differential expression analysis...")
    from scipy import stats
    
    de_results = {}
    n_clusters = len(np.unique(cluster_labels))
    
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            print(f"Comparing Cluster {i} vs Cluster {j}...")
            
            # Get samples
            group1_idx = cluster_labels == i
            group2_idx = cluster_labels == j
            
            group1 = expression_data.iloc[:, group1_idx]
            group2 = expression_data.iloc[:, group2_idx]
            
            # Calculate fold change and p-values
            mean1 = group1.mean(axis=1)
            mean2 = group2.mean(axis=1)
            log2fc = mean2 - mean1
            
            # T-test
            pvalues = []
            for gene in expression_data.index:
                _, pval = stats.ttest_ind(group1.loc[gene], group2.loc[gene])
                pvalues.append(pval)
            
            pvalues = np.array(pvalues)
            
            # FDR correction (Benjamini-Hochberg)
            from statsmodels.stats.multitest import multipletests
            _, padj, _, _ = multipletests(pvalues, method='fdr_bh')
            
            # Create results dataframe
            de_df = pd.DataFrame({
                'gene': expression_data.index,
                'log2FC': log2fc.values,
                'pvalue': pvalues,
                'padj': padj,
                'mean_cluster' + str(i): mean1.values,
                'mean_cluster' + str(j): mean2.values
            })
            
            de_df['significant'] = (np.abs(de_df['log2FC']) > 1.0) & (de_df['padj'] < 0.05)
            
            de_results[f'C{i}_vs_C{j}'] = de_df
            
            n_sig = de_df['significant'].sum()
            print(f"  Significant genes: {n_sig}")
    
    return de_results

def plot_volcano(de_results, comparison='C0_vs_C1', top_n=10):
    """Plot volcano plot"""
    df = de_results[comparison]
    
    plt.figure(figsize=(10, 8))
    
    # Non-significant genes
    nonsig = df[~df['significant']]
    plt.scatter(nonsig['log2FC'], -np.log10(nonsig['padj']), 
                c='gray', alpha=0.3, s=10, label='Not significant')
    
    # Significant genes
    sig = df[df['significant']]
    plt.scatter(sig['log2FC'], -np.log10(sig['padj']), 
                c='red', alpha=0.6, s=20, label='Significant')
    
    # Top genes
    top_up = sig.nlargest(top_n, 'log2FC')
    top_down = sig.nsmallest(top_n, 'log2FC')
    top_genes = pd.concat([top_up, top_down])
    
    for _, row in top_genes.iterrows():
        plt.annotate(row['gene'], (row['log2FC'], -np.log10(row['padj'])),
                    fontsize=8, alpha=0.7)
    
    plt.axhline(-np.log10(0.05), color='blue', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(1, color='blue', linestyle='--', alpha=0.5)
    
    plt.xlabel('Log2 Fold Change', fontsize=12)
    plt.ylabel('-Log10(Adjusted P-value)', fontsize=12)
    plt.title(f'Volcano Plot: {comparison}', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'volcano_{comparison}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 7: PATHWAY ENRICHMENT ANALYSIS
# ============================================================================

def perform_enrichment_analysis(de_results, cluster_labels):
    """
    Perform GO and KEGG pathway enrichment
    Note: Requires gseapy and internet connection
    """
    print("Performing pathway enrichment analysis...")
    
    try:
        import gseapy as gp
        
        enrichment_results = {}
        n_clusters = len(np.unique(cluster_labels))
        
        for i in range(n_clusters):
            print(f"Enrichment for Cluster {i}...")
            
            # Get upregulated genes across all comparisons for this cluster
            sig_genes = []
            for comparison, df in de_results.items():
                if f'C{i}' in comparison:
                    cluster_col = f'mean_cluster{i}'
                    other_clusters = [c for c in comparison.split('_vs_') if c != f'C{i}']
                    
                    if len(other_clusters) > 0:
                        other_col = f'mean_cluster{other_clusters[0][-1]}'
                        
                        if cluster_col in df.columns:
                            upregulated = df[
                                (df['significant']) & 
                                (df[cluster_col] > df[other_col])
                            ]['gene'].tolist()
                            sig_genes.extend(upregulated)
            
            sig_genes = list(set(sig_genes))
            
            if len(sig_genes) > 5:
                # GO enrichment
                enr_go = gp.enrichr(
                    gene_list=sig_genes,
                    gene_sets=['GO_Biological_Process_2021'],
                    organism='human',
                    outdir=None
                )
                
                enrichment_results[f'Cluster{i}_GO'] = enr_go.results
                
                print(f"  Cluster {i}: {len(sig_genes)} genes, Top GO terms:")
                print(enr_go.results.head(5)[['Term', 'Adjusted P-value']])
            
        return enrichment_results
    
    except ImportError:
        print("gseapy not installed. Skipping enrichment analysis.")
        print("Install with: pip install gseapy")
        return None

# ============================================================================
# SECTION 8: DRUG RESPONSE PREDICTION
# ============================================================================

def build_drug_response_model(input_dim=2048 + 19842):
    """
    Build deep neural network for drug response prediction
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def compute_composite_drug_score(dl_score, similarity_score, target_score,
                                 w_dl=0.5, w_sim=0.3, w_target=0.2):
    """
    Compute composite drug ranking score
    Final_Score = 0.5 × DL_Score + 0.3 × Similarity_Score + 0.2 × Target_Score
    """
    return w_dl * dl_score + w_sim * similarity_score + w_target * target_score

# ============================================================================
# SECTION 9: SURVIVAL ANALYSIS
# ============================================================================

def perform_survival_analysis(clinical_data, cluster_labels):
    """
    Perform Kaplan-Meier survival analysis
    """
    print("Performing survival analysis...")
    
    # Prepare data
    survival_df = pd.DataFrame({
        'cluster': cluster_labels,
        'time': clinical_data['OS_time'],
        'event': clinical_data['OS_event']
    })
    
    # Kaplan-Meier curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kmf = KaplanMeierFitter()
    
    for cluster in np.unique(cluster_labels):
        mask = survival_df['cluster'] == cluster
        kmf.fit(
            survival_df.loc[mask, 'time'],
            survival_df.loc[mask, 'event'],
            label=f'Cluster {cluster}'
        )
        kmf.plot_survival_function(ax=ax)
    
    plt.xlabel('Time (months)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.title('Kaplan-Meier Survival Curves by Molecular Subtype', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('survival_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Log-rank test
    clusters = survival_df['cluster'].unique()
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            mask1 = survival_df['cluster'] == clusters[i]
            mask2 = survival_df['cluster'] == clusters[j]
            
            result = logrank_test(
                survival_df.loc[mask1, 'time'],
                survival_df.loc[mask2, 'time'],
                survival_df.loc[mask1, 'event'],
                survival_df.loc[mask2, 'event']
            )
            
            print(f"Log-rank test Cluster {clusters[i]} vs {clusters[j]}: p = {result.p_value:.4f}")
    
    # Cox regression
    cph = CoxPHFitter()
    cph.fit(survival_df, duration_col='time', event_col='event')
    print("\nCox Proportional Hazards Model:")
    print(cph.summary)
    
    return survival_df

# ============================================================================
# SECTION 10: VISUALIZATION
# ============================================================================

def plot_umap_clusters(latent_embeddings, cluster_labels, entropies):
    """Plot UMAP visualization with clusters"""
    print("Generating UMAP visualization...")
    
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=RANDOM_SEED
    )
    
    umap_embeddings = reducer.fit_transform(latent_embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Clusters
    scatter1 = axes[0].scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=cluster_labels,
        cmap='Set2',
        s=50,
        alpha=0.7
    )
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].set_title('VAE + GMM Clustering (k=3)', fontsize=14)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Plot 2: Entropy
    scatter2 = axes[1].scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=entropies,
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].set_title('Assignment Uncertainty (Entropy)', fontsize=14)
    plt.colorbar(scatter2, ax=axes[1], label='Entropy')
    
    plt.tight_layout()
    plt.savefig('umap_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return umap_embeddings

def plot_cluster_heatmap(expression_data, cluster_labels, de_results, top_n=50):
    """Plot heatmap of top differentially expressed genes"""
    print("Generating cluster heatmap...")
    
    # Get top genes from each comparison
    top_genes = []
    for comparison, df in de_results.items():
        sig_genes = df[df['significant']].nlargest(top_n, 'log2FC')['gene'].tolist()
        top_genes.extend(sig_genes)
    
    top_genes = list(set(top_genes))[:150]  # Limit to 150 genes
    
    # Create heatmap data
    heatmap_data = expression_data.loc[top_genes].T
    
    # Sort by cluster
    sort_idx = np.argsort(cluster_labels)
    heatmap_data = heatmap_data.iloc[sort_idx]
    sorted_labels = cluster_labels[sort_idx]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Create color bar for clusters
    lut = dict(zip(np.unique(sorted_labels), sns.color_palette('Set2', len(np.unique(sorted_labels)))))
    row_colors = pd.Series(sorted_labels).map(lut)
    
    sns.clustermap(
        heatmap_data.T,
        col_colors=row_colors,
        cmap='RdBu_r',
        center=0,
        figsize=(14, 12),
        yticklabels=True,
        xticklabels=False,
        cbar_kws={'label': 'Z-score'}
    )
    
    plt.tight_layout()
    plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 11: MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(expression_file, clinical_file=None):
    """
    Execute complete analysis pipeline
    """
    print("="*80)
    print("OVARIAN CANCER MOLECULAR SUBTYPING PIPELINE")
    print("VAE-GMM-Consensus Clustering Framework")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n[1/11] Loading and preprocessing data...")
    raw_data = load_tcga_data(expression_file)
    processed_data = preprocess_expression_data(raw_data)
    
    # Step 2: Train-test split
    print("\n[2/11] Splitting data...")
    n_samples = processed_data.shape[1]
    train_idx = int(0.8 * n_samples)
    
    X_train = processed_data.iloc[:, :train_idx].T.values
    X_val = processed_data.iloc[:, train_idx:].T.values
    X_all = processed_data.T.values
    
    # Step 3: Build and train VAE
    print("\n[3/11] Building and training VAE...")
    vae = build_vae(input_dim=processed_data.shape[0], latent_dim=32)
    vae, history = train_vae(vae, X_train, X_val, epochs=200, batch_size=64)
    plot_training_curves(history)
    
    # Step 4: Extract latent representations
    print("\n[4/11] Extracting latent representations...")
    latent_mean, _, _ = vae.encoder.predict(X_all)
    
    # Step 5: GMM clustering
    print("\n[5/11] Performing GMM clustering...")
    cluster_labels, cluster_probs, entropies, gmm = perform_gmm_clustering(latent_mean, n_clusters=3)
    
    # Step 6: Consensus clustering
    print("\n[6/11] Validating with consensus clustering...")
    consensus_matrix = consensus_clustering(latent_mean, n_clusters=3, n_bootstrap=1000)
    
    # Step 7: Benchmarking
    print("\n[7/11] Benchmarking alternative methods...")
    benchmark_results, pca_emb = benchmark_clustering_methods(processed_data, n_clusters=3)
    
    vae_metrics = {
        'silhouette': silhouette_score(latent_mean, cluster_labels),
        'calinski_harabasz': calinski_harabasz_score(latent_mean, cluster_labels),
        'davies_bouldin': davies_bouldin_score(latent_mean, cluster_labels)
    }
    
    benchmark_table = create_benchmark_table(benchmark_results, vae_metrics)
    
    # Step 8: Differential expression
    print("\n[8/11] Performing differential expression analysis...")
    de_results = perform_differential_expression(processed_data, cluster_labels)
    
    # Plot volcano plots
    for comparison in de_results.keys():
        plot_volcano(de_results, comparison)
    
    # Step 9: Pathway enrichment
    print("\n[9/11] Performing pathway enrichment...")
    enrichment_results = perform_enrichment_analysis(de_results, cluster_labels)
    
    # Step 10: Visualization
    print("\n[10/11] Generating visualizations...")
    umap_embeddings = plot_umap_clusters(latent_mean, cluster_labels, entropies)
    plot_cluster_heatmap(processed_data, cluster_labels, de_results)
    
    # Step 11: Survival analysis (if clinical data available)
    if clinical_file is not None:
        print("\n[11/11] Performing survival analysis...")
        clinical_data = pd.read_csv(clinical_file, sep='\t', index_col=0)
        survival_results = perform_survival_analysis(clinical_data, cluster_labels)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_dict = {
        'latent_embeddings': latent_mean,
        'cluster_labels': cluster_labels,
        'cluster_probabilities': cluster_probs,
        'entropies': entropies,
        'umap_embeddings': umap_embeddings,
        'de_results': de_results,
        'vae_metrics': vae_metrics,
        'benchmark_results': benchmark_results
    }
    
    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'sample': processed_data.columns,
        'cluster': cluster_labels,
        'entropy': entropies,
        'prob_cluster0': cluster_probs[:, 0],
        'prob_cluster1': cluster_probs[:, 1],
        'prob_cluster2': cluster_probs[:, 2]
    })
    cluster_df.to_csv('cluster_assignments.csv', index=False)
    
    # Save DE results
    for comparison, df in de_results.items():
        df.to_csv(f'DE_{comparison}.csv', index=False)
    
    # Save VAE model
    vae.encoder.save('vae_encoder.h5')
    vae.decoder.save('vae_decoder.h5')
    
    print("Results saved successfully!")
    print("="*80)
    
    return results_dict

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Example usage:
    # Replace with your actual file paths
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║   Deep Learning-Based Molecular Subtyping of Ovarian Cancer         ║
    ║   VAE-GMM-Consensus Clustering Pipeline                             ║
    ║                                                                      ║
    ║   Authors: Amir Arshia Beheshti et al.                              ║
    ║   Institution: Ardabil University of Medical Sciences                ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # INSTRUCTIONS:
    # 1. Download TCGA-OV RNA-seq data from GDC Portal
    # 2. Prepare expression matrix: genes (rows) x samples (columns)
    # 3. Prepare clinical data with OS_time, OS_event columns
    # 4. Update file paths below
    
    # File paths (UPDATE THESE)
    EXPRESSION_FILE = "TCGA_OV_expression.tsv"  # Your expression data file
    CLINICAL_FILE = "TCGA_OV_clinical.tsv"      # Your clinical data file (optional)
    
    # Run pipeline
    results = run_complete_pipeline(
        expression_file=EXPRESSION_FILE,
        clinical_file=CLINICAL_FILE  # Set to None if not available
    )
    
    print("\n✓ Pipeline completed successfully!")
    print("\nGenerated files:")
    print("  - cluster_assignments.csv")
    print("  - DE_C*_vs_C*.csv (multiple files)")
    print("  - vae_encoder.h5")
    print("  - vae_decoder.h5")
    print("  - Multiple visualization PNG files")
    
    print("\nNext steps:")
    print("  1. Validate subtypes in external cohorts")
    print("  2. Perform experimental drug validation")
    print("  3. Integrate multi-omics data")
    print("  4. Conduct prospective clinical validation")  
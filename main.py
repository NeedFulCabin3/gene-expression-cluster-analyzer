import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_synthetic_gene_data(n_samples=150, n_genes=1000, random_state=42):
    """Generates synthetic high-dimensional gene expression data for demonstration."""
    print("[INFO] Generating synthetic gene expression dataset...")
    np.random.seed(random_state)
    # Simulate expression values with some built-in cluster patterns
    base_data = np.random.normal(loc=5.0, scale=2.0, size=(n_samples, n_genes))
    # Add artificial groupings to make clustering meaningful
    base_data[50:100, :200] += 4.0
    base_data[100:, 400:600] -= 3.0
    
    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
    sample_ids = [f"Sample_{i+1}" for i in range(n_samples)]
    return pd.DataFrame(base_data, index=sample_ids, columns=gene_names)

def load_data():
    """Prompts user in terminal for data source and loads/generates the dataset."""
    print("\n--- Gene Expression Analysis Tool ---")
    choice = input("Do you want to load a CSV file? (y/n, default=n): ").strip().lower()
    
    if choice == 'y':
        file_path = input("Enter path to CSV file (samples x genes): ").strip()
        try:
            df = pd.read_csv(file_path, index_col=0)
            print(f"[SUCCESS] Loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load file: {e}")
            print("[INFO] Falling back to synthetic data generation.")
            return generate_synthetic_gene_data()
    else:
        return generate_synthetic_gene_data()

def main():
    # 1. Load Data
    df = load_data()
    
    # 2. Input validation
    if df.isnull().values.any():
        print("[WARNING] Missing values detected. Filling missing values with column means.")
        df = df.fillna(df.mean())

    # 3. Standardize Features (crucial for PCA and distance-based clustering)
    print("[INFO] Standardizing gene expression features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # 4. Dimensionality Reduction with PCA
    # Let's retain enough components to explain 80% of the variance, or default to 2 for visualization.
    print("[INFO] Applying PCA for dimensionality reduction...")
    pca_full = PCA().fit(scaled_data)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
    
    print(f" -> Number of components needed for 80% variance: {n_components_80}")
    
    # Use top components (e.g., max(2, n_components_80) capped or user-defined) for clustering
    n_components = max(2, min(n_components_80, 10))
    print(f" -> Reducing down to {n_components} principal components for clustering.")
    
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)

    # 5. Clustering with DBSCAN
    print("[INFO] Applying DBSCAN clustering on reduced components...")
    # Terminal input for DBSCAN hyperparameters with smart defaults
    eps_input = input("Enter DBSCAN epsilon (default 1.5): ").strip()
    eps = float(eps_input) if eps_input else 1.5
    
    min_samples_input = input("Enter DBSCAN min_samples (default 5): ").strip()
    min_samples = int(min_samples_input) if min_samples_input else 5

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(reduced_data)
    
    df['Cluster'] = clusters
    
    # 6. Summarize Results
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print("\n--- Clustering Summary ---")
    print(f"Total Samples Analyzed : {len(df)}")
    print(f"Clusters Detected      : {n_clusters}")
    print(f"Noise/Outlier Points   : {n_noise}")
    print(df['Cluster'].value_counts().to_string())

    # 7. Optional Plotting (if 2 or more components available)
    if n_components >= 2:
        try:
            print("\n[INFO] Generating 2D PCA scatter plot...")
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='tab10', alpha=0.8, edgecolor='k')
            plt.title(f'PCA of Gene Expression (DBSCAN Clusters, eps={eps})')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
            plt.colorbar(scatter, label='Cluster ID (-1 = Noise)')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig('pca_dbscan_clusters.png')
            print("[SUCCESS] Plot saved as 'pca_dbscan_clusters.png'.")
        except Exception as e:
            print(f"[WARNING] Could not generate plot: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Operation cancelled by user.")
        sys.exit(0)
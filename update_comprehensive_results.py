"""
update_comprehensive_results.py
Author: Imran Feisal
Date: 18/01/2025

Description:
This script extends the original comprehensive testing framework to:
 - Read existing VAE outputs and append their results to a **new version** of the `comprehensive_experiments_results.csv` file.
 - Perform clustering for all configurations (VAE, TabNet, and Hybrid) that haven't yet been clustered, using existing outputs.
 - Generate a new CSV file named `comprehensive_experiments_results_v2.csv` by appending new results to the original file.

Usage:
  python update_comprehensive_results.py
"""

import os
import gc
import datetime
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "Data"
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "Experiments")
ORIGINAL_RESULTS_FILE = "comprehensive_experiments_results.csv"
NEW_RESULTS_FILE = "comprehensive_experiments_results_v2.csv"

# Helper function to check if clustering artifacts already exist
def clustering_artifacts_exist(config_id, config_folder):
    plots_folder = os.path.join(config_folder, "plots")
    tsne_file = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
    umap_file = os.path.join(plots_folder, f"umap2d_{config_id}.png")
    return os.path.exists(tsne_file) and os.path.exists(umap_file)

# Function to perform clustering and visualization
def perform_clustering_and_visualization(merged_df, config_id, plots_folder):
    os.makedirs(plots_folder, exist_ok=True)

    # Prepare data
    X_columns = [c for c in merged_df.columns if c not in ("Id", "Predicted_Health_Index")]
    X_full = merged_df[X_columns].values

    if X_full.shape[1] == 0:
        logger.warning(f"[CLUSTERING] No features for config_id={config_id}. Skipping.")
        return {}

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # Perform KMeans clustering
    cluster_range = range(6, 10)
    best_k, best_km, best_metrics = None, None, {}
    for n_clusters in cluster_range:
        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(X_full_scaled)
        sil = silhouette_score(X_full_scaled, labels)
        cal = calinski_harabasz_score(X_full_scaled, labels)
        dav = davies_bouldin_score(X_full_scaled, labels)
        if not best_metrics or sil > best_metrics.get("silhouette", -1):
            best_k, best_km = n_clusters, km
            best_metrics = {"silhouette": sil, "calinski": cal, "davies_bouldin": dav}

    # Assign best KMeans labels
    merged_df["Cluster"] = best_km.labels_

    # Perform DBSCAN
    nbrs = DBSCAN(eps=0.5, min_samples=5)
    labels_db = nbrs.fit_predict(X_full_scaled)
    db_metrics = {
        "dbscan_silhouette": silhouette_score(X_full_scaled, labels_db) if len(set(labels_db)) > 1 else None,
        "dbscan_calinski": calinski_harabasz_score(X_full_scaled, labels_db) if len(set(labels_db)) > 1 else None,
        "dbscan_davies_bouldin": davies_bouldin_score(X_full_scaled, labels_db) if len(set(labels_db)) > 1 else None
    }

    # Save t-SNE plot
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_full_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=merged_df["Cluster"], palette="viridis")
    plt.title(f"t-SNE Clustering for {config_id}")
    tsne_path = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
    plt.savefig(tsne_path)
    plt.close()

    # Save UMAP plot
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X_full_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=merged_df["Cluster"], palette="viridis")
    plt.title(f"UMAP Clustering for {config_id}")
    umap_path = os.path.join(plots_folder, f"umap2d_{config_id}.png")
    plt.savefig(umap_path)
    plt.close()

    return {
        "config_id": config_id,
        "chosen_method": "KMeans",
        "chosen_k": best_k,
        "final_silhouette": best_metrics["silhouette"],
        "final_calinski": best_metrics["calinski"],
        "final_davies_bouldin": best_metrics["davies_bouldin"],
        **db_metrics
    }

# Main function to update results and perform clustering
def main():
    # Load the original results file if it exists
    original_results_path = os.path.join(DATA_DIR, ORIGINAL_RESULTS_FILE)
    if os.path.exists(original_results_path):
        original_results = pd.read_csv(original_results_path)
    else:
        original_results = pd.DataFrame()

    all_results = []

    for config_dir in os.listdir(EXPERIMENTS_DIR):
        config_path = os.path.join(EXPERIMENTS_DIR, config_dir)
        if not os.path.isdir(config_path):
            continue

        config_id = config_dir
        plots_folder = os.path.join(config_path, "plots")

        # Check if clustering artifacts already exist
        if clustering_artifacts_exist(config_id, config_path):
            logger.info(f"[SKIP] Clustering already performed for {config_id}.")
            continue

        # # Load VAE or TabNet outputs
        # latent_path = os.path.join(config_path, f"{config_id}_latent_features.csv")
        # predictions_path = os.path.join(config_path, f"{config_id}_predictions.csv")
        
        latent_path = glob.glob(os.path.join(config_path, "*_latent_features.csv"))
        predictions_path = glob.glob(os.path.join(config_path, "*predictions.csv"))
        
        # We need to ensure that only a single file path is passed to pd.read_csv
        latent_path = latent_path[0] if latent_path else None
        predictions_path = predictions_path[0] if predictions_path else None

        # if os.path.exists(latent_path):
        #     merged_df = pd.read_csv(latent_path)
        # elif os.path.exists(predictions_path):
        #     merged_df = pd.read_csv(predictions_path)
        # else:
        #     logger.warning(f"[MISSING] No output files for {config_id}. Skipping.")
        #     continue

        if latent_path:
            merged_df = pd.read_csv(latent_path)
        elif predictions_path:
            merged_df = pd.read_csv(predictions_path)
        else:
            logger.warning(f"[MISSING] No output files for {config_id}. Skipping.")
            continue
        
        
        
        # Perform clustering
        clustering_results = perform_clustering_and_visualization(merged_df, config_id, plots_folder)
        all_results.append(clustering_results)

    # Combine results with the original file and save the new version
    new_results_df = pd.DataFrame(all_results)
    combined_results = pd.concat([original_results, new_results_df], ignore_index=True)
    combined_results.to_csv(os.path.join(DATA_DIR, NEW_RESULTS_FILE), index=False)
    logger.info(f"[DONE] Results saved to {NEW_RESULTS_FILE}")

if __name__ == "__main__":
    main()

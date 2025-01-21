# vitai_scripts/cluster_utils.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#   Memory-optimised clustering logic and visualisation
#   for each experiment config, returning metrics for
#   K-Means and DBSCAN.

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cluster_and_visualise(
    merged_df: pd.DataFrame,
    config_id: str,
    plots_folder: str
) -> dict:
    """
    Clusters the data using K-Means(6..9) to find the best K, plus DBSCAN.
    Saves t-SNE and UMAP plots in 'plots_folder'.
    Returns a dict with final & DBSCAN metrics.
    """
    os.makedirs(plots_folder, exist_ok=True)

    # Exclude any columns that shouldn't be used for clustering
    exclude_cols = {"Id", "Predicted_Health_Index", "Predicted_CharlsonIndex", "Predicted_ElixhauserIndex"}
    X_cols = [c for c in merged_df.columns if c not in exclude_cols]
    X = merged_df[X_cols].values
    n_rows, n_feats = X.shape
    if n_rows < 2 or n_feats < 2:
        logger.warning(f"[{config_id}] Not enough rows/features ({n_rows}x{n_feats}) -> skipping clustering.")
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    best_k = None
    best_sil = -1
    for k in range(6, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    final_km = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
    final_labels = final_km.predict(X_scaled)
    merged_df["Cluster"] = final_labels

    final_sil = silhouette_score(X_scaled, final_labels)
    final_cal = calinski_harabasz_score(X_scaled, final_labels)
    final_dav = davies_bouldin_score(X_scaled, final_labels)

    # DBSCAN
    db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
    db_labels = db.labels_
    if len(set(db_labels)) < 2:
        db_sil = np.nan
        db_cal = np.nan
        db_dav = np.nan
    else:
        db_sil = silhouette_score(X_scaled, db_labels)
        db_cal = calinski_harabasz_score(X_scaled, db_labels)
        db_dav = davies_bouldin_score(X_scaled, db_labels)

    # Possibly create a 'Severity_Index' if "Predicted_Health_Index" is there
    hue_col = "Cluster"
    if "Predicted_Health_Index" in merged_df.columns:
        cluster_mean = (
            merged_df
            .groupby("Cluster")["Predicted_Health_Index"]
            .mean()
            .sort_values()
            .reset_index()
        )
        cluster_mean["Severity_Index"] = range(1, len(cluster_mean)+1)
        c_map = dict(zip(cluster_mean["Cluster"], cluster_mean["Severity_Index"]))
        merged_df["Severity_Index"] = merged_df["Cluster"].map(c_map)
        hue_col = "Severity_Index"

    # t-SNE
    tsne_2d = TSNE(n_components=2, random_state=42)
    X_tsne = tsne_2d.fit_transform(X_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1],
                    hue=merged_df[hue_col], palette="viridis")
    plt.title(f"t-SNE ({config_id}) - K={best_k}")
    tsne_path = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
    plt.savefig(tsne_path, bbox_inches="tight")
    plt.close()

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1],
                    hue=merged_df[hue_col], palette="viridis")
    plt.title(f"UMAP ({config_id}) - K={best_k}")
    umap_path = os.path.join(plots_folder, f"umap2d_{config_id}.png")
    plt.savefig(umap_path, bbox_inches="tight")
    plt.close()

    return {
        "config_id": config_id,
        "chosen_k": best_k,
        "final_silhouette": final_sil,
        "final_calinski": final_cal,
        "final_davies_bouldin": final_dav,
        "dbscan_silhouette": db_sil,
        "dbscan_calinski": db_cal,
        "dbscan_davies_bouldin": db_dav
    }

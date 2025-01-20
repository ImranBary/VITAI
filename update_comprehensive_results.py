"""
update_comprehensive_results.py
Author: Imran Feisal
Date: 18/01/2025

Description:
Extends the original comprehensive testing framework to:
 - Merge columns from BOTH `patient_data_with_health_index_cci.pkl` (Charlson, etc.)
   and `patient_data_with_health_index.pkl` (Hospital/Med/Abnormal counts, etc.) if they exist.
 - Read existing VAE/TabNet outputs for each config, merges them with the base data,
   then performs clustering if no t-SNE/UMAP plots are found.
 - Appends newly generated clustering rows to `comprehensive_experiments_results_v2.csv`.
"""

import os
import gc
import glob
import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.manifold import TSNE
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "Data"
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "Experiments")

ORIGINAL_RESULTS_FILE = "comprehensive_experiments_results.csv"
NEW_RESULTS_FILE = "comprehensive_experiments_results_v2.csv"

# The two pickles you wish to merge:
CCI_PICKLE = os.path.join(DATA_DIR, "patient_data_with_health_index_cci.pkl")
EXTRA_PICKLE = os.path.join(DATA_DIR, "patient_data_with_health_index.pkl")

###############################################################################
# 1. Check if clustering artifacts exist
###############################################################################
def clustering_artifacts_exist(config_id, config_folder):
    plots_folder = os.path.join(config_folder, "plots")
    tsne_file = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
    umap_file = os.path.join(plots_folder, f"umap2d_{config_id}.png")
    return os.path.exists(tsne_file) and os.path.exists(umap_file)

###############################################################################
# 2. Perform Clustering + Visualisation
###############################################################################
def perform_clustering_and_visualization(merged_df, config_id, plots_folder):
    """
    Clusters with K-Means [6..9], picks best by silhouette. Also runs DBSCAN.
    Excludes 'Id', 'Predicted_Health_Index', 'Predicted_CharlsonIndex' from features.
    If <2 rows/features, skip. Saves t-SNE & UMAP plots if possible.
    Returns a dict of cluster metrics.
    """
    os.makedirs(plots_folder, exist_ok=True)

    exclude_cols = {"Id", "Predicted_Health_Index", "Predicted_CharlsonIndex"}
    X_cols = [c for c in merged_df.columns if c not in exclude_cols]
    X = merged_df[X_cols].values

    n_rows, n_feats = X.shape
    if n_rows < 2 or n_feats < 2:
        logger.warning(f"[{config_id}] Not enough rows/features ({n_rows}x{n_feats}) to cluster.")
        return {}

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means over [6..9], pick best by silhouette
    best_k = None
    best_sil = -1
    for k in range(6, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    # Fit final KMeans
    final_km = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
    final_labels = final_km.predict(X_scaled)
    merged_df["Cluster"] = final_labels

    # Evaluate cluster metrics
    final_sil = silhouette_score(X_scaled, final_labels)
    final_cal = calinski_harabasz_score(X_scaled, final_labels)
    final_dav = davies_bouldin_score(X_scaled, final_labels)

    # DBSCAN for reference
    db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
    db_labels = db.labels_
    if len(set(db_labels)) < 2:
        db_sil, db_cal, db_dav = None, None, None
    else:
        db_sil = silhouette_score(X_scaled, db_labels)
        db_cal = calinski_harabasz_score(X_scaled, db_labels)
        db_dav = davies_bouldin_score(X_scaled, db_labels)

    # t-SNE & UMAP if enough rows
    if n_rows > 1:
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1],
                        hue=merged_df["Cluster"], palette="viridis")
        plt.title(f"t-SNE for {config_id} (K={best_k})")
        tsne_path = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
        plt.savefig(tsne_path, bbox_inches="tight")
        plt.close()

        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1],
                        hue=merged_df["Cluster"], palette="viridis")
        plt.title(f"UMAP for {config_id} (K={best_k})")
        umap_path = os.path.join(plots_folder, f"umap2d_{config_id}.png")
        plt.savefig(umap_path, bbox_inches="tight")
        plt.close()

    return {
        "config_id": config_id,
        "chosen_method": "KMeans",
        "chosen_k": best_k,
        "final_silhouette": final_sil,
        "final_calinski": final_cal,
        "final_davies_bouldin": final_dav,
        "dbscan_silhouette": db_sil,
        "dbscan_calinski": db_cal,
        "dbscan_davies_bouldin": db_dav,
    }

###############################################################################
# 3. Main Update Logic
###############################################################################
def main():
    # Load the original results CSV if present
    orig_csv_path = os.path.join(DATA_DIR, ORIGINAL_RESULTS_FILE)
    if os.path.exists(orig_csv_path):
        original_results = pd.read_csv(orig_csv_path)
    else:
        original_results = pd.DataFrame()

    # Check for existence of both pickles
    if not os.path.exists(CCI_PICKLE):
        logger.error(f"Missing {CCI_PICKLE}; cannot proceed.")
        return
    if not os.path.exists(EXTRA_PICKLE):
        logger.error(f"Missing {EXTRA_PICKLE}; cannot proceed.")
        return

    # Read them
    df_cci = pd.read_pickle(CCI_PICKLE)
    df_extra = pd.read_pickle(EXTRA_PICKLE)
    logger.info(f"[BASE] df_cci shape={df_cci.shape}, df_extra shape={df_extra.shape}")

    # Safely pick only columns from df_extra that exist
    # i.e. if Hospitalizations_Count is missing, we skip it
    wanted_cols = {"Id", "Hospitalizations_Count", "Medications_Count", "Abnormal_Observations_Count"}
    actual_extra_cols = list(set(df_extra.columns).intersection(wanted_cols))

    if len(actual_extra_cols) < 4:
        missing = wanted_cols - set(df_extra.columns)
        if missing:
            logger.warning(f"[BASE] The following expected columns are missing in EXTRA_PICKLE: {missing}")

    # Merge them on "Id"
    base_df = df_cci.merge(df_extra[actual_extra_cols], on="Id", how="left")
    logger.info(f"[BASE] After merge shape={base_df.shape}")

    # Drop unneeded or list-typed columns
    drop_cols = ["SEQUENCE", "PATIENT"]
    for col in drop_cols:
        if col in base_df.columns:
            base_df.drop(columns=[col], inplace=True)

    # If any column holds list data, drop it
    for col in list(base_df.columns):
        if base_df[col].map(type).eq(list).any():
            logger.warning(f"[BASE] Dropping {col} because it contains list data.")
            base_df.drop(columns=[col], inplace=True)

    # Now cluster each config directory
    all_results = []
    for config_dir in sorted(os.listdir(EXPERIMENTS_DIR)):
        config_path = os.path.join(EXPERIMENTS_DIR, config_dir)
        if not os.path.isdir(config_path):
            continue

        config_id = config_dir
        plots_folder = os.path.join(config_path, "plots")

        if clustering_artifacts_exist(config_id, config_path):
            logger.info(f"[SKIP] Clustering already done for {config_id}.")
            continue

        # Look for VAE or TabNet outputs
        lat_files = glob.glob(os.path.join(config_path, "*_latent_features.csv"))
        pred_files = glob.glob(os.path.join(config_path, "*_predictions.csv"))

        if not lat_files and not pred_files:
            logger.warning(f"[MISSING] No VAE or TabNet CSV for {config_id}")
            continue

        merged = base_df.copy()

        # Merge latent if present
        if lat_files:
            df_lat = pd.read_csv(lat_files[0])
            merged = merged.merge(df_lat, on="Id", how="inner")

        # Merge preds if present
        if pred_files:
            df_pred = pd.read_csv(pred_files[0])
            merged = merged.merge(df_pred, on="Id", how="inner")

        logger.info(f"[{config_id}] final merged shape={merged.shape}")

        # Drop columns with list data again if they re-appear
        for col in list(merged.columns):
            if merged[col].map(type).eq(list).any():
                logger.warning(f"[{config_id}] Dropping {col}; it has list data.")
                merged.drop(columns=[col], inplace=True)

        # One-hot encode typical demographic columns if present
        cat_cols = ["GENDER", "RACE", "ETHNICITY", "MARITAL"]
        for c in cat_cols:
            if c in merged.columns:
                merged[c] = merged[c].astype(str)
        merged = pd.get_dummies(merged, columns=[c for c in cat_cols if c in merged.columns], drop_first=True)

        # Perform clustering
        res = perform_clustering_and_visualization(merged, config_id, plots_folder)
        if res:
            all_results.append(res)

        del merged
        gc.collect()

    # Append new results
    if all_results:
        new_df = pd.DataFrame(all_results)
        combined_df = pd.concat([original_results, new_df], ignore_index=True)
        out_csv_path = os.path.join(DATA_DIR, NEW_RESULTS_FILE)
        combined_df.to_csv(out_csv_path, index=False)
        logger.info(f"[DONE] Updated results -> {NEW_RESULTS_FILE}")
    else:
        logger.info("[INFO] No new clustering results were generated.")

if __name__ == "__main__":
    main()

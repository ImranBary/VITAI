"""
update_comprehensive_results.py
Author: Imran Feisal
Date: 18/01/2025

Description:
Extends the original comprehensive testing framework to:
 - Merge columns from BOTH `patient_data_with_health_index_cci.pkl` (Charlson, etc.)
   and `patient_data_with_health_index.pkl` (Hospital/Med/Abnormal counts, etc.).
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

# Two pickles that, when merged, contain everything we need:
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
    Does KMeans in [6..9], picks best by silhouette. Also runs DBSCAN for reference.
    Excludes 'Id', 'Predicted_Health_Index', 'Predicted_CharlsonIndex' from features.
    Skips if <2 rows or <2 features.
    Saves t-SNE and UMAP into {plots_folder}/tsne2d_{config_id}.png, etc.
    Returns a dict of cluster metrics to be appended to final CSV.
    """
    os.makedirs(plots_folder, exist_ok=True)

    exclude_cols = {"Id", "Predicted_Health_Index", "Predicted_CharlsonIndex"}
    X_cols = [c for c in merged_df.columns if c not in exclude_cols]
    X = merged_df[X_cols].values

    n_rows, n_feats = X.shape
    if n_rows < 2 or n_feats < 2:
        logger.warning(
            f"[{config_id}] Cannot cluster: not enough rows/features ({n_rows}x{n_feats})."
        )
        return {}

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans [6..9]
    best_k = None
    best_metrics = None
    for k in range(6, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        dav = davies_bouldin_score(X_scaled, labels)
        # Simple best-silhouette approach
        if (best_metrics is None) or (sil > best_metrics["sil"]):
            best_k = k
            best_metrics = {"sil": sil, "cal": cal, "dav": dav}

    # Fit final KMeans with best_k
    final_km = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
    final_labels = final_km.predict(X_scaled)
    merged_df["Cluster"] = final_labels

    # Evaluate final cluster metrics
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

    # t-SNE & UMAP if enough points
    if n_rows > 1:
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        plt.figure(figsize=(7,5))
        sns.scatterplot(
            x=X_tsne[:,0],
            y=X_tsne[:,1],
            hue=merged_df["Cluster"],
            palette="viridis"
        )
        plt.title(f"t-SNE for {config_id} (K={best_k})")
        tsne_path = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
        plt.savefig(tsne_path, bbox_inches="tight")
        plt.close()

        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        plt.figure(figsize=(7,5))
        sns.scatterplot(
            x=X_umap[:,0],
            y=X_umap[:,1],
            hue=merged_df["Cluster"],
            palette="viridis"
        )
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
        "dbscan_davies_bouldin": db_dav
    }

###############################################################################
# 3. Main Update Logic
###############################################################################
def main():
    # Load the original results file if it exists
    original_results_path = os.path.join(DATA_DIR, ORIGINAL_RESULTS_FILE)
    if os.path.exists(original_results_path):
        original_results = pd.read_csv(original_results_path)
    else:
        original_results = pd.DataFrame()

    # Load both pickles so we can have Charlson + extra derived columns
    if not os.path.exists(CCI_PICKLE):
        logger.error(f"Missing {CCI_PICKLE}; cannot proceed.")
        return
    if not os.path.exists(EXTRA_PICKLE):
        logger.error(f"Missing {EXTRA_PICKLE}; cannot proceed.")
        return

    df_cci = pd.read_pickle(CCI_PICKLE)   # Contains 'CharlsonIndex', etc.
    df_extra = pd.read_pickle(EXTRA_PICKLE)  # Contains 'Hospitalizations_Count', etc.

    # Merge them on "Id" to get all columns
    # If both have the same columns, 'how="outer"' or 'how="left"'; up to you.
    base_df = df_cci.merge(
        df_extra[[
            "Id",
            "Hospitalizations_Count",
            "Medications_Count",
            "Abnormal_Observations_Count"
        ]],
        on="Id",
        how="left"
    )
    logger.info(f"[BASE] Combined shape = {base_df.shape}")

    # If you still have a "SEQUENCE" or "PATIENT" column, drop it if present
    drop_cols = ["SEQUENCE", "PATIENT"]
    for col in drop_cols:
        if col in base_df.columns:
            base_df.drop(columns=[col], inplace=True)

    # For any columns that might contain lists, drop them
    for col in list(base_df.columns):
        if base_df[col].map(type).eq(list).any():
            logger.warning(f"[BASE] Dropping {col} because it contains list data.")
            base_df.drop(columns=[col], inplace=True)

    # Now let's do the main loop
    all_results = []
    config_dirs = sorted(os.listdir(EXPERIMENTS_DIR))
    for config_dir in config_dirs:
        config_path = os.path.join(EXPERIMENTS_DIR, config_dir)
        if not os.path.isdir(config_path):
            continue

        config_id = config_dir
        plots_folder = os.path.join(config_path, "plots")

        # Skip if clustering artifacts exist
        if clustering_artifacts_exist(config_id, config_path):
            logger.info(f"[SKIP] Already has clustering plots for {config_id}.")
            continue

        # Find VAE / TabNet outputs
        latents = glob.glob(os.path.join(config_path, "*_latent_features.csv"))
        preds = glob.glob(os.path.join(config_path, "*_predictions.csv"))

        latent_csv = latents[0] if latents else None
        preds_csv = preds[0] if preds else None

        if not latent_csv and not preds_csv:
            logger.warning(f"[MISSING] No VAE or TabNet CSV for {config_id}")
            continue

        # Merge the base data with whichever we have
        merged = base_df.copy()

        if latent_csv:
            df_lat = pd.read_csv(latent_csv)
            merged = merged.merge(df_lat, on="Id", how="inner")

        if preds_csv:
            df_pred = pd.read_csv(preds_csv)
            merged = merged.merge(df_pred, on="Id", how="inner")

        logger.info(f"[{config_id}] final merged shape={merged.shape}")

        # If any columns hold list data, drop them
        for col in list(merged.columns):
            if merged[col].map(type).eq(list).any():
                logger.warning(f"[{config_id}] Dropping column {col} with list data.")
                merged.drop(columns=[col], inplace=True)

        # One-hot encode the same demographic columns the original script used
        cat_cols = ["GENDER", "RACE", "ETHNICITY", "MARITAL"]
        for c in cat_cols:
            if c in merged.columns:
                merged[c] = merged[c].astype(str)
        merged = pd.get_dummies(
            merged, 
            columns=[c for c in cat_cols if c in merged.columns],
            drop_first=True
        )

        # Perform clustering
        res = perform_clustering_and_visualization(merged, config_id, plots_folder)
        if res:
            all_results.append(res)

        del merged
        gc.collect()

    # Append new rows to the original
    if all_results:
        new_df = pd.DataFrame(all_results)
        combined_df = pd.concat([original_results, new_df], ignore_index=True)
        out_path = os.path.join(DATA_DIR, NEW_RESULTS_FILE)
        combined_df.to_csv(out_path, index=False)
        logger.info(f"[DONE] Wrote updated results -> {NEW_RESULTS_FILE}")
    else:
        logger.info("[INFO] No new clustering results generated.")

if __name__ == "__main__":
    main()

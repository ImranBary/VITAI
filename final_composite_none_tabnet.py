"""
final_composite_none_tabnet.py
Author: You
Date: 21/01/2025

Description:
A script to run the "composite_none_tabnet" config for a final model:
 - Reuses tabnet_model.py for data prep and model code
 - Sets up a final run with longer training (max_epochs) to get best final performance
 - Predicts for the entire dataset (no subset filter, i.e. "none")
 - Clusters the final predictions with K-Means
 - Saves everything to Data/finals/

Usage:
  python final_composite_none_tabnet.py
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from sklearn.manifold import TSNE
import umap.umap_ as umap

# We import  existing tabnet_model code. 
# We'll override the "max_epochs" inside the train function for our final run.
import torch
from tabnet_model import (
    load_data,
    prepare_data,
    hyperparameter_tuning,
    train_tabnet,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
FINALS_DIR = os.path.join(DATA_DIR, "finals")
os.makedirs(FINALS_DIR, exist_ok=True)

# The input file for "composite" scenario typically is "patient_data_with_health_index_cci.pkl",
# or you can just rename it if "tabnet_model.py" expects something else. 
INPUT_PICKLE = "patient_data_with_health_index.pkl"  
TARGET_COL = "Health_Index"
FINAL_MODEL_PREFIX = "final_composite_none_tabnet"


def main():
    # 1) Load data (which has Health_Index, "none" means we do no subset filtering)
    patient_data = load_data(DATA_DIR, INPUT_PICKLE)
    # If you'd like to drop or skip CharlsonIndex, do it here if you want. 
    # But it won't matter if 'prepare_data' doesn't use it.

    # 2) Prepare data for TabNet (currently uses the logic in tabnet_model.py -> prepare_data)
    #    That function by default uses these columns if they exist. 
    #    The target_col is set inside "prepare_data"? 
    # We just need to ensure we have Health_Index. 
    X, y, cat_idxs, cat_dims, feature_columns = prepare_data(patient_data, target_col=TARGET_COL)

    # 3) Train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # 4) Hyperparameter tuning with "normal" ephemeral epochs
    #    Then we override final training with bigger max_epochs
    best_params = hyperparameter_tuning(X_train, y_train, cat_idxs, cat_dims)
    logger.info(f"[FINAL] Best hyperparams from tuning: {best_params}")

    # 5) For final training, we do "train_tabnet" but with bigger max_epochs
    #    We'll monkey-patch the "max_epochs" arg by replacing it in best_params
    #    if "n_steps" or other param is in best_params, we keep them. 
    #    But we want, for example, "max_epochs=400" or "patience=50".
    # So let's do:
    saved_lr = best_params.pop("lr")  # we do the same approach as train_tabnet 
    best_params.update({
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=saved_lr),
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": 1
    })

    # We'll rebuild the regressor with bigger final epochs
    # We can copy the logic from train_tabnet function but override "max_epochs"
    from pytorch_tabnet.tab_model import TabNetRegressor
    final_regressor = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, **best_params)
    final_regressor.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['rmse'],
        max_epochs=400,  # bigger for final
        patience=50,     # also bigger
        batch_size=8192,
        virtual_batch_size=1024
    )
    final_regressor.save_model(os.path.join(FINALS_DIR, f"{FINAL_MODEL_PREFIX}_model"))
    logger.info(f"[FINAL] TabNet model trained with longer epochs and saved to {FINAL_MODEL_PREFIX}_model.zip etc.")

    # Evaluate on test set
    test_preds = final_regressor.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    logger.info(f"[FINAL] Test MSE: {test_mse:.6f}")
    logger.info(f"[FINAL] Test R2: {test_r2:.6f}")

    # Save final metrics
    final_metrics = {
        "test_mse": float(test_mse),
        "test_r2": float(test_r2)
    }
    with open(os.path.join(FINALS_DIR, f"{FINAL_MODEL_PREFIX}_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    # 6) Predict entire dataset
    all_preds = final_regressor.predict(X).flatten()
    patient_ids = patient_data["Id"].values if "Id" in patient_data.columns else np.arange(len(X))
    out_pred = pd.DataFrame({"Id": patient_ids, "Predicted_Health_Index": all_preds})
    out_pred.to_csv(os.path.join(FINALS_DIR, f"{FINAL_MODEL_PREFIX}_predictions.csv"), index=False)
    logger.info("[FINAL] Full-dataset predictions saved.")

    # 7) Clustering on final dataset predictions
    #    We'll merge predictions with the same features used for clustering
    #    If you want "composite" features only, or the entire numeric set, your choice.
    #    For example, cluster on X (scaled or not) plus predicted health index:
    df_cluster = pd.DataFrame(X, columns=feature_columns)
    df_cluster["Predicted_Health_Index"] = all_preds
    df_cluster["Id"] = patient_ids

    # Scale everything except 'Id'
    exclude_cols = {"Id"}
    cluster_cols = [c for c in df_cluster.columns if c not in exclude_cols]
    # We'll do a fresh scale for clustering
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(df_cluster[cluster_cols])

    # K-Means [6..9] pick best by silhouette
    best_k, best_sil = None, -1
    for k in range(6,10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_cluster_scaled)
        s = silhouette_score(X_cluster_scaled, labels)
        if s > best_sil:
            best_sil = s
            best_k = k

    final_km = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_km.fit_predict(X_cluster_scaled)
    df_cluster["Cluster"] = final_labels

    # Optional: t-SNE & UMAP plots
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne_2d = tsne.fit_transform(X_cluster_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_tsne_2d[:,0], y=X_tsne_2d[:,1], hue=final_labels, palette="viridis")
    plt.title(f"t-SNE final K={best_k}")
    tsne_file = os.path.join(FINALS_DIR, f"{FINAL_MODEL_PREFIX}_tsne.png")
    plt.savefig(tsne_file, bbox_inches="tight")
    plt.close()

    # UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap_2d = umap_model.fit_transform(X_cluster_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_umap_2d[:,0], y=X_umap_2d[:,1], hue=final_labels, palette="viridis")
    plt.title(f"UMAP final K={best_k}")
    umap_file = os.path.join(FINALS_DIR, f"{FINAL_MODEL_PREFIX}_umap.png")
    plt.savefig(umap_file, bbox_inches="tight")
    plt.close()

    # Save cluster assignments
    cluster_csv = os.path.join(FINALS_DIR, f"{FINAL_MODEL_PREFIX}_clusters.csv")
    df_cluster[["Id","Predicted_Health_Index","Cluster"]].to_csv(cluster_csv, index=False)
    logger.info("[FINAL] K-Means clusters and predictions saved.")

    logger.info("[DONE] final_composite_none_tabnet pipeline completed successfully.")

if __name__ == "__main__":
    main()

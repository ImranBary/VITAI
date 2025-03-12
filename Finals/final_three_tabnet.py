# final_three_tabnet.py
# Author: Imran Feisal
# Date: 21/01/2025

# Description:
#   This script runs our three final TabNet models on the entire dataset, each restricted to a specific
#   subpopulation:
#     1) "combined_diabetes_tabnet"   -> subset: diabetes,      feature_config: "combined"
#     2) "combined_all_ckd_tabnet"    -> subset: ckd,           feature_config: "combined_all"
#     3) "combined_none_tabnet"       -> subset: none,          feature_config: "combined"

#   For each model, the script performs the following steps:
#     - Hyperparameter tuning on a training split and final model training with increased epochs.
#     - Generation of a continuous predicted Health Index for each patient. This Health Index is a 
#       composite score reflecting the overall clinical status based on demographics, comorbidities,
#       and other health indicators.
#     - Clustering (using K-Means, with additional visualisation via t-SNE/UMAP) on the predicted 
#       subpopulation. The clustering groups patients with similar clinical profiles together, yielding
#       a discrete cluster label for each patient.

#   Output:
#     Each patient is assigned two key outputs:
#       1. A Predicted Health Index – a continuous measure representing the patient’s overall health.
#       2. A Cluster label – a discrete, data-driven categorisation indicating the group to which the 
#          patient belongs, based on similarities in the predicted Health Index and other selected features.
  
#   Classification Approach:
#     - You may use the continuous Health Index directly by defining clinical thresholds (e.g. low, 
#       medium, high risk) to categorise patients.
#     - Alternatively, the cluster label provides a natural classification as it groups similar patients 
#       together. In a dashboard, you might display both: the continuous Health Index for detailed analysis 
#       and the cluster label for a quick overview of the patient’s risk group.

# Usage:
#   python final_three_tabnet.py



import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys 
import joblib

#---------------------------------------------------
# Add the root directory to PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(ROOT_DIR)

# Add `vitai_scripts` to PYTHONPATH
VITAI_SCRIPTS_DIR = os.path.join(ROOT_DIR, "vitai_scripts")
sys.path.append(VITAI_SCRIPTS_DIR)



#---------------------------------------------------

# TabNet, tuning, training
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Import  existing logic
from tabnet_model import load_data, prepare_data, hyperparameter_tuning
from subset_utils import filter_subpopulation
from feature_utils import select_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data")   # Root-level "Data" folder
FINALS_DIR = os.path.join(DATA_DIR, "finals")
os.makedirs(FINALS_DIR, exist_ok=True)

# We’ll load all patients from the final combined file
INPUT_PICKLE = "patient_data_with_all_indices.pkl"

# Here are the three final configs we want:
FINAL_CONFIGS = [
    {
        "model_id": "combined_diabetes_tabnet",
        "subset": "diabetes",
        "feature_config": "combined"
    },
    {
        "model_id": "combined_all_ckd_tabnet",
        "subset": "ckd",
        "feature_config": "combined_all"
    },
    {
        "model_id": "combined_none_tabnet",
        "subset": "none",
        "feature_config": "combined"
    },
]

# The target column for all three is Health Index
TARGET_COL = "Health_Index"

# Final training epochs
FINAL_MAX_EPOCHS = 400
FINAL_PATIENCE = 50

def run_final_model(model_id: str, subset_type: str, feature_config: str, full_df: pd.DataFrame):
    """
    1) Filter the DataFrame by subpopulation (subset_type)
    2) Select features by feature_config
    3) Hyperparameter tune with moderate epochs
    4) Re-train with final large epochs
    5) Predict on subpopulation
    6) Clustering on that subpopulation
    7) Save artifacts to Data/finals/<model_id>/
    """
    logger.info(f"[{model_id}] Starting final run. subset={subset_type}, features={feature_config}")

    # Prepare subfolder
    out_dir = os.path.join(FINALS_DIR, model_id)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Filter subpopulation
    sub_df = filter_subpopulation(full_df, subset_type, DATA_DIR)
    if sub_df.empty:
        logger.warning(f"[{model_id}] No patients in subset={subset_type}. Skipping.")
        return

    # 2) Pick the right features
    feats_df = select_features(sub_df, feature_config)

    # 3) Prepare data for TabNet
    #    (We assume prepare_data can accept a 'target_col' for the final col)
    X, y, cat_idxs, cat_dims, feature_columns = prepare_data(feats_df, target_col=TARGET_COL)
    
    # Extract and save a scaler specifically for this model
    # First, identify categorical and continuous columns
    categorical_columns = ['DECEASED','GENDER','RACE','ETHNICITY','MARITAL']
    continuous_columns = [col for col in feature_columns if col not in categorical_columns]
    
    # Create a dataframe from X with column names for scaling
    x_df = pd.DataFrame(X, columns=feature_columns)
    
    # Create and fit a scaler on continuous features
    scaler = StandardScaler()
    if continuous_columns:  # Only if we have continuous columns
        # Fit scaler (no need to transform, we just want to save the scaler)
        scaler.fit(x_df[continuous_columns])
        
        # Save the model-specific scaler
        scaler_path = os.path.join(out_dir, f"{model_id}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"[{model_id}] Saved feature scaler to {scaler_path}")

    # Train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # 4) Hyperparameter tuning (short-run)
    best_params = hyperparameter_tuning(X_train, y_train, cat_idxs, cat_dims)
    logger.info(f"[{model_id}] Best hyperparams from tuning: {best_params}")

    # 5) Build final regressor with big epochs
    saved_lr = best_params.pop("lr")
    best_params.update({
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=saved_lr),
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": 1
    })

    regressor = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, **best_params)
    regressor.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["rmse"],
        max_epochs=FINAL_MAX_EPOCHS,
        patience=FINAL_PATIENCE,
        batch_size=8192,
        virtual_batch_size=1024
    )

    # Save the model
    regressor.save_model(os.path.join(out_dir, f"{model_id}_model"))
    logger.info(f"[{model_id}] Final TabNet model saved.")

    # Evaluate on test set
    test_preds = regressor.predict(X_test).flatten()
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    logger.info(f"[{model_id}] Test MSE: {test_mse:.6f}")
    logger.info(f"[{model_id}] Test R2:  {test_r2:.6f}")

    # Save final metrics to JSON
    metrics_dict = {
        "test_mse": float(test_mse),
        "test_r2": float(test_r2)
    }
    with open(os.path.join(out_dir, f"{model_id}_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 6) Predict on entire subpopulation (for clustering)
    all_preds = regressor.predict(X).flatten()
    # If 'Id' in feats_df, use it; else fallback
    if "Id" in feats_df.columns:
        pid = feats_df["Id"].values
    else:
        pid = np.arange(len(X))

    pred_df = pd.DataFrame({
        "Id": pid,
        f"Predicted_{TARGET_COL}": all_preds
    })
    pred_csv = os.path.join(out_dir, f"{model_id}_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    logger.info(f"[{model_id}] Full subpopulation predictions saved.")

    # 7) Clustering
    #    Merge predicted values with feature columns used for clustering
    cluster_df = pd.DataFrame(X, columns=feature_columns)
    cluster_df[f"Predicted_{TARGET_COL}"] = all_preds
    cluster_df["Id"] = pid

    # Scale everything except 'Id'
    cluster_cols = [c for c in cluster_df.columns if c != "Id"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df[cluster_cols])

    # K-Means [6..9], pick best by silhouette
    best_k, best_sil = None, -1
    for k in range(6, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        if s > best_sil:
            best_sil = s
            best_k = k

    final_km = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_km.fit_predict(X_scaled)
    cluster_df["Cluster"] = final_labels

    # Evaluate cluster metrics
    sil = silhouette_score(X_scaled, final_labels)
    cal = calinski_harabasz_score(X_scaled, final_labels)
    dav = davies_bouldin_score(X_scaled, final_labels)

    # DBSCAN for reference
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

    cluster_metrics = {
        "chosen_k": best_k,
        "silhouette": float(sil),
        "calinski": float(cal),
        "davies_bouldin": float(dav),
        "dbscan_silhouette": float(db_sil),
        "dbscan_calinski": float(db_cal),
        "dbscan_davies_bouldin": float(db_dav)
    }
    with open(os.path.join(out_dir, f"{model_id}_clusters.json"), "w") as f:
        json.dump(cluster_metrics, f, indent=2)
    logger.info(f"[{model_id}] Clustering metrics: {cluster_metrics}")

    # Save cluster assignments
    clusters_csv = os.path.join(out_dir, f"{model_id}_clusters.csv")
    cluster_df[["Id", f"Predicted_{TARGET_COL}", "Cluster"]].to_csv(clusters_csv, index=False)

    # Optional: t-SNE & UMAP
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=cluster_df["Cluster"], palette="viridis")
    plt.title(f"t-SNE ({model_id}) - K={best_k}")
    tsne_path = os.path.join(out_dir, f"{model_id}_tsne.png")
    plt.savefig(tsne_path, bbox_inches="tight")
    plt.close()

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=cluster_df["Cluster"], palette="viridis")
    plt.title(f"UMAP ({model_id}) - K={best_k}")
    umap_path = os.path.join(out_dir, f"{model_id}_umap.png")
    plt.savefig(umap_path, bbox_inches="tight")
    plt.close()

    logger.info(f"[{model_id}] Clustering plots saved.")
    logger.info(f"[{model_id}] Done.\n")


def main():
    # Load the full dataset
    full_path = os.path.join(DATA_DIR, INPUT_PICKLE)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Cannot find {full_path} - ensure data prep is complete.")

    full_df = pd.read_pickle(full_path)
    logger.info(f"Loaded dataset: {full_df.shape} rows, from {full_path}")

    # Run each final config in turn
    for cfg in FINAL_CONFIGS:
        run_final_model(
            model_id=cfg["model_id"],
            subset_type=cfg["subset"],
            feature_config=cfg["feature_config"],
            full_df=full_df
        )

    logger.info("[All Done] final_three_tabnet pipeline completed successfully.")


if __name__ == "__main__":
    main()

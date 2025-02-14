#!/usr/bin/env python
# final_explain_xai_clustered_lime.py
# Author: Imran Feisal
# Date: 05/02/2025
#
# Description:
#   This script generates advanced global and local explanations for each of the final
#   TabNet models (for diabetes, CKD, and the full population) using a cluster‐based LIME
#   approach. To reduce runtime while maintaining good explanation quality, the script now:
#     - Samples a fixed number of instances for global explanation (SHAP and Integrated Gradients)
#     - Reduces the number of steps in Integrated Gradients.
#   The cluster‐based LIME explanations remain as before.
#
# Requirements:
#   - Final TabNet models in Data/finals/<model_id>/<model_id>_model.zip
#   - Utility scripts: "tabnet_model.py", "subset_utils.py", "feature_utils.py"
#   - The full dataset "patient_data_with_all_indices.pkl" for subpopulation filtering.
#   - pip install shap, alibi, captum, lime
#
# Outputs:
#   In "Data/explain_xai/<model_id>/", the script saves:
#     - <model_id>_shap_values.npy             (global SHAP attributions computed on a subset)
#     - <model_id>_shap_summary.png            (bar plot of mean |SHAP| values)
#     - <model_id>_ig_values.npy               (Integrated Gradients attributions computed on a subset)
#     - <model_id>_deeplift_values.npy         (DeepLIFT attributions for outliers)
#     - <model_id>_anchors_local.csv           (Anchors explanations for critical cases)
#     - <model_id>_tabnet_mask_step0.png       (Bar plot of average TabNet mask)
#     - <model_id>_mask_step0.npy              (Raw step 0 mask array)
#     - <model_id>_cluster_lime_explanations.csv (Cluster-level aggregated LIME explanations)
#
#   These outputs are intended to feed the dashboard, where clinicians can view both global and
#   representative local explanations.

import os
import sys
import csv
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch

# SHAP for model-agnostic attributions
import shap

# Captum for gradient-based methods
from captum.attr import IntegratedGradients, DeepLift

# Alibi for Anchors explanations
from alibi.explainers import AnchorTabular

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

# LIME for local explanations
from lime.lime_tabular import LimeTabularExplainer

# Clustering utilities
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add root and vitai_scripts directories to PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VITAI_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "vitai_scripts")
sys.path.append(PROJECT_ROOT)
sys.path.append(VITAI_SCRIPTS_DIR)

# Project utility scripts
from subset_utils import filter_subpopulation
from feature_utils import select_features
from tabnet_model import prepare_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###########################################
# Configuration
###########################################

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
FINALS_DIR = os.path.join(DATA_DIR, "finals")
EXPLAIN_DIR = os.path.join(DATA_DIR, "explain_xai")
os.makedirs(EXPLAIN_DIR, exist_ok=True)

FULL_DATA_PKL = os.path.join(DATA_DIR, "patient_data_with_all_indices.pkl")

# Define the final models to explain
FINAL_MODELS = [
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

TARGET_COL = "Health_Index"

# SHAP configuration
BACKGROUND_SAMPLE_SIZE = 2000  # Background subsample for KernelExplainer

# Global explanation sampling configuration:
GLOBAL_SAMPLE_SIZE = 500       # Number of instances to sample for global explanations (SHAP & IG)

# Integrated Gradients configuration
IG_N_STEPS = 35                # Reduced number of integration steps to speed up IG computations

# Local explanation configuration
CRITICAL_PERCENTILE = 90       # Top 10% residuals considered critical
ZSCORE_THRESHOLD = 3.0         # Outlier threshold for DeepLIFT

# Clustering configuration for LIME explanations
NUM_CLUSTERS = 5               # Number of clusters to generate for LIME representatives

###########################################
# Helper Functions
###########################################

def load_tabnet_model(model_path: str) -> TabNetRegressor:
    """
    Loads a TabNetRegressor from <model_path>.zip.
    """
    regressor = TabNetRegressor()
    regressor.load_model(model_path + ".zip")
    return regressor

def compute_residuals(model, X, y):
    """
    Returns the absolute residuals per row (absolute difference between prediction and truth).
    Used for identifying critical cases.
    """
    preds = model.predict(X).flatten()
    return np.abs(preds - y.flatten())

def compute_outliers(X, zscore_threshold=3.0):
    """
    Basic outlier detection using a naive z-score rule across features.
    If any feature's z-score exceeds the threshold, the instance is flagged.
    """
    mean_ = X.mean(axis=0)
    std_ = X.std(axis=0) + 1e-9
    zscores = np.abs((X - mean_) / std_)
    outlier_mask = np.any(zscores > zscore_threshold, axis=1)
    return np.where(outlier_mask)[0]

def train_kernel_explainer(model_predict_fn, X, background_size=BACKGROUND_SAMPLE_SIZE):
    """
    Builds a SHAP KernelExplainer using a background (reference) sample.
    """
    if len(X) > background_size:
        idxs = np.random.choice(len(X), size=background_size, replace=False)
        background_data = X[idxs]
    else:
        background_data = X
    logger.info(f"[SHAP] Using {len(background_data)} background samples.")
    explainer = shap.KernelExplainer(model_predict_fn, background_data)
    return explainer

def compute_shap_values(explainer, X):
    """
    Compute SHAP values for X using KernelExplainer.
    Returns a NumPy array of shape (n_samples, n_features).
    """
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.array(shap_values)

def integrated_gradients(model, X, baseline=None, device="cpu"):
    """
    Compute Integrated Gradients attributions for all rows in X using IG_N_STEPS steps.
    If no baseline is provided, use the median of X as reference.
    """
    # Use the underlying PyTorch model stored in 'network'
    ig = IntegratedGradients(model.network)
    X_t = torch.tensor(X, dtype=torch.float, device=device)
    if baseline is None:
        baseline_array = np.median(X, axis=0)
        baseline_t = torch.tensor(baseline_array, dtype=torch.float, device=device)
    else:
        baseline_t = torch.tensor(baseline, dtype=torch.float, device=device)
    # Use the correct keyword "baselines"
    attrs_t = ig.attribute(X_t, baselines=baseline_t, n_steps=IG_N_STEPS)
    return attrs_t.detach().cpu().numpy()

def deep_lift_attributions(model, X, baseline=None, device="cpu"):
    """
    Compute DeepLIFT attributions for X.
    If no baseline is provided, use the median of X as reference.
    """
    # Use the underlying PyTorch model stored in 'network'
    dl = DeepLift(model.network)
    X_t = torch.tensor(X, dtype=torch.float, device=device)
    if baseline is None:
        baseline_array = np.median(X, axis=0)
        baseline_t = torch.tensor(baseline_array, dtype=torch.float, device=device)
    else:
        baseline_t = torch.tensor(baseline, dtype=torch.float, device=device)
    attrs_t = dl.attribute(X_t, baselines=baseline_t)
    return attrs_t.detach().cpu().numpy()

def anchors_local_explanations(model_predict_fn, X, feature_cols, subset_indices, out_csv):
    """
    Use AnchorTabular to generate rule-based local explanations on a subset of rows.
    """
    anchor_exp = AnchorTabular(
        predictor=model_predict_fn,
        feature_names=feature_cols
    )
    anchor_exp.fit(X)  # Fit sampling strategy on X
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["RowIndex", "Precision", "Coverage", "Anchors"])
        for idx in subset_indices:
            explanation = anchor_exp.explain(X[idx], threshold=0.95)
            anchor_str = " AND ".join(explanation.names())
            writer.writerow([idx, f"{explanation.precision:.2f}", f"{explanation.coverage:.2f}", anchor_str])

def plot_feature_bar(data, feature_cols, title, out_png):
    """
    Create a bar plot of mean absolute attributions.
    """
    mean_abs = np.mean(np.abs(data), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    sorted_feats = [feature_cols[i] for i in sorted_idx]
    sorted_vals = mean_abs[sorted_idx]
    plt.figure(figsize=(8,5))
    sns.barplot(x=sorted_vals, y=sorted_feats, orient='h', color="cornflowerblue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def cluster_based_lime_explanations(X, feature_cols, model_predict_fn, num_clusters=NUM_CLUSTERS):
    """
    Cluster patients using KMeans and generate LIME explanations for each cluster representative.
    The representative is chosen as the point closest to the cluster centroid.
    Returns a dictionary mapping cluster labels to the explanation (list of (feature, weight) tuples)
    and an array of cluster labels for each instance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Cluster the scaled data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    # For each cluster, find the representative index (closest to the centroid)
    rep_indices = {}
    for cluster in range(num_clusters):
        cluster_idxs = np.where(cluster_labels == cluster)[0]
        cluster_points = X_scaled[cluster_idxs]
        centroid = centroids[cluster]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        rep_idx = cluster_idxs[np.argmin(distances)]
        rep_indices[cluster] = rep_idx
    # Create a LIME explainer using the entire dataset as background
    lime_explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=feature_cols,
        mode='regression',
        discretize_continuous=True
    )
    # Generate explanations for each representative
    cluster_explanations = {}
    for cluster, rep_idx in rep_indices.items():
        explanation = lime_explainer.explain_instance(
            data_row=X[rep_idx],
            predict_fn=model_predict_fn,
            num_features=len(feature_cols)
        )
        cluster_explanations[cluster] = explanation.as_list()
        logger.info(f"[LIME][Cluster {cluster}] Representative index: {rep_idx}, Explanation: {cluster_explanations[cluster]}")
    return cluster_explanations, cluster_labels

###########################################
# Main Function
###########################################

def main():
    if not os.path.exists(FULL_DATA_PKL):
        raise FileNotFoundError(f"{FULL_DATA_PKL} not found. Ensure data preparation is complete.")
    full_df = pd.read_pickle(FULL_DATA_PKL)
    logger.info(f"Loaded dataset: shape={full_df.shape} from {FULL_DATA_PKL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} for XAI computations.")

    # Iterate through each final model configuration
    for cfg in FINAL_MODELS:
        model_id = cfg["model_id"]
        subset_type = cfg["subset"]
        feature_config = cfg["feature_config"]
        logger.info(f"\n=== EXPLAINING MODEL: {model_id} (subset={subset_type}, features={feature_config}) ===")

        model_out_dir = os.path.join(EXPLAIN_DIR, model_id)
        os.makedirs(model_out_dir, exist_ok=True)

        # 1) Filter subpopulation
        sub_df = filter_subpopulation(full_df, subset_type, DATA_DIR)
        if sub_df.empty:
            logger.warning(f"[{model_id}] No patients in subpopulation '{subset_type}'. Skipping.")
            continue

        # 2) Select features
        feats_df = select_features(sub_df, feature_config)
        if feats_df.empty:
            logger.warning(f"[{model_id}] No data after applying feature config '{feature_config}'. Skipping.")
            continue

        # 3) Prepare data
        X, y, cat_idxs, cat_dims, feature_cols = prepare_data(feats_df, target_col=TARGET_COL)
        logger.info(f"[{model_id}] Prepared data: X shape = {X.shape}, features = {feature_cols}")

        if "Id" in feats_df.columns:
            patient_ids = feats_df["Id"].values
        else:
            patient_ids = np.arange(len(X))

        # 4) Load final TabNet model
        model_path = os.path.join(FINALS_DIR, model_id, f"{model_id}_model")
        if not os.path.exists(model_path + ".zip"):
            logger.warning(f"[{model_id}] Could not find {model_path}.zip - skipping.")
            continue

        tabnet_regressor = load_tabnet_model(model_path)
        logger.info(f"[{model_id}] Loaded model from {model_path}")

        # Define a simple prediction function (required by SHAP and LIME)
        def predict_fn(data):
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            return tabnet_regressor.predict(data).flatten()

        ########################################################
        # (A) GLOBAL EXPLANATIONS
        ########################################################

        # To speed up global explanation computations (SHAP and IG), sample a subset of the data.
        if X.shape[0] > GLOBAL_SAMPLE_SIZE:
            sample_idxs = np.random.choice(X.shape[0], size=GLOBAL_SAMPLE_SIZE, replace=False)
            X_sample = X[sample_idxs]
        else:
            X_sample = X

        # A1) SHAP (KernelExplainer) on sampled data
        logger.info(f"[{model_id}][SHAP] Building KernelExplainer on {X_sample.shape[0]} samples...")
        shap_expl = train_kernel_explainer(model_predict_fn=predict_fn, X=X_sample)
        logger.info(f"[{model_id}][SHAP] Computing SHAP values on sampled data...")
        shap_vals = compute_shap_values(shap_expl, X_sample)
        shap_npy = os.path.join(model_out_dir, f"{model_id}_shap_values.npy")
        np.save(shap_npy, shap_vals)
        logger.info(f"[{model_id}][SHAP] Saved SHAP values -> {shap_npy}")
        shap_png = os.path.join(model_out_dir, f"{model_id}_shap_summary.png")
        plot_feature_bar(shap_vals, feature_cols, f"{model_id} - SHAP Summary", shap_png)
        logger.info(f"[{model_id}][SHAP] Saved SHAP summary plot -> {shap_png}")

        # A2) Integrated Gradients on sampled data
        logger.info(f"[{model_id}][IG] Computing Integrated Gradients with {IG_N_STEPS} steps on sampled data...")
        ig_vals = integrated_gradients(tabnet_regressor, X_sample, baseline=None, device=device)
        ig_npy = os.path.join(model_out_dir, f"{model_id}_ig_values.npy")
        np.save(ig_npy, ig_vals)
        logger.info(f"[{model_id}][IG] Saved Integrated Gradients attributions -> {ig_npy}")

        # A3) TabNet Masks (Intrinsic)
        logger.info(f"[{model_id}][MASKS] Extracting TabNet masks...")
        masks_result = tabnet_regressor.explain(X)
        if isinstance(masks_result, dict) and "masks" in masks_result:
            masks_list = masks_result["masks"]
        else:
            masks_list = masks_result
        if masks_list and len(masks_list) > 0:
            step_0_mask = masks_list[0]
            avg_mask = step_0_mask.mean(axis=0)
            mask_npy = os.path.join(model_out_dir, f"{model_id}_mask_step0.npy")
            np.save(mask_npy, step_0_mask)
            mask_png = os.path.join(model_out_dir, f"{model_id}_tabnet_mask_step0.png")
            plt.figure(figsize=(8,5))
            sns.barplot(x=avg_mask, y=feature_cols, orient='h', color="cornflowerblue")
            plt.title(f"{model_id} - Mean Feature Mask (Step 0)")
            plt.tight_layout()
            plt.savefig(mask_png, dpi=300)
            plt.close()
            logger.info(f"[{model_id}][MASKS] Saved mean mask plot -> {mask_png}")

        ########################################################
        # (B) LOCAL EXPLANATIONS
        ########################################################

        # B1) Critical Cases -> Anchors (for rule-based explanations)
        residuals = compute_residuals(tabnet_regressor, X, y)
        threshold = np.percentile(residuals, CRITICAL_PERCENTILE)
        critical_indices = np.where(residuals > threshold)[0]
        if len(critical_indices) > 0:
            anchors_csv = os.path.join(model_out_dir, f"{model_id}_anchors_local.csv")
            anchors_local_explanations(model_predict_fn=predict_fn, X=X, feature_cols=feature_cols,
                                        subset_indices=critical_indices, out_csv=anchors_csv)
            logger.info(f"[{model_id}][Anchors] Saved local Anchors explanations -> {anchors_csv}")
        else:
            logger.info(f"[{model_id}][Anchors] No critical cases identified; skipping Anchors.")

        # B2) Outliers -> DeepLIFT
        outlier_indices = compute_outliers(X, ZSCORE_THRESHOLD)
        if len(outlier_indices) > 0:
            outlier_data = X[outlier_indices]
            dl_vals = deep_lift_attributions(tabnet_regressor, outlier_data, baseline=None, device=device)
            dl_npy = os.path.join(model_out_dir, f"{model_id}_deeplift_values.npy")
            np.save(dl_npy, dl_vals)
            logger.info(f"[{model_id}][DeepLIFT] Saved DeepLIFT attributions for outliers -> {dl_npy}")
        else:
            logger.info(f"[{model_id}][DeepLIFT] No outliers detected; skipping DeepLIFT.")

        ########################################################
        # (C) CLUSTER-BASED LIME EXPLANATIONS
        ########################################################

        logger.info(f"[{model_id}][LIME] Performing clustering for representative LIME explanations...")
        cluster_expls, cluster_labels = cluster_based_lime_explanations(X, feature_cols, predict_fn, num_clusters=NUM_CLUSTERS)
        # Save cluster-level LIME explanations in a CSV
        lime_csv = os.path.join(model_out_dir, f"{model_id}_cluster_lime_explanations.csv")
        with open(lime_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Cluster", "RepresentativeIndex", "LIME_Explanation"])
            for cluster, explanation in cluster_expls.items():
                # Recompute the representative index for this cluster
                cluster_idxs = np.where(cluster_labels == cluster)[0]
                if len(cluster_idxs) > 0:
                    rep_idx = cluster_idxs[0]
                else:
                    rep_idx = -1  # Fallback if none found
                explanation_str = "; ".join([f"{feat}: {weight:.3f}" for feat, weight in explanation])
                writer.writerow([cluster, rep_idx, explanation_str])
        logger.info(f"[{model_id}][LIME] Saved cluster-level LIME explanations -> {lime_csv}")

        ########################################################
        # Optionally, assign the cluster explanation back to every patient
        ########################################################
        cluster_assignments_df = pd.DataFrame({
            "patient_id": patient_ids,
            "cluster": cluster_labels
        })
        # Merge with cluster explanations
        cluster_expls_list = []
        for cluster, explanation in cluster_expls.items():
            rep_idx = np.where(cluster_labels == cluster)[0][0]  # representative index
            explanation_str = "; ".join([f"{feat}: {weight:.3f}" for feat, weight in explanation])
            cluster_expls_list.append({"cluster": cluster, "rep_idx": rep_idx, "explanation": explanation_str})
        cluster_expls_df = pd.DataFrame(cluster_expls_list)
        patient_explanations_df = cluster_assignments_df.merge(cluster_expls_df, on="cluster", how="left")
        patient_explanations_csv = os.path.join(model_out_dir, f"{model_id}_patient_explanations.csv")
        patient_explanations_df.to_csv(patient_explanations_csv, index=False)
        logger.info(f"[{model_id}][LIME] Saved patient-level cluster explanations -> {patient_explanations_csv}")

    logger.info("\n[ALL DONE] Advanced global and local XAI with cluster-based LIME explanations complete.\n")

if __name__ == "__main__":
    main()

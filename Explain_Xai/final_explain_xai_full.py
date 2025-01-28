# final_explain_xai_full.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#   This script generates advanced global and local explanations for each of your
#   TabNet models (diabetes, CKD, none). It now uses:
#     - SHAP (KernelExplainer) for global attributions
#     - Integrated Gradients, TabNet Masks for additional global insights
#     - Anchors (rule-based) for critical local cases, DeepLIFT for outliers
#   for all patients in each subpopulation.
#
# Usage:
#   python final_explain_xai_full.py
#
# Requirements:
#   - final TabNet models in Data/finals/<model_id>/<model_id>_model.zip
#   - "tabnet_model.py", "subset_utils.py", "feature_utils.py" in your project.
#   - The full dataset "patient_data_with_all_indices.pkl" for subpopulation filtering.
#   - pip install shap, alibi, captum
#
# Outputs:
#   In "Data/explain_xai/<model_id>/", the script saves:
#     - <model_id>_shap_values.npy             (global SHAP attributions)
#     - <model_id>_shap_summary.png            (barplot of mean |SHAP| values)
#     - <model_id>_ig_values.npy               (integrated gradient attributions)
#     - <model_id>_deeplift_values.npy         (outlier attributions)
#     - <model_id>_anchors_local.csv           (critical-case anchor rules)
#     - <model_id>_tabnet_mask_step0.png       (barplot of average mask)
#     - <model_id>_mask_step0.npy              (raw step 0 mask array)
#   and so on.

import os
import sys
import csv
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch

# SHAP for model-agnostic shap values
# pip install shap
import shap

# Captum for gradient-based methods (Integrated Gradients, DeepLIFT)
# pip install captum
from captum.attr import IntegratedGradients, DeepLift

# Alibi for Anchors (local rule-based explanations)
# pip install alibi
from alibi.explainers import AnchorTabular

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

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

# SHAP config
BACKGROUND_SAMPLE_SIZE = 2000  # sample a background subset for KernelExplainer

# Local explanation config
CRITICAL_PERCENTILE = 90   # top 10% residual => "critical" cases
ZSCORE_THRESHOLD = 3.0     # outlier threshold for DeepLIFT

###########################################
# Helper Functions
###########################################

def load_tabnet_model(model_path: str) -> TabNetRegressor:
    """
    Loads a TabNetRegressor from <model_path>.zip
    """
    regressor = TabNetRegressor()
    regressor.load_model(model_path + ".zip")
    return regressor

def compute_residuals(model, X, y):
    """
    Returns absolute residuals per row, used for identifying critical cases
    """
    preds = model.predict(X).flatten()
    return np.abs(preds - y.flatten())

def compute_outliers(X, zscore_threshold=3.0):
    """
    Basic outlier detection using a naive z-score rule across features.
    If any feature's z-score > threshold, mark as outlier.
    """
    mean_ = X.mean(axis=0)
    std_ = X.std(axis=0) + 1e-9
    zscores = np.abs((X - mean_) / std_)
    outlier_mask = np.any(zscores > zscore_threshold, axis=1)
    return np.where(outlier_mask)[0]

def train_kernel_explainer(model_predict_fn, X, background_size=2000):
    """
    Builds a SHAP KernelExplainer with a background (reference) sample.
    """
    # Subsample background if dataset is large
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
    # shap_values can be returned as a list (classification) or array (regression)
    shap_values = explainer.shap_values(X)
    # If using a single output (regression), ensure shap_values is an array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.array(shap_values)

def integrated_gradients(model, X, baseline=None, device="cpu"):
    """
    Compute Integrated Gradients attributions for all rows in X.
    If no baseline is given, use the median of X as the reference.
    """
    ig = IntegratedGradients(model.pytorch_model)
    X_t = torch.tensor(X, dtype=torch.float, device=device)

    if baseline is None:
        baseline_array = np.median(X, axis=0)
        baseline_t = torch.tensor(baseline_array, dtype=torch.float, device=device)
    else:
        baseline_t = torch.tensor(baseline, dtype=torch.float, device=device)

    attrs_t = ig.attribute(X_t, baseline=baseline_t, n_steps=50)
    return attrs_t.detach().cpu().numpy()

def deep_lift_attributions(model, X, baseline=None, device="cpu"):
    """
    Compute DeepLIFT attributions for outlier rows (or any subset).
    If no baseline is given, use the median of X as the reference.
    """
    dl = DeepLift(model.pytorch_model)
    X_t = torch.tensor(X, dtype=torch.float, device=device)

    if baseline is None:
        baseline_array = np.median(X, axis=0)
        baseline_t = torch.tensor(baseline_array, dtype=torch.float, device=device)
    else:
        baseline_t = torch.tensor(baseline, dtype=torch.float, device=device)

    attrs_t = dl.attribute(X_t, baseline=baseline_t)
    return attrs_t.detach().cpu().numpy()

def anchors_local_explanations(model_predict_fn, X, feature_cols, subset_indices, out_csv):
    """
    Use AnchorTabular for local rule-based explanations on a subset of rows (critical).
    """
    anchor_exp = AnchorTabular(
        predictor=model_predict_fn,
        feature_names=feature_cols
    )
    anchor_exp.fit(X)  # fit a sampling strategy to X

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["RowIndex", "Precision", "Coverage", "Anchors"])

        for idx in subset_indices:
            explanation = anchor_exp.explain(X[idx], threshold=0.95)
            anchor_str = " AND ".join(explanation.names())
            writer.writerow([
                idx,
                f"{explanation.precision:.2f}",
                f"{explanation.coverage:.2f}",
                anchor_str
            ])

def plot_feature_bar(data, feature_cols, title, out_png):
    """
    Create a simple bar plot of mean absolute attributions or feature usage.
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

def main():
    if not os.path.exists(FULL_DATA_PKL):
        raise FileNotFoundError(f"{FULL_DATA_PKL} not found. Ensure data prep is complete.")
    full_df = pd.read_pickle(FULL_DATA_PKL)
    logger.info(f"Loaded dataset: shape={full_df.shape} from {FULL_DATA_PKL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device={device} for advanced XAI computations.")

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
            logger.warning(f"[{model_id}] No patients in subpop '{subset_type}'. Skipping.")
            continue

        # 2) Select features
        feats_df = select_features(sub_df, feature_config)
        if feats_df.empty:
            logger.warning(f"[{model_id}] No data after feature config '{feature_config}'. Skipping.")
            continue

        # 3) Prepare data
        X, y, cat_idxs, cat_dims, feature_cols = prepare_data(feats_df, target_col=TARGET_COL)
        logger.info(f"[{model_id}] Prepared shape={X.shape} with columns={feature_cols}")

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
        logger.info(f"[{model_id}] Model loaded -> {model_path}")

        # A simple predict function that returns NumPy arrays (required by SHAP)
        def predict_fn(data):
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            return tabnet_regressor.predict(data)

        ########################################################
        # (A) GLOBAL EXPLANATIONS
        ########################################################

        # A1) SHAP (KernelExplainer)
        logger.info(f"[{model_id}][SHAP] Building explainer with KernelExplainer...")
        shap_expl = train_kernel_explainer(
            model_predict_fn=predict_fn,
            X=X,
            background_size=BACKGROUND_SAMPLE_SIZE
        )

        logger.info(f"[{model_id}][SHAP] Computing SHAP values for entire subset...")
        shap_vals = compute_shap_values(shap_expl, X)
        shap_npy = os.path.join(model_out_dir, f"{model_id}_shap_values.npy")
        np.save(shap_npy, shap_vals)
        logger.info(f"[{model_id}][SHAP] Saved -> {shap_npy}")

        # Optional barplot of mean(|SHAP|)
        shap_png = os.path.join(model_out_dir, f"{model_id}_shap_summary.png")
        plot_feature_bar(shap_vals, feature_cols, f"{model_id} - SHAP", shap_png)
        logger.info(f"[{model_id}][SHAP] Bar plot -> {shap_png}")

        # A2) Integrated Gradients
        logger.info(f"[{model_id}][IG] Computing Integrated Gradients (Captum)...")
        ig_vals = integrated_gradients(tabnet_regressor, X, baseline=None, device=device)
        ig_npy = os.path.join(model_out_dir, f"{model_id}_ig_values.npy")
        np.save(ig_npy, ig_vals)
        logger.info(f"[{model_id}][IG] Saved -> {ig_npy}")

        # A3) TabNet Masks
        logger.info(f"[{model_id}][MASKS] Extracting intrinsic TabNet masks for all rows...")
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
            logger.info(f"[{model_id}][MASKS] Mean mask plot -> {mask_png}")

        ########################################################
        # (B) LOCAL EXPLANATIONS
        ########################################################

        # B1) Critical Cases -> Anchors
        residuals = compute_residuals(tabnet_regressor, X, y)
        threshold = np.percentile(residuals, CRITICAL_PERCENTILE)
        critical_indices = np.where(residuals > threshold)[0]
        if len(critical_indices) > 0:
            anchors_csv = os.path.join(model_out_dir, f"{model_id}_anchors_local.csv")
            anchors_local_explanations(
                model_predict_fn=predict_fn,
                X=X,
                feature_cols=feature_cols,
                subset_indices=critical_indices,
                out_csv=anchors_csv
            )
            logger.info(f"[{model_id}][Anchors] Explanations -> {anchors_csv}")
        else:
            logger.info(f"[{model_id}][Anchors] No critical cases found, skipping anchors.")

        # B2) Outliers -> DeepLIFT
        outlier_indices = compute_outliers(X, ZSCORE_THRESHOLD)
        if len(outlier_indices) > 0:
            outlier_data = X[outlier_indices]
            dl_vals = deep_lift_attributions(tabnet_regressor, outlier_data, baseline=None, device=device)
            dl_npy = os.path.join(model_out_dir, f"{model_id}_deeplift_values.npy")
            np.save(dl_npy, dl_vals)
            logger.info(f"[{model_id}][DeepLIFT] Outlier attributions saved -> {dl_npy}")
        else:
            logger.info(f"[{model_id}][DeepLIFT] No outliers found, skipping.")

        logger.info(f"[{model_id}] Explanations complete. Outputs in {model_out_dir}")

    logger.info("\n[ALL DONE] Full advanced XAI (with SHAP, IG, TabNet Masks, Anchors, DeepLIFT) is complete.\n")


if __name__ == "__main__":
    main()

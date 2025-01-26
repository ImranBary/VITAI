# final_explain_xai_full.py
# Author: Imran Feisal  
# Date: 21/01/2025
#
# Description:
#   This script generates SHAP, LIME, and TabNet intrinsic explanations
#   for each of your three final TabNet models (diabetes, CKD, none),
#   over *all patients* in each subpopulation. No sampling is performed.
#
#   WARNING: For large datasets, this can be extremely time-consuming.
#   Ensure you have sufficient compute resources and time.
#
# Usage:
#   python final_explain_xai_full.py
#
# Requirements:
#   - final TabNet models in Data/finals/<model_id>/<model_id>_model.zip
#   - "tabnet_model.py", "subset_utils.py", "feature_utils.py" in your project.
#   - The full dataset "patient_data_with_all_indices.pkl" for subpopulation filtering.
#
# Outputs:
#   In "Data/explain_xai/<model_id>/", the script saves:
#     - <model_id>_shap_summary.png            (SHAP summary plot)
#     - <model_id>_shap_values.pkl             (raw SHAP values for all rows)
#     - <model_id>_lime_explanations.csv       (one line per row, local explanations)
#     - lime_<model_id>_patient_<Id>.html      (HTML file with that row’s LIME explanation)
#     - <model_id>_tabnet_mask_step0.png       (barplot of average mask across all rows)
#     - <model_id>_mask_step0.npy              (raw step 0 mask array for each row)
#   and so on.
#

import os
import sys
import csv
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LIME / SHAP
import lime.lime_tabular
import shap

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

# Add root and vitai_scripts directories to PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VITAI_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "vitai_scripts")

sys.path.append(PROJECT_ROOT)
sys.path.append(VITAI_SCRIPTS_DIR)

# Import the utility scripts from vitai_scripts
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

# LIME config
NUM_FEATURES_LIME = 6  # number of features to display for each local explanation
SAVE_LIME_HTML = True  # whether to generate per-patient .html files

###########################################
# Helper Functions
###########################################

def load_tabnet_model(model_path: str) -> TabNetRegressor:
    """
    Loads a TabNetRegressor from <model_path>_model.*
    """
    regressor = TabNetRegressor()
    regressor.load_model(model_path)
    return regressor

def shap_explain_entire_dataset(model: TabNetRegressor, X: np.ndarray):
    """
    Approximate approach using shap.KernelExplainer for all rows in X as well.
    This is extremely expensive if X is large. Consider a more efficient approach
    or smaller background set. For demonstration, we do no sampling here.
    """
    def model_predict(data):
        preds = model.predict(data)
        return preds.flatten()

    # By default, KernelExplainer needs a background dataset. We'll (naively) use X itself.
    explainer = shap.KernelExplainer(model_predict, data=X)
    # shap_values will be shape (n_rows, n_features) for regression
    shap_values = explainer.shap_values(X)
    return shap_values

def plot_shap_summary(shap_values, X: np.ndarray, feature_names: list, out_png: str):
    """
    Renders a standard shap.summary_plot and saves to out_png.
    """
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()

def build_lime_explainer(X: np.ndarray, feature_names: list):
    """
    Creates a LimeTabularExplainer for regression, using the entire dataset as reference.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        mode='regression'
    )
    return explainer

def local_lime_explanations(explainer, model_predict, X: np.ndarray, patient_ids, output_csv: str):
    """
    Runs LIME on each row in X, writing results to output_csv.
    Potentially extremely slow if X is large.
    """
    # Overwrite CSV with header
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Patient_ID", "Local_Contributions"])

    for i in range(len(X)):
        explanation = explainer.explain_instance(
            data_row=X[i],
            predict_fn=model_predict,
            num_features=NUM_FEATURES_LIME
        )
        explanation_text = " | ".join(
            f"{feat} => {weight:.4f}"
            for feat, weight in explanation.as_list()
        )
        pid = patient_ids[i]

        # Append to CSV
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([pid, explanation_text])

        # Optionally HTML file
        if SAVE_LIME_HTML:
            explanation.save_to_file(f"lime_{pid}.html")

def main():
    # Load full dataset
    if not os.path.exists(FULL_DATA_PKL):
        raise FileNotFoundError(f"{FULL_DATA_PKL} not found. Ensure data prep is complete.")
    full_df = pd.read_pickle(FULL_DATA_PKL)
    logger.info(f"Loaded dataset: shape={full_df.shape} from {FULL_DATA_PKL}")

    # Run each final model in turn
    for cfg in FINAL_MODELS:
        model_id = cfg["model_id"]
        subset_type = cfg["subset"]
        feature_config = cfg["feature_config"]
        logger.info(f"\n=== EXPLAINING MODEL: {model_id} (subset={subset_type}, features={feature_config}) ===")

        # Subfolder for this model's XAI outputs
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

        # 3) Prepare data exactly as training
        X, y, cat_idxs, cat_dims, feature_cols = prepare_data(feats_df, target_col=TARGET_COL)
        logger.info(f"[{model_id}] Prepared shape={X.shape} with columns={feature_cols}")

        # Identify patient IDs
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

        ###############################
        # (A) SHAP on entire subpop
        ###############################
        logger.info(f"[{model_id}][SHAP] Explaining {len(X)} rows. This may be very slow.")
        shap_values = shap_explain_entire_dataset(tabnet_regressor, X)

        # Save shap_values
        shap_pkl = os.path.join(model_out_dir, f"{model_id}_shap_values.pkl")
        with open(shap_pkl, 'wb') as f:
            pickle.dump(shap_values, f)
        logger.info(f"[{model_id}][SHAP] Raw shap_values saved -> {shap_pkl}")

        # SHAP summary
        shap_png = os.path.join(model_out_dir, f"{model_id}_shap_summary.png")
        plot_shap_summary(shap_values, X, feature_cols, shap_png)
        logger.info(f"[{model_id}][SHAP] Summary plot -> {shap_png}")

        ###############################
        # (B) LIME on entire subpop
        ###############################
        logger.info(f"[{model_id}][LIME] Explaining all {len(X)} rows. This is likely extremely slow.")

        # Build LIME explainer
        lime_exp = build_lime_explainer(X, feature_cols)

        # LIME predict function
        def lime_predict_fn(batch):
            preds = tabnet_regressor.predict(batch)
            return preds.flatten()

        # Where to store local explanations
        lime_csv = os.path.join(model_out_dir, f"{model_id}_lime_explanations.csv")
        local_lime_explanations(lime_exp, lime_predict_fn, X, patient_ids, lime_csv)
        logger.info(f"[{model_id}][LIME] Explanations saved -> {lime_csv}")

        ###############################
        # (C) Intrinsic TabNet Masks
        ###############################
        # "explain" returns a dict or list with step-wise masks
        logger.info(f"[{model_id}][MASKS] Computing TabNet masks for all {len(X)} rows. Might be large.")
        masks_result = tabnet_regressor.explain(X)
        # In some versions: masks_result["masks"] -> list of [step_0, step_1, ...]
        if isinstance(masks_result, dict) and "masks" in masks_result:
            masks_list = masks_result["masks"]
        else:
            masks_list = masks_result

        if masks_list and len(masks_list) > 0:
            # step_0_mask shape = (n_rows, n_features)
            step_0_mask = masks_list[0]
            avg_mask = step_0_mask.mean(axis=0)

            # Plot a bar chart of average mask across entire subpop
            plt.figure(figsize=(8,5))
            sns.barplot(x=avg_mask, y=feature_cols, orient='h', color="cornflowerblue")
            plt.title(f"{model_id} - Mean Feature Mask (Step 0)")
            plt.tight_layout()
            mask_png = os.path.join(model_out_dir, f"{model_id}_tabnet_mask_step0.png")
            plt.savefig(mask_png, dpi=300)
            plt.close()

            # Save raw step_0 mask
            mask_npy = os.path.join(model_out_dir, f"{model_id}_mask_step0.npy")
            np.save(mask_npy, step_0_mask)
            logger.info(f"[{model_id}][MASKS] Step 0 mean mask plot -> {mask_png}")

        logger.info(f"[{model_id}] Explanations completed. Outputs in {model_out_dir}")

    logger.info("\n[ALL DONE] Full-dataset XAI for each final TabNet model is complete.\n")


if __name__ == "__main__":
    main()

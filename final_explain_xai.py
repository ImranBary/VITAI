"""
final_explain_xai.py
Author: Imran Feisal
Date: 21/01/2025

Description:
A runner script for generating both global and per-patient explanations on the
entire dataset, using:
 - SHAP for global feature attributions
 - LIME for local instance explanations
 - TabNet feature masks for intrinsic attention

Logs all LIME explanations to a single CSV file, with columns:
  - Patient_ID
  - Explanation_Text

WARNING:
  Generating local explanations for each row can be large/slow for large datasets.
  Adjust as appropriate.
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle
import csv
import matplotlib.pyplot as plt

# Local imports
from explainability import TabNetExplainability
from tabnet_model import prepare_data  # label-encoding & scaling EXACTLY as in training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
FINALS_DIR = os.path.join(DATA_DIR, "finals")

# The final TabNet model from final_composite_none_tabnet.py
FINAL_TABNET_ZIP = os.path.join(FINALS_DIR, "final_composite_none_tabnet_model.zip")

# The pickled dataset used in training
PATIENT_DATA_PKL = os.path.join(DATA_DIR, "patient_data_with_health_index.pkl")

# For local explanation outputs
LIME_CSV = "lime_explanations.csv"  # Single CSV file with appended rows
SAVE_LIME_HTML = True              # Whether to also save each LIME explanation as HTML

def main():
    # ---------------------------------------------------------------------------------
    # 1) Load your full dataset
    # ---------------------------------------------------------------------------------
    if not os.path.exists(PATIENT_DATA_PKL):
        raise FileNotFoundError(f"Missing data file: {PATIENT_DATA_PKL}")

    patient_data = pd.read_pickle(PATIENT_DATA_PKL)
    logger.info(f"Loaded patient data shape={patient_data.shape}")

    # ---------------------------------------------------------------------------------
    # 2) Prepare data using the EXACT transformations from tabnet_model.prepare_data
    # ---------------------------------------------------------------------------------
    X, y, cat_idxs, cat_dims, feature_cols = prepare_data(
        patient_data=patient_data,
        target_col='Health_Index'  # or 'CharlsonIndex' if your final model used that
    )
    logger.info(f"Prepared data with shape={X.shape}. Feature columns => {feature_cols}")

    # Store patient IDs if available (fallback to index otherwise)
    if 'Id' in patient_data.columns:
        patient_ids = patient_data['Id'].values
    else:
        patient_ids = np.arange(len(X))

    # ---------------------------------------------------------------------------------
    # 3) Initialise TabNetExplainability
    # ---------------------------------------------------------------------------------
    xai = TabNetExplainability(
        model_path=FINAL_TABNET_ZIP,
        column_names=feature_cols,
        class_names=["HealthIndex"],  # not strictly needed for regression
        is_regression=True
    )
    xai.load_model()  # loads final TabNet model from disk

    # ---------------------------------------------------------------------------------
    # 4) SHAP Explanation for the ENTIRE dataset
    # ---------------------------------------------------------------------------------
    logger.info("[SHAP] Generating global SHAP values for the entire dataset.")
    shap_values = xai.shap_explain(
        X_train=X,       # background set
        X_sample=X,      # points to explain
        auto_sample=True,# randomly picks up to sample_size rows as background
        sample_size=200  
    )

    # SHAP Summary
    xai.plot_shap_summary(shap_values, max_display=12)
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("[SHAP] Global summary saved as shap_summary.png")

    # Save raw shap_values for future
    with open("shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)
    logger.info("[SHAP] shap_values saved to shap_values.pkl")

    # ---------------------------------------------------------------------------------
    # 5) LIME Explanation for each patient (LOCAL), logged to a single CSV
    # ---------------------------------------------------------------------------------
    logger.info("[LIME] Generating local explanations for EVERY patient to CSV.")
    # Build LIME explainer once using the entire dataset for distribution
    lime_exp = xai.lime_explainer(X)

    # 5a) Create (or overwrite) the CSV file with headers
    with open(LIME_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Patient_ID", "Explanation_Text"])

    # 5b) Loop over each row, generate local LIME explanation, append to CSV
    for i in range(len(X)):
        explanation = xai.lime_explain_instance(
            explainer=lime_exp,
            instance=X[i],
            label_index=0,  # regression => 0
            num_features=6
        )

        # Convert explanation.as_list() -> a single string
        # e.g. "[('AGE <= 0.52', -0.2), ('Hospitalizations_Count <= 0.1', 0.35), ... ]"
        explanation_pairs = explanation.as_list()
        # custom string formatting:
        explanation_text = " | ".join(f"{feat} => {weight:.4f}" for feat, weight in explanation_pairs)

        # Append one row to CSV
        with open(LIME_CSV, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([patient_ids[i], explanation_text])

        # Optionally also save an HTML file for each patient
        if SAVE_LIME_HTML:
            explanation.save_to_file(f"lime_explanation_{patient_ids[i]}.html")

    logger.info(f"[LIME] CSV-based local explanations saved to {LIME_CSV}")

    # ---------------------------------------------------------------------------------
    # 6) TabNet Intrinsic Feature Masks (Attention) - Demo on first 50
    # ---------------------------------------------------------------------------------
    logger.info("[ATTENTION] Extracting TabNet feature masks for the first 50 rows.")
    X_batch = X[:50]
    masks = xai.get_feature_masks(X_batch)

    # Plot step 0 as a heatmap
    xai.plot_feature_mask_heatmap(masks, step_index=0)
    plt.savefig("feature_mask_step_0.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Optionally save all step masks
    for step_idx, mask_array in enumerate(masks):
        np.save(f"feature_mask_step_{step_idx}.npy", mask_array)

    logger.info("[DONE] Entire-dataset explanations complete. LIME explanations logged to CSV.")

if __name__ == "__main__":
    main()

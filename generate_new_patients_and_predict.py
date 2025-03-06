#!/usr/bin/env python
"""
generate_new_patients_and_predict.py

A script to:
 1) Generate N new synthetic patients (with Synthea).
 2) Preprocess them (so they get a Health_Index, Charlson/Elixhauser, etc.).
 3) Make predictions using the three final TabNet models (diabetes, CKD, none).
 4) Store those predictions in distinct subfolders under Data/new_predictions/<model_id>/.
 5) Run final_explain_xai_clustered_lime.py so that XAI outputs for these new patients
    are saved separately and do not overwrite existing outputs.

No transfer learning is performed; we only load existing final models and predict.
"""

import os
import sys
import argparse
import subprocess
import shutil
from datetime import datetime
import logging
import gc

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# TabNet & utility
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

# Our existing modules
import data_preprocessing
import health_index
from vitai_scripts import data_prep
from vitai_scripts.subset_utils import filter_subpopulation
from vitai_scripts.feature_utils import select_features
from tabnet_model import prepare_data

# XAI module
import Explain_Xai.final_explain_xai_clustered_lime as xai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map each final model to the subset rules & feature config
MODEL_CONFIG_MAP = {
    "combined_diabetes_tabnet": {"subset": "diabetes", "feature_config": "combined"},
    "combined_all_ckd_tabnet":  {"subset": "ckd",      "feature_config": "combined_all"},
    "combined_none_tabnet":     {"subset": "none",     "feature_config": "combined"}
}

DATA_DIR    = os.path.join(os.getcwd(), "Data")
FINALS_DIR  = os.path.join(DATA_DIR, "finals")
PICKLE_PATH = os.path.join(DATA_DIR, "patient_data_with_all_indices.pkl")

#####################################
# 1) Synthea generation logic
#####################################
def run_synthea(pop_size):
    """
    Runs Synthea to generate `pop_size` synthetic patients
    (assuming you have a run_synthea.bat or similar in synthea-master).
    """
    synthea_dir = os.path.join(os.getcwd(), "synthea-master")
    if not os.path.exists(synthea_dir):
        logger.error("synthea-master directory not found. Cannot generate data.")
        sys.exit(1)

    cmd = ["run_synthea.bat", "-p", str(pop_size)]
    logger.info(f"Running Synthea to generate {pop_size} new patients...")
    try:
        subprocess.run(cmd, cwd=synthea_dir, check=True, shell=True)
        logger.info("Synthea generation complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during Synthea execution: {e}")
        sys.exit(1)

def copy_synthea_output():
    """
    Copies the newly generated Synthea CSVs into Data/ with a timestamp suffix,
    so that data_preprocessing sees them as 'diff' files.
    """
    synthea_output = os.path.join(os.getcwd(), "synthea-master", "output", "csv")
    if not os.path.exists(synthea_output):
        logger.error(f"Expected Synthea output directory {synthea_output} not found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DATA_DIR, exist_ok=True)
    expected_files = [
        "patients.csv", "encounters.csv", "conditions.csv",
        "medications.csv", "observations.csv", "procedures.csv"
    ]

    for filename in expected_files:
        src = os.path.join(synthea_output, filename)
        if not os.path.exists(src):
            logger.warning(f"{filename} not found in Synthea output.")
            continue
        base, ext = os.path.splitext(filename)
        dst_filename = f"{base}_diff_{timestamp}{ext}"
        dst = os.path.join(DATA_DIR, dst_filename)
        shutil.copy(src, dst)
        logger.info(f"Copied {src} to {dst}")

#####################################
# 2) Preprocessing
#####################################
def run_full_preprocessing():
    """
    Calls your standard data_preprocessing and health_index scripts,
    plus ensure_preprocessed_data from vitai_scripts/data_prep.py.
    This merges new patients into patient_data_with_all_indices.pkl
    and calculates their Health_Index.
    """
    logger.info("Running data_preprocessing...")
    data_preprocessing.main()

    logger.info("Running health_index...")
    health_index.main()

    logger.info("Running ensure_preprocessed_data...")
    data_prep.ensure_preprocessed_data(DATA_DIR)


#####################################
# 3) Inference (predictions) logic
#####################################
def load_tabnet_model(model_id):
    """
    Loads a final TabNetRegressor from Data/finals/<model_id>/<model_id>_model.zip
    """
    model_dir = os.path.join(FINALS_DIR, model_id)
    model_path = os.path.join(model_dir, f"{model_id}_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find final model at {model_path}")
    regressor = TabNetRegressor()
    regressor.load_model(model_path)
    logger.info(f"Loaded final model: {model_id}")
    return regressor

def run_predictions_on_new_patients():
    """
    1) Load the updated patient_data_with_all_indices.pkl
    2) Isolate only the newly generated patients (NewData=True).
    3) For each final model, filter subpopulation, run inference, and save predictions.
    """
    if not os.path.exists(PICKLE_PATH):
        logger.error(f"{PICKLE_PATH} not found. Please check your preprocessing.")
        sys.exit(1)

    df_all = pd.read_pickle(PICKLE_PATH)
    df_new = df_all[df_all["NewData"] == True].copy()
    logger.info(f"Found {len(df_new)} newly generated patients for inference.")

    # For each final model in your pipeline
    for model_id, config in MODEL_CONFIG_MAP.items():
        subset_type     = config["subset"]
        feature_config  = config["feature_config"]

        logger.info(f"\nPredicting with model={model_id}, subset={subset_type}...")

        # Filter subpopulation (e.g. only patients with diabetes if model=diabetes)
        sub_df = filter_subpopulation(df_new, subset_type, DATA_DIR)
        if sub_df.empty:
            logger.info(f"No new patients belong to subpopulation='{subset_type}'. Skipping.")
            continue

        # Select features for that model
        feats_df = select_features(sub_df, feature_config)
        if feats_df.empty:
            logger.info("No features left after selection. Skipping.")
            continue

        # Prepare arrays for TabNet
        X, _, cat_idxs, cat_dims, feat_names = prepare_data(feats_df, target_col="Health_Index")
        if X.shape[0] == 0:
            logger.info("No rows to predict. Skipping.")
            continue

        # Load the final TabNet model
        try:
            model = load_tabnet_model(model_id)
        except FileNotFoundError:
            logger.warning(f"Model {model_id} not found, skipping.")
            continue

        # Make predictions
        preds = model.predict(X).flatten()

        # Prepare and save predictions in Data/new_predictions/<model_id>/...
        pred_subfolder = os.path.join(DATA_DIR, "new_predictions", model_id)
        os.makedirs(pred_subfolder, exist_ok=True)

        out_df = pd.DataFrame({
            "Id": feats_df["Id"].values,
            "Predicted_Health_Index": preds
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(pred_subfolder, f"{model_id}_predictions_{timestamp}.csv")
        out_df.to_csv(out_csv, index=False)
        logger.info(f"Predictions saved -> {out_csv}")

    # Optionally, set NewData=False once you’re done
    # so these patients won't keep reappearing as new in future runs.
    df_all.loc[df_all["NewData"] == True, "NewData"] = False
    df_all.to_pickle(PICKLE_PATH)
    logger.info("Marked new patients as processed (NewData=False).")


#####################################
# 4) Run XAI on newly added patients
#####################################
def run_explainability():
    """
    Invokes final_explain_xai_clustered_lime.py’s main() function
    so it will generate SHAP, IG, Anchors, LIME, etc. for the updated dataset.

    By default, that script places results into Data/explain_xai/<model_id>/.
    If you want them in a separate subfolder, you can tweak xai.EXPLAIN_DIR here.
    """
    logger.info("Running advanced XAI pipeline on newly added patients.")
    #  redirect outputs to subfolders:
    xai.EXPLAIN_DIR = os.path.join(DATA_DIR, "explain_xai", "new_inference")
    xai.FULL_DATA_PKL = PICKLE_PATH
    xai.main()


#####################################
# Main script
#####################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate new synthetic patients, preprocess, predict with final TabNet models, then run XAI."
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Number of synthetic patients to generate (default=100)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    pop_size = args.population

    # 1) Generate N new patients
    run_synthea(pop_size)
    copy_synthea_output()

    # 2) Preprocess
    run_full_preprocessing()

    # 3) Make predictions with no fine-tuning
    run_predictions_on_new_patients()

    # 4) Run XAI on the updated dataset
    # run_explainability()

    logger.info("Done! Generated new data, made predictions, and ran XAI.")


if __name__ == "__main__":
    main()

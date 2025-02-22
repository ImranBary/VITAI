#!/usr/bin/env python
import os
import sys
import argparse
import logging
import subprocess
import shutil
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import TabNet and our utility functions
from pytorch_tabnet.tab_model import TabNetRegressor
from tabnet_model import prepare_data  # Prepares features and returns X, y, cat_idxs, cat_dims, feature_names

# These utility functions live in your vitai_scripts folder
from vitai_scripts.subset_utils import filter_subpopulation
from vitai_scripts.feature_utils import select_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from model ID substrings to subpopulation and feature configuration
MODEL_CONFIG_MAP = {
    "diabetes": {"subset": "diabetes", "feature_config": "combined"},
    "ckd": {"subset": "ckd", "feature_config": "combined_all"},
    "none": {"subset": "none", "feature_config": "combined"}
}

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transfer Learning Pipeline for Differential Data with Automatic Synthea Generation"
    )
    parser.add_argument("--model_ids", nargs="+", required=True,
                        help="List of final model IDs to update (e.g., combined_diabetes_tabnet combined_all_ckd_tabnet combined_none_tabnet)")
    parser.add_argument("--finetune_epochs", type=int, default=20,
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--synthea_population_size", type=int, default=1000,
                        help="Number of patients to generate via Synthea")
    return parser.parse_args()

def trigger_synthea(pop_size):
    """
    Triggers Synthea to generate synthetic patient data.
    Assumes Synthea is located in the "./Synthea" folder.
    """
    synthea_dir = os.path.join(os.getcwd(), "Synthea")
    if not os.path.exists(synthea_dir):
        logger.error("Synthea directory not found.")
        sys.exit(1)
    # Ensure CSV export is enabled via command-line flag
    synthea_command = ["./run_synthea", "-p", str(pop_size), "--exporter.csv.export=true"]
    logger.info(f"Triggering Synthea to generate {pop_size} patients...")
    try:
        subprocess.run(synthea_command, cwd=synthea_dir, check=True)
        logger.info("Synthea generation complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during Synthea execution: {e}")
        sys.exit(1)

def copy_synthea_output_to_data():
    """
    Copies the generated CSV files from Synthea's output folder to the project's Data folder.
    Expected files: patients.csv, encounters.csv, conditions.csv, medications.csv, observations.csv, procedures.csv
    """
    synthea_output = os.path.join(os.getcwd(), "Synthea", "output", "csv")
    data_dir = os.path.join(os.getcwd(), "Data")
    os.makedirs(data_dir, exist_ok=True)
    expected_files = [
        "patients.csv", "encounters.csv", "conditions.csv",
        "medications.csv", "observations.csv", "procedures.csv"
    ]
    for filename in expected_files:
        src = os.path.join(synthea_output, filename)
        dst = os.path.join(data_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            logger.info(f"Copied {src} to {dst}")
        else:
            logger.error(f"Expected Synthea output file {src} not found.")
            sys.exit(1)

def run_preprocessing_pipeline():
    """
    Runs the entire preprocessing pipeline:
      1) data_preprocessing.py to process CSV files and create patient_data_sequences.pkl
      2) health_index.py to compute the composite Health_Index and save patient_data_with_health_index.pkl
      3) vitai_scripts/data_prep.py to merge indices and produce patient_data_with_all_indices.pkl
    """
    import data_preprocessing
    import health_index
    from vitai_scripts import data_prep
    logger.info("Running data preprocessing...")
    data_preprocessing.main()
    logger.info("Running health index computation...")
    health_index.main()
    logger.info("Ensuring merged data with all indices...")
    data_prep.ensure_preprocessed_data(os.path.join(os.getcwd(), "Data"))
    logger.info("Preprocessing pipeline complete.")

def load_differential_data():
    """
    Loads the merged differential data from Data/patient_data_with_all_indices.pkl.
    Ensures exactly 1,000 patients by sampling if needed.
    """
    differential_pkl = os.path.join(os.getcwd(), "Data", "patient_data_with_all_indices.pkl")
    if not os.path.exists(differential_pkl):
        logger.error(f"Differential data pickle not found at {differential_pkl}")
        sys.exit(1)
    diff_data = pd.read_pickle(differential_pkl)
    logger.info(f"Loaded differential data with shape {diff_data.shape}")
    if diff_data.shape[0] != 1000:
        logger.info(f"Differential data has {diff_data.shape[0]} patients; sampling 1000 patients.")
        diff_data = diff_data.sample(n=1000, random_state=42)
    return diff_data

def load_pretrained_model(model_id, finals_dir):
    """
    Loads the pre-trained TabNet model for a given model_id.
    Assumes the model is stored at: <finals_dir>/<model_id>/<model_id>_model.zip
    """
    model_dir = os.path.join(finals_dir, model_id)
    model_path = os.path.join(model_dir, f"{model_id}_model.zip")
    if not os.path.exists(model_path):
        logger.error(f"Pre-trained model not found at {model_path}")
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
    model = TabNetRegressor()
    model.load_model(model_path)
    logger.info(f"Loaded pre-trained model from {model_path}")
    return model

def finetune_model(model, X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims, learning_rate, epochs):
    """
    Fine-tunes the pre-trained model on the new differential data.
    """
    optimizer_params = {"lr": learning_rate}
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["rmse"],
        max_epochs=epochs,
        patience=5,
        batch_size=4096,
        virtual_batch_size=1024,
        optimizer_params=optimizer_params,
        verbose=1
    )
    return model

def trigger_explainable_ai(differential_pkl):
    """
    Triggers the Explainable AI pipeline for the differential data.
    Sets the FULL_DATA_PKL variable in final_explain_xai_clustered_lime.
    """
    try:
        import Explain_Xai.final_explain_xai_clustered_lime as explainer
        explainer.FULL_DATA_PKL = differential_pkl
        logger.info("Triggering Explainable AI pipeline for differential data...")
        explainer.main()
    except Exception as e:
        logger.error(f"Error triggering Explainable AI pipeline: {e}")

def main():
    args = parse_arguments()
    data_dir = os.path.join(os.getcwd(), "Data")
    finals_dir = os.path.join(data_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)

    # 1. Trigger Synthea to generate new differential data
    trigger_synthea(args.synthea_population_size)
    # 2. Copy Synthea output CSV files to Data folder
    copy_synthea_output_to_data()
    # 3. Run the preprocessing pipeline to generate processed data files
    run_preprocessing_pipeline()
    # 4. Load the differential data (merged with all indices)
    diff_data = load_differential_data()

    # 5. For each model, perform transfer learning on the differential data
    for model_id in args.model_ids:
        logger.info(f"--- Processing model: {model_id} ---")
        config = None
        for key in MODEL_CONFIG_MAP:
            if key in model_id.lower():
                config = MODEL_CONFIG_MAP[key]
                break
        if config is None:
            logger.warning(f"Model ID {model_id} does not match known subset keys. Defaulting to 'none'.")
            config = {"subset": "none", "feature_config": "combined"}
        subset_type = config["subset"]
        feature_config = config["feature_config"]
        logger.info(f"Using subpopulation: {subset_type} and feature configuration: {feature_config}")

        # Filter the differential data by subpopulation using subset_utils
        diff_subset = filter_subpopulation(diff_data, subset_type, data_dir)
        if diff_subset.empty:
            logger.warning(f"No differential data for subpopulation '{subset_type}'. Skipping model {model_id}.")
            continue

        # Select features using feature_utils
        diff_features = select_features(diff_subset, feature_config)
        logger.info(f"Data shape after feature selection: {diff_features.shape}")

        # Prepare data for TabNet using your prepare_data function
        try:
            X, y, cat_idxs, cat_dims, feature_names = prepare_data(diff_features, target_col="Health_Index")
        except Exception as e:
            logger.error(f"Error during data preparation: {e}")
            continue

        # Split the data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_valid.shape}")

        # Load the corresponding pre-trained model from the finals folder
        try:
            model = load_pretrained_model(model_id, finals_dir)
        except Exception as e:
            logger.error(f"Error loading pre-trained model for {model_id}: {e}")
            continue

        # Fine-tune (transfer learn) the model on the new differential data
        logger.info(f"Fine-tuning model {model_id} for {args.finetune_epochs} epochs at learning rate {args.learning_rate}")
        updated_model = finetune_model(model, X_train, y_train, X_valid, y_valid,
                                       cat_idxs, cat_dims,
                                       learning_rate=args.learning_rate,
                                       epochs=args.finetune_epochs)

        # Save the updated model with a new filename in the finals folder
        updated_model_path = os.path.join(finals_dir, model_id, f"{model_id}_updated_model")
        updated_model.save_model(updated_model_path)
        logger.info(f"Updated model saved to {updated_model_path}.zip")

    # 6. Trigger the Explainable AI pipeline on the differential data
    differential_pkl = os.path.join(data_dir, "patient_data_with_all_indices.pkl")
    trigger_explainable_ai(differential_pkl)

if __name__ == "__main__":
    main()

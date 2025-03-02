#!/usr/bin/env python
import os
import sys
import argparse
import logging
import subprocess
import shutil
from datetime import datetime
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gc  # Import the garbage collection module

# TabNet & utility
from pytorch_tabnet.tab_model import TabNetRegressor
from tabnet_model import prepare_data
from vitai_scripts.subset_utils import filter_subpopulation
from vitai_scripts.feature_utils import select_features

# Preprocessing & XAI
import data_preprocessing
import health_index
from vitai_scripts import data_prep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIG_MAP = {
    "diabetes": {"subset": "diabetes", "feature_config": "combined"},
    "ckd": {"subset": "ckd", "feature_config": "combined_all"},
    "none": {"subset": "none", "feature_config": "combined"}
}

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transfer Learning Pipeline for Differential Data with Automatic Synthea Generation"
    )
    parser.add_argument(
        "--model_ids",
        nargs="+",
        default=[
            "combined_diabetes_tabnet",
            "combined_all_ckd_tabnet",
            "combined_none_tabnet"
        ],
        help="List of final model IDs to update/fine-tune"
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=20,
        help="Number of epochs for fine-tuning."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning."
    )
    parser.add_argument(
        "--synthea_population_size",
        type=int,
        default=1000,
        help="Number of patients to generate via Synthea."
    )
    return parser.parse_args()

def trigger_synthea(pop_size):
    """
    Runs Synthea to generate synthetic data (on Windows).
    Expects a run_synthea.bat in ./synthea-master.
    """
    synthea_dir = os.path.join(os.getcwd(), "synthea-master")
    if not os.path.exists(synthea_dir):
        logger.error("synthea-master directory not found.")
        sys.exit(1)
    synthea_command = ["run_synthea.bat", "-p", str(pop_size)]
    logger.info(f"Triggering Synthea to generate {pop_size} patients...")
    try:
        subprocess.run(synthea_command, cwd=synthea_dir, check=True, shell=True)
        logger.info("Synthea generation complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during Synthea execution: {e}")
        sys.exit(1)

def copy_synthea_output_to_data():
    """
    Copies new CSV files from Synthea output to Data folder, adding a timestamp suffix.
    """
    synthea_output = os.path.join(os.getcwd(), "synthea-master", "output", "csv")
    data_dir = os.path.join(os.getcwd(), "Data")
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expected_files = [
        "patients.csv", "encounters.csv", "conditions.csv",
        "medications.csv", "observations.csv", "procedures.csv"
    ]
    for filename in expected_files:
        src = os.path.join(synthea_output, filename)
        base, ext = os.path.splitext(filename)
        dst_filename = f"{base}_diff_{timestamp}{ext}"
        dst = os.path.join(data_dir, dst_filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            logger.info(f"Copied {src} to {dst}")
        else:
            logger.error(f"Expected Synthea output file {src} not found.")
            sys.exit(1)

def load_pretrained_model(model_id, finals_dir):
    """
    Loads the final TabNet model (pretrained) from Data/finals/<model_id>/<model_id>_model.zip
    """
    model_dir = os.path.join(finals_dir, model_id)
    model_path = os.path.join(model_dir, f"{model_id}_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
    regressor = TabNetRegressor()
    regressor.load_model(model_path)
    logger.info(f"Loaded pre-trained model {model_id} from {model_path}")
    return regressor

def finetune_tabnet(
    model: TabNetRegressor,
    X_train, y_train,
    X_valid, y_valid,
    cat_idxs, cat_dims,
    learning_rate=1e-4,
    max_epochs=20,
    batch_size=4096,
    virtual_batch_size=1024,
    patience=5
):
    """
    Fine-tunes (continues training) an already loaded/pre-trained TabNetRegressor `model`
    on (X_train, y_train), validating on (X_valid, y_valid), *without* losing the trained weights.
    
    Args:
        model: A TabNetRegressor that has already been load_model()'ed or trained previously.
        X_train, y_train: Numpy arrays for training
        X_valid, y_valid: Numpy arrays for validation
        cat_idxs, cat_dims: Categorical info (only needed if you originally used it for embeddings)
        learning_rate (float): The LR to use for fine-tuning.
        max_epochs (int): Number of epochs for fine-tuning.
        batch_size (int)
        virtual_batch_size (int)
        patience (int): Early stopping patience.

    Returns:
        The fine-tuned model (same TabNet object, updated).
    """
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["rmse"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        learning_rate=learning_rate,
        device_name="cuda",
        verbose=1
    )

    return model

def trigger_explainable_ai():
    """
    Runs the advanced explainability pipeline from final_explain_xai_clustered_lime.py,
    pointing it at Data/patient_data_with_all_indices.pkl.
    """
    try:
        import Explain_Xai.final_explain_xai_clustered_lime as expl
        logger.info("Running Explainable AI pipeline on the updated data (append mode if supported).")
        expl.FULL_DATA_PKL = os.path.join("Data", "patient_data_with_all_indices.pkl")
        # If your XAI script supports an APPEND_MODE param, you could set that here:
        # expl.APPEND_MODE = True
        expl.main()
    except Exception as e:
        logger.error(f"Error running final_explain_xai_clustered_lime: {e}")

def main():
    """
    1) Generate & copy new Synthea data.
    2) Preprocess fully (data_preprocessing + health_index + ensure_preprocessed_data),
       so that new patients have Charlson/Elixhauser & remain marked as NewData==True.
    3) Load those new patients from patient_data_with_all_indices.pkl, sub-select each model's subpopulation,
       fine-tune with 'combined' features if needed, and save an updated model.
    4) Merge them again if you'd like to set NewData=False (or you can skip that).
    5) Trigger the XAI pipeline last.
    """
    args = parse_arguments()
    data_dir = os.path.join(os.getcwd(), "Data")
    finals_dir = os.path.join(data_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)

    # 1) Generate and copy new Synthea data
    trigger_synthea(args.synthea_population_size)
    copy_synthea_output_to_data()

    # 2) Run the full preprocessing so new patients have Health_Index and Charlson/Elixhauser
    data_preprocessing.main()
    health_index.main()
    data_prep.ensure_preprocessed_data(data_dir)

    # 3) Load from patient_data_with_all_indices.pkl, keeping only NewData==True
    final_pkl = os.path.join(data_dir, "patient_data_with_all_indices.pkl")
    if not os.path.exists(final_pkl):
        logger.error(f"Missing final merged pickle {final_pkl}")
        sys.exit(1)
    df = pd.read_pickle(final_pkl)
    new_data = df[df["NewData"] == True]
    logger.info(f"Found {len(new_data)} new patients for transfer learning.")

    # Drop the full dataset from memory
    del df
    gc.collect()

    # 4) Fine-tune each model on the new data
    for model_id in args.model_ids:
        logger.info(f"\n--- Fine-tuning {model_id} ---")
        # Identify subpopulation & feature config
        matched = False
        for key in MODEL_CONFIG_MAP:
            if key in model_id.lower():
                subset = MODEL_CONFIG_MAP[key]["subset"]
                feat_cfg = MODEL_CONFIG_MAP[key]["feature_config"]
                matched = True
                break
        if not matched:
            subset, feat_cfg = "none", "combined"

        # Filter subpopulation
        sub_df = filter_subpopulation(new_data, subset, data_dir)
        if sub_df.empty:
            logger.info(f"No new data for subpopulation='{subset}'. Skipping {model_id}.")
            continue

        # Select features (CharlsonIndex is available now, so 'combined' is fine)
        sub_feats = select_features(sub_df, feat_cfg)
        if sub_feats.empty:
            logger.info(f"No features left after selection. Skipping {model_id}.")
            continue

        # Prepare data for TabNet
        X, y, cat_idxs, cat_dims, feat_names = prepare_data(sub_feats, target_col="Health_Index")
        if X.shape[0] < 2:
            logger.info("Not enough rows to fine-tune. Skipping.")
            continue

        # Drop intermediate dataframes from memory
        del sub_df, sub_feats
        gc.collect()

        # Train/valid split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Load the pretrained model
        try:
            model = load_pretrained_model(model_id, finals_dir)
        except FileNotFoundError:
            logger.warning(f"No pretrained model found for {model_id}, skipping.")
            continue

        # Fine-tune
        logger.info(f"Fine-tuning {model_id} with {X_train.shape[0]} train rows.")
        model = finetune_tabnet(
            model, X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims,
            args.learning_rate, args.finetune_epochs
        )

        # Save updated model
        updated_model_path = os.path.join(finals_dir, model_id, f"{model_id}_updated_model")
        model.save_model(updated_model_path)
        logger.info(f"Saved updated model -> {updated_model_path}.zip")

        # Drop training data from memory
        del X_train, X_valid, y_train, y_valid, X, y, cat_idxs, cat_dims, feat_names
        gc.collect()

    # 5) Merge new patients back into the main dataset 
    data_prep.ensure_preprocessed_data(data_dir)
    final_df = pd.read_pickle(final_pkl)
    final_df.loc[final_df["NewData"] == True, "NewData"] = False
    final_df.to_pickle(final_pkl)
    logger.info("All new patients now marked as processed (NewData=False).")

    # Drop final_df from memory
    del final_df
    gc.collect()

    # 6) Run the XAI pipeline on the updated data
    trigger_explainable_ai()

    logger.info("\nDone! Transfer learning + XAI pipeline completed in a single pass.\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import os
import sys
import argparse
import logging
import subprocess
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# TabNet and utility imports
from pytorch_tabnet.tab_model import TabNetRegressor
from tabnet_model import prepare_data
from vitai_scripts.subset_utils import filter_subpopulation
from vitai_scripts.feature_utils import select_features

# Preprocessing & XAI modules
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
    parser.add_argument("--model_ids", nargs="+", default=[
        "combined_diabetes_tabnet",
        "combined_all_ckd_tabnet",
        "combined_none_tabnet"
    ],
    help="List of final model IDs to update.")
    parser.add_argument("--finetune_epochs", type=int, default=20,
                        help="Number of epochs for fine-tuning.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for fine-tuning.")
    parser.add_argument("--synthea_population_size", type=int, default=1000,
                        help="Number of patients to generate via Synthea.")
    return parser.parse_args()

def trigger_synthea(pop_size):
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

def run_basic_preprocessing():
    """
    1) data_preprocessing.main() -> patient_data_sequences.pkl
    2) health_index.main()       -> patient_data_with_health_index.pkl
    (We intentionally skip data_prep.ensure_preprocessed_data here so that new patients
     remain marked NewData=True. We'll do the full merge after training.)
    """
    logger.info("Running data_preprocessing...")
    data_preprocessing.main()
    logger.info("Running health_index...")
    health_index.main()

def load_new_data_for_transfer():
    """
    Load from patient_data_with_health_index.pkl, restricting to rows where NewData==True.
    If fewer or more than 1000 new rows exist, we can sample or skip, as desired.
    """
    hi_path = os.path.join("Data", "patient_data_with_health_index.pkl")
    if not os.path.exists(hi_path):
        logger.error(f"{hi_path} not found. Did you run health_index?")
        sys.exit(1)

    df = pd.read_pickle(hi_path)
    new_data = df[df["NewData"] == True]
    logger.info(f"Loaded {df.shape[0]} rows total, found {new_data.shape[0]} new patients.")
    if new_data.empty:
        logger.info("No new patients to transfer-learn on. Exiting.")
        sys.exit(0)
    # If you prefer always exactly 1000, you can sample here:
    # if new_data.shape[0] != 1000:
    #     new_data = new_data.sample(1000, random_state=42)  # or fallback to skip
    return new_data

def load_pretrained_model(model_id, finals_dir):
    model_dir = os.path.join(finals_dir, model_id)
    model_path = os.path.join(model_dir, f"{model_id}_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
    regressor = TabNetRegressor()
    regressor.load_model(model_path)
    logger.info(f"Loaded pre-trained model {model_id} from {model_path}")
    return regressor

def finetune_model(model, X_train, y_train, X_valid, y_valid,
                   cat_idxs, cat_dims,
                   learning_rate, epochs):
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["rmse"],
        max_epochs=epochs,
        patience=5,
        batch_size=4096,
        virtual_batch_size=1024,
        optimizer_params={"lr": learning_rate},
        verbose=1
    )
    return model

def trigger_explainable_ai():
    """
    Calls final_explain_xai_clustered_lime.main(), pointing it to the up-to-date
    patient_data_with_all_indices.pkl.  We assume the code in final_explain_xai_clustered_lime
    is configured to append or overwrite as needed.
    """
    try:
        import Explain_Xai.final_explain_xai_clustered_lime as expl
        logger.info("Running XAI pipeline on updated dataset (append mode).")
        expl.FULL_DATA_PKL = os.path.join("Data", "patient_data_with_all_indices.pkl")
        # If your script supports an APPEND_MODE variable, set it:
        # expl.APPEND_MODE = True
        expl.main()
    except Exception as e:
        logger.error(f"Error running final_explain_xai_clustered_lime: {e}")

def main():
    args = parse_arguments()
    data_dir = os.path.join(os.getcwd(), "Data")
    finals_dir = os.path.join(data_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)

    # 1) Generate and copy new Synthea data
    trigger_synthea(args.synthea_population_size)
    copy_synthea_output_to_data()

    # 2) Basic preprocessing (without full index merging)
    run_basic_preprocessing()

    # 3) Load new data from patient_data_with_health_index.pkl
    new_data = load_new_data_for_transfer()

    # 4) Transfer learning for each model
    for model_id in args.model_ids:
        logger.info(f"\n--- Fine-tuning {model_id} ---")
        # Find subpopulation & feature config
        matched = False
        for key in MODEL_CONFIG_MAP:
            if key in model_id.lower():
                subset = MODEL_CONFIG_MAP[key]["subset"]
                feat_cfg = MODEL_CONFIG_MAP[key]["feature_config"]
                matched = True
                break
        if not matched:
            subset, feat_cfg = "none", "combined"

        # Subset new_data
        sub_df = filter_subpopulation(new_data, subset, data_dir)
        if sub_df.empty:
            logger.info(f"No new data for subpopulation='{subset}'. Skipping {model_id}.")
            continue

        # Feature selection
        sub_feats = select_features(sub_df, feat_cfg)
        if sub_feats.empty:
            logger.info(f"No features left after selection. Skipping {model_id}.")
            continue

        # Prepare data for TabNet
        X, y, cat_idxs, cat_dims, feat_names = prepare_data(sub_feats, target_col="Health_Index")

        if X.shape[0] < 2:
            logger.info("Not enough rows to fine-tune. Skipping.")
            continue

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        # Load pre-trained model
        try:
            model = load_pretrained_model(model_id, finals_dir)
        except FileNotFoundError:
            logger.warning(f"No pretrained model found for {model_id}, skipping.")
            continue

        # Fine-tune
        logger.info(f"Fine-tuning {model_id} with {X_train.shape[0]} train rows.")
        model = finetune_model(
            model, X_train, y_train, X_valid, y_valid,
            cat_idxs, cat_dims,
            args.learning_rate, args.finetune_epochs
        )

        # Save updated model
        updated_model_path = os.path.join(finals_dir, model_id, f"{model_id}_updated_model")
        model.save_model(updated_model_path)
        logger.info(f"Saved updated model -> {updated_model_path}.zip")

    # 5) Now that we have updated any relevant models, do the final merge of new patients
    #    with Charlson & Elixhauser indices, set NewData=False, etc.
    logger.info("\nMerging new patients into the final dataset with charlson/elixhauser indices...")
    data_prep.ensure_preprocessed_data(data_dir)
    logger.info("Finished merging new patients into patient_data_with_all_indices.pkl.")

    # 6) Run XAI pipeline on the updated "patient_data_with_all_indices.pkl"
    trigger_explainable_ai()

    logger.info("\nDone! Transfer learning + XAI pipeline completed in a single pass.\n")

if __name__ == "__main__":
    main()

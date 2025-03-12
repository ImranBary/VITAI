#!/usr/bin/env python
"""
run_patient_group_predictions.py

Script that takes a CSV file with patient features, splits patients into groups
(diabetes, CKD, none), and runs appropriate TabNet models for each group.

Usage:
  python run_patient_group_predictions.py --input-csv=PatientFeatures.csv [--force-cpu]
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# For TabNet inference
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# Import existing utils
from vitai_scripts.subset_utils import filter_subpopulation
from vitai_scripts.feature_utils import select_features
from tabnet_inference import load_model_artifacts, preprocess_features

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Define model configurations for each patient group
MODEL_CONFIG_MAP = {
    "combined_diabetes_tabnet": {"subset": "diabetes", "feature_config": "combined"},
    "combined_all_ckd_tabnet":  {"subset": "ckd",      "feature_config": "combined_all"},
    "combined_none_tabnet":     {"subset": "none",     "feature_config": "combined"}
}

def identify_patient_groups(df):
    """
    Identify which patients belong to which subset (diabetes, CKD, or none)
    This is a simplified version - in practice would call the existing Python utilities
    """
    # This is placeholder logic - would need to be replaced with calls to actual subset utilities
    diabetes_ids = set()
    ckd_ids = set()
    
    # Try to load condition data or use info from the features
    try:
        # Look for Data/conditions_*.csv files (most recent)
        data_dir = "Data"
        cond_files = [f for f in os.listdir(data_dir) if f.startswith("conditions_") and f.endswith(".csv")]
        if cond_files:
            # Sort by timestamp in filename to get most recent
            cond_files.sort(reverse=True)
            conditions_df = pd.read_csv(os.path.join(data_dir, cond_files[0]))
            
            # Extract conditions with diabetes or CKD
            for _, row in conditions_df.iterrows():
                desc = row['DESCRIPTION'].lower() if 'DESCRIPTION' in conditions_df.columns else ''
                patient_id = row['PATIENT'] if 'PATIENT' in conditions_df.columns else None
                
                if patient_id:
                    if "diabetes" in desc:
                        diabetes_ids.add(patient_id)
                    if "chronic kidney disease" in desc or "ckd" in desc:
                        ckd_ids.add(patient_id)
    except Exception as e:
        logger.warning(f"Could not process conditions file: {e}")
    
    # Create group mappings
    patient_groups = {}
    for patient_id in df['Id'].unique():
        if patient_id in diabetes_ids:
            patient_groups[patient_id] = "diabetes"
        elif patient_id in ckd_ids:
            patient_groups[patient_id] = "ckd"
        else:
            patient_groups[patient_id] = "none"
    
    return patient_groups

def run_model_for_group(df, group_name, model_id, force_cpu=False):
    """Run the appropriate model for a patient group"""
    logger.info(f"Running model '{model_id}' for patient group '{group_name}'")
    
    # Load model
    try:
        regressor, _ = load_model_artifacts(model_id, force_cpu)
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        return None
    
    # Preprocess features
    try:
        patient_ids, X = preprocess_features(df, model_id)
        if len(patient_ids) == 0:
            logger.warning(f"No patients in {group_name} group after preprocessing")
            return None
    except Exception as e:
        logger.error(f"Error preprocessing features: {e}")
        return None
    
    # Run inference
    try:
        preds = regressor.predict(X).flatten()
        logger.info(f"Generated {len(preds)} predictions for {group_name} group")
        
        # Create output dataframe
        results_df = pd.DataFrame({
            "Id": patient_ids,
            f"Predicted_Health_Index": preds
        })
        
        return results_df
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return None

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run TabNet models on patient groups")
    parser.add_argument("--input-csv", required=True, help="Input CSV with patient features")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV not found: {args.input_csv}")
        return 1
    
    # Load data
    try:
        df = pd.read_csv(args.input_csv)
        logger.info(f"Loaded {len(df)} patients from {args.input_csv}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return 1
    
    # Identify patient groups
    patient_groups = identify_patient_groups(df)
    
    # Count patients in each group
    group_counts = {"diabetes": 0, "ckd": 0, "none": 0}
    for _, group in patient_groups.items():
        group_counts[group] += 1
    
    logger.info(f"Patient groups: Diabetes={group_counts['diabetes']}, "
                f"CKD={group_counts['ckd']}, None={group_counts['none']}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("Data", "new_predictions")
    os.makedirs(out_dir, exist_ok=True)
    
    # Process each group with its respective model
    for model_id, config in MODEL_CONFIG_MAP.items():
        group_name = config["subset"]
        
        # Filter patients belonging to this group
        group_patient_ids = [pid for pid, group in patient_groups.items() if group == group_name]
        if not group_patient_ids:
            logger.info(f"No patients in '{group_name}' group, skipping model '{model_id}'")
            continue
        
        # Filter dataframe to only include these patients
        group_df = df[df['Id'].isin(group_patient_ids)].copy()
        
        # Run model and get predictions
        results_df = run_model_for_group(group_df, group_name, model_id, args.force_cpu)
        if results_df is not None:
            # Save predictions
            out_path = os.path.join(out_dir, f"{model_id}_predictions_{timestamp}.csv")
            results_df.to_csv(out_path, index=False)
            logger.info(f"Saved predictions to {out_path}")
    
    logger.info("Multi-model inference completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())

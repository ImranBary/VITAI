"""
Utility script to create missing scalers for TabNet models.
This script examines each model in the Data/finals directory,
identifies missing scalers, and creates them.

Usage:
    python create_missing_scalers.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add root directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(ROOT_DIR)

# Path to finals directory
FINALS_DIR = os.path.join(ROOT_DIR, "Data", "finals")

def create_scaler_for_model(model_id):
    """Creates and saves a scaler for the specified model."""
    model_dir = os.path.join(FINALS_DIR, model_id)
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return False
    
    # Check if scaler already exists
    scaler_path = os.path.join(model_dir, f"{model_id}_scaler.joblib")
    if os.path.exists(scaler_path):
        logger.info(f"Scaler already exists for {model_id}")
        return True
    
    # Find prediction CSV for this model
    pred_csv = os.path.join(model_dir, f"{model_id}_predictions.csv")
    if not os.path.exists(pred_csv):
        logger.warning(f"Predictions CSV not found: {pred_csv}")
        return False
    
    try:
        # Find the original features used by looking at all CSVs in the directory
        # Start by loading the prediction CSV to get patient IDs
        preds_df = pd.read_csv(pred_csv)
        logger.info(f"Loaded predictions for {model_id}: {len(preds_df)} rows")
        
        # Look for PatientFeatures.csv in the main directory
        patient_features_path = os.path.join(ROOT_DIR, "PatientFeatures.csv")
        if not os.path.exists(patient_features_path):
            logger.warning(f"PatientFeatures.csv not found at {patient_features_path}")
            return False
        
        # Load feature data
        features_df = pd.read_csv(patient_features_path)
        logger.info(f"Loaded features: {features_df.shape}")
        
        # Define categorical and continuous columns (same as in the model training)
        categorical_columns = ['Gender']  # Simplified from the original which had 'DECEASED','GENDER',etc
        
        # Get all numeric columns except Id and Health_Index
        all_columns = features_df.columns.tolist()
        continuous_columns = [col for col in all_columns 
                             if col not in categorical_columns 
                             and col != 'Id' 
                             and col != 'Health_Index']
        
        logger.info(f"Continuous columns for scaling: {continuous_columns}")
        
        if not continuous_columns:
            logger.warning("No continuous columns found for scaling")
            return False
        
        # Create and fit the scaler
        scaler = StandardScaler()
        scaler.fit(features_df[continuous_columns])
        
        # Save the scaler
        joblib.dump(scaler, scaler_path)
        logger.info(f"Created and saved scaler for {model_id} at {scaler_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating scaler for {model_id}: {e}")
        return False

def main():
    """Main function to create missing scalers for all models."""
    if not os.path.exists(FINALS_DIR):
        logger.error(f"Finals directory not found: {FINALS_DIR}")
        return
    
    # Get list of model directories
    model_dirs = [d for d in os.listdir(FINALS_DIR) 
                 if os.path.isdir(os.path.join(FINALS_DIR, d))]
    
    logger.info(f"Found {len(model_dirs)} model directories")
    
    for model_id in model_dirs:
        logger.info(f"Processing model: {model_id}")
        success = create_scaler_for_model(model_id)
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{status}: Create scaler for {model_id}")
    
    logger.info("Script completed")

if __name__ == "__main__":
    main()

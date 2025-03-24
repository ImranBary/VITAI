"""
TabNet Adapter for Patient Feature Data

This script reorders and formats input features to be compatible with TabNet models.
It handles the specific embedding and feature order requirements of TabNet.
"""

import os
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

def load_and_adapt_data(input_csv, model_id, force_cpu=False):
    """Load data from CSV and adapt it for TabNet model inference"""
    logger.info(f"Loading data for model {model_id} from {input_csv}")
    
    # Read the input data
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        return None, None
        
    # Get the model path
    base_dirs = [
        os.path.join("Data", "finals", model_id),
        os.path.join("Data", "models", model_id),
        "."
    ]
    
    model_path = None
    scaler_path = None
    
    for base_dir in base_dirs:
        model_candidate = os.path.join(base_dir, f"{model_id}_model.zip")
        scaler_candidate = os.path.join(base_dir, f"{model_id}_scaler.joblib")
        
        if os.path.exists(model_candidate):
            model_path = model_candidate
            
        if os.path.exists(scaler_candidate):
            scaler_path = scaler_candidate
            
        if model_path and scaler_path:
            break
    
    if not model_path or not scaler_path:
        logger.error(f"Could not find model or scaler for {model_id}")
        return None, None
    
    # Load the scaler to inspect the expected features
    try:
        scaler = joblib.load(scaler_path)
        expected_features = scaler.feature_names_in_.tolist()
        logger.info(f"Model expects these features: {expected_features}")
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return None, None
    
    # Load the model to inspect embedding dimensions
    device = "cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get model architecture
    try:
        model = TabNetRegressor(device_name=device)
        model.load_model(model_path)
        
        network = model.network
        embedder = network.embedder
        cat_idxs = network.cat_idxs
        cat_dims = network.cat_dims
        
        logger.info(f"Model categorical indices: {cat_idxs}")
        logger.info(f"Model categorical dimensions: {cat_dims}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None
    
    # Reorder columns to match expected feature order
    # These are the columns the model expects in the correct order
    # The first 7 will be scaled, the rest will be used as-is
    model_feature_order = [
        "AGE",                      # Position 0, numeric
        "MARITAL",                  # Position 1, categorical (0-1)
        "RACE",                     # Position 2, categorical (0-1)
        "ETHNICITY",                # Position 3, categorical (0-5)
        "GENDER",                   # Position 4, categorical (0-1)
        "HEALTHCARE_COVERAGE",      # Position 5, categorical (0-4)
        "HEALTHCARE_EXPENSES",      # Position 6, numeric
        "INCOME",                   # Position 7, numeric
        "Hospitalizations_Count",   # Position 8, numeric
        "Medications_Count",        # Position 9, numeric
        "Abnormal_Observations_Count", # Position 10, numeric
        "DECEASED"                  # Position 11, numeric
    ]
    
    # Create a DataFrame with the expected columns in the right order
    if not all(col in df.columns for col in model_feature_order):
        missing = [col for col in model_feature_order if col not in df.columns]
        logger.error(f"Missing required columns: {missing}")
        return None, None
    
    # Select and reorder columns
    df_reordered = df[model_feature_order].copy()
    
    # Ensure categorical values are within bounds
    for idx, dim in zip(cat_idxs, cat_dims):
        col_name = model_feature_order[idx]
        max_valid_value = dim - 1
        
        # Check if any values are out of bounds
        out_of_bounds = (df_reordered[col_name] > max_valid_value).sum()
        if out_of_bounds > 0:
            logger.warning(f"Column {col_name} has {out_of_bounds} values exceeding max valid value {max_valid_value}")
            # Clip values to valid range
            df_reordered[col_name] = df_reordered[col_name].clip(upper=max_valid_value)
    
    # Reset the patient IDs and keep original IDs for mapping back
    patient_ids = df['Id'].tolist()
    
    # Convert to numpy array for model input
    X = df_reordered.values
    
    logger.info(f"Prepared data with shape: {X.shape}")
    return X, patient_ids

def run_inference(input_csv, model_id, force_cpu=False):
    """Run inference with the adapted data"""
    X, patient_ids = load_and_adapt_data(input_csv, model_id, force_cpu)
    if X is None:
        return None
        
    # Get the model path
    base_dirs = [
        os.path.join("Data", "finals", model_id),
        os.path.join("Data", "models", model_id),
        "."
    ]
    
    model_path = None
    
    for base_dir in base_dirs:
        model_candidate = os.path.join(base_dir, f"{model_id}_model.zip")
        
        if os.path.exists(model_candidate):
            model_path = model_candidate
            break
    
    if not model_path:
        logger.error(f"Could not find model for {model_id}")
        return None
    
    # Run inference
    try:
        device = "cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running inference on device: {device}")
        
        model = TabNetRegressor(device_name=device)
        model.load_model(model_path)
        
        preds = model.predict(X).flatten()
        
        # Create results dataframe
        results = pd.DataFrame({
            'Id': patient_ids,
            'Prediction': preds
        })
        
        # Save results
        output_dir = os.path.join("Data", "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{model_id}_predictions.csv")
        results.to_csv(output_path, index=False)
        
        logger.info(f"Saved predictions to {output_path}")
        return results
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--model", "-m", required=True, help="Model ID")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    
    results = run_inference(args.input, args.model, args.force_cpu)
    if results is not None:
        print(f"Successfully ran inference with {len(results)} predictions")
    else:
        print("Inference failed")
        exit(1)

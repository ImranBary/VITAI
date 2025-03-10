# tabnet_inference.py
import sys
import os
import pandas as pd
import numpy as np
import joblib
import logging
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_model_artifacts(model_id, force_cpu=False):
    """Load the TabNet model and any associated artifacts (e.g., scaler)"""
    model_dir = os.path.join("Data","finals", model_id)
    model_file = os.path.join(model_dir, f"{model_id}_model.zip")
    
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        sys.exit(1)
    
    # Load the model
    regressor = TabNetRegressor()
    
    # Force CPU if requested
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Forcing CPU usage for inference")
        
    # Set up PyTorch to catch device-side errors better
    if torch.cuda.is_available() and not force_cpu:
        # Enable device-side assertions where possible
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("CUDA is available - with enhanced error checking")
    else:
        logger.info("Using CPU for inference")
    
    # Check if the model dimensions match the expected shape
    try:
        regressor.load_model(model_file)
        logger.info(f"Loaded TabNet model from {model_file}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Check for scaler
    scaler_file = os.path.join(model_dir, f"{model_id}_scaler.joblib")
    scaler = None
    if os.path.exists(scaler_file):
        try:
            scaler = joblib.load(scaler_file)
            logger.info(f"Loaded feature scaler from {scaler_file}")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}. Will create a new one.")
            scaler = None
    
    return regressor, scaler

def preprocess_features(df, model_id):
    """
    Apply the same preprocessing as during training:
    - Handle categorical columns with label encoding
    - Scale numerical features
    - Handle missing values
    """
    # Determine target column from model_id
    target_col = "Health_Index"  # default
    if "_cci_" in model_id or "charlson" in model_id.lower():
        target_col = "CharlsonIndex"
    elif "_eci_" in model_id or "elixhauser" in model_id.lower():
        target_col = "ElixhauserIndex"
    
    logger.info(f"Inferred target column for this model: {target_col}")
    
    # Extract patient IDs before preprocessing
    patient_ids = df["Id"].values
    
    # Feature columns (match those used in tabnet_model.py)
    categorical_columns = ['DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    continuous_columns = [
        'AGE', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
        'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count'
    ]
    
    # Special handling for combined models
    if "combined" in model_id:
        if "all" in model_id:
            continuous_columns.extend(['Health_Index', 'CharlsonIndex', 'ElixhauserIndex'])
        elif "eci" in model_id:
            continuous_columns.extend(['Health_Index', 'ElixhauserIndex'])
        else:
            continuous_columns.extend(['Health_Index', 'CharlsonIndex'])
    
    # Make a copy to avoid modifying the original dataframe
    features = df.copy()
    
    # Check for required columns
    missing_cols = [col for col in categorical_columns + continuous_columns 
                   if col not in features.columns]
    if missing_cols:
        logger.warning(f"Missing columns in input data: {missing_cols}")
        logger.info(f"Available columns: {features.columns.tolist()}")
    
    # Remove any missing columns from our processing lists
    categorical_columns = [col for col in categorical_columns if col in features.columns]
    continuous_columns = [col for col in continuous_columns if col in features.columns]
    
    # Convert binary columns
    if 'DECEASED' in features.columns:
        features['DECEASED'] = features['DECEASED'].astype(str)
    
    # Encode categorical columns
    for col in categorical_columns:
        if col in features.columns:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
    
    # Load the scaler or create a new one
    model_dir = os.path.join("Data","finals", model_id)
    scaler_file = os.path.join(model_dir, f"{model_id}_scaler.joblib")
    
    if os.path.exists(scaler_file):
        try:
            scaler = joblib.load(scaler_file)
            logger.info(f"Using saved scaler: {scaler_file}")
        except Exception as e:
            logger.warning(f"Error loading scaler: {e}. Creating new scaler.")
            scaler = StandardScaler()
            # Fit the scaler on the available columns
            if continuous_columns:
                features[continuous_columns] = scaler.fit_transform(features[continuous_columns])
    else:
        logger.warning(f"No saved scaler found at {scaler_file}, performing standard scaling")
        scaler = StandardScaler()
        # Fit the scaler on the available columns
        if continuous_columns:
            features[continuous_columns] = scaler.fit_transform(features[continuous_columns])
    
    # Apply scaling to continuous columns
    if continuous_columns:
        features[continuous_columns] = scaler.transform(features[continuous_columns])
    
    # Fill missing values
    features.fillna(0, inplace=True)
    
    # Select only relevant columns for inference
    relevant_cols = categorical_columns + continuous_columns
    
    # Validate that we have features to work with
    if not relevant_cols:
        logger.error("No valid features found for inference!")
        sys.exit(1)
    
    logger.info(f"Processed features: {len(relevant_cols)} columns")
    
    return patient_ids, features[relevant_cols]

def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python tabnet_inference.py <model_id> <csv_for_inference> [--force-cpu]")
        sys.exit(1)

    model_id = sys.argv[1]
    csv_path = sys.argv[2]
    force_cpu = "--force-cpu" in sys.argv
    
    if not os.path.exists(csv_path):
        logger.error(f"Input CSV file not found: {csv_path}")
        sys.exit(1)

    # Load the data
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded input data: {csv_path} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        sys.exit(1)
    
    # Load model
    regressor, _ = load_model_artifacts(model_id, force_cpu)
    
    # Preprocess features
    patient_ids, X = preprocess_features(df, model_id)
    
    # Check dimensions
    feature_count = X.shape[1]
    logger.info(f"Feature shape: {X.shape}")
    
    if feature_count < 7:  # Basic sanity check
        logger.error(f"Feature count too low: {feature_count}. Expected at least 7 features.")
        sys.exit(1)
    
    # Run inference with proper error handling
    try:
        # Convert to numpy explicitly to avoid CUDA errors with certain dataframe formats
        X_values = X.values.astype(np.float32)
        logger.info(f"Running inference with feature shape: {X_values.shape}")
        
        try:
            # Try inference with a small batch first as a test
            if len(X_values) > 1:
                test_pred = regressor.predict(X_values[:1]).flatten()
                logger.info(f"Test prediction successful: {test_pred}")
        except Exception as e:
            if "CUDA" in str(e):
                logger.warning(f"CUDA error in test prediction: {e}. Falling back to CPU.")
                # Force CPU by setting environment variable
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                # Reload model in CPU mode
                regressor = TabNetRegressor(device_name='cpu')
                regressor.load_model(os.path.join("Data","finals", model_id, f"{model_id}_model.zip"))
        
        # Now do the full prediction
        preds = regressor.predict(X_values).flatten()
        logger.info(f"Successfully generated {len(preds)} predictions")
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        sys.exit(1)

    # Create output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("Data", "new_predictions", model_id)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_id}_predictions_{timestamp}.csv")

    # Determine target column name based on model_id
    target_col = "Health_Index"  # default
    if "_cci_" in model_id or "charlson" in model_id.lower():
        target_col = "CharlsonIndex"
    elif "_eci_" in model_id or "elixhauser" in model_id.lower():
        target_col = "ElixhauserIndex"
    
    out_df = pd.DataFrame({
        "Id": patient_ids,
        f"Predicted_{target_col}": preds
    })
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Predictions saved => {out_csv}")
    
    # Optional: If the original data has the ground truth, evaluate
    if target_col in df.columns:
        from sklearn.metrics import mean_squared_error, r2_score
        ground_truth = df[target_col].values
        mse = mean_squared_error(ground_truth, preds)
        r2 = r2_score(ground_truth, preds)
        logger.info(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Save metrics
        metrics_file = os.path.join(out_dir, f"{model_id}_inference_metrics_{timestamp}.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump({"mse": float(mse), "r2": float(r2)}, f)
        logger.info(f"Metrics saved => {metrics_file}")

if __name__ == "__main__":
    main()

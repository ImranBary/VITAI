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

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_model_artifacts(model_id):
    """Load the TabNet model and any associated artifacts (e.g., scaler)"""
    model_dir = os.path.join("Data","finals", model_id)
    model_file = os.path.join(model_dir, f"{model_id}_model.zip")
    
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        sys.exit(1)
    
    # Load the model
    regressor = TabNetRegressor()
    regressor.load_model(model_file)
    logger.info(f"Loaded TabNet model from {model_file}")
    
    # Check for scaler
    scaler_file = os.path.join(model_dir, f"{model_id}_scaler.joblib")
    scaler = None
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        logger.info(f"Loaded feature scaler from {scaler_file}")
    
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
    
    # Make a copy to avoid modifying the original dataframe
    features = df.copy()
    
    # Check for required columns
    missing_cols = [col for col in categorical_columns + continuous_columns 
                   if col not in features.columns]
    if missing_cols:
        logger.warning(f"Missing columns in input data: {missing_cols}")
    
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
        scaler = joblib.load(scaler_file)
        logger.info(f"Using saved scaler: {scaler_file}")
    else:
        logger.warning(f"No saved scaler found at {scaler_file}, performing standard scaling")
        scaler = StandardScaler()
        existing_cont_cols = [col for col in continuous_columns if col in features.columns]
        if existing_cont_cols:
            features[existing_cont_cols] = scaler.fit_transform(features[existing_cont_cols])
    
    # Apply scaling to continuous columns
    existing_cont_cols = [col for col in continuous_columns if col in features.columns]
    if existing_cont_cols:
        features[existing_cont_cols] = scaler.transform(features[existing_cont_cols])
    
    # Fill missing values
    features.fillna(0, inplace=True)
    
    # Select only relevant columns for inference
    relevant_cols = [col for col in categorical_columns + continuous_columns 
                    if col in features.columns]
    
    return patient_ids, features[relevant_cols]

def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python tabnet_inference.py <model_id> <csv_for_inference>")
        sys.exit(1)

    model_id = sys.argv[1]
    csv_path = sys.argv[2]
    
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
    regressor, _ = load_model_artifacts(model_id)
    
    # Preprocess features
    patient_ids, X = preprocess_features(df, model_id)
    
    # Run inference
    try:
        logger.info(f"Running inference with feature shape: {X.shape}")
        preds = regressor.predict(X.values).flatten()
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

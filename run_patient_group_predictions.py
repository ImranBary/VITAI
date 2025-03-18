#!/usr/bin/env python
"""
run_patient_group_predictions.py

Refactored with additional debug prints to help identify data mismatch issues.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import traceback

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# For TabNet inference
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    print("[WARNING] PyTorch TabNet not found. Attempting to continue...")

# ------------------------------------------------------------------------
# 1. Setup logging in DEBUG mode to see all messages
# ------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,  # set to logging.DEBUG to see everything
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# 2. Improved path handling for module imports
# ------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR))
VITAI_SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "vitai_scripts")

# Add necessary paths in the correct order
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, VITAI_SCRIPTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

try:
    # Try to import from various locations with better error reporting
    try:
        from vitai_scripts.subset_utils import filter_subpopulation
        from vitai_scripts.feature_utils import select_features
        logger.info("Loaded utilities from vitai_scripts package")
    except ImportError as e:
        logger.warning(f"Import from package failed: {e}, trying local directory...")
        try:
            sys.path.insert(0, os.path.join(SCRIPT_DIR, "vitai_scripts"))
            from subset_utils import filter_subpopulation
            from feature_utils import select_features
            logger.info("Loaded utilities from local vitai_scripts directory")
        except ImportError as nested_e:
            logger.error(f"Local import also failed: {nested_e}")
            # Re-raise the original exception for better debugging
            raise e

except ImportError as e:
    logger.error(f"Failed to import subset_utils or feature_utils: {e}")
    logger.error(f"sys.path: {sys.path}")
    logger.error(f"Current directory: {os.getcwd()}")
    logger.error(f"Directory contents: {os.listdir(SCRIPT_DIR)}")
    
    # Fallback functions with enhanced logging
    def filter_subpopulation(df, subset_name, _):
        """Fallback function to identify patient subsets"""
        logger.warning(f"Using fallback filter_subpopulation for subset '{subset_name}' with {len(df)} patients")
        if subset_name == "diabetes":
            # Try to load from file if it exists
            try:
                with open("diabetic_patients.txt", "r") as f:
                    diabetic_ids = set(line.strip() for line in f if line.strip() and not line.startswith("#"))
                logger.info(f"Loaded {len(diabetic_ids)} diabetic patient IDs from file")
                return df[df["Id"].isin(diabetic_ids)]
            except Exception as e:
                logger.warning(f"Error loading diabetic patients: {e}")
                # Fallback: ~52% of patients for diabetes
                count = int(len(df) * 0.52)
                logger.info(f"Using percentage-based allocation: {count} patients ({count/len(df)*100:.1f}%)")
                return df.iloc[:count]
        elif subset_name == "ckd":
            # For CKD, ~5%
            count = max(1, int(len(df) * 0.05))
            logger.info(f"CKD subset: {count} patients ({count/len(df)*100:.1f}%)")
            return df.iloc[:count]
        else:  # "none"
            diabetes_df = filter_subpopulation(df, "diabetes", None)
            ckd_df = filter_subpopulation(df, "ckd", None)
            combined_ids = set(diabetes_df["Id"]).union(set(ckd_df["Id"]))
            result = df[~df["Id"].isin(combined_ids)]
            logger.info(f"'None' subset: {len(result)} patients ({len(result)/len(df)*100:.1f}%)")
            return result

    def select_features(df, config):
        logger.warning(f"Using fallback feature selection for config: {config}")
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        logger.info(f"Feature columns: {result_df.columns.tolist()}")
        return result_df

# ------------------------------------------------------------------------
# 3. Model configurations
# ------------------------------------------------------------------------
MODEL_CONFIG_MAP = {
    "combined_diabetes_tabnet": {"subset": "diabetes", "feature_config": "combined"},
    "combined_all_ckd_tabnet":  {"subset": "ckd",      "feature_config": "combined_all"},
    "combined_none_tabnet":     {"subset": "none",     "feature_config": "combined"}
}

def get_model_path(model_id):
    """Find the model path with several fallback locations."""
    potential_paths = [
        os.path.join(SCRIPT_DIR, f"{model_id}_model.zip"),
        os.path.join(SCRIPT_DIR, "Data", "finals", model_id, f"{model_id}_model.zip"),
        os.path.join(SCRIPT_DIR, "Data", "models", f"{model_id}_model.zip"),
        # Check for non-zip
        os.path.join(SCRIPT_DIR, f"{model_id}_model"),
        os.path.join(SCRIPT_DIR, "Data", "finals", model_id, f"{model_id}_model"),
        os.path.join(SCRIPT_DIR, "Data", "models", f"{model_id}_model")
    ]
    for path in potential_paths:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            return path
    logger.warning(f"Could not find model: {model_id}")
    return None

def identify_patient_groups(df):
    """Identify which patients belong to which subset (diabetes, CKD, none)."""
    diabetes_ids = set()
    ckd_ids = set()

    # 1. Try to load from diabetic_patients.txt
    try:
        with open("diabetic_patients.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    diabetes_ids.add(line)
        logger.info(f"Loaded {len(diabetes_ids)} diabetic patients from file")
    except Exception as e:
        logger.warning(f"Could not load diabetic patients from file: {e}")

    # 2. If none found, optionally parse the conditions CSV
    if not diabetes_ids:
        data_dir = "Data"
        try:
            cond_files = [
                f for f in os.listdir(data_dir)
                if f.startswith("conditions_") and f.endswith(".csv")
            ]
            if cond_files:
                cond_files.sort(reverse=True)
                conditions_df = pd.read_csv(os.path.join(data_dir, cond_files[0]))
                for _, row in conditions_df.iterrows():
                    desc = row['DESCRIPTION'].lower() if 'DESCRIPTION' in conditions_df.columns else ''
                    patient_id = row['PATIENT'] if 'PATIENT' in conditions_df.columns else None
                    if patient_id:
                        if "diabetes" in desc:
                            diabetes_ids.add(patient_id)
                        if "chronic kidney disease" in desc or "ckd" in desc:
                            ckd_ids.add(patient_id)
                logger.info(f"Extracted {len(diabetes_ids)} diabetic patients from conditions")
        except Exception as e:
            logger.warning(f"Could not process conditions file: {e}")

    # 3. If we still have no diabetic patients, do a percentage-based approach
    if not diabetes_ids:
        target_count = int(len(df) * 0.52)
        diabetes_ids = set(df['Id'].iloc[:target_count])
        logger.info(f"Assigned {len(diabetes_ids)} patients to diabetes group by percentage")

    # 4. Build final mapping
    patient_groups = {}
    for patient_id in df['Id'].unique():
        if patient_id in diabetes_ids:
            patient_groups[patient_id] = "diabetes"
        elif patient_id in ckd_ids:
            patient_groups[patient_id] = "ckd"
        else:
            patient_groups[patient_id] = "none"
    return patient_groups

def load_model_artifacts(model_id, force_cpu=False):
    """Load TabNet model + scaler from disk."""
    if not HAS_TABNET:
        raise ImportError("PyTorch TabNet is required for model loading.")

    model_path = get_model_path(model_id)
    if not model_path:
        raise FileNotFoundError(f"Model {model_id} not found in any expected location.")

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model {model_id} on device: {device}")

    regressor = TabNetRegressor(device_name=device)
    regressor.load_model(model_path)
    
    # NEW: Check the expected input dimension from the loaded model
    expected_dim = None
    try:
        # Try to access the input dimension from the model
        if hasattr(regressor.network, 'input_dim'):
            expected_dim = regressor.network.input_dim
            logger.info(f"Model expects input dimension: {expected_dim}")
    except Exception as e:
        logger.warning(f"Could not determine model's expected input dimension: {e}")

    # Attempt to load scaler
    scaler = None
    base_dir = os.path.dirname(model_path)
    possible_scalers = [
        os.path.join(base_dir, f"{model_id}_scaler.joblib"),
        os.path.join(SCRIPT_DIR, f"{model_id}_scaler.joblib"),
        os.path.join(SCRIPT_DIR, "tabnet_scaler.joblib")
    ]
    for spath in possible_scalers:
        if os.path.exists(spath):
            try:
                scaler = joblib.load(spath)
                logger.info(f"Loaded scaler from {spath}")
                break
            except Exception as e:
                logger.warning(f"Failed to load scaler from {spath}: {e}")
    return regressor, scaler, expected_dim  # Return the expected dimension

def preprocess_features(df, model_id, expected_dim=None):
    """
    Prepare features for model inference with enhanced feature validation.
    """
    logger.debug(f"[{model_id}] Preprocessing {len(df)} rows with columns: {df.columns.tolist()}")
    logger.debug(f"[{model_id}] Sample data:\n{df.head(3)}")

    # Make a copy for transformations
    preproc_df = df.copy()
    patient_ids = preproc_df['Id'].values

    # Check if we have all expected categorical columns
    cat_cols = ['DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    for col in cat_cols:
        if col not in preproc_df.columns:
            logger.warning(f"[{model_id}] Missing categorical column '{col}'. Adding with default value 0.")
            preproc_df[col] = 0

    # 1. Identify which columns to treat as numeric features
    feature_cols = [
        c for c in preproc_df.columns
        if c not in ['Id', 'Health_Index', 'Predicted_Health_Index']
           and pd.api.types.is_numeric_dtype(preproc_df[c])
    ]
    logger.debug(f"[{model_id}] Numeric feature_cols: {feature_cols}")

    # Check for expected numeric columns and add defaults if missing
    expected_numeric = ['AGE', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 
                        'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']
    for col in expected_numeric:
        if col not in feature_cols and col not in preproc_df.columns:
            logger.warning(f"[{model_id}] Missing numeric column '{col}'. Adding with default value 0.")
            preproc_df[col] = 0
            if col not in feature_cols:
                feature_cols.append(col)

    # 2. Convert categorical columns from string -> ordinal numeric
    for col in cat_cols:
        if col in preproc_df.columns:
            preproc_df[col] = preproc_df[col].astype(str)
            unique_vals = preproc_df[col].unique()
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            preproc_df[col] = preproc_df[col].map(val_to_idx).fillna(0)
            
            # Ensure categorical columns are in feature_cols
            if col not in feature_cols:
                feature_cols.append(col)

    # 3. Construct final X with enhanced error checking
    if not feature_cols:
        logger.error(f"[{model_id}] No suitable feature columns found!")
        logger.error(f"Input columns were: {df.columns.tolist()}")
        raise ValueError(f"No suitable feature columns found for model {model_id}")

    try:
        X = preproc_df[feature_cols].values
        logger.info(f"Selected features: {feature_cols}")
    except Exception as e:
        logger.error(f"[{model_id}] Error selecting features: {str(e)}")
        logger.error(f"Available columns: {preproc_df.columns.tolist()}")
        logger.error(f"Requested features: {feature_cols}")
        raise

    # 4. Fill any leftover NaNs
    X = np.nan_to_num(X)
    original_shape = X.shape
    logger.debug(f"[{model_id}] Original feature shape: {original_shape}")
    
    return patient_ids, X, feature_cols

def run_model_for_group(df, group_name, model_id, force_cpu=False):
    """Loads the model, preprocesses the group's data, and runs inference."""
    logger.info(f"Running model '{model_id}' for group '{group_name}' with {len(df)} patients.")
    logger.debug(f"Group '{group_name}' DataFrame columns: {df.columns.tolist()}")
    logger.debug(f"Group '{group_name}' DataFrame sample:\n{df.head(3)}")

    # 1. Load TabNet model + scaler
    try:
        regressor, scaler, expected_dim = load_model_artifacts(model_id, force_cpu)  # Get expected dimension
        if scaler is None:
            logger.warning(f"No scaler found for model {model_id}. Results may be incorrect.")
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

    # 2. Preprocess features (but don't add padding yet)
    try:
        patient_ids, X, feature_cols = preprocess_features(df, model_id)
        if len(patient_ids) == 0:
            logger.warning(f"No patients in {group_name} after preprocessing.")
            return None
        
        # Log sample of raw feature values before scaling
        logger.debug(f"Sample raw features (first row): {X[0]}")
        
        # Apply scaler if available to the original features
        if scaler is not None:
            logger.info(f"Applying scaler to features for model: {model_id}")
            X = scaler.transform(X)
            logger.debug(f"Sample scaled features (first row): {X[0]}")
            logger.debug(f"Scaled feature min/max: {np.min(X, axis=0)[:3]}.../{np.max(X, axis=0)[:3]}...")
        else:
            logger.warning(f"No scaler available for {model_id}. Using unscaled features.")
        
        # NOW check if we need to pad features (AFTER scaling)
        if expected_dim is not None and X.shape[1] < expected_dim:
            logger.warning(f"Feature count mismatch! Model expects {expected_dim} features but got {X.shape[1]}.")
            logger.warning(f"Adding {expected_dim - X.shape[1]} dummy features with zeros.")
            
            # Create padding with zeros
            padding = np.zeros((X.shape[0], expected_dim - X.shape[1]))
            X = np.hstack((X, padding))
            logger.debug(f"New padded X shape: {X.shape}")
            
        # Check for uniform values that might indicate a problem
        feature_stds = np.std(X, axis=0)
        if np.any(feature_stds < 1e-6):
            logger.warning(f"Some features have near-zero standard deviation: {feature_stds}")

        logger.info(f"Final preprocessed features shape: {X.shape} for {len(patient_ids)} patients")
    except Exception as e:
        logger.error(f"Error preprocessing features: {str(e)}")
        logger.error(traceback.format_exc())
        return None

    # 3. Inference
    try:
        logger.info(f"Inference input shape for '{model_id}': {X.shape}")
        preds = regressor.predict(X).flatten()
        
        # Debug: check if predictions vary
        pred_min = np.min(preds)
        pred_max = np.max(preds)
        pred_std = np.std(preds)
        logger.info(f"Prediction stats - min: {pred_min:.4f}, max: {pred_max:.4f}, std: {pred_std:.4f}")
        
        if pred_std < 1e-6:
            logger.warning(f"WARNING: All predictions are identical ({pred_min:.4f})! This indicates a problem.")
            # Let's check input feature variation as well
            logger.warning(f"Feature std devs: {np.std(X, axis=0)}")
        
        logger.info(f"Generated {len(preds)} predictions for group '{group_name}'")

        # Combine predictions with Id
        results_df = pd.DataFrame({"Id": patient_ids, "Predicted_Health_Index": preds})
        return results_df
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    # 1. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run TabNet models on patient groups")
    parser.add_argument("--input-csv", required=True, help="Input CSV with patient features")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    # 2. Verify input CSV
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV not found: {args.input_csv}")
        return 1

    # 3. Load the entire DataFrame
    try:
        df = pd.read_csv(args.input_csv)
        logger.info(f"Loaded {len(df)} patients total from {args.input_csv}.")
        logger.debug(f"Data columns: {df.columns.tolist()}")
        logger.debug(f"Data sample:\n{df.head(5)}")

        if 'Id' not in df.columns:
            logger.error("CSV must contain an 'Id' column.")
            return 1
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        return 1

    # 4. Identify subgroups
    try:
        patient_groups = identify_patient_groups(df)
    except Exception as e:
        logger.error(f"Error identifying patient groups: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

    # 5. Summarize groups
    group_counts = {"diabetes": 0, "ckd": 0, "none": 0}
    for pid, g in patient_groups.items():
        if g in group_counts:
            group_counts[g] += 1
    logger.info(f"Patient groups => Diabetes={group_counts['diabetes']}, "
                f"CKD={group_counts['ckd']}, None={group_counts['none']}")

    # 6. Output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("Data", "new_predictions")
    os.makedirs(out_dir, exist_ok=True)

    # 7. Check TabNet availability
    if not HAS_TABNET:
        logger.error("TabNet is required but not installed; cannot proceed.")
        return 1

    # 8. For each model in the map, run inference
    all_success = True
    for model_id, cfg in MODEL_CONFIG_MAP.items():
        group_name = cfg["subset"]

        # Filter only that group's patients
        group_ids = [pid for pid, grp in patient_groups.items() if grp == group_name]
        if not group_ids:
            logger.info(f"No patients in '{group_name}' group. Skipping {model_id}.")
            continue

        group_df = df[df['Id'].isin(group_ids)].copy()
        results_df = run_model_for_group(group_df, group_name, model_id, force_cpu=args.force_cpu)

        # If we got predictions, save them
        if results_df is not None:
            out_path = os.path.join(out_dir, f"{model_id}_predictions_{timestamp}.csv")
            results_df.to_csv(out_path, index=False)
            logger.info(f"Saved predictions to {out_path}")
        else:
            logger.error(f"Failed to get predictions for model '{model_id}'")
            all_success = False

    # 9. Final exit code
    if all_success:
        logger.info("Multi-model inference completed successfully.")
        return 0
    else:
        logger.error("Multi-model inference completed with errors.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

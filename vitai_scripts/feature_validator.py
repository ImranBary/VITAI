"""
feature_validator.py
A utility script to validate feature compatibility between C++ and Python code.

This script can be used to:
1. Check that features in the CSV match what the model expects
2. Report any discrepancies 
3. Optionally fix missing or incorrectly formatted features
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import glob

logger = logging.getLogger(__name__)

def validate_features(input_csv, models_dir="Data/finals"):
    """
    Validate that the features in the input CSV match what the model expects.
    
    Args:
        input_csv (str): Path to the input CSV with features
        models_dir (str): Directory containing model metadata
        
    Returns:
        tuple: (is_valid, issues_dict)
    """
    # Load the CSV
    if not os.path.exists(input_csv):
        return False, {"error": f"Input CSV not found: {input_csv}"}
    
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} rows from {input_csv}")
    except Exception as e:
        return False, {"error": f"Failed to load CSV: {str(e)}"}
    
    # Check required Id column
    if "Id" not in df.columns:
        return False, {"error": "Missing required 'Id' column"}
    
    # Get list of model directories
    model_dirs = []
    for item in os.listdir(models_dir):
        if os.path.isdir(os.path.join(models_dir, item)):
            model_dirs.append(os.path.join(models_dir, item))
    
    if not model_dirs:
        return False, {"error": f"No model directories found in {models_dir}"}
    
    # For each model, check feature compatibility
    issues = {}
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        # Look for metadata file
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        if not os.path.exists(metadata_path):
            metadata_candidates = glob.glob(os.path.join(model_dir, "*metadata*.json"))
            if metadata_candidates:
                metadata_path = metadata_candidates[0]
            else:
                issues[model_name] = {"error": "No metadata file found"}
                continue
        
        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get expected features
            expected_features = metadata.get("feature_columns", [])
            if not expected_features:
                issues[model_name] = {"error": "No feature columns in metadata"}
                continue
            
            # Check actual features
            missing_features = [f for f in expected_features if f not in df.columns]
            categorical_cols = metadata.get("categorical_columns", [])
            
            # Check data types
            type_issues = []
            for col in expected_features:
                if col in df.columns:
                    if col in categorical_cols:
                        # Should be string or nominal
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if not all(df[col].apply(lambda x: float(x).is_integer())):
                                type_issues.append(f"{col} (expected categorical but found non-integer values)")
                    elif not pd.api.types.is_numeric_dtype(df[col]):
                        type_issues.append(f"{col} (expected numeric but found non-numeric)")
            
            model_issues = {}
            if missing_features:
                model_issues["missing_features"] = missing_features
            if type_issues:
                model_issues["type_issues"] = type_issues
            
            if model_issues:
                issues[model_name] = model_issues
            
        except Exception as e:
            issues[model_name] = {"error": f"Failed to process metadata: {str(e)}"}
    
    # Return validation result
    if not issues:
        return True, {}
    return False, issues

def fix_features(input_csv, output_csv=None, models_dir="Data/finals"):
    """
    Fix feature issues in the input CSV based on model metadata.
    
    Args:
        input_csv (str): Path to the input CSV with features
        output_csv (str): Path to save the fixed CSV. If None, will update the input CSV.
        models_dir (str): Directory containing model metadata
        
    Returns:
        bool: True if successful, False otherwise
    """
    if output_csv is None:
        output_csv = input_csv
    
    # Validate first
    is_valid, issues = validate_features(input_csv, models_dir)
    if is_valid:
        logger.info(f"No issues found with {input_csv}")
        return True
    
    # Load the CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        return False
    
    # For each model with issues
    for model_name, model_issues in issues.items():
        if "error" in model_issues:
            logger.warning(f"Cannot fix {model_name}: {model_issues['error']}")
            continue
        
        # Load metadata
        metadata_path = os.path.join(models_dir, model_name, f"{model_name}_metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            # Try to find any metadata file
            try:
                metadata_candidates = glob.glob(os.path.join(models_dir, model_name, "*metadata*.json"))
                if metadata_candidates:
                    with open(metadata_candidates[0], 'r') as f:
                        metadata = json.load(f)
                else:
                    continue
            except Exception as e:
                logger.error(f"Cannot load metadata for {model_name}: {str(e)}")
                continue
        
        # Add missing features
        missing_features = model_issues.get("missing_features", [])
        for feature in missing_features:
            logger.info(f"Adding missing feature: {feature}")
            # Check if it's categorical or numeric
            if feature in metadata.get("categorical_columns", []):
                df[feature] = 0  # Default for categorical
            else:
                df[feature] = 0.0  # Default for numeric
        
        # Fix type issues
        type_issues = model_issues.get("type_issues", [])
        for issue in type_issues:
            feature = issue.split(" ")[0]
            if feature in metadata.get("categorical_columns", []):
                # Convert to integer
                logger.info(f"Converting {feature} to categorical (integer)")
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(int)
            else:
                # Convert to float
                logger.info(f"Converting {feature} to numeric")
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)
    
    # Save fixed CSV
    try:
        df.to_csv(output_csv, index=False)
        logger.info(f"Fixed CSV saved to {output_csv}")
        return True
    except Exception as e:
        logger.error(f"Failed to save fixed CSV: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                      format='[%(asctime)s][%(levelname)s] %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
    
    if len(sys.argv) < 2:
        print("Usage: python feature_validator.py input_csv [output_csv]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV not found: {input_csv}")
        sys.exit(1)
    
    # Validate
    valid, issues = validate_features(input_csv)
    if valid:
        print(f"✅ {input_csv} is compatible with all models")
    else:
        print(f"❌ Found issues with {input_csv}:")
        for model_name, model_issues in issues.items():
            print(f"  Model: {model_name}")
            for issue_type, issue_details in model_issues.items():
                if issue_type == "error":
                    print(f"    Error: {issue_details}")
                else:
                    print(f"    {issue_type.replace('_', ' ').title()}:")
                    for detail in issue_details:
                        print(f"      - {detail}")
        
        # Offer to fix
        if output_csv:
            print(f"Attempting to fix issues and save to {output_csv}...")
            if fix_features(input_csv, output_csv):
                print(f"✅ Fixed CSV saved to {output_csv}")
            else:
                print(f"❌ Failed to fix issues")
        else:
            print("To fix these issues, run: python feature_validator.py input_csv output_csv")

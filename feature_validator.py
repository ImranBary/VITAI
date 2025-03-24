#!/usr/bin/env python
"""
Feature Validator and Fixer

This script validates and fixes feature CSV files for TabNet models.
It ensures that:
1. All required columns are present
2. Categorical values are within acceptable ranges
3. Missing values are filled with appropriate defaults

Usage:
    python feature_validator.py input_csv [output_csv]
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define expected features and their types
EXPECTED_FEATURES = [
    "Id",
    "AGE",
    "MARITAL",
    "RACE",
    "ETHNICITY",
    "GENDER",
    "HEALTHCARE_COVERAGE",
    "HEALTHCARE_EXPENSES",
    "INCOME",
    "CharlsonIndex",
    "ElixhauserIndex", 
    "Comorbidity_Score",
    "Hospitalizations_Count",
    "Medications_Count",
    "Abnormal_Observations_Count",
    "DECEASED",
    "Health_Index"
]

# Define categorical feature limits (maximum allowed value)
CATEGORICAL_LIMITS = {
    "MARITAL": 1,      # Binary (0 or 1)
    "RACE": 1,         # Binary (0 or 1)
    "ETHNICITY": 5,    # Values 0-5
    "GENDER": 1,       # Binary (0 or 1)
    "HEALTHCARE_COVERAGE": 4  # Values 0-4
}

def validate_and_fix_csv(input_csv, output_csv=None):
    """Validate and fix feature CSV for TabNet model compatibility"""
    if output_csv is None:
        output_csv = input_csv
        
    logger.info(f"Validating and fixing CSV: {input_csv}")
    
    # Read the CSV
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Read CSV with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return False
    
    # Check for missing columns and add them
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            logger.warning(f"Adding missing column: {col}")
            if col == "Id":
                df[col] = [f"P{i}" for i in range(len(df))]
            elif col in ["CharlsonIndex", "ElixhauserIndex", "Comorbidity_Score", "Health_Index"]:
                df[col] = 0.0
            elif col in ["Hospitalizations_Count", "Medications_Count", "Abnormal_Observations_Count"]:
                df[col] = 0
            elif col == "DECEASED":
                df[col] = 0
            elif col in ["AGE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME"]:
                df[col] = 0.0
            else:
                df[col] = 0
    
    # Check categorical values are within limits
    for col, limit in CATEGORICAL_LIMITS.items():
        if col in df.columns:
            over_limit = (df[col] > limit).sum()
            if over_limit > 0:
                logger.warning(f"Fixing {over_limit} values in {col} that exceed limit {limit}")
                df[col] = df[col].clip(upper=limit)
    
    # Fill NaN values
    for col in df.columns:
        if col in ["CharlsonIndex", "ElixhauserIndex", "Comorbidity_Score", "Health_Index", 
                  "AGE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME"]:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(0)
    
    # Reorder columns to match expected order
    available_cols = [col for col in EXPECTED_FEATURES if col in df.columns]
    df = df[available_cols]
    
    # Save the fixed CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Fixed CSV saved to: {output_csv}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_validator.py input_csv [output_csv]")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else input_csv
    
    if not os.path.exists(input_csv):
        logger.error(f"Input CSV not found: {input_csv}")
        sys.exit(1)
        
    success = validate_and_fix_csv(input_csv, output_csv)
    sys.exit(0 if success else 1)

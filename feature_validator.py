#!/usr/bin/env python
"""
feature_validator.py - Validates patient feature data and identifies potential issues

This script:
1. Reads a CSV file containing patient feature data
2. Checks for unusual or missing values
3. Reports statistics and potential data quality issues
"""

import os
import sys
import csv
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

def validate_features(input_csv):
    """Validate patient feature data for common issues"""
    print(f"Validating patient data in {input_csv}...")
    
    try:
        # Load the data
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} patient records")
        
        # Check required columns
        required_cols = ["Id", "AGE", "GENDER"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {', '.join(missing_cols)}")
            return False
            
        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print("WARNING: Missing values detected:")
            for col in null_counts.index:
                if null_counts[col] > 0:
                    print(f"  - {col}: {null_counts[col]} missing values")
                    
        # Check for zero values in numeric columns
        numeric_cols = ["AGE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", 
                        "INCOME", "Hospitalizations_Count", "Medications_Count",
                        "Abnormal_Observations_Count"]
                        
        available_numeric = [col for col in numeric_cols if col in df.columns]
        zero_counts = {col: (df[col] == 0).sum() for col in available_numeric}
        
        total_zeros = sum(zero_counts.values())
        if total_zeros > 0:
            print(f"WARNING: Found {total_zeros} zero values across {len(available_numeric)} numeric columns:")
            for col, count in zero_counts.items():
                if count > 0:
                    pct = count / len(df) * 100
                    print(f"  - {col}: {count} zeros ({pct:.1f}%)")
                    
        # Check age distribution
        if "AGE" in df.columns:
            age_stats = {
                "min": df["AGE"].min(),
                "max": df["AGE"].max(),
                "mean": df["AGE"].mean(),
                "zeros": (df["AGE"] == 0).sum()
            }
            print(f"Age statistics: min={age_stats['min']}, max={age_stats['max']}, mean={age_stats['mean']:.1f}")
            if age_stats["zeros"] > 0:
                print(f"WARNING: {age_stats['zeros']} patients have age value of 0")
            if age_stats["max"] > 120:
                print(f"WARNING: Maximum age {age_stats['max']} seems unrealistic")
                
        # Check health indices
        if "Health_Index" in df.columns:
            hi_min = df["Health_Index"].min()
            hi_max = df["Health_Index"].max()
            hi_std = df["Health_Index"].std()
            print(f"Health_Index: min={hi_min}, max={hi_max}, std={hi_std:.2f}")
            
            if hi_std < 0.01 and len(df) > 1:
                print("WARNING: Health_Index has near-zero variation - possible calculation error")
                
        # Summary
        print("Validation complete")
        return True
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def fix_features(input_csv, output_csv):
    """Attempt to fix common data issues in the feature file"""
    print(f"Fixing data quality issues in {input_csv}...")
    
    try:
        # Load the data
        df = pd.read_csv(input_csv)
        
        # Fill missing values
        df.fillna(0, inplace=True)
        
        # Ensure categorical columns are integers for model compatibility
        cat_cols = ['DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
        for col in cat_cols:
            if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
                # Convert strings to categorical integers
                if pd.api.types.is_string_dtype(df[col]):
                    categories = df[col].astype(str).unique()
                    cat_map = {cat: i for i, cat in enumerate(categories)}
                    df[col] = df[col].astype(str).map(cat_map).fillna(0).astype(int)
                else:
                    # Convert any non-integer numeric to integer
                    df[col] = df[col].fillna(0).astype(int)
                    
        # Ensure AGE is reasonable
        if "AGE" in df.columns:
            # Fix obviously wrong ages (0 or >120)
            invalid_ages = (df["AGE"] == 0) | (df["AGE"] > 120)
            if invalid_ages.sum() > 0:
                print(f"Fixing {invalid_ages.sum()} invalid age values")
                # Set to median age
                median_age = df.loc[~invalid_ages, "AGE"].median()
                if pd.isna(median_age) or median_age == 0:
                    median_age = 50  # Fallback if median is also invalid
                df.loc[invalid_ages, "AGE"] = median_age
                
        # Save the fixed data
        df.to_csv(output_csv, index=False)
        print(f"Fixed data saved to {output_csv}")
        return True
        
    except Exception as e:
        print(f"ERROR during fix: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate and fix patient feature data")
    parser.add_argument("input_csv", help="Path to input CSV file with patient features")
    parser.add_argument("output_csv", nargs="?", help="Path to save fixed CSV (optional)")
    args = parser.parse_args()
    
    # Always validate
    is_valid = validate_features(args.input_csv)
    
    # Fix if output path is provided
    if args.output_csv:
        fixed = fix_features(args.input_csv, args.output_csv)
        return 0 if fixed else 1
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())

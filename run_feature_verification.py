import os
import pandas as pd
import numpy as np

def verify_feature_ranges(input_csv, categorical_limits):
    """Verify that categorical features in the CSV are within expected embedding ranges"""
    print(f"Verifying feature ranges in {input_csv}...")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    # Check each categorical column against its limit
    all_valid = True
    for col_name, limit in categorical_limits.items():
        if col_name in df.columns:
            try:
                values = df[col_name].astype(int)
                min_val = values.min()
                max_val = values.max()
                unique_vals = sorted(values.unique().tolist())
                
                print(f"Column {col_name}: range [{min_val}, {max_val}], unique values: {unique_vals}")
                
                if max_val >= limit:
                    print(f"[ERROR] Column {col_name} has values >= {limit} (embedding limit)")
                    all_valid = False
            except Exception as e:
                print(f"Error processing column {col_name}: {e}")
                all_valid = False
        else:
            print(f"[WARNING] Column {col_name} not found in CSV")
    
    return all_valid

if __name__ == "__main__":
    # Set up expected embedding dimensions from model_inspector.py
    # Format: column_name: max_allowed_value (embedding_dim - 1)
    categorical_limits = {
        "GENDER": 1,       # embedding_dim=2, values must be 0 or 1
        "MARITAL": 1,      # embedding_dim=2, values must be 0 or 1
        "RACE": 5,         # embedding_dim=6, values must be 0-5
        "ETHNICITY": 1,    # embedding_dim=2, values must be 0 or 1
        "DECEASED": 4      # embedding_dim=5, values must be 0-4
    }
    
    # Verify the generated CSV file
    csv_file = "PatientFeatures.csv"
    if os.path.exists(csv_file):
        is_valid = verify_feature_ranges(csv_file, categorical_limits)
        if is_valid:
            print("✅ All categorical features are within expected embedding limits!")
        else:
            print("❌ Some categorical features exceed embedding limits. Fix needed.")
    else:
        print(f"CSV file not found: {csv_file}")

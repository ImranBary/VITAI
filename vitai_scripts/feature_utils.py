# vitai_scripts/feature_utils.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#   Provides functions for selecting relevant features from
#   the patient DataFrame based on the chosen configuration:
#     - composite (Health_Index)
#     - cci (CharlsonIndex)
#     - eci (ElixhauserIndex)
#     - combined (Health_Index + CharlsonIndex)
#     - combined_eci (Health_Index + ElixhauserIndex)
#
#   You may add your own extra combos if you wish,
#   e.g. "all_indices" for all three.

import pandas as pd

def select_features(df: pd.DataFrame, feature_config: str) -> pd.DataFrame:
    """
    For each feature_config, returns a subset of columns:
      - 'composite' => uses Health_Index
      - 'cci'       => uses CharlsonIndex
      - 'eci'       => uses ElixhauserIndex
      - 'combined'  => uses Health_Index + CharlsonIndex
      - 'combined_eci' => uses Health_Index + ElixhauserIndex
      (You can extend further if needed.)

    Also includes base demographics & hospital/med counts.
    """
    base_cols = [
        "Id", "GENDER", "RACE", "ETHNICITY", "MARITAL",
        "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME",
        "AGE", "DECEASED",
        "Hospitalizations_Count", "Medications_Count", "Abnormal_Observations_Count"
    ]

    # Verify these exist; some might be absent if your data lacks them, so handle carefully.
    for col in base_cols:
        if col not in df.columns:
            df[col] = 0  # fill with 0 or something sensible

    if feature_config == "composite":
        if "Health_Index" not in df.columns:
            raise KeyError("Missing 'Health_Index' for 'composite'.")
        needed = base_cols + ["Health_Index"]

    elif feature_config == "cci":
        if "CharlsonIndex" not in df.columns:
            raise KeyError("Missing 'CharlsonIndex' for 'cci'.")
        needed = base_cols + ["CharlsonIndex"]

    elif feature_config == "eci":
        if "ElixhauserIndex" not in df.columns:
            raise KeyError("Missing 'ElixhauserIndex' for 'eci'.")
        needed = base_cols + ["ElixhauserIndex"]

    elif feature_config == "combined":
        # Health + Charlson
        if ("Health_Index" not in df.columns) or ("CharlsonIndex" not in df.columns):
            raise KeyError("Need both 'Health_Index' & 'CharlsonIndex' for 'combined'.")
        needed = base_cols + ["Health_Index","CharlsonIndex"]

    elif feature_config == "combined_eci":
        # Health + Elixhauser
        if ("Health_Index" not in df.columns) or ("ElixhauserIndex" not in df.columns):
            raise KeyError("Need both 'Health_Index' & 'ElixhauserIndex' for 'combined_eci'.")
        needed = base_cols + ["Health_Index","ElixhauserIndex"]
        
    elif feature_config == "combined_all":
        # Health + Charlson + Elixhauser
        if ("Health_Index" not in df.columns) or ("CharlsonIndex" not in df.columns) or ("ElixhauserIndex" not in df.columns):
            raise KeyError("Need 'Health_Index', 'CharlsonIndex' & 'ElixhauserIndex' for 'combined_all'.")
        needed = base_cols + ["Health_Index","CharlsonIndex","ElixhauserIndex"]

    else:
        raise ValueError(f"Invalid feature_config: {feature_config}")

    return df[needed].copy()

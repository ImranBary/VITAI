# vitai_scripts/data_prep.py
# Author: Imran Feisal 
# Date: 21/01/2025
#
# Description:
#   Ensures the following pickles exist in the Data/ folder:
#     1) patient_data_sequences.pkl
#     2) patient_data_with_health_index.pkl
#   Then merges:
#     - Charlson Comorbidity Index
#     - Elixhauser Comorbidity Index
#   into a single file:
#     patient_data_with_all_indices.pkl
#   containing 'Health_Index', 'CharlsonIndex', 'ElixhauserIndex', etc.
#
#   This uses:
#     data_preprocessing.py -> Preprocess
#     health_index.py       -> Compute Health Index
#     charlson_comorbidity.py
#     elixhauser_comorbidity.py
#
#   The final pickle is 'patient_data_with_all_indices.pkl'.

import os
import logging
import pandas as pd
import gc

# Root-level modules
from data_preprocessing import main as preprocess_main
from health_index import main as health_main
from charlson_comorbidity import load_cci_mapping, compute_cci
from elixhauser_comorbidity import compute_eci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_preprocessed_data(data_dir: str) -> None:
    """
    Ensures these files exist:
      1) patient_data_sequences.pkl
      2) patient_data_with_health_index.pkl
    Then merges both Charlson & Elixhauser Indices into a single
    'patient_data_with_all_indices.pkl'.
    """
    seq_path = os.path.join(data_dir, "patient_data_sequences.pkl")
    hi_path  = os.path.join(data_dir, "patient_data_with_health_index.pkl")
    final_path = os.path.join(data_dir, "patient_data_with_all_indices.pkl")

    # 1) data_preprocessing
    if not os.path.exists(seq_path):
        logger.info("Missing patient_data_sequences.pkl -> Running data_preprocessing.")
        preprocess_main()
    else:
        logger.info("Found patient_data_sequences.pkl.")

    # 2) health_index
    if not os.path.exists(hi_path):
        logger.info("Missing patient_data_with_health_index.pkl -> Running health_index.")
        health_main()
    else:
        logger.info("Found patient_data_with_health_index.pkl.")

    # 3) If final file already exists, skip
    if os.path.exists(final_path):
        logger.info(f"Found {final_path}, skipping further merges.")
        return

    logger.info(f"Creating {final_path} by merging Charlson & Elixhauser.")
    # Load base data
    df = pd.read_pickle(hi_path)

    # Merge Charlson
    conditions_csv = os.path.join(data_dir, "conditions.csv")
    if not os.path.exists(conditions_csv):
        raise FileNotFoundError("conditions.csv not found. Cannot compute Charlson/Elixhauser.")
    conditions = pd.read_csv(conditions_csv, usecols=["PATIENT","CODE","DESCRIPTION"])

    cci_map = load_cci_mapping(data_dir)  # Provided by charlson_comorbidity
    patient_cci = compute_cci(conditions, cci_map)
    merged_cci = df.merge(patient_cci, how="left", left_on="Id", right_on="PATIENT")
    merged_cci.drop(columns="PATIENT", inplace=True)
    merged_cci["CharlsonIndex"] = merged_cci["CharlsonIndex"].fillna(0.0)
    del df, patient_cci
    gc.collect()

    # Merge Elixhauser
    eci_df = compute_eci(conditions)
    merged_eci = merged_cci.merge(eci_df, how="left", left_on="Id", right_on="PATIENT")
    merged_eci.drop(columns="PATIENT", inplace=True, errors="ignore")
    merged_eci["ElixhauserIndex"] = merged_eci["ElixhauserIndex"].fillna(0.0)
    del conditions, eci_df, merged_cci
    gc.collect()

    # Save final
    merged_eci.to_pickle(final_path)
    logger.info(f"[DataPrep] Created {final_path}.")
    del merged_eci
    gc.collect()

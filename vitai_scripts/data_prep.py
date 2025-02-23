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
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
# Root-level modules
from data_preprocessing import main as preprocess_main
from health_index import main as health_main
from charlson_comorbidity import load_cci_mapping, compute_cci
from elixhauser_comorbidity import compute_eci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_preprocessed_data(data_dir: str) -> None:
    """
    Ensures that the processed files exist and updates the final pickle
    'patient_data_with_all_indices.pkl' by appending new patients.
    """
    seq_path = os.path.join(data_dir, "patient_data_sequences.pkl")
    hi_path  = os.path.join(data_dir, "patient_data_with_health_index.pkl")
    final_path = os.path.join(data_dir, "patient_data_with_all_indices.pkl")

    from data_preprocessing import main as preprocess_main
    from health_index import main as health_main
    from charlson_comorbidity import load_cci_mapping, compute_cci
    from elixhauser_comorbidity import compute_eci

    if not os.path.exists(seq_path):
        logger.info("Missing patient_data_sequences.pkl -> Running data_preprocessing.")
        preprocess_main()
    else:
        logger.info("Found patient_data_sequences.pkl.")

    if not os.path.exists(hi_path):
        logger.info("Missing patient_data_with_health_index.pkl -> Running health_index.")
        health_main()
    else:
        logger.info("Found patient_data_with_health_index.pkl.")

    if not os.path.exists(final_path):
        logger.info(f"Creating {final_path} by merging indices.")
        df = pd.read_pickle(hi_path)
        conditions_csv = os.path.join(data_dir, "conditions.csv")
        if not os.path.exists(conditions_csv):
            raise FileNotFoundError("conditions.csv not found. Cannot compute indices.")
        conditions = pd.read_csv(conditions_csv, usecols=["PATIENT", "CODE", "DESCRIPTION"])
        cci_map = load_cci_mapping(data_dir)
        patient_cci = compute_cci(conditions, cci_map)
        merged_cci = df.merge(patient_cci, how="left", left_on="Id", right_on="PATIENT")
        merged_cci.drop(columns="PATIENT", inplace=True)
        merged_cci["CharlsonIndex"] = merged_cci["CharlsonIndex"].fillna(0.0)
        eci_df = compute_eci(conditions)
        merged_eci = merged_cci.merge(eci_df, how="left", left_on="Id", right_on="PATIENT")
        merged_eci.drop(columns="PATIENT", inplace=True, errors="ignore")
        merged_eci["ElixhauserIndex"] = merged_eci["ElixhauserIndex"].fillna(0.0)
        merged_eci.to_pickle(final_path)
        logger.info(f"[DataPrep] Created {final_path}.")
    else:
        logger.info(f"Found existing {final_path}. Updating with new patients if available.")
        existing = pd.read_pickle(final_path)
        current = pd.read_pickle(hi_path)
        # Select rows with NewData==True; if the column is missing, assume no new rows.
        if "NewData" not in current.columns:
            logger.info("No NewData column found in health index data; nothing to update.")
            return
        new_data = current[current["NewData"] == True]
        new_rows = new_data[~new_data["Id"].isin(existing["Id"])]
        if not new_rows.empty:
            logger.info(f"Appending {len(new_rows)} new patients to {final_path}.")
            conditions_csv = os.path.join(data_dir, "conditions.csv")
            conditions = pd.read_csv(conditions_csv, usecols=["PATIENT", "CODE", "DESCRIPTION"])
            cci_map = load_cci_mapping(data_dir)
            patient_cci = compute_cci(conditions, cci_map)
            merged_cci = new_rows.merge(patient_cci, how="left", left_on="Id", right_on="PATIENT")
            merged_cci.drop(columns="PATIENT", inplace=True)
            merged_cci["CharlsonIndex"] = merged_cci["CharlsonIndex"].fillna(0.0)
            eci_df = compute_eci(conditions)
            merged_new = merged_cci.merge(eci_df, how="left", left_on="Id", right_on="PATIENT")
            merged_new.drop(columns="PATIENT", inplace=True, errors="ignore")
            merged_new["ElixhauserIndex"] = merged_new["ElixhauserIndex"].fillna(0.0)
            # Mark new rows as processed
            merged_new["NewData"] = False
            updated = pd.concat([existing, merged_new], ignore_index=True)
            updated.to_pickle(final_path)
            logger.info(f"Updated {final_path} with new patients.")
        else:
            logger.info("No new patients to append.")

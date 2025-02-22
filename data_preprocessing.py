# data_preprocessing.py
# Author: Imran Feisal 
# Date: 31/10/2024
# Description:
# This script loads Synthea data from CSV files, 
# enhances feature extraction from patient demographics,
# handles missing data more robustly,
# aggregates codes from conditions, medications, procedures, and observations, 
# builds sequences of visits for each patient, and saves the processed data for modeling.

import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_and_tag_csv(pattern, base_name):
    """
    Load all CSV files matching the given pattern.
    For files whose name contains '_diff_', add columns:
      - DifferentialTimestamp (parsed from the filename)
      - NewData = True
    For others, NewData = False.
    """
    files = glob.glob(pattern)
    df_list = []
    diff_regex = re.compile(rf"{base_name}_diff_(\d{{8}}_\d{{6}})")
    for file in files:
        try:
            df = pd.read_csv(file)
            # Check if file is a differential file (has _diff_ in its name)
            m = diff_regex.search(os.path.basename(file))
            if m:
                timestamp = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                df["DifferentialTimestamp"] = timestamp
                df["NewData"] = True
            else:
                df["DifferentialTimestamp"] = pd.NaT
                df["NewData"] = False
            df_list.append(df)
            logger.info(f"Loaded {file} with shape {df.shape}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    if df_list:
        combined = pd.concat(df_list, ignore_index=True)
        combined.drop_duplicates(subset=["Id"], inplace=True)
        return combined
    else:
        logger.error(f"No files found for pattern {pattern}")
        return pd.DataFrame()

def load_data(data_dir):
    """
    Load Synthea data from CSV files (both original and differential) and preprocess patient demographics.
    Looks for files matching:
      - patients*.csv, encounters*.csv, etc.
    """
    # For patients, load all CSV files whose names start with "patients"
    patients_pattern = os.path.join(data_dir, "patients*.csv")
    patients = _load_and_tag_csv(patients_pattern, "patients")
    
    # Proceed as before with patients: select columns, convert dates, etc.
    usecols = ['Id', 'BIRTHDATE', 'DEATHDATE', 'GENDER', 'RACE', 'ETHNICITY',
               'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'MARITAL']
    patients = patients[usecols].copy()
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
    patients['DEATHDATE'] = pd.to_datetime(patients['DEATHDATE'], errors='coerce')
    patients = patients[patients['BIRTHDATE'] <= patients['BIRTHDATE'].max()]
    patients = patients[(patients['DEATHDATE'].isnull()) | (patients['DEATHDATE'] >= patients['BIRTHDATE'])]
    
    # For Age, we need encounters dates.
    encounters_pattern = os.path.join(data_dir, "encounters*.csv")
    encounters_dates = _load_and_tag_csv(encounters_pattern, "encounters")
    encounters_dates['START'] = pd.to_datetime(encounters_dates['START']).dt.tz_localize(None)
    latest_date = encounters_dates['START'].max()
    patients['AGE'] = (latest_date - patients['BIRTHDATE']).dt.days / 365.25
    patients['AGE'] = patients['AGE'].fillna(0)
    patients['DECEASED'] = patients['DEATHDATE'].notnull().astype(int)
    patients['AGE_AT_DEATH'] = ((patients['DEATHDATE'] - patients['BIRTHDATE']).dt.days / 365.25).fillna(patients['AGE'])
    patients.drop(columns=['BIRTHDATE', 'DEATHDATE'], inplace=True)
    
    # Impute missing data for numerical and categorical features (as before)
    numerical_features = ['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME']
    for col in numerical_features:
        patients[col].fillna(patients[col].median(), inplace=True)
    categorical_features = ['GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    for col in categorical_features:
        patients[col].fillna('Unknown', inplace=True)
        patients[col] = patients[col].replace('', 'Unknown')
    
    # Load encounters (similarly load all matching files)
    encounters_pattern = os.path.join(data_dir, "encounters*.csv")
    encounters = _load_and_tag_csv(encounters_pattern, "encounters")
    usecols_enc = ['Id', 'PATIENT', 'ENCOUNTERCLASS', 'START', 'STOP', 'REASONCODE', 'REASONDESCRIPTION']
    encounters = encounters[usecols_enc].copy()
    encounters['START'] = pd.to_datetime(encounters['START']).dt.tz_localize(None)
    encounters['STOP'] = pd.to_datetime(encounters['STOP']).dt.tz_localize(None)
    encounters.sort_values(by=['PATIENT', 'START'], inplace=True)
    
    logger.info("Data loaded and preprocessed successfully.")
    return patients, encounters

def aggregate_codes(data_dir):
    # (This function remains unchanged, assuming CSV names are constant)
    # It can use similar globbing if needed, but for brevity we leave it as is.
    # …
    # (Existing code)
    import pandas as pd, os
    conditions = pd.read_csv(os.path.join(data_dir, 'conditions.csv'), usecols=['PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION'])
    medications = pd.read_csv(os.path.join(data_dir, 'medications.csv'), usecols=['PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION'])
    procedures = pd.read_csv(os.path.join(data_dir, 'procedures.csv'), usecols=['PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION'])
    observations = pd.read_csv(os.path.join(data_dir, 'observations.csv'), usecols=['PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'VALUE', 'UNITS'])
    conditions['TYPE'] = 'condition'
    medications['TYPE'] = 'medication'
    procedures['TYPE'] = 'procedure'
    observations['TYPE'] = 'observation'
    codes = pd.concat([conditions, medications, procedures, observations], ignore_index=True)
    codes['CODE'] = codes['CODE'].fillna('UNKNOWN')
    codes['UNIQUE_CODE'] = codes['TYPE'] + '_' + codes['CODE'].astype(str)
    unique_codes = codes['UNIQUE_CODE'].unique()
    code_to_id = {code: idx for idx, code in enumerate(unique_codes)}
    id_to_code = {idx: code for code, idx in code_to_id.items()}
    codes['CODE_ID'] = codes['UNIQUE_CODE'].map(code_to_id)
    logger.info("Codes aggregated successfully.")
    return codes, code_to_id, id_to_code

def build_patient_sequences(encounters, codes):
    """
    Build sequences of visits for each patient.

    Args:
        encounters (pd.DataFrame): Encounters data.
        codes (pd.DataFrame): Aggregated codes.

    Returns:
        patient_sequences (dict): Mapping of patient IDs to sequences of visits.
    """
    # Create a mapping from ENCOUNTER to CODE_IDs
    encounter_code_map = codes.groupby('ENCOUNTER')['CODE_ID'].apply(list)

    # Merge encounters with codes
    encounters_with_codes = encounters[['Id', 'PATIENT']].merge(encounter_code_map, left_on='Id', right_on='ENCOUNTER', how='left')

    # Group by patient and collect sequences
    patient_sequences = encounters_with_codes.groupby('PATIENT')['CODE_ID'].apply(list).to_dict()

    logger.info("Patient sequences built successfully.")

    return patient_sequences

# ------------------------------
# 4. Save Processed Data
# ------------------------------

def save_processed_data(patients, patient_sequences, code_to_id, output_dir):
    patient_sequence_df = pd.DataFrame([
        {'PATIENT': patient_id, 'SEQUENCE': visits}
        for patient_id, visits in patient_sequences.items()
    ])
    patient_data = patients.merge(patient_sequence_df, how='inner', left_on='Id', right_on='PATIENT')
    patient_data.drop(columns=['PATIENT'], inplace=True)
    code_mappings = pd.DataFrame(list(code_to_id.items()), columns=['UNIQUE_CODE', 'CODE_ID'])
    code_mappings.to_csv(os.path.join(output_dir, 'code_mappings.csv'), index=False)
    # IMPORTANT: Preserve any existing processed data by appending new rows.
    pkl_path = os.path.join(output_dir, 'patient_data_sequences.pkl')
    if os.path.exists(pkl_path):
        existing = pd.read_pickle(pkl_path)
        # Append only new patients (based on Id not already present)
        new_rows = patient_data[~patient_data["Id"].isin(existing["Id"])]
        if not new_rows.empty:
            updated = pd.concat([existing, new_rows], ignore_index=True)
            updated.to_pickle(pkl_path)
            logger.info(f"Appended {len(new_rows)} new patients to {pkl_path}.")
        else:
            logger.info("No new patients to append.")
    else:
        patient_data.to_pickle(pkl_path)
        logger.info(f"Saved processed data to {pkl_path}.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data')
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)
    patients, encounters = load_data(data_dir)
    codes, code_to_id, id_to_code = aggregate_codes(data_dir)
    patient_sequences = build_patient_sequences(encounters, codes)
    save_processed_data(patients, patient_sequences, code_to_id, output_dir)

if __name__ == '__main__':
    main()
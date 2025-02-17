# data_preprocessing.py
# Author: Imran Feisal 
# Date: 31/10/2024
# Description:
# This script loads Synthea data from CSV files, 
# enhances feature extraction from patient demographics,
# handles missing data more robustly,
# aggregates codes from conditions, medications, procedures, and observations, 
# builds sequences of visits for each patient, and saves the processed data for modeling.

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Load Data
# ------------------------------

def load_data(data_dir):
    """
    Load Synthea data from CSV files and preprocess patient demographics.

    Args:
        data_dir (str): Directory where Synthea CSV files are stored.

    Returns:
        patients (pd.DataFrame): Processed patient demographics data.
        encounters (pd.DataFrame): Processed encounters data.
    """
    # Load Patients Data
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'), usecols=[
        'Id', 'BIRTHDATE', 'DEATHDATE', 'GENDER', 'RACE', 'ETHNICITY',
        'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'MARITAL'
    ])

    # Convert 'BIRTHDATE' and 'DEATHDATE' to datetime format
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
    patients['DEATHDATE'] = pd.to_datetime(patients['DEATHDATE'], errors='coerce')

    # Check for future birthdates and deaths before births
    patients = patients[patients['BIRTHDATE'] <= patients['BIRTHDATE'].max()]
    patients = patients[(patients['DEATHDATE'].isnull()) | (patients['DEATHDATE'] >= patients['BIRTHDATE'])]

    # Calculate Age using the latest date in encounters as reference
    encounters_dates = pd.read_csv(os.path.join(data_dir, 'encounters.csv'), usecols=['START', 'STOP'])
    encounters_dates['START'] = pd.to_datetime(encounters_dates['START']).dt.tz_localize(None)
    latest_date = encounters_dates['START'].max()
    patients['AGE'] = (latest_date - patients['BIRTHDATE']).dt.days / 365.25
    patients['AGE'] = patients['AGE'].fillna(0)

    # Calculate if patient is deceased
    patients['DECEASED'] = patients['DEATHDATE'].notnull().astype(int)

    # Calculate age at death
    patients['AGE_AT_DEATH'] = ((patients['DEATHDATE'] - patients['BIRTHDATE']).dt.days / 365.25).fillna(patients['AGE'])

    # Drop unnecessary columns
    patients.drop(columns=['BIRTHDATE', 'DEATHDATE'], inplace=True)

    # Handle missing data using median imputation for numerical features
    numerical_features = ['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME']
    for col in numerical_features:
        patients[col].fillna(patients[col].median(), inplace=True)

    # Handle missing data for categorical features by creating an 'Unknown' category
    categorical_features = ['GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    for col in categorical_features:
        patients[col].fillna('Unknown', inplace=True)
        patients[col] = patients[col].replace('', 'Unknown')

    # Load Encounters Data (Visits)
    encounters = pd.read_csv(os.path.join(data_dir, 'encounters.csv'), usecols=[
        'Id', 'PATIENT', 'ENCOUNTERCLASS', 'START', 'STOP', 'REASONCODE', 'REASONDESCRIPTION'
    ])

    # Convert START and STOP to datetime without timezone
    encounters['START'] = pd.to_datetime(encounters['START']).dt.tz_localize(None)
    encounters['STOP'] = pd.to_datetime(encounters['STOP']).dt.tz_localize(None) 

    # Sort encounters by patient and start date
    encounters.sort_values(by=['PATIENT', 'START'], inplace=True)

    logger.info("Data loaded and preprocessed successfully.")

    return patients, encounters

# ------------------------------
# 2. Prepare Visit-Level Data
# ------------------------------

def aggregate_codes(data_dir):
    """
    Aggregate codes from conditions, medications, procedures, and observations.

    Args:
        data_dir (str): Directory where Synthea CSV files are stored.

    Returns:
        codes (pd.DataFrame): Aggregated codes with unified code system.
        code_to_id (dict): Mapping from UNIQUE_CODE to integer IDs.
        id_to_code (dict): Reverse mapping from IDs to UNIQUE_CODE.
    """
    # Load data
    conditions = pd.read_csv(os.path.join(data_dir, 'conditions.csv'), usecols=[
        'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION'
    ])
    medications = pd.read_csv(os.path.join(data_dir, 'medications.csv'), usecols=[
        'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION'
    ])
    procedures = pd.read_csv(os.path.join(data_dir, 'procedures.csv'), usecols=[
        'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION'
    ])
    observations = pd.read_csv(os.path.join(data_dir, 'observations.csv'), usecols=[
        'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'VALUE', 'UNITS'
    ])

    # Combine all codes into a single DataFrame
    conditions['TYPE'] = 'condition'
    medications['TYPE'] = 'medication'
    procedures['TYPE'] = 'procedure'
    observations['TYPE'] = 'observation'

    codes = pd.concat([conditions, medications, procedures, observations], ignore_index=True)

    # Handle missing codes
    codes['CODE'] = codes['CODE'].fillna('UNKNOWN')

    # Create a unified code system
    codes['UNIQUE_CODE'] = codes['TYPE'] + '_' + codes['CODE'].astype(str)

    # Generate a mapping from UNIQUE_CODE to integer IDs
    unique_codes = codes['UNIQUE_CODE'].unique()
    code_to_id = {code: idx for idx, code in enumerate(unique_codes)}
    id_to_code = {idx: code for code, idx in code_to_id.items()}

    # Map codes to IDs
    codes['CODE_ID'] = codes['UNIQUE_CODE'].map(code_to_id)

    logger.info("Codes aggregated successfully.")

    return codes, code_to_id, id_to_code

# ------------------------------
# 3. Build Patient Sequences
# ------------------------------

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
    """
    Save the processed data for modeling.

    Args:
        patients (pd.DataFrame): Patient demographics data.
        patient_sequences (dict): Patient sequences data.
        code_to_id (dict): Code to ID mapping.
        output_dir (str): Directory to save processed data.
    """
    # Convert patient_sequences to a DataFrame
    patient_sequence_df = pd.DataFrame([
        {'PATIENT': patient_id, 'SEQUENCE': visits}
        for patient_id, visits in patient_sequences.items()
    ])

    # Merge with patient demographics
    patient_data = patients.merge(patient_sequence_df, how='inner', left_on='Id', right_on='PATIENT')

    # Drop redundant 'PATIENT' column
    patient_data.drop(columns=['PATIENT'], inplace=True)

    # Save code mappings
    code_mappings = pd.DataFrame(list(code_to_id.items()), columns=['UNIQUE_CODE', 'CODE_ID'])
    code_mappings.to_csv(os.path.join(output_dir, 'code_mappings.csv'), index=False)

    # Save patient data with sequences
    patient_data.to_pickle(os.path.join(output_dir, 'patient_data_sequences.pkl'))

    logger.info("Data aggregation complete. Processed data saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data')
    output_dir = data_dir  # same directory for outputs
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    patients, encounters = load_data(data_dir)

    # Aggregate codes
    codes, code_to_id, id_to_code = aggregate_codes(data_dir)

    # Build patient sequences
    patient_sequences = build_patient_sequences(encounters, codes)

    # Save processed data
    save_processed_data(patients, patient_sequences, code_to_id, output_dir)

if __name__ == '__main__':
    main()
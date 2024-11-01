'''
data_preprocessing.py
Author: Imran Feisal 
Date: 31/10/2024
Description:
This script loads Synthea data from CSV files, 
enhances feature extraction from patient demographics,
handles missing data more robustly,
aggregates codes from conditions, medications, procedures, and observations, 
builds sequences of visits for each patient, and saves the processed data for modeling.

Inspiration:
- Batista, G. E. A. P. A., & Monard, M. C. (2003). An analysis of four missing data treatment methods for supervised learning. Applied Artificial Intelligence, 17(5–6), 519–533.
- Jensen, P. B., Jensen, L. J., & Brunak, S. (2012). Mining electronic health records: towards better research applications and clinical care. Nature Reviews Genetics, 13(6), 395–405.
'''

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
    """Load Synthea data from CSV files."""
    # Load Patients Data
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'), usecols=[
        'Id', 'BIRTHDATE', 'DEATHDATE', 'GENDER', 'RACE', 'ETHNICITY',
        'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'MARITAL'
    ])

    # Convert 'BIRTHDATE' and 'DEATHDATE' to datetime format
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
    patients['DEATHDATE'] = pd.to_datetime(patients['DEATHDATE'], errors='coerce')

    # Check for future birthdates and deaths before births
    patients = patients[patients['BIRTHDATE'] <= datetime.now()]
    patients = patients[(patients['DEATHDATE'].isnull()) | (patients['DEATHDATE'] >= patients['BIRTHDATE'])]

    # Calculate Age
    current_date = datetime.now()
    patients['AGE'] = (current_date - patients['BIRTHDATE']).dt.days / 365.25
    patients['AGE'] = patients['AGE'].fillna(0)

    # Calculate if patient is deceased
    patients['DECEASED'] = patients['DEATHDATE'].notnull().astype(int)

    # Calculate age at death
    patients['AGE_AT_DEATH'] = ((patients['DEATHDATE'] - patients['BIRTHDATE']).dt.days / 365.25).fillna(patients['AGE'])

    # Drop unnecessary columns
    patients.drop(columns=['BIRTHDATE', 'DEATHDATE'], inplace=True)

    # Handle missing data using mean imputation for numerical features
    numerical_features = ['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME']
    for col in numerical_features:
        patients[col].fillna(patients[col].mean(), inplace=True)

    # Handle missing data for categorical features by creating an 'Unknown' category
    categorical_features = ['GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    for col in categorical_features:
        patients[col].fillna('Unknown', inplace=True)

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
    """Aggregate codes from conditions, medications, procedures, and observations."""
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
    codes = pd.concat([
        conditions.assign(TYPE='condition'),
        medications.assign(TYPE='medication'),
        procedures.assign(TYPE='procedure'),
        observations.assign(TYPE='observation')
    ], ignore_index=True)

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
    """Build sequences of visits for each patient."""
    # Create a mapping from ENCOUNTER to CODE_IDs
    encounter_code_map = codes.groupby('ENCOUNTER')['CODE_ID'].apply(list).to_dict()

    # Create a mapping from PATIENT to ENCOUNTER IDs
    patient_encounter_map = encounters.groupby('PATIENT')['Id'].apply(list).to_dict()

    # Build sequences
    patient_sequences = {}
    for patient_id, encounter_ids in patient_encounter_map.items():
        patient_visits = []
        for visit_id in encounter_ids:
            visit_codes = encounter_code_map.get(visit_id, [])
            if visit_codes:
                patient_visits.append(visit_codes)
        if patient_visits:
            patient_sequences[patient_id] = patient_visits

    logger.info("Patient sequences built successfully.")

    return patient_sequences

# ------------------------------
# 4. Save Processed Data
# ------------------------------

def save_processed_data(patients, patient_sequences, code_to_id, output_dir):
    """Save the processed data for modeling."""
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
    data_dir = r'E:\DataGen\synthea\output\csv'  
    output_dir = 'Data'
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

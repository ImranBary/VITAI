# health_index.py
# Author: Imran Feisal
# Date: 31/10/2024
# Description:
# Calculate the composite health index for each patient by grouping SNOMED CT codes
# into clinically meaningful categories, assigning weights, and calculating a health index.

import pandas as pd
import numpy as np
import os
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Load Processed Data
# ------------------------------

def load_processed_data(output_dir):
    """
    Load the processed patient data and code mappings.

    Args:
        output_dir (str): Directory where processed data is stored.

    Returns:
        patient_data (pd.DataFrame): Patient data with sequences.
        code_mappings (pd.DataFrame): Code mappings.
    """
    patient_data = pd.read_pickle(os.path.join(output_dir, 'patient_data_sequences.pkl'))
    code_mappings = pd.read_csv(os.path.join(output_dir, 'code_mappings.csv'))
    logger.info("Processed data loaded successfully.")
    return patient_data, code_mappings

# ------------------------------
# 2. Calculate Health Indicators
# ------------------------------

def calculate_health_indicators(patient_data, data_dir):
    """
    Calculate health indicators for each patient.

    Args:
        patient_data (pd.DataFrame): Patient data with sequences.
        data_dir (str): Directory where raw data is stored.

    Returns:
        patient_data (pd.DataFrame): Patient data with health indicators.
    """
    conditions = pd.read_csv(os.path.join(data_dir, 'conditions.csv'), usecols=[
        'PATIENT', 'CODE', 'DESCRIPTION'
    ])
    encounters = pd.read_csv(os.path.join(data_dir, 'encounters.csv'), usecols=[
        'Id', 'PATIENT', 'ENCOUNTERCLASS'
    ])
    medications = pd.read_csv(os.path.join(data_dir, 'medications.csv'), usecols=[
        'PATIENT', 'CODE', 'DESCRIPTION'
    ])
    observations = pd.read_csv(os.path.join(data_dir, 'observations.csv'), usecols=[
        'PATIENT', 'DESCRIPTION', 'VALUE', 'UNITS'
    ])

    # -----------------------------------------
    # 2.1 Calculate Comorbidity Score using SNOMED CT groups
    # -----------------------------------------
    snomed_groups = {
        'Cardiovascular Diseases': ['53741008', '445118002', '59621000', '22298006', '56265001'],
        'Respiratory Diseases': ['19829001', '233604007', '118940003', '409622000', '13645005'],
        'Diabetes': ['44054006', '73211009', '46635009', '190330002'],
        'Cancer': ['363346000', '254637007', '363406005', '254632001'],
        'Chronic Kidney Disease': ['709044004', '90708001', '46177005'],
        'Neurological Disorders': ['230690007', '26929004', '193003'],
    }

    group_weights = {
        'Cardiovascular Diseases': 3,
        'Respiratory Diseases': 2,
        'Diabetes': 2,
        'Cancer': 3,
        'Chronic Kidney Disease': 2,
        'Neurological Disorders': 1.5,
        'Other': 1
    }

    def find_group(code, snomed_groups):
        for group, codes in snomed_groups.items():
            if str(code) in codes:
                return group
        return 'Other'

    conditions['Group'] = conditions['CODE'].apply(lambda x: find_group(x, snomed_groups))
    conditions['Group_Weight'] = conditions['Group'].map(group_weights)
    conditions['Group_Weight'] = conditions['Group_Weight'].fillna(1)
    comorbidity_scores = conditions.groupby('PATIENT')['Group_Weight'].sum().reset_index()
    comorbidity_scores.rename(columns={'Group_Weight': 'Comorbidity_Score'}, inplace=True)

    # -----------------------------------------
    # 2.2 Calculate Hospitalizations Count
    # -----------------------------------------
    hospitalizations = encounters[encounters['ENCOUNTERCLASS'] == 'inpatient']
    hospitalizations_count = hospitalizations.groupby('PATIENT').size().reset_index(name='Hospitalizations_Count')

    # -----------------------------------------
    # 2.3 Calculate Medications Count
    # -----------------------------------------
    medications_count = medications.groupby('PATIENT')['CODE'].nunique().reset_index(name='Medications_Count')

    # -----------------------------------------
    # 2.4 Calculate Abnormal Observations Count
    # -----------------------------------------
    observation_thresholds = {
        'Systolic Blood Pressure': {'min': 90, 'max': 120},
        'Diastolic Blood Pressure': {'min': 60, 'max': 80},
        'Body Mass Index': {'min': 18.5, 'max': 24.9},
        'Blood Glucose Level': {'min': 70, 'max': 99},
        'Heart Rate': {'min': 60, 'max': 100},
    }
    observation_mappings = {
        'Systolic Blood Pressure': ['Systolic Blood Pressure'],
        'Diastolic Blood Pressure': ['Diastolic Blood Pressure'],
        'Body Mass Index': ['Body mass index (BMI) [Ratio]'],
        'Blood Glucose Level': ['Glucose [Mass/volume] in Blood'],
        'Heart Rate': ['Heart rate'],
    }
    observations['DESCRIPTION'] = observations['DESCRIPTION'].str.strip()
    observations['VALUE'] = pd.to_numeric(observations['VALUE'], errors='coerce')
    observations['IS_ABNORMAL'] = 0
    for standard_desc, desc_list in observation_mappings.items():
        thresholds = observation_thresholds.get(standard_desc, {})
        min_val = thresholds.get('min', -np.inf)
        max_val = thresholds.get('max', np.inf)
        mask = observations['DESCRIPTION'].isin(desc_list)
        observations.loc[
            mask & observations['VALUE'].notna() & ((observations['VALUE'] < min_val) | (observations['VALUE'] > max_val)),
            'IS_ABNORMAL'
        ] = 1
    abnormal_observations_count = observations.groupby('PATIENT')['IS_ABNORMAL'].sum().reset_index()
    abnormal_observations_count.rename(columns={'IS_ABNORMAL': 'Abnormal_Observations_Count'}, inplace=True)

    # -----------------------------------------
    # 2.5 Merge Counts into patient_data
    # -----------------------------------------
    patient_data.set_index('Id', inplace=True)
    patient_data = patient_data.merge(comorbidity_scores.set_index('PATIENT'), left_index=True, right_index=True, how='left')
    patient_data = patient_data.merge(hospitalizations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')
    patient_data = patient_data.merge(medications_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')
    patient_data = patient_data.merge(abnormal_observations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')
    patient_data.reset_index(inplace=True)
    indicators = ['Comorbidity_Score', 'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']
    patient_data[indicators] = patient_data[indicators].fillna(0)
    logger.info("Health indicators calculated successfully.")
    return patient_data

# ------------------------------
# 3. Calculate Composite Health Index
# ------------------------------

def calculate_health_index(patient_data):
    """
    Calculate the composite health index using PCA for weights.

    Args:
        patient_data (pd.DataFrame): Patient data with health indicators.

    Returns:
        patient_data (pd.DataFrame): Patient data with health index.
    """
    indicators = ['AGE', 'Comorbidity_Score', 'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']
    scaler = RobustScaler()
    # Fill NaN values with 0 before PCA to avoid errors
    scaled_indicators = scaler.fit_transform(patient_data[indicators].fillna(0))
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(scaled_indicators)
    weights = pca.components_[0]
    weights = weights / np.sum(np.abs(weights))
    explained_variance = pca.explained_variance_ratio_[0]
    logger.info(f"PCA explained variance ratio: {explained_variance:.4f}")
    patient_data['Health_Index'] = np.dot(scaled_indicators, weights)
    min_hi = patient_data['Health_Index'].min()
    max_hi = patient_data['Health_Index'].max()
    patient_data['Health_Index'] = 1 + 9 * (patient_data['Health_Index'] - min_hi) / (max_hi - min_hi + 1e-8)
    logger.info("Composite health index calculated successfully.")
    return patient_data

# ------------------------------
# 4. Save Health Index
# ------------------------------

def save_health_index(patient_data, output_dir):
    """
    Save the patient data with health index.

    Args:
        patient_data (pd.DataFrame): Patient data with health index.
        output_dir (str): Directory to save the data.
    """
    patient_data.to_pickle(os.path.join(output_dir, 'patient_data_with_health_index.pkl'))
    logger.info("Health index calculated and saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data')
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)
    patient_data, code_mappings = load_processed_data(output_dir)
    patient_data = calculate_health_indicators(patient_data, data_dir)
    patient_data = calculate_health_index(patient_data)
    save_health_index(patient_data, output_dir)

if __name__ == '__main__':
    main()

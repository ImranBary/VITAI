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
    # Load additional data needed
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

    # Define SNOMED CT code groups with actual codes
    snomed_groups = {
        'Cardiovascular Diseases': ['53741008', '445118002', '59621000', '22298006', '56265001'],
        'Respiratory Diseases': ['19829001', '233604007', '118940003', '409622000', '13645005'],
        'Diabetes': ['44054006', '73211009', '46635009', '190330002'],
        'Cancer': ['363346000', '254637007', '363406005', '254632001'],
        'Chronic Kidney Disease': ['709044004', '90708001', '46177005'],
        'Neurological Disorders': ['230690007', '26929004', '193003'],
        # Add more groups and codes as needed
    }

    # Assign weights to groups based on clinical significance
    group_weights = {
        'Cardiovascular Diseases': 3,
        'Respiratory Diseases': 2,
        'Diabetes': 2,
        'Cancer': 3,
        'Chronic Kidney Disease': 2,
        'Neurological Disorders': 1.5,
        'Other': 1  # Assign a default weight to other conditions
        # Adjust weights as appropriate
    }

    # Function to find group for a given code
    def find_group(code, snomed_groups):
        for group, codes in snomed_groups.items():
            if str(code) in codes:
                return group
        return 'Other'

    # Map codes to groups
    conditions['Group'] = conditions['CODE'].apply(lambda x: find_group(x, snomed_groups))

    # Assign weights to conditions
    conditions['Group_Weight'] = conditions['Group'].map(group_weights)
    conditions['Group_Weight'] = conditions['Group_Weight'].fillna(1)  # Assign default weight if not found

    # Sum comorbidity weights per patient
    comorbidity_scores = conditions.groupby('PATIENT')['Group_Weight'].sum().reset_index()
    comorbidity_scores.rename(columns={'Group_Weight': 'Comorbidity_Score'}, inplace=True)

    # -----------------------------------------
    # 2.2 Calculate Hospitalizations Count
    # -----------------------------------------
    # Filter encounters for inpatient class
    hospitalizations = encounters[encounters['ENCOUNTERCLASS'] == 'inpatient']
    hospitalizations_count = hospitalizations.groupby('PATIENT').size().reset_index(name='Hospitalizations_Count')

    # -----------------------------------------
    # 2.3 Calculate Medications Count
    # -----------------------------------------
    medications_count = medications.groupby('PATIENT')['CODE'].nunique().reset_index(name='Medications_Count')

    # -----------------------------------------
    # 2.4 Calculate Abnormal Observations Count
    # -----------------------------------------
    # Define thresholds for abnormal observations based on clinical guidelines
    observation_thresholds = {
        'Systolic Blood Pressure': {'min': 90, 'max': 120},
        'Diastolic Blood Pressure': {'min': 60, 'max': 80},
        'Body Mass Index': {'min': 18.5, 'max': 24.9},
        'Blood Glucose Level': {'min': 70, 'max': 99},
        'Heart Rate': {'min': 60, 'max': 100},
        # Add more observations with thresholds as needed
    }

    # Map observation descriptions to standardized names
    observation_mappings = {
        'Systolic Blood Pressure': ['Systolic Blood Pressure'],
        'Diastolic Blood Pressure': ['Diastolic Blood Pressure'],
        'Body Mass Index': ['Body mass index (BMI) [Ratio]'],
        'Blood Glucose Level': ['Glucose [Mass/volume] in Blood'],
        'Heart Rate': ['Heart rate'],
        # Add more mappings as needed
    }

    # Normalize observation descriptions
    observations['DESCRIPTION'] = observations['DESCRIPTION'].str.strip()

    # Convert 'VALUE' to numeric, coercing errors to NaN
    observations['VALUE'] = pd.to_numeric(observations['VALUE'], errors='coerce')

    # Initialize abnormal flag to zero
    observations['IS_ABNORMAL'] = 0

    # Iterate through each standardized observation and thresholds
    for standard_desc, desc_list in observation_mappings.items():
        thresholds = observation_thresholds.get(standard_desc, {})
        min_val = thresholds.get('min', -np.inf)
        max_val = thresholds.get('max', np.inf)
        mask = observations['DESCRIPTION'].isin(desc_list)

        # Apply threshold checks
        observations.loc[
            mask & observations['VALUE'].notna() & (
                (observations['VALUE'] < min_val) |
                (observations['VALUE'] > max_val)
            ),
            'IS_ABNORMAL'
        ] = 1

    # Group by patient to count abnormal observations
    abnormal_observations_count = observations.groupby('PATIENT')['IS_ABNORMAL'].sum().reset_index()
    abnormal_observations_count.rename(columns={'IS_ABNORMAL': 'Abnormal_Observations_Count'}, inplace=True)

    # -----------------------------------------
    # 2.5 Merge Counts into patient_data
    # -----------------------------------------
    # Set the index of patient_data to 'Id' for efficient merging
    patient_data.set_index('Id', inplace=True)

    # Merge Comorbidity Score
    patient_data = patient_data.merge(comorbidity_scores.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Hospitalizations Count
    patient_data = patient_data.merge(hospitalizations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Medications Count
    patient_data = patient_data.merge(medications_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Abnormal Observations Count
    patient_data = patient_data.merge(abnormal_observations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Reset index to have 'Id' as a column again
    patient_data.reset_index(inplace=True)

    # -----------------------------------------
    # 2.6 Fill NaN values appropriately
    # -----------------------------------------
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
    # Define indicators
    indicators = ['AGE', 'Comorbidity_Score', 'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']

    # Normalize indicators using Robust Scaler to reduce the influence of outliers
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_indicators = scaler.fit_transform(patient_data[indicators])

    # Perform PCA to determine weights
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(scaled_indicators)
    weights = pca.components_[0]
    weights = weights / np.sum(np.abs(weights))  # Normalize weights

    # Check PCA explained variance
    explained_variance = pca.explained_variance_ratio_[0]
    logger.info(f"PCA explained variance ratio: {explained_variance:.4f}")

    # Calculate health index
    patient_data['Health_Index'] = np.dot(scaled_indicators, weights)

    # Scale Health_Index to range 1 to 10
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
    output_dir = data_dir  # same directory for outputs
    os.makedirs(output_dir, exist_ok=True)

    # Load processed data
    patient_data, code_mappings = load_processed_data(output_dir)

    # Calculate health indicators
    patient_data = calculate_health_indicators(patient_data, data_dir)

    # Calculate health index
    patient_data = calculate_health_index(patient_data)

    # Save health index
    save_health_index(patient_data, output_dir)

if __name__ == '__main__':
    main()


'''
import pandas as pd
import numpy as np
import os
import logging
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Load Processed Data
# ------------------------------

def load_processed_data(output_dir):
    """Load the processed patient data and code mappings."""
    patient_data = pd.read_pickle(os.path.join(output_dir, 'patient_data_sequences.pkl'))
    code_mappings = pd.read_csv(os.path.join(output_dir, 'code_mappings.csv'))
    logger.info("Processed data loaded successfully.")
    return patient_data, code_mappings

# ------------------------------
# 2. Calculate Health Indicators
# ------------------------------

def calculate_health_indicators(patient_data, data_dir):
    """Calculate health indicators for each patient."""
    # Load additional data needed
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

    # Define SNOMED CT code groups (Replace with actual codes from your dataset)
    snomed_groups = {
        'Cardiovascular Diseases': ['53741008', '445118002', '59621000'],  # Example SNOMED CT codes
        'Respiratory Diseases': ['19829001', '233604007', '118940003'],
        'Diabetes': ['44054006', '73211009'],
        'Cancer': ['363346000', '254637007'],
        'Chronic Kidney Disease': ['709044004', '90708001'],
        'Neurological Disorders': ['230690007', '26929004'],
        # Add more groups and codes as needed
    }

    # Assign weights to groups based on clinical significance
    group_weights = {
        'Cardiovascular Diseases': 2,
        'Respiratory Diseases': 1.5,
        'Diabetes': 1.5,
        'Cancer': 2,
        'Chronic Kidney Disease': 1.5,
        'Neurological Disorders': 1,
        'Other': 0.5  # Assign a small weight to other conditions
        # Adjust weights as appropriate
    }

    # Function to find group for a given code
    def find_group(code, snomed_groups):
        for group, codes in snomed_groups.items():
            if str(code) in codes:
                return group
        return 'Other'

    # Map codes to groups
    conditions['Group'] = conditions['CODE'].apply(lambda x: find_group(x, snomed_groups))

    # Assign weights to conditions
    conditions['Group_Weight'] = conditions['Group'].map(group_weights)
    conditions['Group_Weight'] = conditions['Group_Weight'].fillna(0)  # Assign 0 weight if not found

    # Sum comorbidity weights per patient
    comorbidity_scores = conditions.groupby('PATIENT')['Group_Weight'].sum().reset_index()
    comorbidity_scores.rename(columns={'Group_Weight': 'Comorbidity_Score'}, inplace=True)

    # -----------------------------------------
    # 2.2 Calculate Hospitalizations Count
    # -----------------------------------------
    # Filter encounters for inpatient class
    hospitalizations = encounters[encounters['ENCOUNTERCLASS'] == 'inpatient']
    hospitalizations_count = hospitalizations.groupby('PATIENT').size().reset_index(name='Hospitalizations_Count')

    # -----------------------------------------
    # 2.3 Calculate Medications Count
    # -----------------------------------------
    medications_count = medications.groupby('PATIENT')['CODE'].nunique().reset_index(name='Medications_Count')

    # -----------------------------------------
    # 2.4 Calculate Abnormal Observations Count
    # -----------------------------------------
    # Define thresholds for abnormal observations based on clinical guidelines
    observation_thresholds = {
        'Systolic Blood Pressure': {'min': 90, 'max': 120},
        'Diastolic Blood Pressure': {'min': 60, 'max': 80},
        'Body mass index (BMI) [Ratio]': {'min': 18.5, 'max': 24.9},
        'Glucose [Mass/volume] in Blood': {'min': 70, 'max': 99},
        'Heart rate': {'min': 60, 'max': 100},
        # Add more observations with thresholds as needed
    }

    # Convert 'VALUE' to numeric, coercing errors to NaN
    observations['VALUE'] = pd.to_numeric(observations['VALUE'], errors='coerce')

    # Initialize abnormal flag to zero
    observations['IS_ABNORMAL'] = 0

    # Iterate through each observation description and threshold
    for obs_desc, thresholds in observation_thresholds.items():
        # Create a mask for the current observation
        mask = observations['DESCRIPTION'] == obs_desc

        # Apply threshold checks only on rows with numeric 'VALUE'
        observations.loc[
            mask & observations['VALUE'].notna() & (
                (observations['VALUE'] < thresholds['min']) |
                (observations['VALUE'] > thresholds['max'])
            ),
            'IS_ABNORMAL'
        ] = 1

    # Group by patient to count abnormal observations
    abnormal_observations_count = observations.groupby('PATIENT')['IS_ABNORMAL'].sum().reset_index()
    abnormal_observations_count.rename(columns={'IS_ABNORMAL': 'Abnormal_Observations_Count'}, inplace=True)

    # -----------------------------------------
    # 2.5 Merge Counts into patient_data
    # -----------------------------------------
    # Ensure that the 'Id' column in patient_data matches 'PATIENT' in counts

    # Set the index of patient_data to 'Id' for efficient merging
    patient_data.set_index('Id', inplace=True)

    # Merge Comorbidity Score
    patient_data = patient_data.merge(comorbidity_scores.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Hospitalizations Count
    patient_data = patient_data.merge(hospitalizations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Medications Count
    patient_data = patient_data.merge(medications_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Abnormal Observations Count
    patient_data = patient_data.merge(abnormal_observations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Reset index to have 'Id' as a column again
    patient_data.reset_index(inplace=True)

    # -----------------------------------------
    # 2.6 Fill NaN values with zeros
    # -----------------------------------------
    indicators = ['Comorbidity_Score', 'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']
    patient_data[indicators] = patient_data[indicators].fillna(0)

    logger.info("Health indicators calculated successfully.")

    return patient_data

# ------------------------------
# 3. Calculate Composite Health Index
# ------------------------------

def calculate_health_index(patient_data):
    """Calculate the composite health index using PCA for weights."""
    # Define indicators
    indicators = ['AGE', 'Comorbidity_Score', 'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']

    # Normalize indicators using Robust Scaler to reduce the influence of outliers
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    patient_data[indicators] = scaler.fit_transform(patient_data[indicators])

    # Perform PCA to determine weights
    pca = PCA(n_components=1)
    pca.fit(patient_data[indicators])
    weights = pca.components_[0]
    weights = weights / np.sum(np.abs(weights))  # Normalize weights

    # Calculate health index
    patient_data['Health_Index'] = np.dot(patient_data[indicators], weights)

    # Scale Health_Index to range 1 to 10
    min_hi = patient_data['Health_Index'].min()
    max_hi = patient_data['Health_Index'].max()
    patient_data['Health_Index'] = 1 + 9 * (patient_data['Health_Index'] - min_hi) / (max_hi - min_hi + 1e-8)

    logger.info("Composite health index calculated successfully.")

    return patient_data

# ------------------------------
# 4. Save Health Index
# ------------------------------

def save_health_index(patient_data, output_dir):
    """Save the patient data with health index."""
    patient_data.to_pickle(os.path.join(output_dir, 'patient_data_with_health_index.pkl'))
    logger.info("Health index calculated and saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    output_dir = 'Data'
    data_dir = r'E:\DataGen\synthea\output\csv'

    # Load processed data
    patient_data, code_mappings = load_processed_data(output_dir)

    # Calculate health indicators
    patient_data = calculate_health_indicators(patient_data, data_dir)

    # Calculate health index
    patient_data = calculate_health_index(patient_data)

    # Save health index
    save_health_index(patient_data, output_dir)

if __name__ == '__main__':
    main()

'''
'''
# health_index.py
# Author: Imran Feisal
# Date: 31/10/2024
# Description: Calculate the composite health index for each patient and save it for use in modeling.

import pandas as pd
import numpy as np
import os

# ------------------------------
# 1. Load Processed Data
# ------------------------------

def load_processed_data(output_dir):
    """Load the processed patient data and code mappings."""
    patient_data = pd.read_pickle(os.path.join(output_dir, 'patient_data_sequences.pkl'))
    code_mappings = pd.read_csv(os.path.join(output_dir, 'code_mappings.csv'))
    return patient_data, code_mappings

# ------------------------------
# 2. Calculate Health Indicators
# ------------------------------

def calculate_health_indicators(patient_data, data_dir):
    """Calculate health indicators for each patient."""
    # Load additional data needed
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
    # 2.1 Calculate Chronic Conditions Count
    # -----------------------------------------
    # Define a list of chronic conditions based on your data
    chronic_conditions_list = [
        # From your unique condition descriptions, extract chronic conditions
        'Asthma',
        'Chronic sinusitis (disorder)',
        'Chronic obstructive lung disease',
        'Chronic kidney disease stage 1 (disorder)',
        'Chronic kidney disease stage 2 (disorder)',
        'Chronic kidney disease stage 3 (disorder)',
        'Chronic kidney disease stage 4 (disorder)',
        'Chronic neck pain (finding)',
        'Chronic low back pain (finding)',
        'Chronic congestive heart failure (disorder)',
        'Chronic pain',
        'Chronic obstructive bronchitis (disorder)',
        'Chronic hepatitis C (disorder)',
        'Chronic intractable migraine without aura',
        'Chronic type B viral hepatitis (disorder)',
        'Diabetes mellitus type 2 (disorder)',
        'Essential hypertension (disorder)',
        'Heart failure (disorder)',
        'Human immunodeficiency virus infection (disorder)',
        'Ischemic heart disease (disorder)',
        'Metabolic syndrome X (disorder)',
        'Multiple myeloma (disorder)',
        'Osteoarthritis of knee',
        'Osteoarthritis of hip',
        'Pulmonary emphysema (disorder)',
        'Rheumatoid arthritis',
        'Chronic pain (finding)',
        'Epilepsy (disorder)',
        'Fibromyalgia (disorder)',
        'Obstructive sleep apnea syndrome (disorder)',
        'Seizure disorder',
        'Sleep apnea (disorder)',
        'Cerebral palsy (disorder)',
        'Spasticity (finding)',
        'Stroke',
        'Depression',
        'Anemia (disorder)',
        'Gout',
        'Hypoxemia (disorder)',
        'Neuropathy due to type 2 diabetes mellitus (disorder)',
        'Retinopathy due to type 2 diabetes mellitus (disorder)',
        'Chronic renal failure (disorder)',
        'Chronic respiratory failure (disorder)',
        # Add more conditions as needed
    ]

    # Map condition descriptions to chronic conditions
    conditions['IS_CHRONIC'] = conditions['DESCRIPTION'].isin(chronic_conditions_list).astype(int)
    chronic_conditions_count = conditions.groupby('PATIENT')['IS_CHRONIC'].sum().reset_index()
    chronic_conditions_count.rename(columns={'IS_CHRONIC': 'Chronic_Conditions_Count'}, inplace=True)

    # -----------------------------------------
    # 2.2 Calculate Hospitalizations Count
    # -----------------------------------------
    # Filter encounters for inpatient class
    hospitalizations = encounters[encounters['ENCOUNTERCLASS'] == 'inpatient']
    hospitalizations_count = hospitalizations.groupby('PATIENT').size().reset_index(name='Hospitalizations_Count')

    # -----------------------------------------
    # 2.3 Calculate Medications Count
    # -----------------------------------------
    medications_count = medications.groupby('PATIENT')['CODE'].nunique().reset_index(name='Medications_Count')

    # -----------------------------------------
    # 2.4 Calculate Abnormal Observations Count
    # -----------------------------------------
    # Define abnormal observations based on descriptions
    abnormal_observations_list = [
        # From your unique observation descriptions, identify those indicating abnormal results
        'Hemoglobin A1c/Hemoglobin.total in Blood',
        'Glucose [Mass/volume] in Blood',
        'Cholesterol [Mass/volume] in Serum or Plasma',
        'Triglycerides',
        'Low Density Lipoprotein Cholesterol',
        'Cholesterol in HDL [Mass/volume] in Serum or Plasma',
        'Blood Pressure',
        'Body mass index (BMI) [Ratio]',
        'Oxygen saturation in Arterial blood',
        'Troponin I.cardiac [Mass/volume] in Serum or Plasma by High sensitivity method',
        'C reactive protein [Mass/volume] in Serum or Plasma',
        'Prothrombin time (PT)',
        'INR in Platelet poor plasma by Coagulation assay',
        'Creatinine [Mass/volume] in Blood',
        'Urea nitrogen [Mass/volume] in Blood',
        'Glomerular filtration rate/1.73 sq M.predicted',
        # Add more observations as needed
    ]

    # Assume any recorded value for these observations indicates an abnormal result
    observations['IS_ABNORMAL'] = observations['DESCRIPTION'].isin(abnormal_observations_list).astype(int)
    abnormal_observations_count = observations.groupby('PATIENT')['IS_ABNORMAL'].sum().reset_index()
    abnormal_observations_count.rename(columns={'IS_ABNORMAL': 'Abnormal_Observations_Count'}, inplace=True)

    # -----------------------------------------
    # 2.5 Merge Counts into patient_data
    # -----------------------------------------
    # Ensure that the 'Id' column in patient_data matches 'PATIENT' in counts
    # So, we'll merge on 'Id' and 'PATIENT' appropriately

    # First, set the index of patient_data to 'Id' for efficient merging
    patient_data.set_index('Id', inplace=True)

    # Merge Chronic Conditions Count
    patient_data = patient_data.merge(chronic_conditions_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Hospitalizations Count
    patient_data = patient_data.merge(hospitalizations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Medications Count
    patient_data = patient_data.merge(medications_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Merge Abnormal Observations Count
    patient_data = patient_data.merge(abnormal_observations_count.set_index('PATIENT'), left_index=True, right_index=True, how='left')

    # Reset index to have 'Id' as a column again
    patient_data.reset_index(inplace=True)

    # -----------------------------------------
    # 2.6 Fill NaN values with zeros
    # -----------------------------------------
    indicators = ['Chronic_Conditions_Count', 'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']
    patient_data[indicators] = patient_data[indicators].fillna(0)

    return patient_data

# ------------------------------
# 3. Calculate Composite Health Index
# ------------------------------

def calculate_health_index(patient_data):
    """Calculate the composite health index."""
    # Define weights
    weights = {
        'AGE': 0.2,
        'Chronic_Conditions_Count': 0.3,
        'Hospitalizations_Count': 0.25,
        'Medications_Count': 0.15,
        'Abnormal_Observations_Count': 0.1
    }

    # Normalize indicators
    indicators = ['AGE', 'Chronic_Conditions_Count', 'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']
    for col in indicators:
        if patient_data[col].max() != patient_data[col].min():
            # Min-Max scale with a small epsilon to avoid division by zero
            epsilon = 1e-8
            patient_data[col] = (patient_data[col] - patient_data[col].min()) / (patient_data[col].max() - patient_data[col].min() + epsilon)

            #patient_data[col] = (patient_data[col] - patient_data[col].min()) / (patient_data[col].max() - patient_data[col].min())
        else:
            patient_data[col] = 0  # If all values are the same, set normalized value to 0

    # Calculate health index
    patient_data['Health_Index'] = (
        patient_data['AGE'] * weights['AGE'] +
        patient_data['Chronic_Conditions_Count'] * weights['Chronic_Conditions_Count'] +
        patient_data['Hospitalizations_Count'] * weights['Hospitalizations_Count'] +
        patient_data['Medications_Count'] * weights['Medications_Count'] +
        patient_data['Abnormal_Observations_Count'] * weights['Abnormal_Observations_Count']
    )

    return patient_data

# ------------------------------
# 4. Save Health Index
# ------------------------------

def save_health_index(patient_data, output_dir):
    """Save the patient data with health index."""
    patient_data.to_pickle(os.path.join(output_dir, 'patient_data_with_health_index.pkl'))
    print("Health index calculated and saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    output_dir = 'Data'
    data_dir = r'E:\DataGen\synthea\output\csv' 

    # Load processed data
    patient_data, code_mappings = load_processed_data(output_dir)

    # Calculate health indicators
    patient_data = calculate_health_indicators(patient_data, data_dir)

    # Calculate health index
    patient_data = calculate_health_index(patient_data)

    # Save health index
    save_health_index(patient_data, output_dir)

if __name__ == '__main__':
    main()


Comments on improvements:
> Abnormal observations Calculations:
. For a more accurate count of abnormal observations, consider implementing logic to compare observation values against normal reference ranges.
. This would require defining normal ranges for each observation and checking if the patient's value falls outside that range.
. Implementing this would involve more detailed data analysis and possibly domain expertise.

> Medical Categorisation:
. You might consider categorising medications into classes (e.g., antihypertensives, antidiabetics) and counting medications in specific categories.
. This could provide more insight into the medication burden related to chronic conditions.

> Encounter Data:
. Besides hospitalisations, consider analyzing emergency visits or other encounter types that may indicate poor health status.
'''
# aggregation_script.py

import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------
# 1. Load Data
# ------------------------------

# 1.1 Load Patients Data
patients = pd.read_csv(r"D:\DataGen\synthea\output\csv\patients.csv", usecols=[
    'Id', 'BIRTHDATE', 'DEATHDATE', 'GENDER', 'RACE', 'ETHNICITY',
    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME'
])

# Convert BIRTHDATE and DEATHDATE to datetime without timezone
patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE']).dt.tz_localize(None)
patients['DEATHDATE'] = pd.to_datetime(patients['DEATHDATE']).dt.tz_localize(None)

# Calculate Age
current_date = datetime.now()
patients['AGE'] = (current_date - patients['BIRTHDATE']).dt.days / 365.25
patients['AGE'] = patients['AGE'].fillna(0)

# Calculate if patient is deceased
patients['DECEASED'] = patients['DEATHDATE'].notnull().astype(int)

# Drop unnecessary columns
patients.drop(columns=['BIRTHDATE', 'DEATHDATE'], inplace=True)

# ------------------------------
# 2. Aggregate Conditions
# ------------------------------

# Load Conditions Data
conditions = pd.read_csv(r"D:\DataGen\synthea\output\csv\conditions.csv", usecols=['PATIENT', 'DESCRIPTION', 'START', 'STOP'])

# Convert START and STOP to datetime without timezone
conditions['START'] = pd.to_datetime(conditions['START']).dt.tz_localize(None)
conditions['STOP'] = pd.to_datetime(conditions['STOP']).dt.tz_localize(None)

# Calculate duration of conditions
conditions['DURATION_DAYS'] = (conditions['STOP'] - conditions['START']).dt.days
conditions['DURATION_DAYS'] = conditions['DURATION_DAYS'].fillna(0)

# Count of conditions per patient
condition_counts = conditions.groupby('PATIENT').size().reset_index(name='num_conditions')

# Total duration of conditions per patient
condition_duration = conditions.groupby('PATIENT')['DURATION_DAYS'].sum().reset_index(name='total_condition_duration')

# Conditions to search for
condition_descriptions = {
    'diabetes': 'diabetes',
    'hypertension': 'hypertension',
    'asthma': 'asthma'
}

# Create binary flags for conditions
for name, keyword in condition_descriptions.items():
    patients[name] = patients['Id'].isin(
        conditions.loc[
            conditions['DESCRIPTION'].str.contains(keyword, case=False, na=False), 'PATIENT'
        ]
    ).astype(int)

# ------------------------------
# 3. Aggregate Medications
# ------------------------------

# Load Medications Data
medications = pd.read_csv(r"D:\DataGen\synthea\output\csv\medications.csv", usecols=['PATIENT', 'DESCRIPTION', 'START', 'STOP'])

# Convert START and STOP to datetime without timezone
medications['START'] = pd.to_datetime(medications['START']).dt.tz_localize(None)
medications['STOP'] = pd.to_datetime(medications['STOP']).dt.tz_localize(None)

# Calculate duration of medications
medications['DURATION_DAYS'] = (medications['STOP'] - medications['START']).dt.days
medications['DURATION_DAYS'] = medications['DURATION_DAYS'].fillna(0)

# Count of medications per patient
medication_counts = medications.groupby('PATIENT').size().reset_index(name='num_medications')

# Total duration of medications per patient
medication_duration = medications.groupby('PATIENT')['DURATION_DAYS'].sum().reset_index(name='total_medication_duration')

# Medications to search for
medication_descriptions = {
    'metformin': 'metformin',
    'lisinopril': 'lisinopril',
    'albuterol': 'albuterol'
}

# Create binary flags for medications
for name, keyword in medication_descriptions.items():
    patients[name] = patients['Id'].isin(
        medications.loc[
            medications['DESCRIPTION'].str.contains(keyword, case=False, na=False), 'PATIENT'
        ]
    ).astype(int)

# ------------------------------
# 4. Aggregate Observations
# ------------------------------

# Load Observations Data
observations = pd.read_csv(r"D:\DataGen\synthea\output\csv\observations.csv", usecols=[
    'PATIENT', 'DATE', 'DESCRIPTION', 'VALUE', 'UNITS'
])

# Convert DATE to datetime without timezone
observations['DATE'] = pd.to_datetime(observations['DATE']).dt.tz_localize(None)

# Filter for vital signs of interest
vital_signs_descriptions = {
    'weight': 'Body Weight',
    'height': 'Body Height',
    'diastolic_bp': 'Diastolic Blood Pressure',
    'systolic_bp': 'Systolic Blood Pressure',
    'body_temperature': 'Body Temperature',
    'heart_rate': 'Heart rate',
    'respiratory_rate': 'Respiratory rate'
}

vitals = observations[observations['DESCRIPTION'].isin(vital_signs_descriptions.values())].copy()

# Convert VALUE to numeric
vitals['VALUE'] = pd.to_numeric(vitals['VALUE'], errors='coerce')

# Calculate statistical measures per patient and vital sign
vitals_agg = vitals.groupby(['PATIENT', 'DESCRIPTION']).agg({
    'VALUE': ['mean', 'std', 'min', 'max', 'last'],
    'DATE': ['last']
}).reset_index()

# Flatten MultiIndex columns
vitals_agg.columns = ['PATIENT', 'DESCRIPTION', 'mean', 'std', 'min', 'max', 'last_value', 'last_date']

# Ensure 'last_date' is timezone-naive
vitals_agg['last_date'] = vitals_agg['last_date'].dt.tz_localize(None)

# Calculate days since last observation
vitals_agg['days_since_last'] = (current_date - vitals_agg['last_date']).dt.days

# Pivot to wide format
vitals_pivot = vitals_agg.pivot(index='PATIENT', columns='DESCRIPTION')

# Flatten MultiIndex columns
vitals_pivot.columns = ['_'.join(col).strip() for col in vitals_pivot.columns.values]

# Reset index
vitals_pivot.reset_index(inplace=True)

# ------------------------------
# 5. Aggregate Procedures
# ------------------------------

# Load Procedures Data
procedures = pd.read_csv(r"D:\DataGen\synthea\output\csv\procedures.csv", usecols=['PATIENT', 'DESCRIPTION', 'START', 'STOP'])

# Convert START and STOP to datetime without timezone
procedures['START'] = pd.to_datetime(procedures['START']).dt.tz_localize(None)
procedures['STOP'] = pd.to_datetime(procedures['STOP']).dt.tz_localize(None)

# Calculate duration of procedures
procedures['DURATION_DAYS'] = (procedures['STOP'] - procedures['START']).dt.days
procedures['DURATION_DAYS'] = procedures['DURATION_DAYS'].fillna(0)

# Count of procedures per patient
procedure_counts = procedures.groupby('PATIENT').size().reset_index(name='num_procedures')

# Total duration of procedures per patient
procedure_duration = procedures.groupby('PATIENT')['DURATION_DAYS'].sum().reset_index(name='total_procedure_duration')

# Procedures to search for
procedure_descriptions = {
    'appendectomy': 'appendectomy',
    'blood_transfusion': 'blood transfusion'
}

# Create binary flags for procedures
for name, keyword in procedure_descriptions.items():
    patients[name] = patients['Id'].isin(
        procedures.loc[
            procedures['DESCRIPTION'].str.contains(keyword, case=False, na=False), 'PATIENT'
        ]
    ).astype(int)

# ------------------------------
# 6. Aggregate Allergies
# ------------------------------

# Load Allergies Data
allergies = pd.read_csv(r"D:\DataGen\synthea\output\csv\allergies.csv", usecols=['PATIENT', 'DESCRIPTION'])

# Count of allergies per patient
allergy_counts = allergies.groupby('PATIENT').size().reset_index(name='num_allergies')

# Allergies to search for
allergy_descriptions = {
    'penicillin_allergy': 'penicillin',
    'peanut_allergy': 'peanut'
}

# Create binary flags for allergies
for name, keyword in allergy_descriptions.items():
    patients[name] = patients['Id'].isin(
        allergies.loc[
            allergies['DESCRIPTION'].str.contains(keyword, case=False, na=False), 'PATIENT'
        ]
    ).astype(int)

# ------------------------------
# 7. Aggregate Imaging Studies
# ------------------------------

# Load Imaging Studies Data
imaging_studies = pd.read_csv(r"D:\DataGen\synthea\output\csv\imaging_studies.csv", usecols=[
    'PATIENT', 'DATE', 'MODALITY_CODE'
])

# Convert DATE to datetime without timezone
imaging_studies['DATE'] = pd.to_datetime(imaging_studies['DATE']).dt.tz_localize(None)

# Count of imaging studies per patient
imaging_counts = imaging_studies.groupby('PATIENT').size().reset_index(name='num_imaging_studies')

# Counts of imaging modalities per patient
modalities = imaging_studies.groupby(['PATIENT', 'MODALITY_CODE']).size().unstack(fill_value=0)
modalities.reset_index(inplace=True)

# Map modality codes to descriptive names
modality_codes = {
    'CT': 'ct_scans',
    'MR': 'mri_scans',
    'DX': 'xray_scans',
    'US': 'ultrasound_scans',
    'NM': 'nuclear_medicine_scans',
    'OT': 'other_scans'
}

# Ensure modality columns are correctly named
modalities.rename(columns=modality_codes, inplace=True)

# ------------------------------
# 8. Aggregate Encounters
# ------------------------------

# Load Encounters Data
encounters = pd.read_csv(r"D:\DataGen\synthea\output\csv\encounters.csv", usecols=['PATIENT', 'ENCOUNTERCLASS', 'START', 'STOP'])

# Convert START and STOP to datetime without timezone
encounters['START'] = pd.to_datetime(encounters['START']).dt.tz_localize(None)
encounters['STOP'] = pd.to_datetime(encounters['STOP']).dt.tz_localize(None)

# Calculate duration of encounters
encounters['DURATION_DAYS'] = (encounters['STOP'] - encounters['START']).dt.days
encounters['DURATION_DAYS'] = encounters['DURATION_DAYS'].fillna(0)

# Count of encounters per patient
encounter_counts = encounters.groupby('PATIENT').size().reset_index(name='num_encounters')

# Total duration of encounters per patient
encounter_duration = encounters.groupby('PATIENT')['DURATION_DAYS'].sum().reset_index(name='total_encounter_duration')

# Counts of encounter types per patient
encounter_types = encounters.groupby(['PATIENT', 'ENCOUNTERCLASS']).size().unstack(fill_value=0)
encounter_types.reset_index(inplace=True)

# ------------------------------
# 9. Merge All Data
# ------------------------------

# Start with patients DataFrame
merged_df = patients.copy()

# Merge condition counts and duration
merged_df = merged_df.merge(
    condition_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
).merge(
    condition_duration,
    how='left',
    on='PATIENT'
)
merged_df['num_conditions'] = merged_df['num_conditions'].fillna(0).astype(int)
merged_df['total_condition_duration'] = merged_df['total_condition_duration'].fillna(0)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge medication counts and duration
merged_df = merged_df.merge(
    medication_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
).merge(
    medication_duration,
    how='left',
    on='PATIENT'
)
merged_df['num_medications'] = merged_df['num_medications'].fillna(0).astype(int)
merged_df['total_medication_duration'] = merged_df['total_medication_duration'].fillna(0)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge procedure counts and duration
merged_df = merged_df.merge(
    procedure_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
).merge(
    procedure_duration,
    how='left',
    on='PATIENT'
)
merged_df['num_procedures'] = merged_df['num_procedures'].fillna(0).astype(int)
merged_df['total_procedure_duration'] = merged_df['total_procedure_duration'].fillna(0)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge allergy counts
merged_df = merged_df.merge(
    allergy_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df['num_allergies'] = merged_df['num_allergies'].fillna(0).astype(int)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge imaging counts
merged_df = merged_df.merge(
    imaging_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df['num_imaging_studies'] = merged_df['num_imaging_studies'].fillna(0).astype(int)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge imaging modalities
merged_df = merged_df.merge(
    modalities,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge encounter counts and duration
merged_df = merged_df.merge(
    encounter_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
).merge(
    encounter_duration,
    how='left',
    on='PATIENT'
)
merged_df['num_encounters'] = merged_df['num_encounters'].fillna(0).astype(int)
merged_df['total_encounter_duration'] = merged_df['total_encounter_duration'].fillna(0)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge encounter types
merged_df = merged_df.merge(
    encounter_types,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge vital signs
merged_df = merged_df.merge(
    vitals_pivot,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df.drop(columns=['PATIENT'], inplace=True)

# ------------------------------
# 10. Enhance Feature Engineering
# ------------------------------

# Adjust counts by age
merged_df['age_years'] = merged_df['AGE']
merged_df['num_conditions_per_year'] = merged_df['num_conditions'] / merged_df['age_years']
merged_df['num_medications_per_year'] = merged_df['num_medications'] / merged_df['age_years']
merged_df['num_encounters_per_year'] = merged_df['num_encounters'] / merged_df['age_years']

# Replace infinities with zeros (in case of division by zero)
merged_df.replace([np.inf, -np.inf], 0, inplace=True)

# Log transformation of counts
counts_cols = ['num_conditions', 'num_medications', 'num_procedures', 'num_allergies', 'num_imaging_studies', 'num_encounters']
for col in counts_cols:
    merged_df[col + '_log'] = np.log1p(merged_df[col])

# ------------------------------
# 11. Final Data Preparation
# ------------------------------

# Collect all additional columns
additional_columns = list(condition_descriptions.keys()) + list(medication_descriptions.keys()) + \
    list(procedure_descriptions.keys()) + list(allergy_descriptions.keys())

# Ensure all columns exist in merged_df
final_columns = [
    'Id', 'AGE', 'DECEASED', 'GENDER', 'RACE', 'ETHNICITY',
    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
    'num_conditions', 'total_condition_duration', 'num_medications', 'total_medication_duration',
    'num_procedures', 'total_procedure_duration', 'num_allergies', 'num_imaging_studies', 'num_encounters', 'total_encounter_duration',
    'age_years', 'num_conditions_per_year', 'num_medications_per_year', 'num_encounters_per_year'
] + counts_cols + [col + '_log' for col in counts_cols] + additional_columns + list(encounter_types.columns[1:]) + list(modalities.columns[1:]) + list(vitals_pivot.columns[1:])

final_columns = [col for col in final_columns if col in merged_df.columns]

final_df = merged_df[final_columns]

# ------------------------------
# 12. Save to CSV
# ------------------------------

# Save the final DataFrame to a CSV file
final_df.to_csv('aggregated_patient_data_improved.csv', index=False)

print("Aggregation and feature engineering complete. Data saved to 'aggregated_patient_data_improved.csv'.")

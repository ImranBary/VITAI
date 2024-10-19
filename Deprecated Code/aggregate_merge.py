import pandas as pd
import numpy as np

# ------------------------------
# 1. Load Data
# ------------------------------

# 1.1 Load Patients Data
patients = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/patients.csv', usecols=[
    'Id', 'BIRTHDATE', 'DEATHDATE', 'GENDER', 'RACE', 'ETHNICITY',
    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME'
])

# Convert BIRTHDATE and DEATHDATE to datetime
patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
patients['DEATHDATE'] = pd.to_datetime(patients['DEATHDATE'])

# Calculate Age
current_date = pd.to_datetime('today')
patients['AGE'] = (current_date - patients['BIRTHDATE']).dt.days / 365.25
patients['AGE'] = patients['AGE'].fillna(0)

# Calculate if patient is deceased
patients['DECEASED'] = patients['DEATHDATE'].notnull().astype(int)

# ------------------------------
# 2. Aggregate Conditions
# ------------------------------

# Load Conditions Data
conditions = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/conditions.csv', usecols=['PATIENT', 'DESCRIPTION'])

# Count of conditions per patient
condition_counts = conditions.groupby('PATIENT').size().reset_index(name='num_conditions')

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
medications = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/medications.csv', usecols=['PATIENT', 'DESCRIPTION'])

# Count of medications per patient
medication_counts = medications.groupby('PATIENT').size().reset_index(name='num_medications')

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
observations = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/observations.csv', usecols=[
    'PATIENT', 'DATE', 'CODE', 'DESCRIPTION', 'VALUE', 'UNITS'
])

# Convert DATE to datetime
observations['DATE'] = pd.to_datetime(observations['DATE'])

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

# Aggregate the last recorded value per patient and vital sign
vitals_sorted = vitals.sort_values(by=['PATIENT', 'DATE'])
vitals_last = vitals_sorted.groupby(['PATIENT', 'DESCRIPTION']).tail(1)

# Pivot to wide format
vitals_pivot = vitals_last.pivot(index='PATIENT', columns='DESCRIPTION', values='VALUE')
vitals_pivot.rename(columns={v: k for k, v in vital_signs_descriptions.items()}, inplace=True)
vitals_pivot.reset_index(inplace=True)

# ------------------------------
# 5. Aggregate Procedures
# ------------------------------

# Load Procedures Data
procedures = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/procedures.csv', usecols=['PATIENT', 'DESCRIPTION'])

# Count of procedures per patient
procedure_counts = procedures.groupby('PATIENT').size().reset_index(name='num_procedures')

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
allergies = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/allergies.csv', usecols=['PATIENT', 'DESCRIPTION'])

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
imaging_studies = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/imaging_studies.csv', usecols=[
    'PATIENT', 'DATE', 'MODALITY_CODE'
])

# Count of imaging studies per patient
imaging_counts = imaging_studies.groupby('PATIENT').size().reset_index(name='num_imaging_studies')

# Counts of imaging modalities per patient
modalities = imaging_studies.groupby(['PATIENT', 'MODALITY_CODE']).size().unstack(fill_value=0)
modalities.reset_index(inplace=True)

# Map modality codes to descriptive names (based on your data)
modality_codes = {
    'CT': 'ct_scans',
    'MR': 'mri_scans',
    'DX': 'xray_scans'
}

# Ensure modality columns are correctly named
modalities.rename(columns=modality_codes, inplace=True)

# ------------------------------
# 8. Aggregate Encounters
# ------------------------------

# Load Encounters Data
encounters = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/encounters.csv', usecols=['PATIENT', 'ENCOUNTERCLASS'])

# Count of encounters per patient
encounter_counts = encounters.groupby('PATIENT').size().reset_index(name='num_encounters')

# Counts of encounter types per patient
encounter_types = encounters.groupby(['PATIENT', 'ENCOUNTERCLASS']).size().unstack(fill_value=0)
encounter_types.reset_index(inplace=True)

# ------------------------------
# 9. Merge All Data
# ------------------------------

# Start with patients DataFrame
merged_df = patients.copy()

# Merge condition counts
merged_df = merged_df.merge(
    condition_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df['num_conditions'] = merged_df['num_conditions'].fillna(0).astype(int)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge medication counts
merged_df = merged_df.merge(
    medication_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df['num_medications'] = merged_df['num_medications'].fillna(0).astype(int)
merged_df.drop(columns=['PATIENT'], inplace=True)

# Merge procedure counts
merged_df = merged_df.merge(
    procedure_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df['num_procedures'] = merged_df['num_procedures'].fillna(0).astype(int)
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

# Merge encounter counts
merged_df = merged_df.merge(
    encounter_counts,
    how='left',
    left_on='Id',
    right_on='PATIENT'
)
merged_df['num_encounters'] = merged_df['num_encounters'].fillna(0).astype(int)
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

# Fill missing vital signs with mean values
vital_signs = list(vital_signs_descriptions.keys())
for sign in vital_signs:
    if sign in merged_df.columns:
        merged_df[sign] = merged_df[sign].fillna(merged_df[sign].mean())

# Fill missing modality counts with zeros
modality_columns = list(modality_codes.values())
for modality in modality_columns:
    if modality in merged_df.columns:
        merged_df[modality] = merged_df[modality].fillna(0).astype(int)

# Fill missing encounter types with zeros
encounter_type_columns = encounters['ENCOUNTERCLASS'].unique().tolist()
for encounter_type in encounter_type_columns:
    if encounter_type in merged_df.columns:
        merged_df[encounter_type] = merged_df[encounter_type].fillna(0).astype(int)

# ------------------------------
# 10. Include Free-Text Data (if available)
# ------------------------------

# Assuming there's a 'notes.csv' file containing free-text clinical notes
try:
    notes = pd.read_csv('/Users/ImranBary/DataGen/synthea/output/csv/notes.csv', usecols=['PATIENT', 'NOTE'])
    # Simple feature: Count of words in notes per patient
    notes['word_count'] = notes['NOTE'].str.split().str.len()
    notes_agg = notes.groupby('PATIENT')['word_count'].sum().reset_index()
    merged_df = merged_df.merge(
        notes_agg,
        how='left',
        left_on='Id',
        right_on='PATIENT'
    )
    merged_df['word_count'] = merged_df['word_count'].fillna(0).astype(int)
    merged_df.drop(columns=['PATIENT'], inplace=True)
except FileNotFoundError:
    print("No free-text data found to include.")

# ------------------------------
# 11. Final Data Preparation
# ------------------------------

# Collect all additional columns
additional_columns = list(condition_descriptions.keys()) + list(medication_descriptions.keys()) + \
    list(procedure_descriptions.keys()) + list(allergy_descriptions.keys()) + vital_signs + \
    modality_columns + encounter_type_columns

# Ensure all columns exist in merged_df
final_columns = [
    'Id', 'AGE', 'DECEASED', 'GENDER', 'RACE', 'ETHNICITY',
    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
    'num_conditions', 'num_medications', 'num_procedures', 'num_allergies',
    'num_imaging_studies', 'num_encounters'
] + additional_columns

final_columns = [col for col in final_columns if col in merged_df.columns]

final_df = merged_df[final_columns]

# Verify the columns now contain expected values
# Uncomment the following lines to check the sums of binary flag columns
# print("Sum of binary flag columns:")
# binary_columns = list(condition_descriptions.keys()) + list(medication_descriptions.keys()) + \
#     list(procedure_descriptions.keys()) + list(allergy_descriptions.keys())
# print(final_df[binary_columns].sum())

# ------------------------------
# 12. Save to CSV
# ------------------------------

# Save the final DataFrame to a CSV file
final_df.to_csv('aggregated_patient_data.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Load the combined data
combined_df = pd.read_csv('combined_data.csv')

# Define the columns
structured_cols = [
    'AGE', 'DECEASED', 'GENDER', 'RACE', 'ETHNICITY',
    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
    'num_conditions', 'num_medications', 'num_procedures',
    'num_allergies', 'num_imaging_studies', 'num_encounters',
    'diabetes', 'hypertension', 'asthma', 'metformin',
    'lisinopril', 'albuterol', 'appendectomy', 'blood_transfusion',
    'penicillin_allergy', 'peanut_allergy', 'weight', 'height',
    'diastolic_bp', 'systolic_bp', 'body_temperature', 'heart_rate',
    'respiratory_rate', 'ct_scans', 'xray_scans', 'ambulatory',
    'wellness', 'outpatient', 'emergency', 'inpatient', 'urgentcare',
    'virtual', 'hospice', 'snf', 'home'
]

# Exclude 'Id' from features
# Explicitly create a copy to avoid SettingWithCopyWarning
structured_data = combined_df.loc[:, structured_cols].copy()

# Handle missing values
# For numerical columns, use mean imputation
num_cols = structured_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
imputer = SimpleImputer(strategy='mean')
structured_data.loc[:, num_cols] = imputer.fit_transform(structured_data[num_cols])

# For categorical columns, use most frequent value
cat_cols = ['GENDER', 'RACE', 'ETHNICITY']
imputer_cat = SimpleImputer(strategy='most_frequent')
structured_data.loc[:, cat_cols] = imputer_cat.fit_transform(structured_data[cat_cols])

# One-Hot Encode categorical variables
structured_data = pd.get_dummies(structured_data, columns=cat_cols)

# **Convert numerical columns to float64**
structured_data[num_cols] = structured_data[num_cols].astype('float64')

# Normalize numerical features to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
structured_data.loc[:, num_cols] = scaler.fit_transform(structured_data[num_cols])

# Save the preprocessed structured data
structured_data.to_csv('structured_data_preprocessed.csv', index=False)






'''
Depricated code. 




import pandas as pd
import numpy as np
#Adjust scaling for using sigmoid activation function [0,1]
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
#joblib is used to save the label encoders
import joblib

# Load the combined data
combined_df = pd.read_csv('combined_data.csv')

# Separate structured data
structured_cols = [
    'AGE', 'DECEASED', 'GENDER', 'RACE', 'ETHNICITY',
    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
    'num_conditions', 'num_medications', 'num_procedures',
    'num_allergies', 'num_imaging_studies', 'num_encounters',
    'diabetes', 'hypertension', 'asthma', 'metformin',
    'lisinopril', 'albuterol', 'appendectomy', 'blood_transfusion',
    'penicillin_allergy', 'peanut_allergy', 'weight', 'height',
    'diastolic_bp', 'systolic_bp', 'body_temperature', 'heart_rate',
    'respiratory_rate', 'ct_scans', 'xray_scans', 'ambulatory',
    'wellness', 'outpatient', 'emergency', 'inpatient', 'urgentcare',
    'virtual', 'hospice', 'snf', 'home'
]
structured_data = combined_df[structured_cols]

# Handle missing values
# For numerical columns, use mean imputation
num_cols = structured_data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
structured_data[num_cols] = imputer.fit_transform(structured_data[num_cols])

# For categorical columns, use most frequent value
cat_cols = ['GENDER', 'RACE', 'ETHNICITY']
imputer_cat = SimpleImputer(strategy='most_frequent')
structured_data[cat_cols] = imputer_cat.fit_transform(structured_data[cat_cols])

# Encode categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    structured_data[col] = le.fit_transform(structured_data[col])
    label_encoders[col] = le  # Save encoder for future use

# Normalize numerical features to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
structured_data[num_cols] = scaler.fit_transform(structured_data[num_cols])

# Save the preprocessed structured data
structured_data.to_csv('structured_data_preprocessed.csv', index=False)

# Save the scaler
joblib.dump(scaler, 'min_max_scaler.pkl')

# Save label encoders
for col, le in label_encoders.items():
    joblib.dump(le, f'label_encoder_{col}.pkl')
'''
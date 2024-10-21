# preprocessing_script.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  # Using SimpleImputer for speed
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# ------------------------------
# 1. Load Aggregated Data
# ------------------------------

# Load the improved aggregated data
combined_df = pd.read_csv('aggregated_patient_data_improved.csv')

# Exclude 'Id' from features
structured_data = combined_df.drop(columns=['Id']).copy()

# ------------------------------
# 2. Handle Date Columns
# ------------------------------

# Identify columns containing 'last_date'
date_cols = [col for col in structured_data.columns if 'last_date' in col]

# Option 1: Drop the date columns
# structured_data.drop(columns=date_cols, inplace=True)

# Option 2: Convert date columns to numeric format (e.g., days since a reference date)
# For this example, we'll convert dates to days since '1970-01-01'

for col in date_cols:
    # Convert to datetime
    structured_data[col] = pd.to_datetime(structured_data[col], errors='coerce')
    # Replace NaT with the current date to avoid NaNs
    structured_data[col] = structured_data[col].fillna(pd.to_datetime('today'))
    # Convert to days since 1970-01-01
    structured_data[col] = (structured_data[col] - pd.Timestamp('1970-01-01')).dt.total_seconds() / (24 * 3600)

# ------------------------------
# 3. Handle Missing Values
# ------------------------------

# Identify numerical columns
num_cols = structured_data.select_dtypes(include=[np.number]).columns.tolist()

# Remove columns with more than 50% missing values
threshold = len(structured_data) * 0.5
cols_to_drop = [col for col in num_cols if structured_data[col].isnull().sum() > threshold]
structured_data.drop(columns=cols_to_drop, inplace=True)
num_cols = [col for col in num_cols if col not in cols_to_drop]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(structured_data[num_cols])

# Convert back to DataFrame with proper columns and index
X_imputed_df = pd.DataFrame(X_imputed, columns=num_cols, index=structured_data.index)

# Assign back to structured_data
structured_data[num_cols] = X_imputed_df

# ------------------------------
# 4. Encode Categorical Variables
# ------------------------------

# Identify categorical columns
cat_cols = ['GENDER', 'RACE', 'ETHNICITY']

# Fill missing categorical values with 'Unknown'
structured_data[cat_cols] = structured_data[cat_cols].fillna('Unknown')

# Frequency Encoding for categorical variables
for col in cat_cols:
    freq_encoding = structured_data[col].value_counts(normalize=True)
    structured_data[col + '_freq_enc'] = structured_data[col].map(freq_encoding)

# Drop original categorical columns
structured_data.drop(columns=cat_cols, inplace=True)

# ------------------------------
# 5. Ensure All Features Are Numeric
# ------------------------------

# Identify non-numeric columns
non_numeric_cols = structured_data.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric_cols:
    print(f"Non-numeric columns found and will be dropped: {non_numeric_cols}")
    structured_data.drop(columns=non_numeric_cols, inplace=True)

# Update num_cols to include new numeric features
num_cols = structured_data.columns.tolist()

# ------------------------------
# 6. Feature Scaling
# ------------------------------

# Initialize scaler
scaler = StandardScaler()

# Scale features
structured_data[num_cols] = scaler.fit_transform(structured_data[num_cols])

# ------------------------------
# 7. Feature Selection
# ------------------------------

# Remove low variance features
selector = VarianceThreshold(threshold=0.01)
selector.fit(structured_data)
structured_data = structured_data.loc[:, selector.get_support()]

# Remove highly correlated features
corr_matrix = structured_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper.columns if any(upper[column] > 0.9)]
structured_data.drop(columns=high_corr_features, inplace=True)

# ------------------------------
# 8. Save Preprocessed Data
# ------------------------------

# Save the preprocessed structured data
structured_data.to_csv('structured_data_preprocessed.csv', index=False)

print("Preprocessing complete. Data saved to 'structured_data_preprocessed.csv'.")

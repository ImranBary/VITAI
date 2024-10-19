# preprocessing_script.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
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
# 2. Handle Missing Values
# ------------------------------

# Identify numerical columns
num_cols = structured_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Initialize KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)

# Impute missing values
structured_data[num_cols] = knn_imputer.fit_transform(structured_data[num_cols])

# Add missingness indicators (if desired)
# for col in num_cols:
#     structured_data[col + '_missing'] = structured_data[col].isnull().astype(int)

# ------------------------------
# 3. Encode Categorical Variables
# ------------------------------

# Identify categorical columns
cat_cols = ['GENDER', 'RACE', 'ETHNICITY']

# Fill missing categorical values with 'Unknown'
structured_data[cat_cols] = structured_data[cat_cols].fillna('Unknown')

# Reconsider encoding strategy
# Frequency Encoding for categorical variables
for col in cat_cols:
    freq_encoding = structured_data[col].value_counts(normalize=True)
    structured_data[col + '_freq_enc'] = structured_data[col].map(freq_encoding)

# Drop original categorical columns
structured_data.drop(columns=cat_cols, inplace=True)

# ------------------------------
# 4. Feature Scaling
# ------------------------------

# Define features to scale (excluding frequency encoded columns)
features_to_scale = num_cols + [col for col in structured_data.columns if '_freq_enc' in col]

# Initialize scaler
scaler = StandardScaler()

# Scale features
structured_data[features_to_scale] = scaler.fit_transform(structured_data[features_to_scale])

# ------------------------------
# 5. Feature Selection
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
# 6. Save Preprocessed Data
# ------------------------------

# Save the preprocessed structured data
structured_data.to_csv('structured_data_preprocessed.csv', index=False)

print("Preprocessing complete. Data saved to 'structured_data_preprocessed.csv'.")

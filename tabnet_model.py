'''
tabnet_model.py
Author: Imran Feisal
Date: 31/10/2024
Description: This script builds and trains a TabNet model
using the composite health index, and save the trained model and results
'''

import numpy as np
import pandas as pd
import joblib
import os
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load Data
# ------------------------------

def load_data(output_dir):
    """Load patient data with health index."""
    patient_data = pd.read_pickle(os.path.join(output_dir, 'patient_data_with_health_index.pkl'))
    return patient_data

# ------------------------------
# 2. Prepare Data for TabNet
# ------------------------------

def prepare_data(patient_data):
    """Prepare data for training the TabNet model."""
    # Extract features and target
    features = patient_data[['AGE', 'DECEASED', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME']]
    features = pd.get_dummies(features, columns=['DECEASED'])
    features.fillna(0, inplace=True)
    target = patient_data['Health_Index']

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Save the scaler
    joblib.dump(scaler, 'scaler_tabnet.joblib')

    return X, target.values

# ------------------------------
# 3. Train TabNet Model
# ------------------------------

def train_tabnet(X, y):
    """Train the TabNet model."""
    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize TabNetRegressor
    regressor = TabNetRegressor(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        mask_type='entmax',
        verbose=10,
    )

    # Train the model
    regressor.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    # Save the model
    regressor.save_model('tabnet_model')

# ------------------------------
# 4. Save Predictions
# ------------------------------

def save_predictions(regressor, X, patient_ids):
    """Save the predictions from the TabNet model."""
    predictions = regressor.predict(X)
    predictions_df = pd.DataFrame({'Id': patient_ids, 'Predicted_Health_Index': predictions})
    predictions_df.to_csv('tabnet_predictions.csv', index=False)
    print("TabNet predictions saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    output_dir = 'Data'

    # Load data
    patient_data = load_data(output_dir)
    patient_ids = patient_data['Id'].values

    # Prepare data
    X, y = prepare_data(patient_data)

    # Train TabNet model
    train_tabnet(X, y)

    # Load trained model
    regressor = TabNetRegressor()
    regressor.load_model('tabnet_model.zip')

    # Save predictions
    save_predictions(regressor, X, patient_ids)

if __name__ == '__main__':
    main()

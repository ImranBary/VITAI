# tabnet_model.py
# Author: Imran Feisal
# Date: 31/10/2024
# Description:
# This script builds and trains a TabNet model using hyperparameter tuning,
# includes cross-validation, extracts feature importances, and saves the
# trained model and results. 
# Now accepts an output_prefix param to avoid overwriting artifacts,
# and target_col param to decide which column to predict (Health_Index or CharlsonIndex).

import numpy as np
import pandas as pd
import joblib
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import logging
import json
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(output_dir, input_file):
    data_path = os.path.join(output_dir, input_file)
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")
    patient_data = pd.read_pickle(data_path)
    logger.info("Patient data loaded.")
    return patient_data

def prepare_data(patient_data, target_col='Health_Index'):
    """
    Prepare the dataset for TabNet:
      - features: columns that define the model inputs
      - target: the column we want to predict (Health_Index or CharlsonIndex)
    """
    # Feature columns
    features = patient_data[[
        'AGE','DECEASED','GENDER','RACE','ETHNICITY','MARITAL',
        'HEALTHCARE_EXPENSES','HEALTHCARE_COVERAGE','INCOME',
        'Hospitalizations_Count','Medications_Count','Abnormal_Observations_Count'
    ]].copy()

    # Target column is chosen based on `target_col`
    if target_col not in patient_data.columns:
        raise KeyError(f"Column '{target_col}' not found in patient_data!")
    target = patient_data[target_col]

    # Setup categorical columns
    categorical_columns = ['DECEASED','GENDER','RACE','ETHNICITY','MARITAL']
    cat_idxs = [i for i,col in enumerate(features.columns) if col in categorical_columns]
    cat_dims = []

    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        cat_dims.append(features[col].nunique())

    # Scale continuous columns
    continuous_columns = [col for col in features.columns if col not in categorical_columns]
    scaler = StandardScaler()
    features[continuous_columns] = scaler.fit_transform(features[continuous_columns])
    joblib.dump(scaler, 'tabnet_scaler.joblib')

    # Handle missing
    features.fillna(0, inplace=True)

    X = features.values
    y = target.values.reshape(-1, 1)
    logger.info(f"Data prepared for TabNet (target_col='{target_col}').")

    return X, y, cat_idxs, cat_dims, features.columns.tolist()

def objective(trial, X, y, cat_idxs, cat_dims):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=trial.suggest_float('lr',1e-4,1e-2,log=True)),
        'cat_emb_dim': trial.suggest_int('cat_emb_dim',1,5),
        'n_shared': trial.suggest_int('n_shared',1,5),
        'n_independent': trial.suggest_int('n_independent',1,5),
        'device_name': 'cuda',
        'verbose': 0,
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mse_list = []

    for train_idx, valid_idx in kf.split(X):
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        model = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, **params)
        model.fit(
            X_train=X_train_fold, y_train=y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            eval_metric=['rmse'],
            max_epochs=50,
            patience=10,
            batch_size=4096,
            virtual_batch_size=512
        )
        preds = model.predict(X_valid_fold)
        mse = mean_squared_error(y_valid_fold, preds)
        mse_list.append(mse)
    return np.mean(mse_list)

def hyperparameter_tuning(X, y, cat_idxs, cat_dims):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y, cat_idxs, cat_dims), n_trials=10)
    logger.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params

def train_tabnet(X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims, best_params, output_prefix='tabnet'):
    optimizer_fn = torch.optim.Adam
    optimizer_params = {'lr': best_params.pop('lr')}
    best_params.update({
        'optimizer_fn': optimizer_fn,
        'optimizer_params': optimizer_params,
        'device_name': 'cuda',
        'verbose': 1
    })
    regressor = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, **best_params)
    regressor.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=8192,
        virtual_batch_size=1024
    )
    regressor.save_model(f'{output_prefix}_model')
    logger.info(f"TabNet model trained and saved -> {output_prefix}_model.zip (among others).")
    return regressor

def main(input_file='patient_data_with_health_index.pkl',
         output_prefix='tabnet',
         target_col='Health_Index'):
    """
    Args:
        input_file (str): Pickle file containing patient data
        output_prefix (str): Unique prefix to avoid overwriting model artifacts
        target_col (str): Which column to predict ('Health_Index' or 'CharlsonIndex')
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Data')
    patient_data = load_data(output_dir, input_file)

    # Prepare data for the specified target column
    X, y, cat_idxs, cat_dims, feature_columns = prepare_data(patient_data, target_col=target_col)

    # Train/valid/test splits
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train, y_train, cat_idxs, cat_dims)
    regressor = train_tabnet(X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims, best_params, output_prefix=output_prefix)

    # Evaluate on test set
    test_preds = regressor.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test R2: {test_r2:.4f}")

    # Save predictions
    # If 'Id' is present in the DF, we can map it; otherwise we do a simple index-based approach
    num_test = len(X_test)
    pred_col_name = ("Predicted_Health_Index" if target_col == "Health_Index" 
                     else "Predicted_CharlsonIndex")

    if 'Id' in patient_data.columns:
        # Take the last 'num_test' rows as test IDs
        test_ids = patient_data.iloc[-num_test:]['Id'].values
    else:
        # Fallback if not present
        test_ids = np.arange(num_test)

    predictions_df = pd.DataFrame({
        'Id': test_ids,
        pred_col_name: test_preds.flatten()
    })
    pred_csv = f'{output_prefix}_predictions.csv'
    predictions_df.to_csv(pred_csv, index=False)
    logger.info(f"TabNet predictions saved -> {pred_csv}")

    # Save metrics
    metrics = {
        "test_mse": test_mse,
        "test_r2": test_r2
    }
    metrics_file = f"{output_prefix}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
    logger.info(f"TabNet metrics saved -> {metrics_file}")

if __name__ == '__main__':
    main()

'''
import numpy as np
import pandas as pd
import joblib
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Load Data
# ------------------------------

def load_data(output_dir):
    data_path = os.path.join(output_dir, 'patient_data_with_health_index.pkl')
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")
    patient_data = pd.read_pickle(data_path)
    logger.info("Patient data loaded successfully.")
    return patient_data

# ------------------------------
# 2. Prepare Data for TabNet
# ------------------------------

def prepare_data(patient_data):
    features = patient_data[['AGE', 'DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL',
                             'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
                             'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']].copy()
    target = patient_data['Health_Index']

    categorical_columns = ['DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    cat_idxs = [i for i, col in enumerate(features.columns) if col in categorical_columns]
    cat_dims = []

    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        cat_dims.append(features[col].nunique())

    continuous_columns = [col for col in features.columns if col not in categorical_columns]
    scaler = StandardScaler()
    features[continuous_columns] = scaler.fit_transform(features[continuous_columns])
    joblib.dump(scaler, 'tabnet_scaler.joblib')

    features.fillna(0, inplace=True)
    X = features.values
    y = target.values.reshape(-1, 1)

    logger.info("Data prepared for TabNet model.")
    return X, y, cat_idxs, cat_dims, features.columns.tolist()

# ------------------------------
# 3. Hyperparameter Tuning with Optuna
# ------------------------------

def objective(trial, X, y, cat_idxs, cat_dims):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True)),
        'cat_emb_dim': trial.suggest_int('cat_emb_dim', 1, 5),
        'n_shared': trial.suggest_int('n_shared', 1, 5),
        'n_independent': trial.suggest_int('n_independent', 1, 5),
        'device_name': 'cuda',
        'verbose': 0,
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mse_list = []

    for train_index, valid_index in kf.split(X):
        X_train_fold, X_valid_fold = X[train_index], X[valid_index]
        y_train_fold, y_valid_fold = y[train_index], y[valid_index]

        model = TabNetRegressor(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            **params
        )

        model.fit(
            X_train=X_train_fold, y_train=y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            eval_metric=['rmse'],
            max_epochs=50,  # Reduce epochs during tuning for faster results
            patience=10,
            batch_size=4096,
            virtual_batch_size=512
        )

        preds = model.predict(X_valid_fold)
        mse = mean_squared_error(y_valid_fold, preds)
        mse_list.append(mse)

    return np.mean(mse_list)

def hyperparameter_tuning(X, y, cat_idxs, cat_dims):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y, cat_idxs, cat_dims), n_trials=20)  # Reduce trials for faster results
    logger.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params

# ------------------------------
# 4. Train TabNet Model
# ------------------------------

def train_tabnet(X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims, best_params):
    optimizer_fn = torch.optim.Adam
    optimizer_params = {'lr': best_params.pop('lr')}
    best_params.update({
        'optimizer_fn': optimizer_fn,
        'optimizer_params': optimizer_params,
        'device_name': 'cuda',
        'verbose': 1,
    })

    regressor = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        **best_params
    )

    regressor.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=8192,  # Larger batch size for GPU
        virtual_batch_size=1024
    )

    regressor.save_model('tabnet_model')
    logger.info("TabNet model trained and saved successfully.")
    return regressor

# ------------------------------
# 5. Main Execution
# ------------------------------

def main():
    output_dir = 'Data'
    patient_data = load_data(output_dir)
    X, y, cat_idxs, cat_dims, feature_columns = prepare_data(patient_data)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    best_params = hyperparameter_tuning(X_train, y_train, cat_idxs, cat_dims)
    regressor = train_tabnet(X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims, best_params)

    test_preds = regressor.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test R2 Score: {test_r2:.4f}")

if __name__ == '__main__':
    main()

'''
'''
# tabnet_model.py
# Author: Imran Feisal
# Date: 31/10/2024
# Description:
# This script builds and trains a TabNet model using hyperparameter tuning,
# includes cross-validation, extracts feature importances, and saves the trained model and results.

import numpy as np
import pandas as pd
import joblib
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import optuna
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Load Data
# ------------------------------

def load_data(output_dir):
    """Load patient data with health index."""
    data_path = os.path.join(output_dir, 'patient_data_with_health_index.pkl')
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")
    patient_data = pd.read_pickle(data_path)
    logger.info("Patient data loaded successfully.")
    return patient_data

# ------------------------------
# 2. Prepare Data for TabNet
# ------------------------------

def prepare_data(patient_data):
    """Prepare data for training the TabNet model."""
    # Extract features and target without the Charlson Comorbidity Score
    features = patient_data[['AGE', 'DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL',
                             'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
                             'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']]
    target = patient_data['Health_Index']

    # Identify categorical columns
    categorical_columns = ['DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    cat_idxs = [i for i, col in enumerate(features.columns) if col in categorical_columns]
    
    # Label encode categorical columns and update cat_dims
    cat_dims = []
    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        cat_dims.append(features[col].nunique())

    # Fill missing values
    features.fillna(0, inplace=True)
    
    # Convert DataFrame to numpy array
    X = features.values
    y = target.values.reshape(-1, 1)  # Reshape target to 2D array

    logger.info("Data prepared for TabNet model.")
    return X, y, cat_idxs, cat_dims

# ------------------------------
# 3. Hyperparameter Tuning with Optuna
# ------------------------------

def objective(trial, X, y, cat_idxs, cat_dims):
    """Objective function for hyperparameter tuning with Optuna."""
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True)),
        'cat_emb_dim': trial.suggest_int('cat_emb_dim', 1, 5),
        'n_shared': trial.suggest_int('n_shared', 1, 5),
        'n_independent': trial.suggest_int('n_independent', 1, 5),
    }

    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mse_list = []

    for train_index, valid_index in kf.split(X):
        X_train_fold, X_valid_fold = X[train_index], X[valid_index]
        y_train_fold, y_valid_fold = y[train_index], y[valid_index]

        model = TabNetRegressor(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            **params
        )

        model.fit(
            X_train=X_train_fold, y_train=y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            eval_metric=['rmse'],
            max_epochs=100,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        preds = model.predict(X_valid_fold)
        mse = mean_squared_error(y_valid_fold, preds)
        mse_list.append(mse)

    return np.mean(mse_list)

def hyperparameter_tuning(X, y, cat_idxs, cat_dims):
    """Perform hyperparameter tuning using Optuna."""
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y, cat_idxs, cat_dims), n_trials=20)

    logger.info(f"Best trial: {study.best_trial.params}")

    return study.best_trial.params

# ------------------------------
# 4. Train TabNet Model with Best Hyperparameters
# ------------------------------

def train_tabnet(X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims, best_params):
    """Train the TabNet model with the best hyperparameters."""
    # Extract 'lr' from best_params and create 'optimizer_params'
    optimizer_params = {'lr': best_params.pop('lr')}
    
    # Now pass optimizer_params to TabNetRegressor
    regressor = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_params=optimizer_params,
        **best_params
    )

    regressor.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    # Save the model
    regressor.save_model('tabnet_model')
    logger.info("TabNet model trained and saved successfully.")

    return regressor


# ------------------------------
# 5. Evaluate Model
# ------------------------------

def evaluate_model(regressor, X_valid, y_valid):
    """Evaluate the TabNet model on the validation set."""
    predictions = regressor.predict(X_valid)
    mse = mean_squared_error(y_valid, predictions)
    mae = mean_absolute_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)
    logger.info(f"Validation MSE: {mse:.4f}")
    logger.info(f"Validation MAE: {mae:.4f}")
    logger.info(f"Validation R2 Score: {r2:.4f}")

    # Feature Importances
    feature_importances = regressor.feature_importances_
    logger.info("Feature importances extracted.")
    return feature_importances

# ------------------------------
# 6. Save Predictions and Feature Importances
# ------------------------------

def save_results(regressor, X, patient_ids, feature_columns, feature_importances):
    """Save the predictions and feature importances."""
    predictions = regressor.predict(X)
    predictions = predictions.flatten()  # Flatten predictions to 1D array
    predictions_df = pd.DataFrame({'Id': patient_ids, 'Predicted_Health_Index': predictions})
    predictions_df.to_csv('tabnet_predictions.csv', index=False)
    logger.info("TabNet predictions saved.")

    # Save feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    })
    feature_importances_df.to_csv('tabnet_feature_importances.csv', index=False)
    logger.info("Feature importances saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    output_dir = 'Data'

    # Load data
    patient_data = load_data(output_dir)
    patient_ids = patient_data['Id'].values

    # Prepare data
    X, y, cat_idxs, cat_dims = prepare_data(patient_data)
    feature_columns = [
        'AGE', 'DECEASED', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL',
        'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
        'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count'
    ]

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X, y, cat_idxs, cat_dims)

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train TabNet model with best hyperparameters
    regressor = train_tabnet(X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims, best_params)

    # Evaluate model
    feature_importances = evaluate_model(regressor, X_valid, y_valid)

    # Save predictions and feature importances
    save_results(regressor, X, patient_ids, feature_columns, feature_importances)

if __name__ == '__main__':
    main()
    '''
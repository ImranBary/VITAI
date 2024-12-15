# vae_model.py
# Author: Imran Feisal
# Date: 31/10/2024
# Description:
# Updated script to address variable duplication warnings in TensorFlow.

import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Load Data
# ------------------------------

def load_data(output_dir):
    """
    Load patient data with health index.
    """
    patient_data = pd.read_pickle(os.path.join(output_dir, 'patient_data_with_health_index.pkl'))
    return patient_data

# ------------------------------
# 2. Prepare Data for VAE
# ------------------------------

def prepare_data(patient_data):
    """
    Prepare data for training the VAE, using embedding layers for categorical variables.
    """
    # Extract features without the Health Index
    features = patient_data[['AGE', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL',
                             'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
                             'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']].copy()

    patient_ids = patient_data['Id'].values

    # Identify categorical variables
    categorical_features = ['GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    continuous_features = [col for col in features.columns if col not in categorical_features]

    # Initialize embedding info
    embedding_info = {}
    input_data = {}

    # Process categorical features
    for col in categorical_features:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        # Save the LabelEncoder
        joblib.dump(le, f'label_encoder_{col}.joblib')

        vocab_size = features[col].nunique()
        embedding_dim = min(50, (vocab_size + 1) // 2)

        embedding_info[col] = {'vocab_size': vocab_size, 'embedding_dim': embedding_dim}
        input_data[col] = features[col].values

    # Process continuous features
    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(features[continuous_features])
    joblib.dump(scaler, 'scaler_vae.joblib')
    input_data['continuous'] = scaled_continuous

    logger.info("Data prepared for VAE model.")

    return input_data, embedding_info, patient_ids, continuous_features, categorical_features

# ------------------------------
# 3. Build VAE Model
# ------------------------------

def build_vae(input_dim, embedding_info, continuous_dim, latent_dim=20):
    """
    Build the VAE model without variable duplication.
    """
    # Inputs
    inputs = {}
    encoded_features = []

    # Categorical inputs and embeddings
    for col, info in embedding_info.items():
        input_cat = keras.Input(shape=(1,), name=f'input_{col}')
        embedding_layer = layers.Embedding(
            input_dim=info['vocab_size'],
            output_dim=info['embedding_dim'],
            name=f'embedding_{col}'
        )(input_cat)
        flat_embedding = layers.Flatten()(embedding_layer)
        inputs[f'input_{col}'] = input_cat
        encoded_features.append(flat_embedding)

    # Continuous inputs
    input_cont = keras.Input(shape=(continuous_dim,), name='input_continuous')
    inputs['input_continuous'] = input_cont
    encoded_features.append(input_cont)

    # Concatenate features
    concatenated_features = layers.concatenate(encoded_features)

    # Encoder
    h = layers.Dense(256, activation='relu')(concatenated_features)
    h = layers.Dense(128, activation='relu')(h)
    z_mean = layers.Dense(latent_dim, name='z_mean')(h)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)

    # Sampling Layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    h_decoder = layers.Dense(128, activation='relu')(decoder_inputs)
    h_decoder = layers.Dense(256, activation='relu')(h_decoder)
    reconstructed = layers.Dense(input_dim, activation='linear')(h_decoder)

    # Define encoder and decoder models
    encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder')
    decoder = keras.Model(inputs=decoder_inputs, outputs=reconstructed, name='decoder')

    # VAE Model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs=inputs, outputs=outputs, name='vae')

    # Loss
    reconstruction_loss = tf.reduce_mean(tf.square(concatenated_features - outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    logger.info("VAE model built successfully.")

    return vae, encoder, decoder

# ------------------------------
# 4. Train VAE Model
# ------------------------------

def train_vae(vae, input_data):
    """
    Train the VAE model.
    """
    # Prepare input data for training
    x_train = {f'input_{key}': value for key, value in input_data.items() if key != 'continuous'}
    x_train['input_continuous'] = input_data['continuous']

    # Early stopping and model checkpoint
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint('vae_best_model.h5', monitor='val_loss', save_best_only=True)

    # Since the VAE reconstructs the inputs, we don't need to provide targets
    vae.fit(
        x=x_train,
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint]
    )
    vae.save('vae_model', save_format='tf')

    logger.info("VAE model trained and saved successfully.")

# ------------------------------
# 5. Save Latent Features
# ------------------------------

def save_latent_features(encoder, input_data, patient_ids):
    """
    Save the latent features from the encoder.
    """
    # Prepare input data for prediction
    x_pred = {f'input_{key}': value for key, value in input_data.items() if key != 'continuous'}
    x_pred['input_continuous'] = input_data['continuous']

    z_mean, _, _ = encoder.predict(x_pred)
    latent_features_df = pd.DataFrame(z_mean)
    latent_features_df['Id'] = patient_ids
    latent_features_df.to_csv('latent_features_vae.csv', index=False)
    logger.info("Latent features saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    output_dir = 'Data'

    # Load data
    patient_data = load_data(output_dir)

    # Prepare data
    input_data, embedding_info, patient_ids, continuous_features, categorical_features = prepare_data(patient_data)

    # Build VAE model
    input_dim = sum(info['embedding_dim'] for info in embedding_info.values()) + len(continuous_features)
    continuous_dim = len(continuous_features)
    vae, encoder, decoder = build_vae(input_dim, embedding_info, continuous_dim)

    # Train VAE model
    train_vae(vae, input_data)

    # Save encoder and decoder separately
    encoder.save('vae_encoder', save_format='tf')
    decoder.save('vae_decoder', save_format='tf')

    # Save latent features
    save_latent_features(encoder, input_data, patient_ids)

if __name__ == '__main__':
    main()

'''
# vae_model.py
# Author: Imran Feisal
# Date: 31/10/2024
# Description:
# This script loads patient data with health index,
# enhances data preparation by including more features and embedding sequences,
# builds a more complex VAE model with hyperparameter tuning and early stopping,
# trains the model, and saves the entire model for future use.

# Inspiration:
# - Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. International Conference on Learning Representations.
# - Miotto, R., et al. (2016). Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records. Scientific Reports, 6, 26094.

import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1. Load Data
# ------------------------------

def load_data(output_dir):
    """Load patient data with health index."""
    patient_data = pd.read_pickle(os.path.join(output_dir, 'patient_data_with_health_index.pkl'))
    return patient_data

# ------------------------------
# 2. Prepare Data for VAE
# ------------------------------

def prepare_data(patient_data):
    """Prepare data for training the VAE."""
    # Extract features without the Charlson Comorbidity Score
    features = patient_data[['AGE', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL', 
                             'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
                             'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count']]

    # One-hot encode categorical variables
    categorical_features = ['GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    features = pd.get_dummies(features, columns=categorical_features)

    # Fill missing values
    features.fillna(0, inplace=True)

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Save the scaler
    joblib.dump(scaler, 'scaler_vae.joblib')

    logger.info("Data prepared for VAE model.")

    return X

# ------------------------------
# 3. Build VAE Model
# ------------------------------

def build_vae(input_dim, latent_dim=20):
    """Build the VAE model."""
    # Encoder
    inputs = keras.Input(shape=(input_dim,))
    h = layers.Dense(256, activation='relu')(inputs)
    h = layers.Dense(128, activation='relu')(h)
    z_mean = layers.Dense(latent_dim, name='z_mean')(h)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)

    # Sampling Layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    h_decoder = layers.Dense(128, activation='relu')(latent_inputs)
    h_decoder = layers.Dense(256, activation='relu')(h_decoder)
    outputs = layers.Dense(input_dim, activation='linear')(h_decoder)

    # Define models
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = keras.Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])

    # VAE Model
    vae = keras.Model(inputs, outputs, name='vae')

    # Loss
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    logger.info("VAE model built successfully.")

    return vae, encoder, decoder

# ------------------------------
# 4. Train VAE Model
# ------------------------------

def train_vae(vae, X):
    """Train the VAE model."""
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    vae.fit(X, epochs=100, batch_size=256, validation_split=0.2, callbacks=[early_stopping])
    vae.save('vae_model.h5')

    logger.info("VAE model trained and saved successfully.")

# ------------------------------
# 5. Save Latent Features
# ------------------------------

def save_latent_features(encoder, X, patient_ids):
    """Save the latent features from the encoder."""
    z_mean, _, _ = encoder.predict(X)
    latent_features_df = pd.DataFrame(z_mean)
    latent_features_df['Id'] = patient_ids
    latent_features_df.to_csv('latent_features_vae.csv', index=False)
    logger.info("Latent features saved.")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    output_dir = 'Data'

    # Load data
    patient_data = load_data(output_dir)
    patient_ids = patient_data['Id'].values

    # Prepare data
    X = prepare_data(patient_data)

    # Build VAE model
    input_dim = X.shape[1]
    vae, encoder, decoder = build_vae(input_dim)

    # Train VAE model
    train_vae(vae, X)

    # Save latent features
    save_latent_features(encoder, X, patient_ids)

if __name__ == '__main__':
    main()
'''
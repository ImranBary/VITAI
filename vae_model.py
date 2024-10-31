'''
vae_model.py
Author: Imran Feisal
Date: 31/10/2024
Description: This script loads patient data with health index,
prepares the data for training a Variational Autoencoder (VAE),
builds the VAE model, trains the model, and saves the latent features.

'''

import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

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
    # Extract features
    features = patient_data[['AGE', 'DECEASED', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'Health_Index']]
    features = pd.get_dummies(features, columns=['DECEASED'])
    features.fillna(0, inplace=True)

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Save the scaler
    joblib.dump(scaler, 'scaler_vae.joblib')

    return X

# ------------------------------
# 3. Build VAE Model
# ------------------------------

def build_vae(input_dim, latent_dim=10):
    """Build the VAE model."""
    # Encoder
    inputs = keras.Input(shape=(input_dim,))
    h = layers.Dense(128, activation='relu')(inputs)
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

    return vae, encoder, decoder

# ------------------------------
# 4. Train VAE Model
# ------------------------------

def train_vae(vae, X):
    """Train the VAE model."""
    vae.fit(X, epochs=50, batch_size=256, validation_split=0.2)
    vae.save_weights('vae_weights.h5')

# ------------------------------
# 5. Save Latent Features
# ------------------------------

def save_latent_features(encoder, X, patient_ids):
    """Save the latent features from the encoder."""
    z_mean, _, _ = encoder.predict(X)
    latent_features_df = pd.DataFrame(z_mean)
    latent_features_df['Id'] = patient_ids
    latent_features_df.to_csv('latent_features_vae.csv', index=False)
    print("Latent features saved.")

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

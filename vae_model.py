# vae_model.py
# Author: Imran Feisal
# Date: 31/10/2024
# Description:
# Updated script to address variable duplication warnings in TensorFlow.
# This script trains a VAE model, now accepts an input_file and output_prefix parameters
# so as to avoid overwriting model artifacts. 
# 
# UPDATe 19/01/2025 this script now saves a JSON file with final
# training and validation losses, named <output_prefix>_vae_metrics.json.

import numpy as np
import pandas as pd
import joblib
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(output_dir, input_file):
    patient_data = pd.read_pickle(os.path.join(output_dir, input_file))
    return patient_data

def prepare_data(patient_data):
    features = patient_data[[
        'AGE', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL',
        'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
        'Hospitalizations_Count', 'Medications_Count', 'Abnormal_Observations_Count'
    ]].copy()

    patient_ids = patient_data['Id'].values
    categorical_features = ['GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
    continuous_features = [col for col in features.columns if col not in categorical_features]

    embedding_info = {}
    input_data = {}

    for col in categorical_features:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        joblib.dump(le, f'label_encoder_{col}.joblib')
        vocab_size = features[col].nunique()
        embedding_dim = min(50, (vocab_size + 1)//2)
        embedding_info[col] = {'vocab_size': vocab_size, 'embedding_dim': embedding_dim}
        input_data[col] = features[col].values

    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(features[continuous_features])
    joblib.dump(scaler, 'scaler_vae.joblib')
    input_data['continuous'] = scaled_continuous

    logger.info("Data prepared for VAE.")
    return input_data, embedding_info, patient_ids, continuous_features, categorical_features

def build_vae(input_dim, embedding_info, continuous_dim, latent_dim=20):
    inputs = {}
    encoded_features = []

    # Embeddings for categorical
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

    # Continuous input
    input_cont = keras.Input(shape=(continuous_dim,), name='input_continuous')
    inputs['input_continuous'] = input_cont
    encoded_features.append(input_cont)

    concatenated_features = layers.concatenate(encoded_features)
    h = layers.Dense(256, activation='relu')(concatenated_features)
    h = layers.Dense(128, activation='relu')(h)
    z_mean = layers.Dense(latent_dim, name='z_mean')(h)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5*z_log_var)*epsilon

    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    h_decoder = layers.Dense(128, activation='relu')(decoder_inputs)
    h_decoder = layers.Dense(256, activation='relu')(h_decoder)
    reconstructed = layers.Dense(input_dim, activation='linear')(h_decoder)

    encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder')
    decoder = keras.Model(inputs=decoder_inputs, outputs=reconstructed, name='decoder')

    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs=inputs, outputs=outputs, name='vae')

    reconstruction_loss = tf.reduce_mean(tf.square(concatenated_features - outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    logger.info("VAE model built.")
    return vae, encoder, decoder

def train_vae(vae, input_data, output_prefix='vae'):
    x_train = {
        f'input_{key}': value for key, value in input_data.items() 
        if key != 'continuous'
    }
    x_train['input_continuous'] = input_data['continuous']

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        f'{output_prefix}_best_model.h5', 
        monitor='val_loss', 
        save_best_only=True
    )

    # Fit returns a History object with training & validation losses
    history = vae.fit(
        x_train, 
        epochs=100, 
        batch_size=512, 
        validation_split=0.2, 
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    vae.save(f'{output_prefix}_model', save_format='tf')
    logger.info(f"VAE trained and saved with prefix={output_prefix}.")

    # Extract final losses from history
    # Because of early stopping, 'val_loss' might not correspond to the final epoch
    # We take the minimal val_loss across epochs as a reference
    final_train_loss = float(history.history['loss'][-1])  # last epoch's training loss
    final_val_loss = float(min(history.history['val_loss']))  # best validation loss

    # Save them to a JSON for easier retrieval
    metrics_json = {
        "final_train_loss": final_train_loss,
        "best_val_loss": final_val_loss
    }
    with open(f"{output_prefix}_vae_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"[METRICS] VAE training/validation losses saved to {output_prefix}_vae_metrics.json")

def save_latent_features(encoder, input_data, patient_ids, output_prefix='vae'):
    x_pred = {
        f'input_{key}': value for key, value in input_data.items() 
        if key != 'continuous'
    }
    x_pred['input_continuous'] = input_data['continuous']
    z_mean, _, _ = encoder.predict(x_pred)

    df = pd.DataFrame(z_mean)
    df['Id'] = patient_ids
    csv_name = f'{output_prefix}_latent_features.csv'
    df.to_csv(csv_name, index=False)
    logger.info(f"Latent features saved to {csv_name}.")

def main(input_file='patient_data_with_health_index.pkl', output_prefix='vae'):
    """
    Args:
        input_file (str): Name of the input pickle file containing patient data.
        output_prefix (str): A unique prefix for saving model artifacts 
                             (latent CSV, model files, etc.).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Data')
    patient_data = load_data(output_dir, input_file)
    input_data, embedding_info, patient_ids, continuous_features, categorical_features = prepare_data(patient_data)

    input_dim = sum(info['embedding_dim'] for info in embedding_info.values()) + len(continuous_features)
    continuous_dim = len(continuous_features)
    vae, encoder, decoder = build_vae(input_dim, embedding_info, continuous_dim)

    train_vae(vae, input_data, output_prefix=output_prefix)
    encoder.save(f'{output_prefix}_encoder', save_format='tf')
    decoder.save(f'{output_prefix}_decoder', save_format='tf')

    save_latent_features(encoder, input_data, patient_ids, output_prefix=output_prefix)

if __name__ == '__main__':
    main()

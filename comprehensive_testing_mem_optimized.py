"""
comprehensive_testing_mem_optimized.py
Author: Imran Feisal
Date: 12/01/2025

Description:
This memory-optimized script orchestrates multiple experiments to evaluate:
 - Feature configurations: composite, cci, combined
 - Subsets: none, diabetes, ckd
 - Model approaches: vae, tabnet, hybrid

It references:
 - data_preprocessing.py   -> generating patient_data_sequences.pkl
 - health_index.py         -> computing the composite health index
 - charlson_comorbidity.py -> integrating Charlson Comorbidity Index (CCI)
 - vae_model.py            -> training VAE & saving latent features
 - tabnet_model.py         -> training TabNet & saving predictions

Memory-Saving Approaches:
 - Skips hierarchical clustering entirely (AgglomerativeClustering).
 - Uses the entire dataset (no subsampling) for DBSCAN, t-SNE, UMAP, etc.
 - Employs joblib to dump intermediate DataFrames to disk, free memory, and reload as needed.

Usage:
  python comprehensive_testing_mem_optimized.py
"""

import os
import sys
import gc
import datetime
import logging
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump, load

# Clustering & metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler

# Dimensionality reduction
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Local imports
from data_preprocessing import main as preprocess_main
from health_index import main as health_main
from charlson_comorbidity import load_cci_mapping, compute_cci
from vae_model import main as vae_main
from tabnet_model import main as tabnet_main

# Basic config
OUTPUT_RESULTS_CSV = "comprehensive_experiments_results.csv"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


###############################################################################
# 1. Ensure Data Preprocessing & CCI
###############################################################################
def ensure_preprocessed_data(data_dir):
    """
    Ensures we have patient_data_sequences.pkl,
    patient_data_with_health_index.pkl, and
    patient_data_with_health_index_cci.pkl.
    If missing, calls relevant scripts.
    """
    pkl_sequences = os.path.join(data_dir, "patient_data_sequences.pkl")
    pkl_health_index = os.path.join(data_dir, "patient_data_with_health_index.pkl")
    pkl_health_cci = os.path.join(data_dir, "patient_data_with_health_index_cci.pkl")

    # data_preprocessing step
    if not os.path.exists(pkl_sequences):
        logger.info("Missing patient_data_sequences.pkl -> Running data_preprocessing.py")
        preprocess_main()
    else:
        logger.info("Preprocessed sequences found.")

    # health_index step
    if not os.path.exists(pkl_health_index):
        logger.info("Missing patient_data_with_health_index.pkl -> Running health_index.py")
        health_main()
    else:
        logger.info("Health index data found.")

    # CCI step
    if not os.path.exists(pkl_health_cci):
        logger.info("Missing patient_data_with_health_index_cci.pkl -> merging CCI.")
        conditions_csv = os.path.join(data_dir, "conditions.csv")
        if not os.path.exists(conditions_csv):
            raise FileNotFoundError("conditions.csv not found. Cannot compute CCI.")

        conditions = pd.read_csv(conditions_csv, usecols=["PATIENT","CODE","DESCRIPTION"])
        cci_map = load_cci_mapping(data_dir)
        patient_cci = compute_cci(conditions, cci_map)

        patient_data = pd.read_pickle(pkl_health_index)
        merged = patient_data.merge(
            patient_cci, how="left", left_on="Id", right_on="PATIENT"
        )
        merged.drop(columns="PATIENT", inplace=True)
        merged["CharlsonIndex"] = merged["CharlsonIndex"].fillna(0.0)
        merged.to_pickle(pkl_health_cci)
        logger.info("[INFO] CCI merged & saved -> %s", pkl_health_cci)
        del conditions, cci_map, patient_cci, patient_data, merged
        gc.collect()
    else:
        logger.info("CCI data found.")


###############################################################################
# 2. Subset Filtering Logic
###############################################################################
def load_conditions(data_dir):
    cpath = os.path.join(data_dir, "conditions.csv")
    if not os.path.exists(cpath):
        raise FileNotFoundError(f"conditions.csv not found at {cpath}.")
    return pd.read_csv(cpath, usecols=['PATIENT','CODE','DESCRIPTION'])

def subset_diabetes(patient_data, data_dir):
    conditions = load_conditions(data_dir)
    mask = conditions['DESCRIPTION'].str.lower().str.contains('diabetes', na=False)
    diabetes_patients = conditions.loc[mask, 'PATIENT'].unique()
    sub = patient_data[patient_data['Id'].isin(diabetes_patients)].copy()
    logger.info(f"[INFO] Diabetes subset shape: {sub.shape}")
    del conditions
    gc.collect()
    return sub

def subset_ckd(patient_data, data_dir):
    conditions = load_conditions(data_dir)
    ckd_snomed = {431855005, 431856006, 433144002, 431857002, 46177005}
    code_mask = conditions['CODE'].isin(ckd_snomed)
    text_mask = conditions['DESCRIPTION'].str.lower().str.contains('chronic kidney disease', na=False)
    ckd_mask = code_mask | text_mask
    ckd_patients = conditions.loc[ckd_mask, 'PATIENT'].unique()
    sub = patient_data[patient_data['Id'].isin(ckd_patients)].copy()
    logger.info(f"[INFO] CKD subset shape: {sub.shape}")
    del conditions
    gc.collect()
    return sub

def filter_subpopulation(patient_data, subset_type, data_dir):
    if subset_type.lower() == "none":
        return patient_data
    elif subset_type.lower() == "diabetes":
        return subset_diabetes(patient_data, data_dir)
    elif subset_type.lower() == "ckd":
        return subset_ckd(patient_data, data_dir)
    else:
        logger.warning(f"Unknown subset_type={subset_type}, returning full data.")
        return patient_data


###############################################################################
# 3. Feature Selection
###############################################################################
def select_features(patient_data, feature_config="composite"):
    """
    Grabs only the columns relevant for each feature config.
    We'll one-hot encode them later, after merging with the model outputs,
    to avoid numeric-categorical mismatch.
    """
    chosen_cols = ['Id']
    base_demo = [
        'GENDER','RACE','ETHNICITY','MARITAL',
        'HEALTHCARE_EXPENSES','HEALTHCARE_COVERAGE','INCOME',
        'AGE','DECEASED','Hospitalizations_Count','Medications_Count','Abnormal_Observations_Count'
    ]
    chosen_cols.extend(base_demo)

    if feature_config == "composite":
        if 'Health_Index' not in patient_data.columns:
            raise KeyError("Missing 'Health_Index' for feature_config='composite'")
        chosen_cols.append('Health_Index')
    elif feature_config == "cci":
        if 'CharlsonIndex' not in patient_data.columns:
            raise KeyError("Missing 'CharlsonIndex' for feature_config='cci'")
        chosen_cols.append('CharlsonIndex')
    elif feature_config == "combined":
        if 'Health_Index' not in patient_data.columns or 'CharlsonIndex' not in patient_data.columns:
            raise KeyError("Missing 'Health_Index' or 'CharlsonIndex' for feature_config='combined'")
        chosen_cols.extend(['Health_Index', 'CharlsonIndex'])
    else:
        raise ValueError(f"Invalid feature_config: {feature_config}")

    return patient_data[chosen_cols].copy()


###############################################################################
# 4. Model Runners
###############################################################################
def run_vae(pkl_file, output_prefix):
    from vae_model import main as vae_main
    logger.info(f"Running VAE on {pkl_file} with prefix={output_prefix}.")
    vae_main(input_file=pkl_file, output_prefix=output_prefix)
    latent_csv = f"{output_prefix}_latent_features.csv"
    if not os.path.exists(latent_csv):
        logger.warning("[WARN] latent features CSV missing after VAE.")
        return None
    return latent_csv

def run_tabnet(pkl_file, output_prefix):
    from tabnet_model import main as tabnet_main
    logger.info(f"Running TabNet on {pkl_file} with prefix={output_prefix}.")
    tabnet_main(input_file=pkl_file, output_prefix=output_prefix)
    preds_csv = f"{output_prefix}_predictions.csv"
    if not os.path.exists(preds_csv):
        logger.warning("[WARN] TabNet predictions CSV missing after training.")
        return None
    return preds_csv


###############################################################################
# 5. Clustering & Visualization
###############################################################################
def memory_optimized_clustering_and_visualization(merged_df, config_id, plots_folder):
    """
    Clusters on the entire dataset, saving 2D t-SNE/UMAP plots to 'plots_folder'.
    One-hot encoding must already be done if we have categorical columns.
    """
    os.makedirs(plots_folder, exist_ok=True)

    # Exclude 'Id' and 'Predicted_Health_Index' from clustering
    X_columns = [c for c in merged_df.columns if c not in ('Id', 'Predicted_Health_Index')]
    X_full = merged_df[X_columns].values
    logger.info(f"[CLUSTER] Data shape: {X_full.shape}")

    if X_full.shape[1] == 0:
        logger.warning(f"[CLUSTER] 0 features for {config_id}. Skipping clustering.")
        return {}

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # 1) K-Means
    cluster_range = range(6, 10)
    kmeans_results = []
    for n in cluster_range:
        km = KMeans(n_clusters=n, random_state=42)
        labels_km = km.fit_predict(X_full_scaled)
        s = silhouette_score(X_full_scaled, labels_km)
        c = calinski_harabasz_score(X_full_scaled, labels_km)
        d = davies_bouldin_score(X_full_scaled, labels_km)
        kmeans_results.append((n, s, c, d))

    kmeans_df = pd.DataFrame(kmeans_results, columns=['k','silhouette','calinski','davies_bouldin'])
    kmeans_df['method'] = 'KMeans'
    kmeans_df['sil_rank'] = kmeans_df['silhouette'].rank(ascending=False)
    kmeans_df['ch_rank'] = kmeans_df['calinski'].rank(ascending=False)
    kmeans_df['db_rank'] = kmeans_df['davies_bouldin'].rank(ascending=True)
    kmeans_df['avg_rank'] = kmeans_df[['sil_rank','ch_rank','db_rank']].mean(axis=1)

    best_k = int(kmeans_df.loc[kmeans_df['avg_rank'].idxmin(), 'k'])
    best_km = KMeans(n_clusters=best_k, random_state=42)
    best_km.fit(X_full_scaled)
    final_labels_kmeans = best_km.predict(X_full_scaled)
    merged_df['Cluster'] = final_labels_kmeans

    # 2) DBSCAN
    neighbors = 5
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X_full_scaled)
    dist, idx = nbrs.kneighbors(X_full_scaled)
    dist = np.sort(dist[:, neighbors-1], axis=0)
    epsilon = dist[int(0.9 * len(dist))]
    dbscan = DBSCAN(eps=epsilon, min_samples=5)
    db_labels = dbscan.fit_predict(X_full_scaled)

    def cluster_scores(arr, labels):
        uset = set(labels)
        if len(uset) < 2:
            return (np.nan, np.nan, np.nan)
        return (
            silhouette_score(arr, labels),
            calinski_harabasz_score(arr, labels),
            davies_bouldin_score(arr, labels)
        )

    db_sil, db_cal, db_dav = cluster_scores(X_full_scaled, db_labels)

    # 3) Severity Index if Predicted_Health_Index is present
    if 'Predicted_Health_Index' in merged_df.columns:
        cluster_mean = (
            merged_df
            .groupby('Cluster')['Predicted_Health_Index']
            .mean()
            .sort_values()
            .reset_index()
        )
        cluster_mean['Severity_Index'] = range(1, len(cluster_mean)+1)
        c_map = dict(zip(cluster_mean['Cluster'], cluster_mean['Severity_Index']))
        merged_df['Severity_Index'] = merged_df['Cluster'].map(c_map)

    # 4) t-SNE & UMAP Visualization
    hue_col = 'Severity_Index' if 'Severity_Index' in merged_df.columns else 'Cluster'
    hue_vals = merged_df[hue_col].values

    tsne_2d = TSNE(n_components=2, random_state=42)
    X_tsne_2d = tsne_2d.fit_transform(X_full_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_tsne_2d[:,0], y=X_tsne_2d[:,1], hue=hue_vals, palette='viridis')
    plt.title(f"t-SNE 2D - KMeans best_k={best_k}, config={config_id}")
    tsne_path = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
    plt.savefig(tsne_path, bbox_inches='tight')
    plt.close()

    umap_2d = umap.UMAP(n_components=2, random_state=42)
    X_umap_2d = umap_2d.fit_transform(X_full_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_umap_2d[:,0], y=X_umap_2d[:,1], hue=hue_vals, palette='viridis')
    plt.title(f"UMAP 2D - {config_id}")
    umap_path = os.path.join(plots_folder, f"umap2d_{config_id}.png")
    plt.savefig(umap_path, bbox_inches='tight')
    plt.close()

    # 5) Final K-Means metrics
    final_sil = silhouette_score(X_full_scaled, final_labels_kmeans)
    final_ch = calinski_harabasz_score(X_full_scaled, final_labels_kmeans)
    final_db = davies_bouldin_score(X_full_scaled, final_labels_kmeans)

    del X_full, X_full_scaled
    gc.collect()

    return {
        "config_id": config_id,
        "chosen_method": "KMeans",
        "chosen_k": best_k,
        "final_silhouette": final_sil,
        "final_calinski": final_ch,
        "final_davies_bouldin": final_db,
        "dbscan_silhouette": db_sil,
        "dbscan_calinski": db_cal,
        "dbscan_davies_bouldin": db_dav
    }


###############################################################################
# 6. TabNet Regression Performance
###############################################################################
def evaluate_regression_performance(config_id, output_prefix):
    """
    Reads TabNet's metrics JSON if available.
    """
    metrics_file = f"{output_prefix}_metrics.json"
    mse_val, r2_val = np.nan, np.nan
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            mse_val = float(data.get("test_mse", np.nan))
            r2_val = float(data.get("test_r2", np.nan))
        except Exception as e:
            logger.warning(f"Could not parse {metrics_file}: {e}")
    else:
        logger.info(f"No TabNet metrics file {metrics_file} found; skipping.")
    return {"config_id": config_id, "tabnet_mse": mse_val, "tabnet_r2": r2_val}


###############################################################################
# 7. Save Overall Experiment Results
###############################################################################
def save_results_to_csv(output_path, results_list):
    df = pd.DataFrame(results_list)
    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode='a', header=write_header, index=False)


###############################################################################
# 8. Main Execution
###############################################################################
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Data")

    # Ensure data is ready
    ensure_preprocessed_data(data_dir)
    cci_path = os.path.join(data_dir, "patient_data_with_health_index_cci.pkl")
    if not os.path.exists(cci_path):
        raise FileNotFoundError("patient_data_with_health_index_cci.pkl missing after preprocessing.")

    full_df = pd.read_pickle(cci_path)
    logger.info("[MAIN] Loaded data shape=%s", full_df.shape)

    feature_configs = ["composite", "cci", "combined"]
    subset_types = ["none", "diabetes", "ckd"]
    model_approaches = ["vae", "tabnet", "hybrid"]

    all_results = []
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # For each (feature_config, subset, model_approach), we do an experiment
    for fc, ss, ma in itertools.product(feature_configs, subset_types, model_approaches):
        config_id = f"{fc}_{ss}_{ma}"
        logger.info("\n========================================")
        logger.info(" Running config: %s", config_id)
        logger.info("========================================")

        # Create a per-config subfolder
        config_folder = os.path.join(data_dir, "Experiments", config_id)
        os.makedirs(config_folder, exist_ok=True)

        # 1) Filter subset
        sub_df = filter_subpopulation(full_df, ss, data_dir)
        # 2) Feature selection
        use_df = select_features(sub_df, fc)

        # Dump intermediate to disk, then free memory
        temp_file = os.path.join(config_folder, f"temp_{config_id}_{run_timestamp}.pkl")
        use_df.to_pickle(temp_file)
        del sub_df, use_df
        gc.collect()

        # 3) Train or load models
        latent_csv = None
        tabnet_csv = None

        if ma == "vae":
            vae_prefix = os.path.join(config_folder, f"{config_id}_{run_timestamp}_vae")
            latent_csv = run_vae(temp_file, vae_prefix)

        elif ma == "tabnet":
            tabnet_prefix = os.path.join(config_folder, f"{config_id}_{run_timestamp}_tabnet")
            tabnet_csv = run_tabnet(temp_file, tabnet_prefix)

        elif ma == "hybrid":
            vae_prefix = os.path.join(config_folder, f"{config_id}_{run_timestamp}_vae")
            tabnet_prefix = os.path.join(config_folder, f"{config_id}_{run_timestamp}_tabnet")
            latent_csv = run_vae(temp_file, vae_prefix)
            tabnet_csv = run_tabnet(temp_file, tabnet_prefix)

        # 4) Merge outputs & cluster
        if latent_csv or tabnet_csv:
            frames = []
            if latent_csv and os.path.exists(latent_csv):
                df_lat = pd.read_csv(latent_csv)
                frames.append(df_lat)
            if tabnet_csv and os.path.exists(tabnet_csv):
                df_tab = pd.read_csv(tabnet_csv)
                frames.append(df_tab)

            if frames:
                # (a) Merge any model outputs together
                merged_df = frames[0]
                for fdf in frames[1:]:
                    merged_df = merged_df.merge(fdf, on='Id', how='inner')

                # (b) Re-merge the original training features
                temp_data = pd.read_pickle(temp_file)
                merged_df = temp_data.merge(merged_df, on='Id', how='inner')

                # >>>>>>  (b2) One-Hot Encode Categorical Columns  <<<<<<
                cat_cols = ['GENDER', 'RACE', 'ETHNICITY', 'MARITAL']
                # Only encode if these columns exist
                existing_cat = [c for c in cat_cols if c in merged_df.columns]
                if existing_cat:
                    # Convert to string before pd.get_dummies
                    for col in existing_cat:
                        merged_df[col] = merged_df[col].astype(str)
                    merged_df = pd.get_dummies(
                        merged_df, 
                        columns=existing_cat, 
                        drop_first=True  # avoids dummy trap
                    )
                # End of one-hot step

                # (c) Cluster if we have > 0 features
                features_cols = [c for c in merged_df.columns if c not in ('Id','Predicted_Health_Index')]
                if len(features_cols) == 0:
                    logger.warning(f"No features to cluster on for {config_id}, skipping clustering.")
                else:
                    # Use a config-specific plots folder
                    plots_folder = os.path.join(config_folder, "plots")
                    clust_res = memory_optimized_clustering_and_visualization(
                        merged_df,
                        config_id,
                        plots_folder=plots_folder
                    )
                    all_results.append(clust_res)

                # Clean up
                del merged_df
                for fdf in frames:
                    del fdf
                gc.collect()
            else:
                logger.info("[INFO] No CSV frames to merge for clustering.")
        else:
            logger.info(f"[INFO] No model output CSV to cluster/visualize for {config_id}")

        # 5) Evaluate TabNet if relevant
        if ma in ["tabnet", "hybrid"]:
            tabnet_prefix = os.path.join(config_folder, f"{config_id}_{run_timestamp}_tabnet")
            reg_res = evaluate_regression_performance(config_id, tabnet_prefix)
            all_results.append(reg_res)

        # 6) Remove temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        gc.collect()

    # 7) Save all results (at the top level or inside data_dir)
    results_csv_path = os.path.join(data_dir, OUTPUT_RESULTS_CSV)
    if all_results:
        save_results_to_csv(results_csv_path, all_results)
        logger.info("[MAIN] All experiment results appended to %s", results_csv_path)
    else:
        logger.info("[WARN] No results collected. Check logs for issues.")


if __name__ == "__main__":
    main()


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
 - Uses sampling for DBSCAN, t-SNE, UMAP if the dataset is large.
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
PLOTS_FOLDER = "plots"
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
    """
    Runs the VAE script; returns path to the latent CSV.
    """
    from vae_model import main as vae_main
    logger.info(f"Running VAE on {pkl_file} with prefix={output_prefix}.")
    vae_main(input_file=pkl_file, output_prefix=output_prefix)
    latent_csv = f"{output_prefix}_latent_features.csv"
    if not os.path.exists(latent_csv):
        logger.warning("[WARN] latent features CSV missing after VAE.")
        return None
    return latent_csv

def run_tabnet(pkl_file, output_prefix):
    """
    Runs the TabNet script; returns path to predictions CSV.
    """
    from tabnet_model import main as tabnet_main
    logger.info(f"Running TabNet on {pkl_file} with prefix={output_prefix}.")
    tabnet_main(input_file=pkl_file, output_prefix=output_prefix)
    preds_csv = f"{output_prefix}_predictions.csv"
    if not os.path.exists(preds_csv):
        logger.warning("[WARN] TabNet predictions CSV missing after training.")
        return None
    return preds_csv


###############################################################################
# 5. Clustering & Visualization (Memory-Optimized, No Hierarchical)
###############################################################################
def ensure_plots_folder():
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)

def memory_optimized_clustering_and_visualization(merged_df, config_id,
                                                  max_rows_for_clustering=10000,
                                                  max_rows_for_visuals=5000):
    """
    - Use KMeans + DBSCAN, skipping hierarchical (Agg).
    - Subsample for large data to avoid memory blow-ups.
    - Also subsample for t-SNE / UMAP.
    """
    ensure_plots_folder()

    # 1) Basic prep
    X_columns = [c for c in merged_df.columns if c not in ('Id','Predicted_Health_Index')]
    X_full = merged_df[X_columns].values
    logger.info(f"[CLUSTER] Data shape: {X_full.shape}")
    scaler = StandardScaler()

    # 2) Subsample for clustering (KMeans, DBSCAN)
    if len(X_full) > max_rows_for_clustering:
        idx_clust = np.random.choice(len(X_full), size=max_rows_for_clustering, replace=False)
        X_clust = X_full[idx_clust]
        logger.info(f"[CLUSTER] Subsampled {len(X_clust)} of {len(X_full)} for KMeans/DBSCAN.")
    else:
        X_clust = X_full

    X_clust_scaled = scaler.fit_transform(X_clust)

    # 2.a KMeans with range(6..10)
    cluster_range = range(6,10)
    kmeans_results = []
    for n in cluster_range:
        km = KMeans(n_clusters=n, random_state=42)
        labels_km = km.fit_predict(X_clust_scaled)
        s = silhouette_score(X_clust_scaled, labels_km)
        c = calinski_harabasz_score(X_clust_scaled, labels_km)
        d = davies_bouldin_score(X_clust_scaled, labels_km)
        kmeans_results.append((n, s, c, d))

    kmeans_df = pd.DataFrame(kmeans_results, columns=['k','silhouette','calinski','davies_bouldin'])
    kmeans_df['method'] = 'KMeans'
    kmeans_df['sil_rank'] = kmeans_df['silhouette'].rank(ascending=False)
    kmeans_df['ch_rank'] = kmeans_df['calinski'].rank(ascending=False)
    kmeans_df['db_rank'] = kmeans_df['davies_bouldin'].rank(ascending=True)
    kmeans_df['avg_rank'] = kmeans_df[['sil_rank','ch_rank','db_rank']].mean(axis=1)

    best_k = int(kmeans_df.loc[kmeans_df['avg_rank'].idxmin(),'k'])
    # Final KMeans
    best_km = KMeans(n_clusters=best_k, random_state=42)
    best_km.fit(X_clust_scaled)
    # If we want cluster labels for the entire dataset:
    # transform the full data
    X_full_scaled = scaler.transform(X_full) if len(X_full) != len(X_clust) else X_clust_scaled
    final_labels_kmeans = best_km.predict(X_full_scaled)

    # 2.b DBSCAN on sample
    neighbors = 5
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X_clust_scaled)
    dist, idx = nbrs.kneighbors(X_clust_scaled)
    dist = np.sort(dist[:,neighbors-1], axis=0)
    epsilon = dist[int(0.9 * len(dist))]
    dbscan = DBSCAN(eps=epsilon, min_samples=5)
    db_labels_sample = dbscan.fit_predict(X_clust_scaled)

    def cluster_scores(arr, labels):
        uset = set(labels)
        if len(uset) < 2:
            return (np.nan, np.nan, np.nan)
        return (
            silhouette_score(arr, labels),
            calinski_harabasz_score(arr, labels),
            davies_bouldin_score(arr, labels)
        )

    db_sil, db_cal, db_dav = cluster_scores(X_clust_scaled, db_labels_sample)

    # 3) Assign final cluster method: We only keep KMeans (since hierarchical is discarded).
    # For demonstration, let's store the KMeans cluster labels in merged_df:
    merged_df['Cluster'] = final_labels_kmeans

    # 4) Severity index if we have 'Predicted_Health_Index'
    if 'Predicted_Health_Index' in merged_df.columns:
        cluster_mean = (
            merged_df.groupby('Cluster')['Predicted_Health_Index'].mean()
            .sort_values().reset_index()
        )
        cluster_mean['Severity_Index'] = range(1, len(cluster_mean)+1)
        c_map = dict(zip(cluster_mean['Cluster'], cluster_mean['Severity_Index']))
        merged_df['Severity_Index'] = merged_df['Cluster'].map(c_map)

    # 5) Visualization with t-SNE / UMAP on a subsample
    hue_col = 'Severity_Index' if 'Severity_Index' in merged_df.columns else 'Cluster'

    # 5.a Subsample for visuals
    if len(X_full) > max_rows_for_visuals:
        idx_vis = np.random.choice(len(X_full), size=max_rows_for_visuals, replace=False)
        X_vis = X_full[idx_vis]
        hue_vals = merged_df.iloc[idx_vis][hue_col].values
        logger.info(f"[VIS] Subsampled {len(X_vis)} for t-SNE/UMAP from {len(X_full)}.")
    else:
        X_vis = X_full
        hue_vals = merged_df[hue_col].values

    X_vis_scaled = scaler.transform(X_vis)

    # t-SNE
    tsne_2d = TSNE(n_components=2, random_state=42)
    X_tsne_2d = tsne_2d.fit_transform(X_vis_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_tsne_2d[:,0], y=X_tsne_2d[:,1],
                    hue=hue_vals, palette='viridis')
    plt.title(f"t-SNE 2D - KMeans best_k={best_k}, config={config_id}")
    tsne_path = os.path.join(PLOTS_FOLDER, f"tsne2d_{config_id}.png")
    plt.savefig(tsne_path, bbox_inches='tight')
    plt.close()

    # UMAP 2D
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    X_umap_2d = umap_2d.fit_transform(X_vis_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_umap_2d[:,0], y=X_umap_2d[:,1],
                    hue=hue_vals, palette='viridis')
    plt.title(f"UMAP 2D - {config_id}")
    umap2d_path = os.path.join(PLOTS_FOLDER, f"umap2d_{config_id}.png")
    plt.savefig(umap2d_path, bbox_inches='tight')
    plt.close()

    # measure final cluster metrics on the full KMeans assignment
    final_sil = silhouette_score(X_full_scaled, final_labels_kmeans)
    final_ch = calinski_harabasz_score(X_full_scaled, final_labels_kmeans)
    final_db = davies_bouldin_score(X_full_scaled, final_labels_kmeans)

    # Cleanup big arrays
    del X_full, X_clust, X_clust_scaled, X_full_scaled, X_vis, X_vis_scaled
    gc.collect()

    cluster_metrics = {
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
    return cluster_metrics


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

    for fc, ss, ma in itertools.product(feature_configs, subset_types, model_approaches):
        config_id = f"{fc}_{ss}_{ma}"
        logger.info("\n========================================")
        logger.info(" Running config: %s", config_id)
        logger.info("========================================")

        # 1) Filter subset
        sub_df = filter_subpopulation(full_df, ss, data_dir)
        # 2) Feature selection
        use_df = select_features(sub_df, fc)

        # Dump intermediate to disk, then free memory
        temp_file = os.path.join(data_dir, f"temp_{config_id}_{run_timestamp}.pkl")
        use_df.to_pickle(temp_file)
        del sub_df, use_df
        gc.collect()

        # 3) Train or load models
        latent_csv = None
        tabnet_csv = None

        if ma == "vae":
            vae_prefix = f"{config_id}_{run_timestamp}_vae"
            latent_csv = run_vae(temp_file, vae_prefix)

        elif ma == "tabnet":
            tabnet_prefix = f"{config_id}_{run_timestamp}_tabnet"
            tabnet_csv = run_tabnet(temp_file, tabnet_prefix)

        elif ma == "hybrid":
            vae_prefix = f"{config_id}_{run_timestamp}_vae"
            tabnet_prefix = f"{config_id}_{run_timestamp}_tabnet"
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
                merged_df = frames[0]
                for f in frames[1:]:
                    merged_df = merged_df.merge(f, on='Id', how='inner')

                # Memory-optimized clustering & visualization (no hierarchical)
                clust_res = memory_optimized_clustering_and_visualization(merged_df, config_id)
                all_results.append(clust_res)

                # Clean up
                del merged_df
                for fdf in frames:
                    del fdf
                gc.collect()
            else:
                logger.info("[INFO] No CSV frames to merge for clustering.")
        else:
            logger.info("[INFO] No model output CSV to cluster/visualize for %s", config_id)

        # 5) Evaluate TabNet if relevant
        if ma in ["tabnet", "hybrid"]:
            tabnet_prefix = f"{config_id}_{run_timestamp}_tabnet"
            reg_res = evaluate_regression_performance(config_id, tabnet_prefix)
            all_results.append(reg_res)

        # 6) Remove temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        gc.collect()

    # 7) Save all results
    if all_results:
        save_results_to_csv(OUTPUT_RESULTS_CSV, all_results)
        logger.info("[MAIN] All experiment results appended to %s", OUTPUT_RESULTS_CSV)
    else:
        logger.info("[WARN] No results collected. Check logs for issues.")


if __name__ == "__main__":
    main()



### THIS NEEDED WAY TOO MUCH RAM - CANNOT DO THIS ANYMORE 
# """
# comprehensive_testing.py
# Author: Imran Feisal 
# Date: 11/01/2025

# Description:
# This script orchestrates a series of experiments to evaluate multiple configurations 
# of data (composite health index alone, CCI alone, combined), multiple models (VAE, 
# TabNet, or a hybrid approach), and multiple population subsets 
# (e.g., the entire dataset, 'diabetes' patients, or 'ckd' patients).

# It references the following scripts/modules:
#   - data_preprocessing.py   -> For generating patient_data_sequences.pkl
#   - health_index.py         -> For computing composite health index (1–10)
#   - charlson_comorbidity.py -> For integrating Charlson Comorbidity Index (CCI)
#   - vae_model.py            -> For training VAE & saving latent features
#   - tabnet_model.py         -> For training TabNet & saving predictions

# It then performs advanced clustering & visualization on the merged model outputs.

# Usage:
#   python comprehensive_testing.py
# """

# import os
# import sys
# import pandas as pd
# import numpy as np
# import itertools
# import datetime
# import logging
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Clustering and metrics
# from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
# from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
#                              davies_bouldin_score)
# from sklearn.preprocessing import StandardScaler

# # Dimensionality reduction
# from sklearn.manifold import TSNE
# import umap.umap_ as umap

# # 3D plotting
# from mpl_toolkits.mplot3d import Axes3D

# # Local imports
# from data_preprocessing import main as preprocess_main
# from health_index import main as health_main
# from charlson_comorbidity import load_cci_mapping, compute_cci
# from vae_model import main as vae_main
# from tabnet_model import main as tabnet_main

# # Basic config
# OUTPUT_RESULTS_CSV = "comprehensive_experiments_results.csv"
# PLOTS_FOLDER = "plots"
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def ensure_preprocessed_data(data_dir):
#     """
#     Ensure we have patient_data_sequences.pkl, patient_data_with_health_index.pkl,
#     and patient_data_with_health_index_cci.pkl. If missing, run relevant scripts.
#     """
#     pkl_sequences = os.path.join(data_dir, "patient_data_sequences.pkl")
#     pkl_health_index = os.path.join(data_dir, "patient_data_with_health_index.pkl")
#     pkl_health_cci = os.path.join(data_dir, "patient_data_with_health_index_cci.pkl")

#     # data_preprocessing
#     if not os.path.exists(pkl_sequences):
#         logger.info("patient_data_sequences.pkl not found -> Running data_preprocessing.py...")
#         preprocess_main()
#     else:
#         logger.info("Preprocessed sequences found.")

#     # health_index
#     if not os.path.exists(pkl_health_index):
#         logger.info("patient_data_with_health_index.pkl not found -> Running health_index.py...")
#         health_main()
#     else:
#         logger.info("Health index data found.")

#     # CCI
#     if not os.path.exists(pkl_health_cci):
#         logger.info("patient_data_with_health_index_cci.pkl not found -> Merging CCI.")
#         conditions_path = os.path.join(data_dir, "conditions.csv")
#         if not os.path.exists(conditions_path):
#             raise FileNotFoundError("conditions.csv not found. Cannot compute CCI.")
#         conditions = pd.read_csv(conditions_path, usecols=["PATIENT","CODE","DESCRIPTION"])

#         cci_map = load_cci_mapping(data_dir)
#         patient_cci = compute_cci(conditions, cci_map)

#         patient_data = pd.read_pickle(pkl_health_index)
#         merged = patient_data.merge(
#             patient_cci, how="left", left_on="Id", right_on="PATIENT"
#         )
#         merged.drop(columns="PATIENT", inplace=True)
#         merged["CharlsonIndex"] = merged["CharlsonIndex"].fillna(0.0)
#         merged.to_pickle(pkl_health_cci)
#         logger.info(f"[INFO] CCI merged & saved -> {pkl_health_cci}")
#     else:
#         logger.info("CCI data found.")


# def ensure_plots_folder():
#     """
#     Create a 'plots/' folder if missing, for storing outputs.
#     """
#     if not os.path.exists(PLOTS_FOLDER):
#         os.makedirs(PLOTS_FOLDER)


# # ------------------------------------------------------------------------------
# # Subset Filtering Logic
# # ------------------------------------------------------------------------------

# def load_conditions(data_dir):
#     """
#     A small utility to load conditions.csv if needed multiple times.
#     """
#     cpath = os.path.join(data_dir, "conditions.csv")
#     if not os.path.exists(cpath):
#         raise FileNotFoundError(f"conditions.csv not found at {cpath}. Cannot filter.")
#     return pd.read_csv(cpath, usecols=['PATIENT','CODE','DESCRIPTION'])


# def subset_diabetes(patient_data, data_dir):
#     """
#     Subset to patients with 'diabetes' anywhere in the condition description (case-insensitive).
#     """
#     conditions = load_conditions(data_dir)
#     diabetes_mask = conditions['DESCRIPTION'].str.lower().str.contains('diabetes', na=False)
#     diabetes_patients = conditions.loc[diabetes_mask, 'PATIENT'].unique()
#     # Filter
#     sub = patient_data[patient_data['Id'].isin(diabetes_patients)].copy()
#     logger.info(f"[INFO] Diabetes subset shape: {sub.shape}")
#     return sub


# def subset_ckd(patient_data, data_dir):
#     """
#     Subset to patients with 'chronic kidney disease' (CKD) or ESRD.
#     For demonstration, we check any stage 1–4, ESRD, or CKD mentions by code or text.
#     E.g., SNOMED codes: 
#        stage 1 (431855005), stage 2 (431856006), stage 3 (433144002),
#        stage 4 (431857002), ESRD (46177005), or 'chronic kidney disease' in text.

#     You can refine or expand codes to match your scenario. 
#     """
#     conditions = load_conditions(data_dir)
#     ckd_snomed = {431855005, 431856006, 433144002, 431857002, 46177005}  # etc.
#     # Alternatively, searching text:
#     # ckd_mask = conditions['DESCRIPTION'].str.lower().str.contains('chronic kidney disease', na=False)

#     # We'll do either approach:
#     code_mask = conditions['CODE'].isin(ckd_snomed)
#     text_mask = conditions['DESCRIPTION'].str.lower().str.contains('chronic kidney disease', na=False)
#     ckd_mask = code_mask | text_mask

#     ckd_patients = conditions.loc[ckd_mask, 'PATIENT'].unique()
#     sub = patient_data[patient_data['Id'].isin(ckd_patients)].copy()
#     logger.info(f"[INFO] CKD subset shape: {sub.shape}")
#     return sub


# def filter_subpopulation(patient_data, subset_type, data_dir):
#     """
#     Return a specific subpopulation:
#        - 'none': entire dataset
#        - 'diabetes': only diabetic patients
#        - 'ckd': only CKD patients
#     """
#     if subset_type.lower() == "none":
#         return patient_data
#     elif subset_type.lower() == "diabetes":
#         return subset_diabetes(patient_data, data_dir)
#     elif subset_type.lower() == "ckd":
#         return subset_ckd(patient_data, data_dir)
#     else:
#         logger.warning(f"Subset '{subset_type}' not recognized -> returning full data.")
#         return patient_data


# # ------------------------------------------------------------------------------
# # Feature Selection
# # ------------------------------------------------------------------------------

# def select_features(patient_data, feature_config="composite"):
#     """
#     Grab columns from the dataset: base demographics, plus
#     - composite (Health_Index)
#     - cci (CharlsonIndex)
#     - combined (both)
#     """
#     chosen_cols = ['Id']
#     base_demo = [
#         'GENDER','RACE','ETHNICITY','MARITAL',
#         'HEALTHCARE_EXPENSES','HEALTHCARE_COVERAGE','INCOME',
#         'AGE','DECEASED','Hospitalizations_Count','Medications_Count','Abnormal_Observations_Count'
#     ]
#     chosen_cols.extend(base_demo)

#     if feature_config == "composite":
#         if 'Health_Index' not in patient_data.columns:
#             raise KeyError("Missing 'Health_Index' for feature_config='composite'.")
#         chosen_cols.append('Health_Index')
#     elif feature_config == "cci":
#         if 'CharlsonIndex' not in patient_data.columns:
#             raise KeyError("Missing 'CharlsonIndex' for feature_config='cci'.")
#         chosen_cols.append('CharlsonIndex')
#     elif feature_config == "combined":
#         if 'Health_Index' not in patient_data.columns or 'CharlsonIndex' not in patient_data.columns:
#             raise KeyError("Missing 'Health_Index' or 'CharlsonIndex' for feature_config='combined'.")
#         chosen_cols.extend(['Health_Index','CharlsonIndex'])
#     else:
#         raise ValueError(f"Invalid feature_config: {feature_config}")

#     return patient_data[chosen_cols].copy()


# # ------------------------------------------------------------------------------
# # Model Runners
# # ------------------------------------------------------------------------------

# def run_vae(patient_data_pkl, output_prefix):
#     """
#     Runs the VAE script with a given output_prefix and returns the path 
#     to the latent CSV file produced.
#     """
#     logger.info(f"[INFO] Running VAE on {patient_data_pkl} with prefix={output_prefix} ...")
#     # We call the VAE main script via an OS command or by importing and calling main().
#     # If you're calling it as a function, do:
#     from vae_model import main as vae_main
#     vae_main(input_file=patient_data_pkl, output_prefix=output_prefix)

#     latent_csv = f"{output_prefix}_latent_features.csv"  
#     if not os.path.exists(latent_csv):
#         logger.warning("[WARN] latent features CSV not found after VAE training.")
#         return None
#     return latent_csv


# def run_tabnet(patient_data_pkl, output_prefix):
#     """
#     Runs the TabNet script with a given output_prefix and returns the path
#     to the predictions CSV.
#     """
#     logger.info(f"[INFO] Running TabNet on {patient_data_pkl} with prefix={output_prefix} ...")
#     from tabnet_model import main as tabnet_main
#     tabnet_main(input_file=patient_data_pkl, output_prefix=output_prefix)

#     preds_csv = f"{output_prefix}_predictions.csv"
#     if not os.path.exists(preds_csv):
#         logger.warning("[WARN] TabNet predictions CSV not found after TabNet training.")
#         return None
#     return preds_csv


# # ------------------------------------------------------------------------------
# # Clustering & Visualization (based on your analysis notebook)
# # ------------------------------------------------------------------------------

# def ensure_plots_folder():
#     if not os.path.exists(PLOTS_FOLDER):
#         os.makedirs(PLOTS_FOLDER)


# def perform_clustering_and_visualization(merged_df, config_id):
#     """
#     1) K-Means, Agglomerative, DBSCAN
#     2) Evaluate silhouette, calinski, davies-bouldin
#     3) Choose best method
#     4) Create cluster-based severity index using mean Predicted_Health_Index
#     5) t-SNE & UMAP (2D, 3D) plots saved to 'plots/' folder

#     Returns a dict of final cluster metrics for the given config_id.
#     """
#     ensure_plots_folder()

#     X_columns = [c for c in merged_df.columns if c not in ('Id','Predicted_Health_Index')]
#     X = merged_df[X_columns].values
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     cluster_range = range(6,10)
#     # 1) KMeans
#     kmeans_results = []
#     for n in cluster_range:
#         km = KMeans(n_clusters=n, random_state=42)
#         labels = km.fit_predict(X_scaled)
#         sil = silhouette_score(X_scaled, labels)
#         ch = calinski_harabasz_score(X_scaled, labels)
#         db = davies_bouldin_score(X_scaled, labels)
#         kmeans_results.append((n, sil, ch, db))
#     kmeans_df = pd.DataFrame(kmeans_results, columns=['k','silhouette','calinski','davies_bouldin'])
#     kmeans_df['method'] = 'KMeans'
#     kmeans_df['sil_rank'] = kmeans_df['silhouette'].rank(ascending=False)
#     kmeans_df['ch_rank'] = kmeans_df['calinski'].rank(ascending=False)
#     kmeans_df['db_rank'] = kmeans_df['davies_bouldin'].rank(ascending=True)
#     kmeans_df['avg_rank'] = kmeans_df[['sil_rank','ch_rank','db_rank']].mean(axis=1)
#     best_k = int(kmeans_df.loc[kmeans_df['avg_rank'].idxmin(),'k'])

#     # 2) Agg
#     agg_results = []
#     for n in cluster_range:
#         agg = AgglomerativeClustering(n_clusters=n)
#         labels = agg.fit_predict(X_scaled)
#         sil = silhouette_score(X_scaled, labels)
#         ch = calinski_harabasz_score(X_scaled, labels)
#         db = davies_bouldin_score(X_scaled, labels)
#         agg_results.append((n, sil, ch, db))
#     agg_df = pd.DataFrame(agg_results, columns=['k','silhouette','calinski','davies_bouldin'])
#     agg_df['method'] = 'Agglomerative'
#     agg_df['sil_rank'] = agg_df['silhouette'].rank(ascending=False)
#     agg_df['ch_rank'] = agg_df['calinski'].rank(ascending=False)
#     agg_df['db_rank'] = agg_df['davies_bouldin'].rank(ascending=True)
#     agg_df['avg_rank'] = agg_df[['sil_rank','ch_rank','db_rank']].mean(axis=1)
#     best_k_agg = int(agg_df.loc[agg_df['avg_rank'].idxmin(),'k'])

#     # 3) DBSCAN
#     from sklearn.neighbors import NearestNeighbors
#     neighbors = 5
#     nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X_scaled)
#     dist, idx = nbrs.kneighbors(X_scaled)
#     dist = np.sort(dist[:,neighbors-1], axis=0)
#     epsilon = dist[int(0.9 * len(dist))]
#     dbscan = DBSCAN(eps=epsilon, min_samples=5)
#     db_labels = dbscan.fit_predict(X_scaled)

#     def cluster_scores(arr, lbls):
#         uset = set(lbls)
#         if len(uset) < 2:
#             return (np.nan, np.nan, np.nan)
#         s_ = silhouette_score(arr, lbls)
#         c_ = calinski_harabasz_score(arr, lbls)
#         d_ = davies_bouldin_score(arr, lbls)
#         return (s_, c_, d_)

#     sil_db, ch_db, db_db = cluster_scores(X_scaled, db_labels)

#     all_eval = pd.concat([
#         kmeans_df[['method','k','silhouette','calinski','davies_bouldin','avg_rank']],
#         agg_df[['method','k','silhouette','calinski','davies_bouldin','avg_rank']]
#     ], ignore_index=True)
#     # add DBSCAN row
#     dbscan_row = pd.DataFrame({
#         'method':['DBSCAN'],
#         'k':[np.nan],
#         'silhouette':[sil_db],
#         'calinski':[ch_db],
#         'davies_bouldin':[db_db],
#         'avg_rank':[np.nan]
#     })
#     all_eval = pd.concat([all_eval, dbscan_row], ignore_index=True)

#     # Decide best from KMeans / Agg
#     method_opts = all_eval.dropna(subset=['avg_rank'])
#     best_idx = method_opts['avg_rank'].idxmin()
#     best_method = method_opts.loc[best_idx,'method']
#     if best_method == 'KMeans':
#         final_clusterer = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
#         final_labels = final_clusterer.labels_
#         chosen_k = best_k
#     else:
#         final_clusterer = AgglomerativeClustering(n_clusters=best_k_agg).fit(X_scaled)
#         final_labels = final_clusterer.labels_
#         chosen_k = best_k_agg

#     merged_df['Cluster'] = final_labels

#     # 4) Severity index
#     if 'Predicted_Health_Index' in merged_df.columns:
#         c_map = (
#             merged_df.groupby('Cluster')['Predicted_Health_Index'].mean()
#             .sort_values().reset_index()
#         )
#         c_map['Severity_Index'] = range(1, len(c_map)+1)
#         mp = dict(zip(c_map['Cluster'], c_map['Severity_Index']))
#         merged_df['Severity_Index'] = merged_df['Cluster'].map(mp)

#     # 5) Visualizations
#     # t-SNE (2D)
#     tsne_2d = TSNE(n_components=2, random_state=42)
#     X_tsne_2d = tsne_2d.fit_transform(X_scaled)
#     plt.figure(figsize=(8,6))
#     hue_col = 'Severity_Index' if 'Severity_Index' in merged_df.columns else 'Cluster'
#     sns.scatterplot(x=X_tsne_2d[:,0], y=X_tsne_2d[:,1],
#                     hue=merged_df[hue_col], palette='viridis')
#     plt.title(f"t-SNE 2D - {best_method} config={config_id}")
#     tsne_path = os.path.join(PLOTS_FOLDER, f"tsne2d_{config_id}.png")
#     plt.savefig(tsne_path, bbox_inches='tight')
#     plt.close()

#     # UMAP (2D)
#     umap_2d = umap.UMAP(n_components=2, random_state=42)
#     X_umap_2d = umap_2d.fit_transform(X_scaled)
#     plt.figure(figsize=(8,6))
#     sns.scatterplot(x=X_umap_2d[:,0], y=X_umap_2d[:,1],
#                     hue=merged_df[hue_col], palette='viridis')
#     plt.title(f"UMAP 2D - {best_method} config={config_id}")
#     umap2d_path = os.path.join(PLOTS_FOLDER, f"umap2d_{config_id}.png")
#     plt.savefig(umap2d_path, bbox_inches='tight')
#     plt.close()

#     # UMAP (3D)
#     umap_3d = umap.UMAP(n_components=3, random_state=42)
#     X_umap_3d = umap_3d.fit_transform(X_scaled)
#     fig = plt.figure(figsize=(10,7))
#     ax = fig.add_subplot(111, projection='3d')
#     sc = ax.scatter(
#         X_umap_3d[:,0], X_umap_3d[:,1], X_umap_3d[:,2],
#         c=merged_df[hue_col], cmap='viridis', alpha=0.8
#     )
#     ax.set_title(f"UMAP 3D - {best_method} config={config_id}")
#     ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_zlabel("UMAP3")
#     cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.09)
#     cbar.set_label(hue_col)
#     umap3d_path = os.path.join(PLOTS_FOLDER, f"umap3d_{config_id}.png")
#     plt.savefig(umap3d_path, bbox_inches='tight')
#     plt.close()

#     # measure final cluster metrics
#     final_sil = silhouette_score(X_scaled, final_labels)
#     final_ch  = calinski_harabasz_score(X_scaled, final_labels)
#     final_db  = davies_bouldin_score(X_scaled, final_labels)

#     cluster_metrics = {
#         "config_id": config_id,
#         "chosen_method": best_method,
#         "chosen_k": chosen_k if best_method in ['KMeans','Agglomerative'] else np.nan,
#         "final_silhouette": final_sil,
#         "final_calinski": final_ch,
#         "final_davies_bouldin": final_db,
#         "dbscan_silhouette": sil_db,
#         "dbscan_calinski": ch_db,
#         "dbscan_davies_bouldin": db_db
#     }
#     return cluster_metrics


# def evaluate_regression_performance(config_id):
#     """
#     Read TabNet's test metrics from 'tabnet_metrics.json' (if available).
#     """
#     mse_val = np.nan
#     r2_val = np.nan

#     metrics_file = "tabnet_metrics.json"
#     if os.path.exists(metrics_file):
#         try:
#             with open(metrics_file, "r") as f:
#                 metrics_data = json.load(f)
#             mse_val = float(metrics_data.get("test_mse", np.nan))
#             r2_val = float(metrics_data.get("test_r2", np.nan))
#         except Exception as e:
#             logger.warning(f"Could not parse {metrics_file}: {e}")
#     else:
#         logger.info("No tabnet_metrics.json found; using placeholder MSE & R^2.")

#     return {
#         "config_id": config_id,
#         "tabnet_mse": mse_val,
#         "tabnet_r2": r2_val
#     }



# def save_results_to_csv(output_path, results_list):
#     df = pd.DataFrame(results_list)
#     write_header = not os.path.exists(output_path)
#     df.to_csv(output_path, mode='a', header=write_header, index=False)


# # ------------------------------------------------------------------------------
# # Main Execution
# # ------------------------------------------------------------------------------
# def main():
#     """
#     1) Ensure preprocessed data with CCI
#     2) Loop over (feature_config × subset_type × model_approach)
#     3) For each config, subset the data, select columns, save to .pkl
#     4) Train VAE/TabNet/hybrid
#     5) Merge outputs, do advanced clustering & visuals
#     6) Save metrics to CSV
#     """
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(script_dir, "Data")

#     # Make sure we have the needed data
#     ensure_preprocessed_data(data_dir)
#     cci_path = os.path.join(data_dir, "patient_data_with_health_index_cci.pkl")
#     if not os.path.exists(cci_path):
#         raise FileNotFoundError("patient_data_with_health_index_cci.pkl not found after preprocessing.")

#     full_data = pd.read_pickle(cci_path)

#     # We add 'ckd' to the subset types
#     feature_configs = ["composite", "cci", "combined"]
#     subset_types = ["none", "diabetes", "ckd"]  # extended
#     model_approaches = ["vae", "tabnet", "hybrid"]

#     all_results = []
#     run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#     for fc, ss, ma in itertools.product(feature_configs, subset_types, model_approaches):
#         config_id = f"{fc}_{ss}_{ma}"
#         logger.info(f"\n--- Running config: {config_id} ---")

#         # (1) Filter data
#         data_sub = filter_subpopulation(full_data, ss, data_dir=data_dir)

#         # (2) Select features
#         data_sel = select_features(data_sub, fc)

#         # (3) Save to .pkl
#         tmp_pkl_name = f"temp_data_{config_id}_{run_timestamp}.pkl"
#         tmp_pkl_path = os.path.join(data_dir, tmp_pkl_name)
#         data_sel.to_pickle(tmp_pkl_path)
#         logger.info(f"[INFO] Saved subset data -> {tmp_pkl_path}")

#         # (4) Train model(s)
#         latent_csv = None
#         tabnet_csv = None

#         if ma == "vae":
#             prefix_vae = f"{config_id}_{run_timestamp}_vae"
#             latent_csv = run_vae(tmp_pkl_name, prefix_vae)

#         elif ma == "tabnet":
#             prefix_tabnet = f"{config_id}_{run_timestamp}_tabnet"
#             tabnet_csv = run_tabnet(tmp_pkl_name, prefix_tabnet)

#         elif ma == "hybrid":
#             prefix_vae = f"{config_id}_{run_timestamp}_vae"
#             prefix_tabnet = f"{config_id}_{run_timestamp}_tabnet"
#             latent_csv = run_vae(tmp_pkl_name, prefix_vae)
#             tabnet_csv = run_tabnet(tmp_pkl_name, prefix_tabnet)

#         # (5) Merge & cluster
#         if latent_csv or tabnet_csv:
#             frames = []
#             if latent_csv and os.path.exists(latent_csv):
#                 lat_df = pd.read_csv(latent_csv)
#                 frames.append(lat_df)
#             if tabnet_csv and os.path.exists(tabnet_csv):
#                 tb_df = pd.read_csv(tabnet_csv)
#                 frames.append(tb_df)

#             if frames:
#                 merged_df = frames[0]
#                 for f in frames[1:]:
#                     merged_df = merged_df.merge(f, on='Id', how='inner')
#                 # Clustering & visuals
#                 cluster_res = perform_clustering_and_visualization(merged_df, config_id)
#                 all_results.append(cluster_res)
#             else:
#                 logger.info("[INFO] No CSV frames to merge for clustering.")
#         else:
#             logger.info("[INFO] No model output CSV to evaluate for config=" + config_id)

#         # (6) Evaluate TabNet if relevant
#         if ma in ["tabnet", "hybrid"]:
#             # We read from "xx_tabnet_metrics.json" depending on prefix_tabnet
#             metrics_file = f"{config_id}_{run_timestamp}_tabnet_metrics.json"
#             if os.path.exists(metrics_file):
#                 try:
#                     with open(metrics_file, "r") as f:
#                         metrics_data = json.load(f)
#                     mse_val = float(metrics_data.get("test_mse", np.nan))
#                     r2_val = float(metrics_data.get("test_r2", np.nan))
#                 except Exception as e:
#                     logger.warning(f"Could not parse {metrics_file}: {e}")
#                     mse_val, r2_val = np.nan, np.nan
#             else:
#                 logger.info(f"No TabNet metrics file found for config={config_id}; using placeholder.")
#                 mse_val, r2_val = np.nan, np.nan

#             reg_res = {
#                 "config_id": config_id,
#                 "tabnet_mse": mse_val,
#                 "tabnet_r2": r2_val
#             }
#             all_results.append(reg_res)

#         # Clean up if you like
#         os.remove(tmp_pkl_path)

#     # (7) Save all results
#     if all_results:
#         save_results_to_csv(OUTPUT_RESULTS_CSV, all_results)
#         logger.info(f"[INFO] Comprehensive testing done. Results appended to {OUTPUT_RESULTS_CSV}.")
#     else:
#         logger.info("[WARN] No results collected. Check model outputs.")

# if __name__ == "__main__":
#     main()

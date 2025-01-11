"""
comprehensive_testing.py
Author: [Your Name]
Date: [DD/MM/YYYY]

Description:
This script orchestrates a series of experiments to evaluate multiple configurations 
of data (composite health index alone, CCI alone, combined), multiple models (VAE, 
TabNet, or a hybrid approach), and multiple population subsets 
(e.g., the entire dataset, 'diabetes' patients, or 'ckd' patients).

It references the following scripts/modules:
  - data_preprocessing.py   -> For generating patient_data_sequences.pkl
  - health_index.py         -> For computing composite health index (1–10)
  - charlson_comorbidity.py -> For integrating Charlson Comorbidity Index (CCI)
  - vae_model.py            -> For training VAE & saving latent features
  - tabnet_model.py         -> For training TabNet & saving predictions

It then performs advanced clustering & visualization on the merged model outputs.

Usage:
  python comprehensive_testing.py
"""

import os
import sys
import pandas as pd
import numpy as np
import itertools
import datetime
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering and metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.preprocessing import StandardScaler

# Dimensionality reduction
from sklearn.manifold import TSNE
import umap.umap_ as umap

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D

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


def ensure_preprocessed_data(data_dir):
    """
    Ensure we have patient_data_sequences.pkl, patient_data_with_health_index.pkl,
    and patient_data_with_health_index_cci.pkl. If missing, run relevant scripts.
    """
    pkl_sequences = os.path.join(data_dir, "patient_data_sequences.pkl")
    pkl_health_index = os.path.join(data_dir, "patient_data_with_health_index.pkl")
    pkl_health_cci = os.path.join(data_dir, "patient_data_with_health_index_cci.pkl")

    # data_preprocessing
    if not os.path.exists(pkl_sequences):
        logger.info("patient_data_sequences.pkl not found -> Running data_preprocessing.py...")
        preprocess_main()
    else:
        logger.info("Preprocessed sequences found.")

    # health_index
    if not os.path.exists(pkl_health_index):
        logger.info("patient_data_with_health_index.pkl not found -> Running health_index.py...")
        health_main()
    else:
        logger.info("Health index data found.")

    # CCI
    if not os.path.exists(pkl_health_cci):
        logger.info("patient_data_with_health_index_cci.pkl not found -> Merging CCI.")
        conditions_path = os.path.join(data_dir, "conditions.csv")
        if not os.path.exists(conditions_path):
            raise FileNotFoundError("conditions.csv not found. Cannot compute CCI.")
        conditions = pd.read_csv(conditions_path, usecols=["PATIENT","CODE","DESCRIPTION"])

        cci_map = load_cci_mapping(data_dir)
        patient_cci = compute_cci(conditions, cci_map)

        patient_data = pd.read_pickle(pkl_health_index)
        merged = patient_data.merge(
            patient_cci, how="left", left_on="Id", right_on="PATIENT"
        )
        merged.drop(columns="PATIENT", inplace=True)
        merged["CharlsonIndex"] = merged["CharlsonIndex"].fillna(0.0)
        merged.to_pickle(pkl_health_cci)
        logger.info(f"[INFO] CCI merged & saved -> {pkl_health_cci}")
    else:
        logger.info("CCI data found.")


def ensure_plots_folder():
    """
    Create a 'plots/' folder if missing, for storing outputs.
    """
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)


# ------------------------------------------------------------------------------
# Subset Filtering Logic
# ------------------------------------------------------------------------------

def load_conditions(data_dir):
    """
    A small utility to load conditions.csv if needed multiple times.
    """
    cpath = os.path.join(data_dir, "conditions.csv")
    if not os.path.exists(cpath):
        raise FileNotFoundError(f"conditions.csv not found at {cpath}. Cannot filter.")
    return pd.read_csv(cpath, usecols=['PATIENT','CODE','DESCRIPTION'])


def subset_diabetes(patient_data, data_dir):
    """
    Subset to patients with 'diabetes' anywhere in the condition description (case-insensitive).
    """
    conditions = load_conditions(data_dir)
    diabetes_mask = conditions['DESCRIPTION'].str.lower().str.contains('diabetes', na=False)
    diabetes_patients = conditions.loc[diabetes_mask, 'PATIENT'].unique()
    # Filter
    sub = patient_data[patient_data['Id'].isin(diabetes_patients)].copy()
    logger.info(f"[INFO] Diabetes subset shape: {sub.shape}")
    return sub


def subset_ckd(patient_data, data_dir):
    """
    Subset to patients with 'chronic kidney disease' (CKD) or ESRD.
    For demonstration, we check any stage 1–4, ESRD, or CKD mentions by code or text.
    E.g., SNOMED codes: 
       stage 1 (431855005), stage 2 (431856006), stage 3 (433144002),
       stage 4 (431857002), ESRD (46177005), or 'chronic kidney disease' in text.

    You can refine or expand codes to match your scenario. 
    """
    conditions = load_conditions(data_dir)
    ckd_snomed = {431855005, 431856006, 433144002, 431857002, 46177005}  # etc.
    # Alternatively, searching text:
    # ckd_mask = conditions['DESCRIPTION'].str.lower().str.contains('chronic kidney disease', na=False)

    # We'll do either approach:
    code_mask = conditions['CODE'].isin(ckd_snomed)
    text_mask = conditions['DESCRIPTION'].str.lower().str.contains('chronic kidney disease', na=False)
    ckd_mask = code_mask | text_mask

    ckd_patients = conditions.loc[ckd_mask, 'PATIENT'].unique()
    sub = patient_data[patient_data['Id'].isin(ckd_patients)].copy()
    logger.info(f"[INFO] CKD subset shape: {sub.shape}")
    return sub


def filter_subpopulation(patient_data, subset_type, data_dir):
    """
    Return a specific subpopulation:
       - 'none': entire dataset
       - 'diabetes': only diabetic patients
       - 'ckd': only CKD patients
    """
    if subset_type.lower() == "none":
        return patient_data
    elif subset_type.lower() == "diabetes":
        return subset_diabetes(patient_data, data_dir)
    elif subset_type.lower() == "ckd":
        return subset_ckd(patient_data, data_dir)
    else:
        logger.warning(f"Subset '{subset_type}' not recognized -> returning full data.")
        return patient_data


# ------------------------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------------------------

def select_features(patient_data, feature_config="composite"):
    """
    Grab columns from the dataset: base demographics, plus
    - composite (Health_Index)
    - cci (CharlsonIndex)
    - combined (both)
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
            raise KeyError("Missing 'Health_Index' for feature_config='composite'.")
        chosen_cols.append('Health_Index')
    elif feature_config == "cci":
        if 'CharlsonIndex' not in patient_data.columns:
            raise KeyError("Missing 'CharlsonIndex' for feature_config='cci'.")
        chosen_cols.append('CharlsonIndex')
    elif feature_config == "combined":
        if 'Health_Index' not in patient_data.columns or 'CharlsonIndex' not in patient_data.columns:
            raise KeyError("Missing 'Health_Index' or 'CharlsonIndex' for feature_config='combined'.")
        chosen_cols.extend(['Health_Index','CharlsonIndex'])
    else:
        raise ValueError(f"Invalid feature_config: {feature_config}")

    return patient_data[chosen_cols].copy()


# ------------------------------------------------------------------------------
# Model Runners
# ------------------------------------------------------------------------------

def run_vae(patient_data_pkl):
    logger.info(f"[INFO] Running VAE on {patient_data_pkl} ...")
    vae_main(input_file=patient_data_pkl)
    latent_csv = "latent_features_vae.csv"
    if not os.path.exists(latent_csv):
        logger.warning("[WARN] latent_features_vae.csv not found after VAE training.")
    return latent_csv


def run_tabnet(patient_data_pkl):
    logger.info(f"[INFO] Running TabNet on {patient_data_pkl} ...")
    tabnet_main(input_file=patient_data_pkl)
    preds_csv = "tabnet_predictions.csv"
    if not os.path.exists(preds_csv):
        logger.warning("[WARN] tabnet_predictions.csv not found after TabNet training.")
    return preds_csv


# ------------------------------------------------------------------------------
# Clustering & Visualization (based on your analysis notebook)
# ------------------------------------------------------------------------------

def ensure_plots_folder():
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)


def perform_clustering_and_visualization(merged_df, config_id):
    """
    1) K-Means, Agglomerative, DBSCAN
    2) Evaluate silhouette, calinski, davies-bouldin
    3) Choose best method
    4) Create cluster-based severity index using mean Predicted_Health_Index
    5) t-SNE & UMAP (2D, 3D) plots saved to 'plots/' folder

    Returns a dict of final cluster metrics for the given config_id.
    """
    ensure_plots_folder()

    X_columns = [c for c in merged_df.columns if c not in ('Id','Predicted_Health_Index')]
    X = merged_df[X_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cluster_range = range(6,10)
    # 1) KMeans
    kmeans_results = []
    for n in cluster_range:
        km = KMeans(n_clusters=n, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        kmeans_results.append((n, sil, ch, db))
    kmeans_df = pd.DataFrame(kmeans_results, columns=['k','silhouette','calinski','davies_bouldin'])
    kmeans_df['method'] = 'KMeans'
    kmeans_df['sil_rank'] = kmeans_df['silhouette'].rank(ascending=False)
    kmeans_df['ch_rank'] = kmeans_df['calinski'].rank(ascending=False)
    kmeans_df['db_rank'] = kmeans_df['davies_bouldin'].rank(ascending=True)
    kmeans_df['avg_rank'] = kmeans_df[['sil_rank','ch_rank','db_rank']].mean(axis=1)
    best_k = int(kmeans_df.loc[kmeans_df['avg_rank'].idxmin(),'k'])

    # 2) Agg
    agg_results = []
    for n in cluster_range:
        agg = AgglomerativeClustering(n_clusters=n)
        labels = agg.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        agg_results.append((n, sil, ch, db))
    agg_df = pd.DataFrame(agg_results, columns=['k','silhouette','calinski','davies_bouldin'])
    agg_df['method'] = 'Agglomerative'
    agg_df['sil_rank'] = agg_df['silhouette'].rank(ascending=False)
    agg_df['ch_rank'] = agg_df['calinski'].rank(ascending=False)
    agg_df['db_rank'] = agg_df['davies_bouldin'].rank(ascending=True)
    agg_df['avg_rank'] = agg_df[['sil_rank','ch_rank','db_rank']].mean(axis=1)
    best_k_agg = int(agg_df.loc[agg_df['avg_rank'].idxmin(),'k'])

    # 3) DBSCAN
    from sklearn.neighbors import NearestNeighbors
    neighbors = 5
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X_scaled)
    dist, idx = nbrs.kneighbors(X_scaled)
    dist = np.sort(dist[:,neighbors-1], axis=0)
    epsilon = dist[int(0.9 * len(dist))]
    dbscan = DBSCAN(eps=epsilon, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)

    def cluster_scores(arr, lbls):
        uset = set(lbls)
        if len(uset) < 2:
            return (np.nan, np.nan, np.nan)
        s_ = silhouette_score(arr, lbls)
        c_ = calinski_harabasz_score(arr, lbls)
        d_ = davies_bouldin_score(arr, lbls)
        return (s_, c_, d_)

    sil_db, ch_db, db_db = cluster_scores(X_scaled, db_labels)

    all_eval = pd.concat([
        kmeans_df[['method','k','silhouette','calinski','davies_bouldin','avg_rank']],
        agg_df[['method','k','silhouette','calinski','davies_bouldin','avg_rank']]
    ], ignore_index=True)
    # add DBSCAN row
    dbscan_row = pd.DataFrame({
        'method':['DBSCAN'],
        'k':[np.nan],
        'silhouette':[sil_db],
        'calinski':[ch_db],
        'davies_bouldin':[db_db],
        'avg_rank':[np.nan]
    })
    all_eval = pd.concat([all_eval, dbscan_row], ignore_index=True)

    # Decide best from KMeans / Agg
    method_opts = all_eval.dropna(subset=['avg_rank'])
    best_idx = method_opts['avg_rank'].idxmin()
    best_method = method_opts.loc[best_idx,'method']
    if best_method == 'KMeans':
        final_clusterer = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
        final_labels = final_clusterer.labels_
        chosen_k = best_k
    else:
        final_clusterer = AgglomerativeClustering(n_clusters=best_k_agg).fit(X_scaled)
        final_labels = final_clusterer.labels_
        chosen_k = best_k_agg

    merged_df['Cluster'] = final_labels

    # 4) Severity index
    if 'Predicted_Health_Index' in merged_df.columns:
        c_map = (
            merged_df.groupby('Cluster')['Predicted_Health_Index'].mean()
            .sort_values().reset_index()
        )
        c_map['Severity_Index'] = range(1, len(c_map)+1)
        mp = dict(zip(c_map['Cluster'], c_map['Severity_Index']))
        merged_df['Severity_Index'] = merged_df['Cluster'].map(mp)

    # 5) Visualizations
    # t-SNE (2D)
    tsne_2d = TSNE(n_components=2, random_state=42)
    X_tsne_2d = tsne_2d.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    hue_col = 'Severity_Index' if 'Severity_Index' in merged_df.columns else 'Cluster'
    sns.scatterplot(x=X_tsne_2d[:,0], y=X_tsne_2d[:,1],
                    hue=merged_df[hue_col], palette='viridis')
    plt.title(f"t-SNE 2D - {best_method} config={config_id}")
    tsne_path = os.path.join(PLOTS_FOLDER, f"tsne2d_{config_id}.png")
    plt.savefig(tsne_path, bbox_inches='tight')
    plt.close()

    # UMAP (2D)
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    X_umap_2d = umap_2d.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_umap_2d[:,0], y=X_umap_2d[:,1],
                    hue=merged_df[hue_col], palette='viridis')
    plt.title(f"UMAP 2D - {best_method} config={config_id}")
    umap2d_path = os.path.join(PLOTS_FOLDER, f"umap2d_{config_id}.png")
    plt.savefig(umap2d_path, bbox_inches='tight')
    plt.close()

    # UMAP (3D)
    umap_3d = umap.UMAP(n_components=3, random_state=42)
    X_umap_3d = umap_3d.fit_transform(X_scaled)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        X_umap_3d[:,0], X_umap_3d[:,1], X_umap_3d[:,2],
        c=merged_df[hue_col], cmap='viridis', alpha=0.8
    )
    ax.set_title(f"UMAP 3D - {best_method} config={config_id}")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_zlabel("UMAP3")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.09)
    cbar.set_label(hue_col)
    umap3d_path = os.path.join(PLOTS_FOLDER, f"umap3d_{config_id}.png")
    plt.savefig(umap3d_path, bbox_inches='tight')
    plt.close()

    # measure final cluster metrics
    final_sil = silhouette_score(X_scaled, final_labels)
    final_ch  = calinski_harabasz_score(X_scaled, final_labels)
    final_db  = davies_bouldin_score(X_scaled, final_labels)

    cluster_metrics = {
        "config_id": config_id,
        "chosen_method": best_method,
        "chosen_k": chosen_k if best_method in ['KMeans','Agglomerative'] else np.nan,
        "final_silhouette": final_sil,
        "final_calinski": final_ch,
        "final_davies_bouldin": final_db,
        "dbscan_silhouette": sil_db,
        "dbscan_calinski": ch_db,
        "dbscan_davies_bouldin": db_db
    }
    return cluster_metrics


def evaluate_regression_performance(config_id):
    """
    Read TabNet's test metrics from 'tabnet_metrics.json' (if available).
    """
    mse_val = np.nan
    r2_val = np.nan

    metrics_file = "tabnet_metrics.json"
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
            mse_val = float(metrics_data.get("test_mse", np.nan))
            r2_val = float(metrics_data.get("test_r2", np.nan))
        except Exception as e:
            logger.warning(f"Could not parse {metrics_file}: {e}")
    else:
        logger.info("No tabnet_metrics.json found; using placeholder MSE & R^2.")

    return {
        "config_id": config_id,
        "tabnet_mse": mse_val,
        "tabnet_r2": r2_val
    }



def save_results_to_csv(output_path, results_list):
    df = pd.DataFrame(results_list)
    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode='a', header=write_header, index=False)


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    """
    1) Ensure preprocessed data with CCI
    2) Loop over (feature_config × subset_type × model_approach)
    3) For each config, subset the data, select columns, save to .pkl
    4) Train VAE/TabNet/hybrid
    5) Merge outputs, do advanced clustering & visuals
    6) Save metrics to CSV
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Data")

    # Make sure we have the needed data
    ensure_preprocessed_data(data_dir)
    cci_path = os.path.join(data_dir, "patient_data_with_health_index_cci.pkl")
    if not os.path.exists(cci_path):
        raise FileNotFoundError("patient_data_with_health_index_cci.pkl not found after preprocessing.")

    full_data = pd.read_pickle(cci_path)

    # We add 'ckd' to the subset types
    feature_configs = ["composite", "cci", "combined"]
    subset_types = ["none", "diabetes", "ckd"]  # extended
    model_approaches = ["vae", "tabnet", "hybrid"]

    all_results = []
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for fc, ss, ma in itertools.product(feature_configs, subset_types, model_approaches):
        config_id = f"{fc}_{ss}_{ma}"
        logger.info(f"\n--- Running config: {config_id} ---")

        # 1) Filter data to subset
        data_sub = filter_subpopulation(full_data, ss, data_dir=data_dir)

        # 2) Select features
        data_sel = select_features(data_sub, fc)

        # 3) Save to .pkl
        tmp_pkl_name = f"temp_data_{config_id}_{run_timestamp}.pkl"
        tmp_pkl_path = os.path.join(data_dir, tmp_pkl_name)
        data_sel.to_pickle(tmp_pkl_path)
        logger.info(f"[INFO] Saved subset data for config={config_id} -> {tmp_pkl_path}")

        # 4) Train model(s)
        latent_csv, tabnet_csv = None, None
        if ma == "vae":
            latent_csv = run_vae(tmp_pkl_name)
        elif ma == "tabnet":
            tabnet_csv = run_tabnet(tmp_pkl_name)
        elif ma == "hybrid":
            latent_csv = run_vae(tmp_pkl_name)
            tabnet_csv = run_tabnet(tmp_pkl_name)

        # 5) Merge outputs & cluster
        if latent_csv or tabnet_csv:
            frames = []
            if latent_csv and os.path.exists(latent_csv):
                lat_df = pd.read_csv(latent_csv)
                frames.append(lat_df)
            if tabnet_csv and os.path.exists(tabnet_csv):
                tb_df = pd.read_csv(tabnet_csv)
                frames.append(tb_df)

            if frames:
                merged_df = frames[0]
                for f in frames[1:]:
                    merged_df = merged_df.merge(f, on='Id', how='inner')
                # Clustering & visuals
                cluster_res = perform_clustering_and_visualization(merged_df, config_id)
                all_results.append(cluster_res)
            else:
                logger.info("[INFO] No CSV frames to merge for clustering.")
        else:
            logger.info("[INFO] No model output CSV to evaluate for config=" + config_id)

        # 6) Evaluate TabNet regression if relevant
        if ma in ["tabnet","hybrid"]:
            reg_res = evaluate_regression_performance(config_id)
            all_results.append(reg_res)

        # Clean up if you like:
        os.remove(tmp_pkl_path)

    # 7) Save all experiment results
    if all_results:
        save_results_to_csv(OUTPUT_RESULTS_CSV, all_results)
        logger.info(f"[INFO] Comprehensive testing done. Results appended to {OUTPUT_RESULTS_CSV}.")
    else:
        logger.info("[WARN] No results collected. Check model outputs.")


if __name__ == "__main__":
    main()

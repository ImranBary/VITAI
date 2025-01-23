# vitai_scripts/run_vitai_tests_main.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#   A single master script to:
#     1) Ensure the data is prepared (including Elixhauser).
#     2) For each (feature_config, subset_type, model_approach),
#        run VAE/TabNet as needed, merge outputs, do clustering.
#     3) Gather model metrics & cluster metrics into ONE final CSV.
#
# Usage Example:
#   python run_vitai.py \
#       --data-dir Data \
#       --output-file final_vitai_results.csv \
#       --feature-configs composite cci eci combined combined_eci combined_all\
#       --subset-types none diabetes ckd \
#       --model-approaches vae tabnet hybrid

import os
import sys
import gc
import argparse
import logging
import datetime
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm

# vitai_scripts modules
from data_prep import ensure_preprocessed_data
from subset_utils import filter_subpopulation
from feature_utils import select_features
from model_utils import run_vae, run_tabnet, gather_vae_metrics, gather_tabnet_metrics
from cluster_utils import cluster_and_visualise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def file_is_fully_written(file_path: str, min_size=1, max_age=30) -> bool:
    """
    Checks if a file is large enough and not modified in the last 'max_age' seconds.
    Used to avoid partial files.
    """
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) < min_size:
        return False
    import time
    mtime = os.path.getmtime(file_path)
    now = time.time()
    if (now - mtime) < max_age:
        return False
    return True

def vae_done(vae_prefix: str) -> bool:
    latent_csv = f"{vae_prefix}_latent_features.csv"
    return file_is_fully_written(latent_csv, min_size=10)

def tabnet_done(tabnet_prefix: str) -> bool:
    preds_csv = f"{tabnet_prefix}_predictions.csv"
    metrics_json = f"{tabnet_prefix}_metrics.json"
    preds_ok = file_is_fully_written(preds_csv, min_size=10)
    metrics_ok = file_is_fully_written(metrics_json, min_size=2)
    return preds_ok and metrics_ok

def cluster_done(config_id: str, config_folder: str) -> bool:
    plots_folder = os.path.join(config_folder, "plots")
    tsne_path = os.path.join(plots_folder, f"tsne2d_{config_id}.png")
    umap_path = os.path.join(plots_folder, f"umap2d_{config_id}.png")
    tsne_ok = file_is_fully_written(tsne_path, min_size=1000)
    umap_ok = file_is_fully_written(umap_path, min_size=1000)
    return (tsne_ok and umap_ok)

def all_done(fc, ss, ma, config_folder, run_ts):
    """
    Checks if VAE, TabNet, and clustering are all done for this combo.
    """
    config_id = f"{fc}_{ss}_{ma}"
    vae_prefix = os.path.join(config_folder, f"{config_id}_{run_ts}_vae")
    tabnet_prefix = os.path.join(config_folder, f"{config_id}_{run_ts}_tabnet")

    if ma == "vae":
        return vae_done(vae_prefix) and cluster_done(config_id, config_folder)
    elif ma == "tabnet":
        return tabnet_done(tabnet_prefix) and cluster_done(config_id, config_folder)
    elif ma == "hybrid":
        return vae_done(vae_prefix) and tabnet_done(tabnet_prefix) and cluster_done(config_id, config_folder)
    return False

def run_vitai_pipeline(data_dir, output_file, feature_configs, subset_types, model_approaches):
    data_dir = os.path.abspath(data_dir)
    out_csv  = os.path.join(data_dir, output_file)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output CSV: {out_csv}")
    logger.info(f"Feature configs: {feature_configs}")
    logger.info(f"Subset types: {subset_types}")
    logger.info(f"Model approaches: {model_approaches}")

    # 1) Ensure data is fully prepped (including Elixhauser)
    ensure_preprocessed_data(data_dir)

    # 2) Load the combined pickle with all indices
    final_pkl = os.path.join(data_dir, "patient_data_with_all_indices.pkl")
    if not os.path.exists(final_pkl):
        raise FileNotFoundError("No patient_data_with_all_indices.pkl after data prep.")
    full_df = pd.read_pickle(final_pkl)
    logger.info(f"[Main] Loaded final dataset shape={full_df.shape}")

    # 3) Build combos
    combos = list(itertools.product(feature_configs, subset_types, model_approaches))
    total = len(combos)

    # We'll store results for each combo, then write to a single CSV
    all_results = []
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for (fc, ss, ma) in tqdm(combos, total=total, desc="Overall progress"):
        config_id = f"{fc}_{ss}_{ma}"
        logger.info(f"\n===== Running {config_id} =====")
        config_folder = os.path.join(data_dir, "Experiments", config_id)
        os.makedirs(config_folder, exist_ok=True)

        # If all done, skip
        if all_done(fc, ss, ma, config_folder, run_ts):
            logger.info(f"[Skip] All done for {config_id}.")
            continue

        # Subset
        sub_df = filter_subpopulation(full_df, ss, data_dir)
        # Feature selection
        feats_df = select_features(sub_df, fc)

        # Save temp
        temp_pkl = os.path.join(config_folder, f"temp_{config_id}_{run_ts}.pkl")
        feats_df.to_pickle(temp_pkl)
        del sub_df, feats_df
        gc.collect()

        # Model prefixes
        vae_prefix = os.path.join(config_folder, f"{config_id}_{run_ts}_vae")
        tabnet_prefix = os.path.join(config_folder, f"{config_id}_{run_ts}_tabnet")

        # Possibly run VAE
        if ma in ["vae","hybrid"] and not vae_done(vae_prefix):
            run_vae(temp_pkl, vae_prefix)

        # Possibly run TabNet
        if ma in ["tabnet","hybrid"] and not tabnet_done(tabnet_prefix):
            # Decide target_col
            if fc == "cci":
                run_tabnet(temp_pkl, tabnet_prefix, target_col="CharlsonIndex")
            elif fc == "eci":
                run_tabnet(temp_pkl, tabnet_prefix, target_col="ElixhauserIndex")
            else:
                # composite, combined, combined_eci -> typically predict Health_Index
                run_tabnet(temp_pkl, tabnet_prefix, target_col="Health_Index")

        # Gather partial results (model metrics)
        result_dict = {"config_id": config_id}
        if ma in ["tabnet","hybrid"]:
            tb_metrics = gather_tabnet_metrics(tabnet_prefix)
            result_dict.update(tb_metrics)
        if ma in ["vae","hybrid"]:
            vae_metrics = gather_vae_metrics(vae_prefix)
            result_dict.update(vae_metrics)

        # Merge outputs to do clustering
        frames = []
        vae_csv = f"{vae_prefix}_latent_features.csv"
        if os.path.exists(vae_csv) and os.path.getsize(vae_csv) > 10:
            frames.append(pd.read_csv(vae_csv))
        tabnet_csv = f"{tabnet_prefix}_predictions.csv"
        if os.path.exists(tabnet_csv) and os.path.getsize(tabnet_csv) > 10:
            frames.append(pd.read_csv(tabnet_csv))

        if frames:
            temp_data = pd.read_pickle(temp_pkl)
            merged_df = temp_data
            for fdf in frames:
                merged_df = merged_df.merge(fdf, on="Id", how="inner")

            # One-hot encode typical demographics
            cat_cols = ["GENDER","RACE","ETHNICITY","MARITAL"]
            existing_cats = [c for c in cat_cols if c in merged_df.columns]
            for c in existing_cats:
                merged_df[c] = merged_df[c].astype(str)
            if existing_cats:
                merged_df = pd.get_dummies(merged_df, columns=existing_cats, drop_first=True)

            # Clustering
            if not cluster_done(config_id, config_folder):
                clus_res = cluster_and_visualise(merged_df, config_id, os.path.join(config_folder, "plots"))
                result_dict.update(clus_res)
            else:
                logger.info(f"[Skip] Clustering for {config_id} already done.")
            del merged_df, temp_data
        else:
            logger.info(f"[{config_id}] No model output to cluster, skipping.")

        # Clean up
        if os.path.exists(temp_pkl):
            os.remove(temp_pkl)

        all_results.append(result_dict)

    # 4) Write everything to one final CSV
    if all_results:
        df_final = pd.DataFrame(all_results)
        # If file doesn't exist, write header
        write_header = not os.path.exists(out_csv)
        df_final.to_csv(out_csv, mode='a', header=write_header, index=False)
        logger.info(f"Saved final results to {out_csv}")
    else:
        logger.info("No new results to write. All done.")

def main():
    parser = argparse.ArgumentParser(description="Run the entire VITAI pipeline (incl. Elixhauser) from one script.")
    parser.add_argument("--data-dir", type=str, default="Data",
                        help="Path to the Data folder.")
    parser.add_argument("--output-file", type=str, default="vitai_final_results.csv",
                        help="Name of the final CSV to produce.")
    parser.add_argument("--feature-configs", nargs="+",
                        default=["composite","cci","eci","combined","combined_eci", "combined_all"],
                        help="Which feature configs to test.")
    parser.add_argument("--subset-types", nargs="+",
                        default=["none","diabetes","ckd"],
                        help="Which subpopulations to test.")
    parser.add_argument("--model-approaches", nargs="+",
                        default=["vae","tabnet","hybrid"],
                        help="Which model approaches to test.")
    args = parser.parse_args()

    run_vitai_pipeline(args.data_dir, args.output_file, args.feature_configs, args.subset_types, args.model_approaches)

if __name__ == "__main__":
    main()
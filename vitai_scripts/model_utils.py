# vitai_scripts/model_utils.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#   Provides utility functions to run the VAE and TabNet models,
#   plus gather their metrics from JSON files.

import os
import glob
import json
import logging
import numpy as np
import pandas as pd
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
# Root-level models
from vae_model import main as vae_main
from tabnet_model import main as tabnet_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_vae(input_pkl: str, output_prefix: str) -> str:
    """
    Runs VAE on 'input_pkl' and saves results with prefix=output_prefix.
    Returns the path to the latent CSV or None if missing.
    """
    logger.info(f"[VAE] Starting with prefix={output_prefix}")
    vae_main(input_file=input_pkl, output_prefix=output_prefix)
    latent_csv = f"{output_prefix}_latent_features.csv"
    if os.path.exists(latent_csv) and os.path.getsize(latent_csv) > 10:
        return latent_csv
    logger.warning("[VAE] Latent features CSV not found or too small.")
    return None

def run_tabnet(input_pkl: str, output_prefix: str, target_col: str = "Health_Index") -> str:
    """
    Runs TabNet on 'input_pkl' with the specified 'target_col',
    returns the path to the predictions CSV or None if missing.
    """
    logger.info(f"[TabNet] Starting with prefix={output_prefix}, target={target_col}")
    tabnet_main(input_file=input_pkl, output_prefix=output_prefix, target_col=target_col)
    preds_csv = f"{output_prefix}_predictions.csv"
    if os.path.exists(preds_csv) and os.path.getsize(preds_csv) > 10:
        return preds_csv
    logger.warning("[TabNet] Predictions CSV not found or too small.")
    return None

def gather_tabnet_metrics(prefix: str) -> dict:
    """
    Reads TabNet metrics from the JSON file if present.
    Returns a dict with: tabnet_mse, tabnet_r2
    """
    mf = f"{prefix}_metrics.json"
    out = {"tabnet_mse": np.nan, "tabnet_r2": np.nan}
    if not os.path.exists(mf) or os.path.getsize(mf) < 2:
        return out

    try:
        with open(mf, "r") as f:
            data = json.load(f)
        out["tabnet_mse"] = float(data.get("test_mse", np.nan))
        out["tabnet_r2"]  = float(data.get("test_r2", np.nan))
    except Exception as e:
        logger.warning(f"[TabNet] Error reading metrics JSON: {e}")
    return out

def gather_vae_metrics(prefix: str) -> dict:
    """
    Reads VAE metrics from the JSON file if present.
    Returns a dict with: vae_final_train_loss, vae_best_val_loss
    """
    pattern = f"{prefix}_vae_metrics.json"
    if not os.path.exists(pattern):
        maybe = glob.glob(prefix + "_vae_metrics.json")
        if not maybe:
            return {"vae_final_train_loss": np.nan, "vae_best_val_loss": np.nan}
        pattern = maybe[0]

    out = {"vae_final_train_loss": np.nan, "vae_best_val_loss": np.nan}
    if os.path.getsize(pattern) < 2:
        return out

    try:
        with open(pattern, "r") as f:
            data = json.load(f)
        out["vae_final_train_loss"] = float(data.get("final_train_loss", np.nan))
        out["vae_best_val_loss"]    = float(data.get("best_val_loss", np.nan))
    except Exception as e:
        logger.warning(f"[VAE] Error reading metrics JSON: {e}")

    return out

"""
update_model_metrics.py
Author: Imran Feisal
Date: 19/01/2025

Description:
1) Reads 'comprehensive_experiments_results_v2.csv' (clustering + partial TabNet metrics).
2) Merges base data from 'patient_data_with_health_index_cci.pkl' + 'patient_data_with_health_index.pkl'.
3) For each config_id that is 'vae' or 'hybrid' but missing VAE outputs, re-run VAE with a temp pkl.
4) Grab new VAE metrics (final_train_loss, best_val_loss) + TabNet MSE/R2 from JSONs.
5) Update the table as 'comprehensive_experiments_results_v3.csv'.
"""

import os
import glob
import json
import logging
import numpy as np
import pandas as pd
import time

from vae_model import main as vae_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Paths
###############################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "Experiments")

OLD_RESULTS_CSV = "comprehensive_experiments_results_v2.csv"
NEW_RESULTS_CSV = "comprehensive_experiments_results_v3.csv"

CCI_PICKLE = os.path.join(DATA_DIR, "patient_data_with_health_index_cci.pkl")
EXTRA_PICKLE = os.path.join(DATA_DIR, "patient_data_with_health_index.pkl")

TEMP_PKL_DIR = os.path.join(DATA_DIR, "temp_pkls")
os.makedirs(TEMP_PKL_DIR, exist_ok=True)

###############################################################################
# Utility: Check if file is fully written
###############################################################################
def file_is_fully_written(file_path, min_size=1, max_age_seconds=30):
    """True iff file_path is large enough and not freshly modified."""
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) < min_size:
        return False
    mtime = os.path.getmtime(file_path)
    now = time.time()
    if (now - mtime) < max_age_seconds:
        return False
    return True

def step_vae_artifacts_ok(config_path):
    """
    Return True iff *both* '*_latent_features.csv' & '*_vae_metrics.json' exist & are valid size.
    """
    latents = glob.glob(os.path.join(config_path, "*_latent_features.csv"))
    metrics = glob.glob(os.path.join(config_path, "*_vae_metrics.json"))
    if not latents or not metrics:
        return False
    lat_csv_ok = any(file_is_fully_written(f, min_size=10) for f in latents)
    json_ok = any(file_is_fully_written(f, min_size=2) for f in metrics)
    return lat_csv_ok and json_ok

###############################################################################
# Merge Base Data
###############################################################################
def load_and_merge_base_data():
    """Load cci + extra pickles, merge if columns exist, drop list columns."""
    if not os.path.exists(CCI_PICKLE):
        raise FileNotFoundError(f"Missing {CCI_PICKLE}")
    if not os.path.exists(EXTRA_PICKLE):
        raise FileNotFoundError(f"Missing {EXTRA_PICKLE}")

    df_cci = pd.read_pickle(CCI_PICKLE)   # has CharlsonIndex, Health_Index
    df_extra = pd.read_pickle(EXTRA_PICKLE)  # has Hospitalizations_Count, etc.

    # Only merge columns that exist
    desired = {
        "Id", 
        "Hospitalizations_Count", 
        "Medications_Count", 
        "Abnormal_Observations_Count"
    }
    actual_extra_cols = list(set(df_extra.columns).intersection(desired))
    missing = desired - set(df_extra.columns)
    if missing:
        logger.warning(f"[BASE] The following columns not found in {EXTRA_PICKLE}: {missing}")

    base_df = df_cci.merge(df_extra[actual_extra_cols], on="Id", how="left")

    # Drop any list columns
    for col in ["SEQUENCE", "PATIENT"]:
        if col in base_df.columns:
            base_df.drop(columns=[col], inplace=True)

    for col in list(base_df.columns):
        if base_df[col].map(type).eq(list).any():
            logger.warning(f"[BASE] Dropping {col}, it contains list data.")
            base_df.drop(columns=[col], inplace=True)

    return base_df

###############################################################################
# Subset + Feature Selection
###############################################################################
def subset_ckd(df):
    ckd_codes = {431855005, 431856006, 433144002, 431857002, 46177005}
    cond_path = os.path.join(DATA_DIR, "conditions.csv")
    if not os.path.exists(cond_path):
        raise FileNotFoundError("conditions.csv not found (subset_ckd).")

    cdf = pd.read_csv(cond_path, usecols=["PATIENT","CODE","DESCRIPTION"])
    code_mask = cdf["CODE"].isin(ckd_codes)
    text_mask = cdf["DESCRIPTION"].str.lower().str.contains("chronic kidney disease", na=False)
    ckd_patients = cdf.loc[code_mask | text_mask, "PATIENT"].unique()
    return df[df["Id"].isin(ckd_patients)].copy()

def subset_diabetes(df):
    cond_path = os.path.join(DATA_DIR, "conditions.csv")
    if not os.path.exists(cond_path):
        raise FileNotFoundError("conditions.csv not found (subset_diabetes).")

    cdf = pd.read_csv(cond_path, usecols=["PATIENT","DESCRIPTION"])
    mask = cdf["DESCRIPTION"].str.lower().str.contains("diabetes", na=False)
    diab_patients = cdf.loc[mask, "PATIENT"].unique()
    return df[df["Id"].isin(diab_patients)].copy()

def filter_subpopulation(df, subset_type):
    ss = subset_type.lower()
    if ss == "none":
        return df
    elif ss == "ckd":
        return subset_ckd(df)
    elif ss == "diabetes":
        return subset_diabetes(df)
    else:
        logger.warning(f"[subset] Unknown subset={subset_type}, returning full.")
        return df

def select_features(df, feature_config):
    """
    We want base_cols = [
      'Id','GENDER','RACE','ETHNICITY','MARITAL',
      'HEALTHCARE_EXPENSES','HEALTHCARE_COVERAGE','INCOME',
      'AGE','DECEASED',
      'Hospitalizations_Count','Medications_Count','Abnormal_Observations_Count'
    ]
    plus Health_Index or CharlsonIndex or both.
    If some columns are missing, fill them with zeros.
    """
    base_cols = [
        'Id','GENDER','RACE','ETHNICITY','MARITAL',
        'HEALTHCARE_EXPENSES','HEALTHCARE_COVERAGE','INCOME',
        'AGE','DECEASED',
        'Hospitalizations_Count','Medications_Count','Abnormal_Observations_Count'
    ]
    # Fill missing base cols with zero
    missing_in_df = set(base_cols) - set(df.columns)
    for col in missing_in_df:
        df[col] = 0

    if feature_config == "composite":
        if "Health_Index" not in df.columns:
            raise KeyError("Missing 'Health_Index' for feature_config='composite'")
        final_cols = base_cols + ["Health_Index"]
    elif feature_config == "cci":
        if "CharlsonIndex" not in df.columns:
            raise KeyError("Missing 'CharlsonIndex' for feature_config='cci'")
        final_cols = base_cols + ["CharlsonIndex"]
    elif feature_config == "combined":
        if ("Health_Index" not in df.columns or "CharlsonIndex" not in df.columns):
            raise KeyError("Missing 'Health_Index' or 'CharlsonIndex' for 'combined'")
        final_cols = base_cols + ["Health_Index","CharlsonIndex"]
    else:
        raise ValueError(f"Invalid feature_config: {feature_config}")

    # Only use columns that actually exist now
    final_cols = [c for c in final_cols if c in df.columns]
    return df[final_cols].copy()

###############################################################################
# Main
###############################################################################
def main():
    old_path = os.path.join(DATA_DIR, OLD_RESULTS_CSV)
    if not os.path.exists(old_path):
        logger.error(f"[ERROR] {OLD_RESULTS_CSV} not found; nothing to update.")
        return

    df_results = pd.read_csv(old_path)
    logger.info(f"[INFO] Loaded results shape={df_results.shape}")

    # Merge base data
    base_df = load_and_merge_base_data()
    logger.info(f"[INFO] Base data shape={base_df.shape}")

    updated_info = {}

    # 1) Check each row => config_id => parse fc, ss, ma => if ma in [vae, hybrid], 
    #    check if we have VAE artifacts. If not, rebuild temp pkl & run VAE.
    for i, row in df_results.iterrows():
        cid = row.get("config_id","")
        if not cid:
            continue

        parts = cid.split("_")
        if len(parts) != 3:
            continue  # or do something else
        fc, ss, ma = parts

        if ma not in ["vae","hybrid"]:
            continue

        config_path = os.path.join(EXPERIMENTS_DIR, cid)
        if not os.path.isdir(config_path):
            # No folder => skip
            continue

        # If we already have VAE artifacts, skip
        if step_vae_artifacts_ok(config_path):
            logger.info(f"[SKIP] {cid} => VAE artifacts exist.")
            continue

        # Otherwise, re-train
        logger.info(f"[RETRAIN] {cid} => missing VAE => generating temp pkl & run VAE.")
        sub_df = filter_subpopulation(base_df, ss)
        use_df = select_features(sub_df, fc)

        temp_path = os.path.join(TEMP_PKL_DIR, f"temp_{cid}.pkl")
        use_df.to_pickle(temp_path)

        # VAE prefix
        vae_prefix = os.path.join(config_path, f"{cid}_manual_vae")
        try:
            vae_main(input_file=temp_path, output_prefix=vae_prefix)
            logger.info(f"[DONE] VAE re-trained for {cid}")
        except Exception as ex:
            logger.error(f"[FAIL] Could not run VAE for {cid}: {ex}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 2) Gather TabNet & VAE metrics
    config_dirs = sorted(os.listdir(EXPERIMENTS_DIR))
    for conf in config_dirs:
        conf_path = os.path.join(EXPERIMENTS_DIR, conf)
        if not os.path.isdir(conf_path):
            continue

        updated_info[conf] = {}

        # TabNet
        t_json = glob.glob(os.path.join(conf_path, "*_tabnet_metrics.json"))
        tabnet_mse = np.nan
        tabnet_r2 = np.nan
        if t_json:
            tfile = t_json[0]
            if file_is_fully_written(tfile, min_size=2):
                try:
                    with open(tfile,"r") as f:
                        data = json.load(f)
                    tabnet_mse = float(data.get("test_mse", np.nan))
                    tabnet_r2 = float(data.get("test_r2", np.nan))
                except Exception as e:
                    logger.warning(f"[{conf}] parse error on {tfile}: {e}")
        updated_info[conf]["tabnet_mse"] = tabnet_mse
        updated_info[conf]["tabnet_r2"] = tabnet_r2

        # VAE
        v_json = glob.glob(os.path.join(conf_path, "*_vae_metrics.json"))
        vae_loss = np.nan
        vae_val = np.nan
        if v_json:
            vfile = v_json[0]
            if file_is_fully_written(vfile, min_size=2):
                try:
                    with open(vfile,"r") as f:
                        data = json.load(f)
                    vae_loss = float(data.get("final_train_loss", np.nan))
                    vae_val  = float(data.get("best_val_loss", np.nan))
                except Exception as e:
                    logger.warning(f"[{conf}] parse error on {vfile}: {e}")
        updated_info[conf]["vae_loss"] = vae_loss
        updated_info[conf]["vae_val_loss"] = vae_val

    # 3) Insert into df_results
    for col in ["tabnet_mse","tabnet_r2","vae_loss","vae_val_loss"]:
        if col not in df_results.columns:
            df_results[col] = np.nan

    for i in range(len(df_results)):
        configid = df_results.at[i,"config_id"]
        if configid in updated_info:
            # TabNet
            if pd.isna(df_results.at[i,"tabnet_mse"]):
                df_results.at[i,"tabnet_mse"] = updated_info[configid]["tabnet_mse"]
            if pd.isna(df_results.at[i,"tabnet_r2"]):
                df_results.at[i,"tabnet_r2"] = updated_info[configid]["tabnet_r2"]
            # VAE
            df_results.at[i,"vae_loss"] = updated_info[configid]["vae_loss"]
            df_results.at[i,"vae_val_loss"] = updated_info[configid]["vae_val_loss"]

    # 4) Save
    out_csv = os.path.join(DATA_DIR, NEW_RESULTS_CSV)
    df_results.to_csv(out_csv, index=False)
    logger.info(f"[DONE] Wrote {NEW_RESULTS_CSV} with updated metrics.")

if __name__ == "__main__":
    main()

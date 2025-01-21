# vitai_scripts/subset_utils.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#   Provides logic for filtering the patient DataFrame
#   into specific sub-populations (none, diabetes, ckd).

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_conditions(data_dir: str) -> pd.DataFrame:
    cond_path = os.path.join(data_dir, "conditions.csv")
    if not os.path.exists(cond_path):
        raise FileNotFoundError(f"Cannot find conditions.csv at {cond_path}")
    return pd.read_csv(cond_path, usecols=["PATIENT","CODE","DESCRIPTION"])

def subset_diabetes(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    conditions = _load_conditions(data_dir)
    mask = conditions["DESCRIPTION"].str.lower().str.contains("diabetes", na=False)
    diabetic_patients = conditions.loc[mask, "PATIENT"].unique()
    sub = df[df["Id"].isin(diabetic_patients)].copy()
    logger.info(f"Subset 'diabetes' shape: {sub.shape}")
    return sub

def subset_ckd(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    conditions = _load_conditions(data_dir)
    ckd_codes = {431855005, 431856006, 433144002, 431857002, 46177005}
    code_mask = conditions["CODE"].isin(ckd_codes)
    text_mask = conditions["DESCRIPTION"].str.lower().str.contains("chronic kidney disease", na=False)
    ckd_patients = conditions.loc[code_mask | text_mask, "PATIENT"].unique()
    sub = df[df["Id"].isin(ckd_patients)].copy()
    logger.info(f"Subset 'ckd' shape: {sub.shape}")
    return sub

def filter_subpopulation(df: pd.DataFrame, subset_type: str, data_dir: str) -> pd.DataFrame:
    """
    Subset the DataFrame by 'none', 'diabetes', or 'ckd'.
    If unknown subset, returns the full data.
    """
    st = subset_type.lower().strip()
    if st == "none":
        return df
    elif st == "diabetes":
        return subset_diabetes(df, data_dir)
    elif st == "ckd":
        return subset_ckd(df, data_dir)
    else:
        logger.warning(f"Unknown subset='{subset_type}', returning full dataset.")
        return df

# comorbidity_validation.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#  A validation script that compares final TabNet-based health severity outputs
#  against Charlson Comorbidity Index (CCI) and Elixhauser Comorbidity Index (ECI),
#  using the consistent merge approach from data_preprocessing.py.
#
# Steps:
#   1) Loads final cluster/prediction CSV (columns: 'Id', 'Predicted_Health_Index', 'Cluster').
#   2) Loads conditions.csv (keyed by 'PATIENT').
#   3) Computes Charlson & Elixhauser indices. Returns data keyed by 'PATIENT'.
#   4) Merges them with final TabNet outputs, using left_on='Id', right_on='PATIENT';
#      drops 'PATIENT', keeps 'Id' as main identifier.
#   5) Performs descriptive stats, correlations, ANOVA or Kruskal across clusters.
#   6) Saves outputs in a 'Validations/' subfolder.
#
# Usage:
#   python comorbidity_validation.py
#
# Requirements:
#   - charlson_comorbidity.py and elixhauser_comorbidity.py
#   - final clusters/predictions CSV with columns: "Id", "Predicted_Health_Index", "Cluster"
#   - conditions.csv with columns: "PATIENT", "CODE" (and more if needed)
#   - comorbidity mapping files for Charlson if needed (e.g. res195-comorbidity-cci-snomed.csv in Data)
#

import os
import logging
import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal

# Local imports
import charlson_comorbidity
import elixhauser_comorbidity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################
# USER CONFIG: Paths & Filenames
########################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) The final TabNet model's cluster/prediction output
FINAL_CLUSTERS_FILE = os.path.join(SCRIPT_DIR, "Data", "finals", "final_composite_none_tabnet_clusters.csv")
# This CSV is expected to have columns like: "Id", "Predicted_Health_Index", "Cluster"

# 2) Conditions CSV
CONDITIONS_FILE = os.path.join(SCRIPT_DIR, "Data", "conditions.csv")
# Must have columns: "PATIENT", "CODE"

# 3) Directory to save validation outputs
VALIDATION_DIR = os.path.join(SCRIPT_DIR, "Validations")
os.makedirs(VALIDATION_DIR, exist_ok=True)

def main():
    # -------------------------------
    # 1) Load final TabNet predictions/clusters
    # -------------------------------
    if not os.path.exists(FINAL_CLUSTERS_FILE):
        raise FileNotFoundError(f"Missing final clusters file at {FINAL_CLUSTERS_FILE}")
    df_tabnet = pd.read_csv(FINAL_CLUSTERS_FILE)
    logger.info(f"Loaded TabNet results shape={df_tabnet.shape}, columns={list(df_tabnet.columns)}")

    required_cols = {'Id', 'Predicted_Health_Index'}
    if not required_cols.issubset(df_tabnet.columns):
        raise KeyError(f"Your final clusters file must contain at least {required_cols}.")

    # -------------------------------
    # 2) Load conditions and compute Charlson/Elixhauser
    # -------------------------------
    if not os.path.exists(CONDITIONS_FILE):
        raise FileNotFoundError(f"Conditions file missing at {CONDITIONS_FILE}")
    conditions = pd.read_csv(CONDITIONS_FILE)
    logger.info(f"Loaded conditions shape={conditions.shape}, columns={list(conditions.columns)}")

    # Must have columns "PATIENT" (the patient ID) and "CODE" (the SNOMED code).
    if "PATIENT" not in conditions.columns:
        raise KeyError("conditions.csv must have a 'PATIENT' column.")
    if "CODE" not in conditions.columns:
        raise KeyError("conditions.csv must have a 'CODE' column.")

    # Compute Charlson
    cci_map = charlson_comorbidity.load_cci_mapping(os.path.join(SCRIPT_DIR, "Data"))
    df_cci = charlson_comorbidity.compute_cci(conditions, cci_map)
    logger.info(f"Charlson CCI computed: shape={df_cci.shape}, columns={list(df_cci.columns)}")

    # Compute Elixhauser
    df_eci = elixhauser_comorbidity.compute_eci(conditions)
    logger.info(f"Elixhauser ECI computed: shape={df_eci.shape}, columns={list(df_eci.columns)}")

    # -------------------------------
    # 3) Merge all together
    # -------------------------------
    # In data_preprocessing.py, we always keep 'Id' for patient data,
    # and conditions reference 'PATIENT'. So we do:
    #   left_on='Id', right_on='PATIENT', then drop 'PATIENT'.
    # This ensures we maintain 'Id' as the unique ID in the final merged dataset.

    # Merge final TabNet with Charlson
    # df_cci: columns = ['PATIENT','CharlsonIndex']
    df_merged = df_tabnet.merge(df_cci, how='left', left_on='Id', right_on='PATIENT')
    df_merged.drop(columns=['PATIENT'], inplace=True)

    # Merge with Elixhauser
    # df_eci: columns = ['PATIENT','ElixhauserIndex']
    df_merged = df_merged.merge(df_eci, how='left', left_on='Id', right_on='PATIENT')
    df_merged.drop(columns=['PATIENT'], inplace=True)

    # Fill missing Charlson or Elixhauser with 0 if no comorbidities
    df_merged['CharlsonIndex'] = df_merged['CharlsonIndex'].fillna(0)
    df_merged['ElixhauserIndex'] = df_merged['ElixhauserIndex'].fillna(0)

    logger.info(f"Merged shape={df_merged.shape}, columns={list(df_merged.columns)}")
    logger.info("[MERGE] Retaining 'Id' as the patient identifier. Charlson/Elixhauser merged successfully.")

    # -------------------------------
    # 4) Descriptive Stats
    # -------------------------------
    numeric_cols = ['Predicted_Health_Index', 'CharlsonIndex', 'ElixhauserIndex']
    desc_stats = df_merged[numeric_cols].describe().T  # transpose for clarity

    desc_stats_path = os.path.join(VALIDATION_DIR, "validation_stats.csv")
    desc_stats.to_csv(desc_stats_path)
    logger.info(f"[STATS] Descriptive stats saved -> {desc_stats_path}")

    # -------------------------------
    # 5) Correlations
    # -------------------------------
    # We'll do pairs: 
    #  (Predicted_Health_Index, CharlsonIndex), 
    #  (Predicted_Health_Index, ElixhauserIndex),
    #  (CharlsonIndex, ElixhauserIndex).

    pairs = [
        ('Predicted_Health_Index', 'CharlsonIndex'),
        ('Predicted_Health_Index', 'ElixhauserIndex'),
        ('CharlsonIndex', 'ElixhauserIndex')
    ]
    corrs = []

    for (col_x, col_y) in pairs:
        sub = df_merged[[col_x, col_y]].dropna()
        if len(sub) < 2:
            # Not enough data
            corrs.append({
                "Variable_X": col_x,
                "Variable_Y": col_y,
                "Pearson_R": np.nan, "Pearson_pvalue": np.nan,
                "Spearman_R": np.nan, "Spearman_pvalue": np.nan
            })
            continue

        pear_r, pear_p = pearsonr(sub[col_x], sub[col_y])
        spear_r, spear_p = spearmanr(sub[col_x], sub[col_y])
        corrs.append({
            "Variable_X": col_x,
            "Variable_Y": col_y,
            "Pearson_R": pear_r,
            "Pearson_pvalue": pear_p,
            "Spearman_R": spear_r,
            "Spearman_pvalue": spear_p
        })

    df_corrs = pd.DataFrame(corrs)
    corr_path = os.path.join(VALIDATION_DIR, "validation_correlations.csv")
    df_corrs.to_csv(corr_path, index=False)
    logger.info(f"[CORRELATIONS] Pearson & Spearman saved -> {corr_path}")

    # -------------------------------
    # 6) (Optional) ANOVA / Kruskal across TabNet clusters
    # -------------------------------
    # If "Cluster" is a meaningful grouping in df_merged, we can test if
    # CharlsonIndex or ElixhauserIndex differ among clusters.

    results_anova = []
    if 'Cluster' in df_merged.columns:
        cluster_col = 'Cluster'
        for measure in ['CharlsonIndex', 'ElixhauserIndex', 'Predicted_Health_Index']:
            groups = []
            for cluster_id, grp in df_merged.groupby(cluster_col):
                groups.append(grp[measure].dropna().values)
            if len(groups) > 1:
                # ANOVA
                f_stat, p_val = f_oneway(*groups)
                results_anova.append({
                    "Measure": measure,
                    "Test": "ANOVA",
                    "F_stat": f_stat,
                    "p_value": p_val
                })
                # Kruskal
                h_stat, p_val2 = kruskal(*groups)
                results_anova.append({
                    "Measure": measure,
                    "Test": "Kruskal-Wallis",
                    "H_stat": h_stat,
                    "p_value": p_val2
                })
            else:
                # only one cluster => no variation
                results_anova.append({
                    "Measure": measure,
                    "Test": "ANOVA",
                    "F_stat": np.nan,
                    "p_value": np.nan
                })
                results_anova.append({
                    "Measure": measure,
                    "Test": "Kruskal-Wallis",
                    "H_stat": np.nan,
                    "p_value": np.nan
                })

        df_anova = pd.DataFrame(results_anova)
        anova_path = os.path.join(VALIDATION_DIR, "validation_anova.csv")
        df_anova.to_csv(anova_path, index=False)
        logger.info(f"[ANOVA] ANOVA & Kruskal results saved -> {anova_path}")
    else:
        logger.info("[ANOVA] 'Cluster' column not found. Skipping cluster-based stats.")

    # Done
    logger.info("[DONE] Comorbidity validation script completed successfully.")
    logger.info(f"Validation outputs saved in: {VALIDATION_DIR}")

if __name__ == "__main__":
    main()

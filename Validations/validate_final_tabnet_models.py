# validate_final_tabnet_models.py
# Author: Imran Feisal
# Date: 21/01/2025
#
# Description:
#   A validation script that compares each of the three final TabNet models'
#   predictions/clusters against Charlson (CCI) & Elixhauser (ECI) indices.
#
#   Specifically, for each final model:
#     1) Load its final cluster CSV (e.g. "Id", "Predicted_Health_Index", "Cluster").
#     2) Compute Charlson & Elixhauser from "conditions.csv" using your existing
#        "charlson_comorbidity.py" & "elixhauser_comorbidity.py".
#     3) Merge them on patient ID ("Id" vs. "PATIENT").
#     4) Perform correlations, descriptive stats, ANOVA/Kruskal across clusters.
#     5) Generate scatter/box/violin plots.
#     6) Save everything in "Data/validations/<model_id>/".
#
# Usage:
#   python validate_final_tabnet_models.py

import os
import logging
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Local imports for CCI/ECI.
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.append(PROJECT_ROOT)

import charlson_comorbidity
import elixhauser_comorbidity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################
# CONFIG
########################

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
VALIDATIONS_DIR = os.path.join(DATA_DIR, "validations")
os.makedirs(VALIDATIONS_DIR, exist_ok=True)

# Where your conditions.csv lives
CONDITIONS_CSV = os.path.join(DATA_DIR, "conditions.csv")

# These are the three final model outputs for "diabetes", "ckd", and "none".
# Adjust names/paths if needed. Each CSV must have: "Id", "Predicted_Health_Index", "Cluster" (optional).
FINAL_MODEL_CSVS = {
    "combined_diabetes_tabnet":    "finals/combined_diabetes_tabnet/combined_diabetes_tabnet_clusters.csv",
    "combined_all_ckd_tabnet":     "finals/combined_all_ckd_tabnet/combined_all_ckd_tabnet_clusters.csv",
    "combined_none_tabnet":        "finals/combined_none_tabnet/combined_none_tabnet_clusters.csv"
}

########################
# FUNCTIONS
########################

def load_and_merge_comorbidity(final_csv: str) -> pd.DataFrame:
    """
    Loads a final model's cluster CSV, merges with Charlson & Elixhauser indices
    computed from conditions.csv, then returns the merged DataFrame.
    """

    # 1) Load final TabNet predictions/clusters
    df_model = pd.read_csv(final_csv)
    logger.info(f"Loaded model output from {final_csv}: shape={df_model.shape}")

    # Check for required columns
    if "Id" not in df_model.columns or "Predicted_Health_Index" not in df_model.columns:
        raise KeyError(f"{final_csv} must contain 'Id' and 'Predicted_Health_Index' columns.")
    # 'Cluster' is optional, but let's check
    has_cluster = ("Cluster" in df_model.columns)

    # 2) Compute Charlson & Elixhauser from conditions
    if not os.path.exists(CONDITIONS_CSV):
        raise FileNotFoundError(f"Cannot find conditions.csv at {CONDITIONS_CSV}")

    conditions = pd.read_csv(CONDITIONS_CSV)
    if "PATIENT" not in conditions.columns or "CODE" not in conditions.columns:
        raise KeyError("conditions.csv must have 'PATIENT' and 'CODE' columns.")

    # Compute Charlson
    cci_map = charlson_comorbidity.load_cci_mapping(DATA_DIR)
    df_cci = charlson_comorbidity.compute_cci(conditions, cci_map)

    # Compute Elixhauser
    df_eci = elixhauser_comorbidity.compute_eci(conditions)

    # Merge
    df_merged = df_model.merge(df_cci, how="left", left_on="Id", right_on="PATIENT")
    df_merged.drop(columns=["PATIENT"], inplace=True, errors="ignore")
    df_merged = df_merged.merge(df_eci, how="left", left_on="Id", right_on="PATIENT")
    df_merged.drop(columns=["PATIENT"], inplace=True, errors="ignore")

    # Fill missing
    df_merged["CharlsonIndex"] = df_merged["CharlsonIndex"].fillna(0)
    df_merged["ElixhauserIndex"] = df_merged["ElixhauserIndex"].fillna(0)

    logger.info(f"Merged shape={df_merged.shape} after adding CCI/ECI.")
    return df_merged


def save_correlations(df: pd.DataFrame, out_path: str):
    """
    Computes Pearson & Spearman correlations among:
      - Predicted_Health_Index
      - CharlsonIndex
      - ElixhauserIndex
    Saves them to a CSV at out_path.
    Also returns the correlation DataFrame.
    """
    pairs = [
        ("Predicted_Health_Index", "CharlsonIndex"),
        ("Predicted_Health_Index", "ElixhauserIndex"),
        ("CharlsonIndex", "ElixhauserIndex"),
    ]
    corrs = []
    for (xcol, ycol) in pairs:
        data_xy = df[[xcol, ycol]].dropna()
        if len(data_xy) < 2:
            corrs.append({
                "VarX": xcol, "VarY": ycol,
                "Pearson_R": np.nan, "Pearson_pvalue": np.nan,
                "Spearman_R": np.nan, "Spearman_pvalue": np.nan
            })
            continue
        pear_r, pear_p = pearsonr(data_xy[xcol], data_xy[ycol])
        spear_r, spear_p = spearmanr(data_xy[xcol], data_xy[ycol])
        corrs.append({
            "VarX": xcol, "VarY": ycol,
            "Pearson_R": pear_r, "Pearson_pvalue": pear_p,
            "Spearman_R": spear_r, "Spearman_pvalue": spear_p
        })

    df_corrs = pd.DataFrame(corrs)
    df_corrs.to_csv(out_path, index=False)
    return df_corrs


def boxplot_by_cluster(df: pd.DataFrame, column: str, cluster_col: str, out_file: str):
    """
    Generates a boxplot of 'column' grouped by 'cluster_col' in df,
    saves it to out_file. If cluster_col not present or only one cluster,
    it does nothing.
    """
    if cluster_col not in df.columns or df[cluster_col].nunique() < 2:
        return

    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, x=cluster_col, y=column, palette="Set2")
    plt.title(f"{column} by {cluster_col}")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def scatter_plot(df: pd.DataFrame, xcol: str, ycol: str, out_file: str, hue_col: str = None):
    """
    Generates a scatter plot of df[xcol] vs df[ycol], optional hue_col.
    """
    if xcol not in df.columns or ycol not in df.columns:
        return

    plt.figure(figsize=(6,5))
    if hue_col and hue_col in df.columns:
        sns.scatterplot(data=df, x=xcol, y=ycol, hue=hue_col, palette="viridis")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(data=df, x=xcol, y=ycol)
    plt.title(f"{ycol} vs {xcol}")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def distribution_plot(df: pd.DataFrame, column: str, out_file: str):
    """
    Generates a simple histogram + KDE of the given column.
    """
    if column not in df.columns:
        return

    plt.figure(figsize=(7,5))
    sns.histplot(df[column], kde=True, color="blue")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def anova_kruskal_across_clusters(df: pd.DataFrame, cluster_col: str, measure: str):
    """
    If 'cluster_col' is in df, perform one-way ANOVA and Kruskal-Wallis on the
    given 'measure' across cluster groups. Return a dict of stats.
    """
    if cluster_col not in df.columns or df[cluster_col].nunique() < 2:
        return {"Measure": measure, "Test": "ANOVA", "F_stat": np.nan, "p_value": np.nan}

    groups = [grp[measure].dropna().values for _, grp in df.groupby(cluster_col)]
    if len(groups) < 2:
        return {"Measure": measure, "Test": "ANOVA", "F_stat": np.nan, "p_value": np.nan}

    # ANOVA
    f_stat, p_val = f_oneway(*groups)
    # Kruskal
    h_stat, p_val2 = kruskal(*groups)

    return {
        "ANOVA_F": f_stat, "ANOVA_p": p_val,
        "Kruskal_H": h_stat, "Kruskal_p": p_val2
    }


def main():
    # Ensure conditions.csv is present
    if not os.path.exists(CONDITIONS_CSV):
        raise FileNotFoundError(f"Cannot find conditions.csv at {CONDITIONS_CSV}")

    # Loop over each final model CSV
    for model_id, rel_path in FINAL_MODEL_CSVS.items():
        logger.info(f"=== Validating model: {model_id} ===")
        csv_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(csv_path):
            logger.warning(f"[{model_id}] Missing final cluster CSV at {csv_path} - skipping.")
            continue

        # Subfolder for this model's validation outputs
        model_val_dir = os.path.join(VALIDATIONS_DIR, model_id)
        os.makedirs(model_val_dir, exist_ok=True)

        # Merge final predictions/clusters with CCI/ECI
        df_merged = load_and_merge_comorbidity(csv_path)

        # Basic descriptive stats
        numeric_cols = ["Predicted_Health_Index", "CharlsonIndex", "ElixhauserIndex"]
        desc_stats = df_merged[numeric_cols].describe().T
        desc_stats_file = os.path.join(model_val_dir, f"{model_id}_desc_stats.csv")
        desc_stats.to_csv(desc_stats_file)
        logger.info(f"[{model_id}] Descriptive stats saved -> {desc_stats_file}")

        # Correlations
        corr_file = os.path.join(model_val_dir, f"{model_id}_correlations.csv")
        df_corrs = save_correlations(df_merged, corr_file)
        logger.info(f"[{model_id}] Correlations saved -> {corr_file}")

        # ANOVA / Kruskal across clusters
        # (if "Cluster" is present & >1 unique value)
        anova_results = []
        if "Cluster" in df_merged.columns and df_merged["Cluster"].nunique() > 1:
            for measure in ["Predicted_Health_Index", "CharlsonIndex", "ElixhauserIndex"]:
                stats_dict = anova_kruskal_across_clusters(df_merged, "Cluster", measure)
                stats_dict["Measure"] = measure
                anova_results.append(stats_dict)

            df_anova = pd.DataFrame(anova_results)
            anova_file = os.path.join(model_val_dir, f"{model_id}_anova.csv")
            df_anova.to_csv(anova_file, index=False)
            logger.info(f"[{model_id}] ANOVA/Kruskal results -> {anova_file}")

        # Visualisations
        # 1) Distribution of predicted HI, CCI, ECI
        for c in numeric_cols:
            dist_file = os.path.join(model_val_dir, f"{model_id}_dist_{c}.png")
            distribution_plot(df_merged, c, dist_file)

        # 2) Boxplot of each measure by cluster
        if "Cluster" in df_merged.columns and df_merged["Cluster"].nunique() > 1:
            for c in numeric_cols:
                box_file = os.path.join(model_val_dir, f"{model_id}_box_{c}.png")
                boxplot_by_cluster(df_merged, c, "Cluster", box_file)

        # 3) Scatter plots (HI vs. Charlson, HI vs. ECI, Charlson vs. ECI)
        scatter_pairs = [
            ("Predicted_Health_Index", "CharlsonIndex"),
            ("Predicted_Health_Index", "ElixhauserIndex"),
            ("CharlsonIndex", "ElixhauserIndex")
        ]
        for xcol, ycol in scatter_pairs:
            scatter_path = os.path.join(model_val_dir, f"{model_id}_scatter_{xcol}_vs_{ycol}.png")
            scatter_plot(df_merged, xcol, ycol, scatter_path, hue_col="Cluster")

        logger.info(f"[{model_id}] Validation visuals saved in {model_val_dir}\n")

    logger.info("[All Done] Validation across final TabNet models completed.")


if __name__ == "__main__":
    main()

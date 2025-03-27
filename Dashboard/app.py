# app.py
# --------------------------------
# Enhanced so each model's df includes Health_Index from base data
# ensuring the Model Details scatter plot works for all subpopulations.

import os
import json
import base64
import logging
import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xai_formatter import format_explanation

# Add these imports at the top with the other imports
import gc
from sklearn.cluster import KMeans
import warnings
import psutil
import math

# Add these imports at the top of the file
from flask_caching import Cache
from memory_utils import MemoryMonitor, DataSampler
import time
from dash.dash import no_update
import dask.dataframe as dd

# -----------------------------
# Logging & Directories Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
FINALS_DIR = os.path.join(DATA_DIR, "finals")
EXPLAIN_XAI_DIR = os.path.join(DATA_DIR, "explain_xai")

PICKLE_ALL = os.path.join(DATA_DIR, "patient_data_with_all_indices.pkl")
CSV_PATIENTS = os.path.join(DATA_DIR, "patients.csv")

# The "none" subpopulation is labeled "General" in the dashboard
final_groups = [
    {"model": "combined_diabetes_tabnet", "label": "Diabetes"},
    {"model": "combined_all_ckd_tabnet", "label": "CKD"},
    {"model": "combined_none_tabnet", "label": "General"}
]

# -----------------------------
# Load Base Data
# -----------------------------
if os.path.exists(PICKLE_ALL):
    # Use dask for chunked reading of large files
    df_all = dd.read_pickle(PICKLE_ALL).compute()
    # Convert to smaller dtypes where possible to save memory
    for col in df_all.select_dtypes(include=['float64']).columns:
        df_all[col] = df_all[col].astype('float32')
    for col in df_all.select_dtypes(include=['int64']).columns:
        df_all[col] = df_all[col].astype('int32')
    logger.info(f"Loaded enriched data from {PICKLE_ALL}.")
    # Only keep essential columns initially for faster loading
    essential_cols = ["Id", "BIRTHDATE", "AGE", "GENDER", "INCOME", 
                      "Health_Index", "CharlsonIndex", "ElixhauserIndex",
                      "Risk_Category", "ZIP", "LAT", "LON"]
    df_all = df_all[[col for col in essential_cols if col in df_all.columns]]
else:
    logger.warning("Enriched pickle not found; falling back to patients CSV.")
    df_all = pd.read_csv(CSV_PATIENTS)

    # Minimal fallback transformations
    df_all["BIRTHDATE"] = pd.to_datetime(df_all["BIRTHDATE"], errors="coerce")
    df_all["AGE"] = ((pd.Timestamp("today") - df_all["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    np.random.seed(42)
    df_all["Health_Index"] = np.random.uniform(1, 10, len(df_all)).round(2)
    df_all["CharlsonIndex"] = np.random.uniform(0, 5, len(df_all)).round(2)
    df_all["ElixhauserIndex"] = np.random.uniform(0, 15, len(df_all)).round(2)
    df_all["Cluster"] = np.random.choice([0, 1, 2], len(df_all))
    df_all["Predicted_Health_Index"] = (
        df_all["Health_Index"] + np.random.normal(0, 0.5, len(df_all))
    ).round(2)
    df_all["Actual"] = df_all["Health_Index"]

# Merge location or income if missing
missing_loc_cols = any(col not in df_all.columns for col in ["ZIP", "LAT", "LON"])
if missing_loc_cols and os.path.exists(CSV_PATIENTS):
    df_loc = pd.read_csv(CSV_PATIENTS, usecols=["Id", "BIRTHDATE", "ZIP", "LAT", "LON", "INCOME"])
    df_loc["BIRTHDATE"] = pd.to_datetime(df_loc["BIRTHDATE"], errors="coerce")
    df_loc["AGE"] = ((pd.Timestamp("today") - df_loc["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    df_all = pd.merge(df_all, df_loc, on="Id", how="left", suffixes=("", "_csv"))

# Make sure "Health_Index" is present
if "Health_Index" not in df_all.columns:
    df_all["Health_Index"] = np.random.uniform(1, 10, len(df_all)).round(2)

# Calculate age groups for demographic analysis
df_all["Age_Group"] = pd.cut(
    df_all["AGE"] if "AGE" in df_all.columns else 0, 
    bins=[0, 18, 35, 50, 65, 80, 120],
    labels=["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]
)

# Calculate risk categories based on health indices
df_all["Risk_Category"] = pd.cut(
    df_all["Health_Index"], 
    bins=[0, 3, 6, 8, 10],
    labels=["High Risk", "Moderate Risk", "Low Risk", "Very Low Risk"]
)

# -----------------------------
# Helper Functions
# -----------------------------
def encode_image(image_file):
    """Base64-encode an image file for embedding in HTML."""
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return None

# Initialize cache
cache = Cache()

# Replace load_final_model_outputs with a cached version
@cache.memoize()
def load_final_model_outputs():
    """
    Cached function to load model outputs - won't reload if called again with same args
    """
    models_data = {}

    # We'll keep a small subset of base columns to merge in.
    base_cols = ["Id", "Health_Index", "CharlsonIndex", "ElixhauserIndex"]

    for grp in final_groups:
        model_name = grp["model"]
        label_name = grp["label"]
        model_dir = os.path.join(FINALS_DIR, model_name)

        preds_csv    = os.path.join(model_dir, f"{model_name}_predictions.csv")
        clusters_csv = os.path.join(model_dir, f"{model_name}_clusters.csv")

        metrics_json = os.path.join(model_dir, f"{model_name}_metrics.json")
        cluster_json = os.path.join(model_dir, f"{model_name}_clusters.json")

        tsne_png = os.path.join(model_dir, f"{model_name}_tsne.png")
        umap_png = os.path.join(model_dir, f"{model_name}_umap.png")

        # Merge predictions + clusters
        if os.path.exists(preds_csv) and os.path.exists(clusters_csv):
            df_preds    = pd.read_csv(preds_csv)
            df_clusters = pd.read_csv(clusters_csv)
            df_model    = pd.merge(df_preds, df_clusters, on="Id", how="outer", suffixes=("", "_cluster"))
            df_model.rename(
                columns={
                    "Predicted_Health_Index": "PredictedHI_final",
                    "Cluster": "Cluster_final"
                },
                inplace=True
            )
            # Now also merge with the base data so we have "Health_Index"
            df_model = pd.merge(
                df_model,
                df_all[base_cols],
                on="Id",
                how="left"
            )
        else:
            df_model = pd.DataFrame()

        # Load & combine metrics
        if os.path.exists(metrics_json):
            with open(metrics_json, "r") as f:
                metrics_primary = json.load(f)
        else:
            metrics_primary = {"test_mse": "N/A", "test_r2": "N/A"}

        # Load cluster metrics
        if os.path.exists(cluster_json):
            with open(cluster_json, "r") as f:
                metrics_cluster = json.load(f)
            metrics_primary["Silhouette"]         = metrics_cluster.get("silhouette", "N/A")
            metrics_primary["Calinski_Harabasz"]  = metrics_cluster.get("calinski", "N/A")
            metrics_primary["Davies_Bouldin"]     = metrics_cluster.get("davies_bouldin", "N/A")
        else:
            metrics_primary.setdefault("Silhouette", "N/A")
            metrics_primary.setdefault("Calinski_Harabasz", "N/A")
            metrics_primary.setdefault("Davies_Bouldin", "N/A")

        # Encode t-SNE / UMAP images
        tsne_img = encode_image(tsne_png)
        umap_img = encode_image(umap_png)

        models_data[label_name] = {
            "df": df_model,
            "metrics": metrics_primary,
            "tsne_img": tsne_img,
            "umap_img": umap_img
        }

    return models_data

logger.info("Merging final model outputs with base data if needed.")
final_models_data = load_final_model_outputs()

# Optionally unify them in df_all:
list_models = []
for label, mdata in final_models_data.items():
    if not mdata["df"].empty:
        mdata["df"]["Group"] = label
        list_models.append(mdata["df"])
if list_models:
    df_models = pd.concat(list_models, ignore_index=True)
    df_all = pd.merge(
        df_all,
        df_models.drop(columns=["Health_Index", "CharlsonIndex", "ElixhauserIndex"], errors="ignore"),
        on="Id",
        how="left"
    )
    logger.info(f"Master df_all shape after merges: {df_all.shape}")

# -----------------------------
# Global KPIs
# -----------------------------
TOTAL_PATIENTS    = len(df_all)
AVG_AGE           = round(df_all["AGE"].mean(), 1) if "AGE" in df_all.columns else "N/A"
AVG_INCOME        = round(df_all["INCOME"].mean(), 0) if "INCOME" in df_all.columns else "N/A"
AVG_HEALTH_INDEX  = round(df_all["Health_Index"].mean(), 2) if "Health_Index" in df_all.columns else "N/A"
AVG_CHARLSON      = round(df_all["CharlsonIndex"].mean(), 2) if "CharlsonIndex" in df_all.columns else "N/A"
AVG_ELIXHAUSER    = round(df_all["ElixhauserIndex"].mean(), 2) if "ElixhauserIndex" in df_all.columns else "N/A"

# -----------------------------
# Color / Theme Config
# -----------------------------
nhs_colors = {
    "background": "#F7F7F7",
    "text": "#333333", 
    "primary": "#005EB8",
    "secondary": "#FFFFFF",
    "accent": "#00843D",
    "highlight": "#FFB81C",
    "risk_high": "#DA291C",
    "risk_medium": "#ED8B00",
    "risk_low": "#00843D",
    "risk_verylow": "#0072CE"
}
external_stylesheets = [dbc.themes.FLATLY]

# Color scales for consistent styling
risk_colors = [nhs_colors["risk_high"], nhs_colors["risk_medium"], 
               nhs_colors["risk_low"], nhs_colors["risk_verylow"]]

# -----------------------------
# Enhanced Visualization Functions
# -----------------------------
def apply_2decimal_format(fig):
    """
    Forces x-axis & y-axis ticks to show 2 decimals (ex: 3 -> 3.00).
    If you want to only show decimals if non-integer, see 'tickformat' ~r approach.
    """
    fig.update_layout(
        xaxis=dict(tickformat=".2f"),
        yaxis=dict(tickformat=".2f")
    )
    return fig

def create_demographic_charts(df):
    """Create demographic analysis charts"""
    # Gender distribution pie chart (if available)
    gender_fig = None
    if "GENDER" in df.columns:
        # Filter out NaN values
        df_gender = df.dropna(subset=["GENDER"])
        if len(df_gender) > 0:
            gender_counts = df_gender["GENDER"].value_counts().reset_index()
            gender_fig = px.pie(
                gender_counts, 
                values="count", 
                names="GENDER", 
                title="Gender Distribution",
                hole=0.4,
                color_discrete_sequence=[nhs_colors["primary"], nhs_colors["accent"]]
            )
            gender_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    
    # Age group breakdown
    age_fig = None
    if "Age_Group" in df.columns or "AGE" in df.columns:
        age_column = "Age_Group" if "Age_Group" in df.columns else "AGE"
        df_age = df.dropna(subset=[age_column])
        
        if len(df_age) > 0:
            age_fig = px.histogram(
                df_age, 
                x=age_column, 
                title="Age Distribution",
                color="Risk_Category" if "Risk_Category" in df.columns else None,
                color_discrete_sequence=risk_colors,
                category_orders={"Age_Group": ["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]},
                barmode="group"
            )
            age_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    
    # Risk distribution
    risk_fig = None
    if "Risk_Category" in df.columns:
        df_risk = df.dropna(subset=["Risk_Category"])
        if len(df_risk) > 0:
            risk_fig = px.pie(
                df_risk["Risk_Category"].value_counts().reset_index(),
                values="count",
                names="Risk_Category",
                title="Risk Distribution",
                hole=0.4,
                color="Risk_Category",
                color_discrete_map={
                    "High Risk": nhs_colors["risk_high"],
                    "Moderate Risk": nhs_colors["risk_medium"],
                    "Low Risk": nhs_colors["risk_low"],
                    "Very Low Risk": nhs_colors["risk_verylow"]
                }
            )
            risk_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    
    return gender_fig, age_fig, risk_fig

# -----------------------------
# RAM Optimization Functions
# -----------------------------
def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def log_memory_usage(label):
    """Log memory usage with a label for debugging"""
    logger.info(f"Memory usage at {label}: {get_memory_usage():.2f} MB")

def smart_sample_dataframe(df, max_points=5000, min_points=500, method='random'):
    """
    Intelligently sample a dataframe to reduce memory usage in visualizations
    
    Parameters:
    - df: DataFrame to sample
    - max_points: Maximum number of points to include in visualization
    - min_points: Minimum number of points to include (won't sample below this)
    - method: Sampling method ('random', 'stratified', 'cluster')
    
    Returns:
    - Sampled DataFrame
    """
    if df is None or df.empty:
        return df
    
    if len(df) <= max_points:
        return df  # No sampling needed
    
    sample_size = min(max_points, max(min_points, int(len(df) * 0.1)))
    
    if method == 'random':
        # Simple random sampling
        return df.sample(sample_size, random_state=42)
    
    elif method == 'stratified' and 'Risk_Category' in df.columns:
        # Stratified sampling by risk category to maintain distribution
        result = pd.DataFrame()
        for category in df['Risk_Category'].unique():
            category_df = df[df['Risk_Category'] == category]
            category_size = max(1, int(sample_size * len(category_df) / len(df)))
            result = pd.concat([result, category_df.sample(min(category_size, len(category_df)), random_state=42)])
        return result
    
    elif method == 'cluster' and df.select_dtypes(include=['number']).shape[1] >= 2:
        # Cluster-based sampling to maintain data structure
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            # Select a subset of numeric columns for clustering if we have many
            if len(numeric_cols) > 5:
                numeric_cols = numeric_cols[:5]
            
            # Handle NaN values
            cluster_data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Determine number of clusters based on sample size
            n_clusters = min(int(math.sqrt(sample_size)), 50)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_copy = df.copy()
                df_copy['cluster'] = kmeans.fit_predict(cluster_data)
                
                # Sample from each cluster proportionally
                result = pd.DataFrame()
                for cluster in range(n_clusters):
                    cluster_df = df_copy[df_copy['cluster'] == cluster]
                    cluster_size = max(1, int(sample_size * len(cluster_df) / len(df)))
                    result = pd.concat([result, cluster_df.sample(min(cluster_size, len(cluster_df)), random_state=42)])
                
                return result.drop(columns=['cluster'])
        except Exception as e:
            logger.warning(f"Cluster sampling failed, falling back to random: {e}")
            return df.sample(sample_size, random_state=42)
    
    # Default to random sampling
    return df.sample(sample_size, random_state=42)

def release_memory():
    """Force garbage collection to free up memory"""
    gc.collect()

def sanitize_datatable_values(df, max_rows=1000):
    """
    More efficient sanitization with row limiting
    """
    if df is None or df.empty:
        return df
    
    # Limit rows for faster rendering
    df_sample = df.head(max_rows)
    df_copy = df_sample.copy()
    
    for col in df_copy.columns:
        # Only check a small sample for non-scalar values
        sample = df_copy[col].dropna().head(5)
        if any(isinstance(x, (list, dict, np.ndarray)) for x in sample):
            df_copy[col] = df_copy[col].astype(str)
        # Use more efficient type conversion for numeric columns
        elif df_copy[col].dtype.kind in 'iuf':
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    return df_copy

def create_indices_comparison(df, max_points=3000):
    """Create a comparison of health indices with memory optimization"""
    if all(col in df.columns for col in ["Health_Index", "CharlsonIndex", "ElixhauserIndex"]):
        # Filter out NaN values to avoid plotly errors
        df_filtered = df.dropna(subset=["Health_Index", "CharlsonIndex", "ElixhauserIndex"])
        
        # Apply smart sampling to reduce memory usage
        df_filtered = smart_sample_dataframe(df_filtered, max_points=max_points, method='cluster')
        
        # Handle AGE for sizing - replace NaN with median or drop if all NaN
        if "AGE" in df_filtered.columns:
            if df_filtered["AGE"].isna().all():
                size_param = None  # Don't use AGE if all values are NaN
            else:
                # Fill NaN values with median age for visualization purposes only
                df_filtered["AGE_for_plot"] = df_filtered["AGE"].fillna(df_filtered["AGE"].median())
                size_param = "AGE_for_plot"
        else:
            size_param = None
            
        # Create scatter plot comparing indices
        fig = px.scatter(
            df_filtered,
            x="Health_Index",
            y="CharlsonIndex",
            color="ElixhauserIndex",
            color_continuous_scale="Turbo",
            opacity=0.7,
            size=size_param,
            hover_data=["Id", "Risk_Category"] + (["AGE"] if "AGE" in df_filtered.columns else []),
            title=f"Health Indices Comparison (Sampled: {len(df_filtered)} of {len(df)} points)"
        )
        apply_2decimal_format(fig)
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        
        # Release memory after creating plot
        df_filtered = None
        release_memory()
        
        return fig
    return None

def create_income_health_chart(df, max_points=3000):
    """Create a chart showing relationship between income and health with memory optimization"""
    if "INCOME" in df.columns and "Health_Index" in df.columns:
        # Filter out rows with missing values
        df_filtered = df.dropna(subset=["INCOME", "Health_Index"])
        
        if len(df_filtered) == 0:
            return go.Figure().update_layout(
                title="No valid income-health data available"
            )
            
        # Apply smart sampling for large datasets
        df_filtered = smart_sample_dataframe(df_filtered, max_points=max_points, method='stratified')
            
        fig = px.scatter(
            df_filtered,
            x="INCOME",
            y="Health_Index",
            color="Risk_Category" if "Risk_Category" in df_filtered.columns else None,
            color_discrete_sequence=risk_colors if "Risk_Category" in df_filtered.columns else None,
            opacity=0.7,
            trendline="ols",
            title=f"Income vs Health Index (Sampled: {len(df_filtered)} of {len(df)} points)"
        )
        apply_2decimal_format(fig)
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        
        # Release memory
        df_filtered = None
        release_memory()
        
        return fig
    return None

def create_compact_map(df, height=300, max_points=2000):
    """Create a compact map visualization with memory optimization"""
    if "LAT" not in df.columns or "LON" not in df.columns:
        return None
    
    # Filter out rows with missing lat/lon
    df_map = df.dropna(subset=["LAT", "LON"])
    
    if len(df_map) == 0:
        return go.Figure().update_layout(
            title="No valid location data available",
            height=height
        )
    
    # Apply intelligent sampling for maps - use cluster sampling to maintain geographic distribution
    original_count = len(df_map)
    df_map = smart_sample_dataframe(df_map, max_points=max_points, method='cluster')
    
    color_column = "Risk_Category" if "Risk_Category" in df_map.columns else "Health_Index"
    
    fig = px.scatter_mapbox(
        df_map,
        lat="LAT",
        lon="LON",
        color=color_column,
        color_discrete_sequence=risk_colors if color_column == "Risk_Category" else None,
        color_continuous_scale=px.colors.sequential.Viridis if color_column == "Health_Index" else None,
        zoom=5,
        height=height,
        hover_data=["Id", "Health_Index"] + (["AGE"] if "AGE" in df_map.columns else [])
    )
    fig.update_layout(
        mapbox_style="carto-positron", 
        margin={"r":0, "t":40, "l":0, "b":0},
        title=f"Patient Geographic Distribution (Sampled: {len(df_map)} of {original_count} points)"
    )
    
    # Release memory
    df_map = None
    release_memory()
    
    return fig

def create_correlation_matrix(df, max_cols=15):
    """Create an optimized correlation matrix for large datasets"""
    # Only include numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Select the most relevant columns if we have too many
    if len(numeric_df.columns) > max_cols:
        # Focus on key health indices and other important metrics
        priority_cols = [col for col in ["Health_Index", "CharlsonIndex", "ElixhauserIndex", 
                               "AGE", "INCOME", "PredictedHI_final"] if col in numeric_df.columns]
        # Add other columns to reach max_cols
        other_cols = [col for col in numeric_df.columns if col not in priority_cols]
        selected_cols = priority_cols + other_cols[:max_cols - len(priority_cols)]
        numeric_df = numeric_df[selected_cols]
    
    # Compute the correlation matrix
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        corr_fig = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto", 
            title=f"Feature Correlations (Top {len(corr)} features)",
            color_continuous_scale=px.colors.diverging.RdBu,
            zmin=-1, 
            zmax=1
        )
        apply_2decimal_format(corr_fig)
        corr_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        return corr_fig
    return None

# Add after the create_indices_comparison function and before create_income_health_chart

def create_health_trend_chart(df, max_points=3000):
    """Create a chart showing health trend by age group with memory optimization"""
    if "Age_Group" in df.columns and "Health_Index" in df.columns:
        # Drop rows with missing values in required columns
        df_filtered = df.dropna(subset=["Age_Group", "Health_Index"])
        
        if len(df_filtered) == 0:
            return go.Figure().update_layout(
                title="No valid data for health trend chart"
            )
        
        # For line charts showing aggregates, we usually don't need to sample
        # But we can optimize large datasets by pre-aggregating before visualization
        if len(df_filtered) > max_points:
            # For age group trends, we're only looking at means by group
            # So we can pre-aggregate rather than sampling
            health_by_age = df_filtered.groupby("Age_Group")["Health_Index"].agg(
                ['mean', 'count', 'std']
            ).reset_index()
            
            # Add count information for hover data
            health_by_age.rename(columns={'mean': 'Health_Index'}, inplace=True)
        else:
            # For smaller datasets, just do simple grouping
            health_by_age = df_filtered.groupby("Age_Group")["Health_Index"].mean().reset_index()
        
        # Create the line chart
        fig = px.line(
            health_by_age,
            x="Age_Group",
            y="Health_Index",
            markers=True,
            title="Average Health Index by Age Group",
            category_orders={"Age_Group": ["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]},
            # Add hover data with patient counts if available
            hover_data=['count', 'std'] if 'count' in health_by_age.columns else None
        )
        apply_2decimal_format(fig)
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        
        # Release memory after creating plot
        df_filtered = None
        health_by_age = None
        release_memory()
        
        return fig
    return None

# After the definition of create_correlation_matrix, add:

def kpi_card(title, value, color=None):
    """Create a styled KPI card with a title and value"""
    return html.Div(
        style={
            "padding": "15px",
            "margin": "5px",
            "backgroundColor": nhs_colors["secondary"],
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "borderRadius": "5px",
            "textAlign": "center",
            "borderLeft": f"4px solid {color or nhs_colors['primary']}"
        },
        children=[
            html.H5(title, style={"color": nhs_colors["text"], "marginBottom": "8px", "fontSize": "14px"}),
            html.H3(value, style={"color": color or nhs_colors["primary"], "fontWeight": "bold", "margin": "0"})
        ]
    )
    
# Also add the collapsible_card function if it's missing:
def collapsible_card(title, content, id_prefix, initially_open=True):
    """Create a collapsible card section for the dashboard"""
    return html.Div(
        className="collapsible-card",
        style={
            "backgroundColor": nhs_colors["secondary"],
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "borderRadius": "5px",
            "marginBottom": "15px"
        },
        children=[
            html.Div(
                id=f"{id_prefix}-header",
                className="collapsible-header",
                style={
                    "padding": "12px 15px",
                    "backgroundColor": nhs_colors["primary"],
                    "color": nhs_colors["secondary"],
                    "borderRadius": "5px 5px 0 0",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center"
                },
                children=[
                    html.H4(title, style={"margin": 0, "fontSize": "16px"}),
                    html.I(className=f"fa {'fa-chevron-up' if initially_open else 'fa-chevron-down'}")
                ]
            ),
            html.Div(
                id=f"{id_prefix}-content",
                className="collapsible-content",
                style={
                    "padding": "15px",
                    "display": "block" if initially_open else "none"
                },
                children=content
            )
        ]
    )

# Also add the patient_modal definition:
patient_modal = html.Div([
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Patient Details")),
            dbc.ModalBody(id="patient-detail-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ms-auto")
            ),
        ],
        id="patient-modal",
        is_open=False,
        size="lg",
    ),
])

# ------ ADD THIS SECTION BEFORE THE FILTER PANEL DEFINITION ------
# Define model dropdown options early since they're needed for the filter panel
model_dropdown_options = [{"label": g["label"], "value": g["label"]} for g in final_groups]

# XAI model dropdown
xai_dropdown = dcc.Dropdown(
    id="xai-model-dropdown",
    options=model_dropdown_options,
    value=model_dropdown_options[0]["value"] if model_dropdown_options else None,
    clearable=False
)

# Create the main filter panel
filter_panel = html.Div([
    html.H5("Dashboard Filters", style={"color": nhs_colors["primary"], "marginBottom": "15px"}),
    html.Label("Model Group:"),
    dcc.Dropdown(
        id="global-model-dropdown",
        options=[{"label": "All", "value": "All"}] + model_dropdown_options,
        value="All",
        clearable=False
    ),
    html.Label("Risk Category:", style={"marginTop": "15px"}),
    dcc.Dropdown(
        id="global-risk-dropdown",
        options=[
            {"label": "All", "value": "All"},
            {"label": "High Risk", "value": "High Risk"},
            {"label": "Moderate Risk", "value": "Moderate Risk"},
            {"label": "Low Risk", "value": "Low Risk"},
            {"label": "Very Low Risk", "value": "Very Low Risk"}
        ],
        value="All",
        clearable=False
    ),
    html.Label("Age Range:", style={"marginTop": "15px"}),
    dcc.RangeSlider(
        id="age-range-slider",
        min=df_all["AGE"].min() if "AGE" in df_all.columns else 0,
        max=df_all["AGE"].max() if "AGE" in df_all.columns else 100,
        step=1,
        marks={i: str(i) for i in range(0, 101, 20)},
        value=[
            df_all["AGE"].min() if "AGE" in df_all.columns else 0,
            df_all["AGE"].max() if "AGE" in df_all.columns else 100
        ],
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Income Range:", style={"marginTop": "15px"}),
    dcc.RangeSlider(
        id="income-range-slider",
        min=df_all["INCOME"].min() if "INCOME" in df_all.columns else 0,
        max=df_all["INCOME"].max() if "INCOME" in df_all.columns else 100000,
        step=1000,
        marks={i: f"£{i:,}" for i in range(0, 100001, 25000)},
        value=[
            df_all["INCOME"].min() if "INCOME" in df_all.columns else 0,
            df_all["INCOME"].max() if "INCOME" in df_all.columns else 100000
        ],
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Health Index Range:", style={"marginTop": "15px"}),
    dcc.RangeSlider(
        id="health-index-slider",
        min=df_all["Health_Index"].min(),
        max=df_all["Health_Index"].max(),
        step=0.1,
        marks={i: str(i) for i in range(0, 11, 2)},
        value=[df_all["Health_Index"].min(), df_all["Health_Index"].max()],
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Div([
        dbc.Button(
            "Apply Filters", 
            id="apply-filters-btn", 
            color="primary", 
            className="mr-1",
            style={"marginRight": "5px"}
        ),
        dbc.Button(
            "Reset Filters", 
            id="reset-filters-btn", 
            color="secondary",
            className="mr-1",
            style={"marginRight": "5px"}
        ),
    ], style={"marginTop": "20px", "display": "flex", "justifyContent": "space-between"})
], style={
    "backgroundColor": nhs_colors["secondary"],
    "padding": "15px",
    "borderRadius": "5px",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
})

# Top KPI row
kpi_row = html.Div([
    kpi_card("Total Patients", f"{TOTAL_PATIENTS:,}"),
    kpi_card("Average Age", f"{AVG_AGE}", nhs_colors["accent"]),
    kpi_card("Average Income", f"£{AVG_INCOME:,.0f}"),
    kpi_card("Health Index", f"{AVG_HEALTH_INDEX:,.2f}", nhs_colors["primary"]),
    kpi_card("Charlson Index", f"{AVG_CHARLSON:,.2f}", nhs_colors["highlight"]),
    kpi_card("Elixhauser Index", f"{AVG_ELIXHAUSER:,.2f}", nhs_colors["risk_high"]),
], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between", "marginBottom": "15px"})

# Initialize the app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets + [
    {'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css', 'rel': 'stylesheet'}
])
server = app.server

# Initialize cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 3600  # Cache timeout in seconds (1 hour)
})

# After initializing the app (right after app = dash.Dash(...)), add caching
app.server.config.update({
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 3600  # Cache timeout in seconds (1 hour)
})
cache = Cache(app.server)

log_memory_usage("Before visualization creation")

# Create the demographic analysis charts with optimized memory usage
gender_fig, age_fig, risk_fig = create_demographic_charts(df_all)
log_memory_usage("After demographics charts")

# Create optimized indices comparison with sampling
indices_comparison_fig = create_indices_comparison(df_all)
log_memory_usage("After indices comparison")

# Create optimized health trend chart
health_trend_fig = create_health_trend_chart(df_all)
log_memory_usage("After health trend")

# Create optimized income vs health chart with sampling
income_health_fig = create_income_health_chart(df_all)
log_memory_usage("After income health chart")

# Create optimized map with sampling
compact_map_fig = create_compact_map(df_all)
log_memory_usage("After map creation")

# Generate optimized correlation matrix
corr_fig = create_correlation_matrix(df_all)
log_memory_usage("After correlation matrix")

# Initialize layout with placeholders for heavy components
app.layout = html.Div([
    # Header with title and subtitle
    html.Div([
        html.H1("VITAI Healthcare Analytics Dashboard", style={"color": nhs_colors["primary"], "marginBottom": "5px"}),
        html.P("Comprehensive Patient Health Analytics & Predictive Modeling", 
               style={"color": nhs_colors["text"], "marginBottom": "15px"}),
        # Top KPI Cards Row
        kpi_row,
    ], style={
        "backgroundColor": nhs_colors["secondary"],
        "padding": "15px 20px",
        "borderRadius": "5px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "marginBottom": "15px"
    }),

    # Main content area with sidebar and dashboard panels
    dcc.Tabs([
        dcc.Tab(label="Overview", children=[
            html.Div([
                # Left sidebar with filters
                html.Div([filter_panel], style={"width": "20%", "minWidth": "250px", "marginRight": "15px"}),
                
                # Demographics Panel - this is relatively lightweight
                html.Div([
                    collapsible_card(
                        "Patient Demographics & Risk Distribution", 
                        html.Div(id="demographics-content", children=[
                            dcc.Loading(id="demographics-loading")
                        ]),
                        "demographics",
                        initially_open=True
                    ),
                ], style={"width": "80%"})
            ], style={"display": "flex"})
        ]),
        
        # Health Indices tab - loads only when selected
        dcc.Tab(label="Health Indices", children=[
            html.Div(id="health-indices-tab-content", children=[
                dcc.Loading(id="health-indices-loading")
            ])
        ]),
        
        # Geographic tab - loads only when selected
        dcc.Tab(label="Geographic Distribution", children=[
            html.Div(id="geo-tab-content", children=[
                dcc.Loading(id="geo-loading")
            ])
        ]),
        
        # Model Performance tab - loads only when selected
        dcc.Tab(label="Model Performance", children=[
            html.Div(id="model-tab-content", children=[
                dcc.Loading(id="model-loading")
            ])
        ]),
        
        # XAI tab - loads only when selected
        dcc.Tab(label="XAI Insights", children=[
            html.Div(id="xai-tab-content", children=[
                dcc.Loading(id="xai-loading")
            ])
        ]),
        
        # Patient data tab - loads only when selected
        dcc.Tab(label="Patient Data", children=[
            html.Div(id="patient-tab-content", children=[
                dcc.Loading(id="patient-loading")
            ])
        ]),
    ]),
    
    # Store components for shared state
    dcc.Store(id="filtered-data-store", storage_type="memory"),
    dcc.Store(id="selected-model-store", storage_type="memory"),
    
    # Memory management components
    html.Div(id="memory-management", style={"display": "none"}),
    patient_modal
], style={"backgroundColor": nhs_colors["background"], "padding": "15px"})

# -----------------------------
# Callbacks
# -----------------------------
# Collapsible sections functionality
for section in ["demographics", "health-indices", "geographic", "model-performance", "xai-insights", "patient-data"]:
    @app.callback(
        [Output(f"{section}-content", "style"), Output(f"{section}-header", "children")],
        [Input(f"{section}-header", "n_clicks")],
        [State(f"{section}-content", "style"), State(f"{section}-header", "children")]
    )
    def toggle_collapse(n, current_style, current_header_children):
        if not n:
            raise PreventUpdate

        display_now = current_style.get("display", "block")
        new_display = "none" if display_now == "block" else "block"

        # Toggle icon
        # current_header_children should be [html.H4(...), html.I(...)]
        icon_element = current_header_children[1]
        icon_class = icon_element["props"].get("className", "")
        if "fa-chevron-up" in icon_class:
            icon_class = icon_class.replace("fa-chevron-up", "fa-chevron-down")
        else:
            icon_class = icon_class.replace("fa-chevron-down", "fa-chevron-up")
        icon_element["props"]["className"] = icon_class

        # Update style and header children
        current_style["display"] = new_display
        new_header_children = [current_header_children[0], icon_element]
        return current_style, new_header_children

# Update model details
@app.callback(
    [Output("model-metrics-display", "children"),
     Output("model-scatter-plot", "children"),
     Output("model-cluster-display", "children")],
    [Input("model-performance-dropdown", "value"),
     Input("model-viz-toggle", "value")]
)
def update_model_performance(selected_model, viz_mode):
    if not selected_model:
        return "No model selected", no_update, no_update

    if viz_mode is None:
        viz_mode = "default"

    all_models = load_final_model_outputs()
    mdata = all_models.get(selected_model, {})

    # Return placeholders or minimal content
    return "Metrics placeholder", "Scatter placeholder", "Cluster placeholder"

# XAI Callback - Updated for better layout
@app.callback(
    Output("xai-content-area", "children"),
    [Input("xai-model-dropdown", "value")]
)
def update_xai_insights(selected_model):
    if not selected_model:
        return "No model selected"

    # Return minimal placeholder
    return "XAI insights placeholder"

# Filter reset callback
@app.callback(
    [Output("global-model-dropdown", "value"),
     Output("global-risk-dropdown", "value"),
     Output("age-range-slider", "value"),
     Output("income-range-slider", "value"),
     Output("health-index-slider", "value")],
    [Input("reset-filters-btn", "n_clicks")]
)
def reset_filters(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    return "All", "All", [
        df_all["AGE"].min() if "AGE" in df_all.columns else 0,
        df_all["AGE"].max() if "AGE" in df_all.columns else 100
    ], [
        df_all["INCOME"].min() if "INCOME" in df_all.columns else 0,
        df_all["INCOME"].max() if "INCOME" in df_all.columns else 100000
    ], [
        df_all["Health_Index"].min(),
        df_all["Health_Index"].max()
    ]

# Patient modal callback
@app.callback(
    [Output("patient-modal", "is_open"), Output("patient-detail-body", "children")],
    [Input("patient-data-table", "active_cell"), Input("close-modal", "n_clicks")],
    [State("patient-modal", "is_open"), State("patient-data-table", "data")]
)
def toggle_patient_modal(active_cell, close_click, is_open, table_data):
    ctx = callback_context

    if not ctx.triggered:
        return is_open, ""

    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == "patient-data-table" and active_cell:
        row_data = table_data[active_cell["row"]]
        patient_id = row_data.get("Id")
        patient_records = df_all[df_all["Id"] == patient_id]
        if patient_records.empty:
            return True, [html.P("Patient not found.")]
        patient = patient_records.iloc[0]
        
        # Create formatted details with more comprehensive information
        details_rows = []

        # Basic info section
        basic_info = html.Div([
            html.H5("Basic Information", style={"borderBottom": f"2px solid {nhs_colors['primary']}"}),
            html.Div([
                html.Strong("Patient ID: "),
                html.Span(patient.get('Id', 'N/A'))
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Strong("Age: "),
                html.Span(patient.get('AGE', 'N/A'))
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Strong("Gender: "),
                html.Span(patient.get('GENDER', 'N/A'))
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Strong("Income: £"),
                html.Span(f"{patient.get('INCOME', 'N/A'):,.2f}" if 'INCOME' in patient else 'N/A')
            ], style={"marginBottom": "8px"}),
        ])
        details_rows.append(basic_info)

        # Health indices section
        health_indices = html.Div([
            html.H5("Health Indices", style={"borderBottom": f"2px solid {nhs_colors['primary']}", "marginTop": "15px"}),
            html.Div([
                html.Strong("Health Index: "),
                html.Span(f"{patient.get('Health_Index', 'N/A'):.2f}" if 'Health_Index' in patient else 'N/A')
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Strong("Charlson Index: "),
                html.Span(f"{patient.get('CharlsonIndex', 'N/A'):.2f}" if 'CharlsonIndex' in patient else 'N/A')
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Strong("Elixhauser Index: "),
                html.Span(f"{patient.get('ElixhauserIndex', 'N/A'):.2f}" if 'ElixhauserIndex' in patient else 'N/A')
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Strong("Risk Category: "),
                html.Span(patient.get('Risk_Category', 'N/A'), 
                          style={"color": nhs_colors["risk_high"] if patient.get('Risk_Category') == "High Risk" else 
                                 nhs_colors["risk_medium"] if patient.get('Risk_Category') == "Moderate Risk" else 
                                 nhs_colors["risk_low"] if patient.get('Risk_Category') == "Low Risk" else 
                                 nhs_colors["risk_verylow"] if patient.get('Risk_Category') == "Very Low Risk" else 
                                 "inherit"})
            ], style={"marginBottom": "8px"}),
        ])
        details_rows.append(health_indices)

        # Prediction details section if available
        if any(col in patient for col in ['Group', 'PredictedHI_final', 'Cluster_final']):
            prediction_details = html.Div([
                html.H5("Model Predictions", style={"borderBottom": f"2px solid {nhs_colors['primary']}", "marginTop": "15px"}),
                html.Div([
                    html.Strong("Model Group: "),
                    html.Span(patient.get('Group', 'N/A'))
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Strong("Predicted Health Index: "),
                    html.Span(f"{patient.get('PredictedHI_final', 'N/A'):.2f}" if 'PredictedHI_final' in patient else 'N/A')
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Strong("Assigned Cluster: "),
                    html.Span(patient.get('Cluster_final', 'N/A'))
                ], style={"marginBottom": "8px"}),
            ])
            details_rows.append(prediction_details)
        
        # Location information if available
        if all(col in patient for col in ['ZIP', 'LAT', 'LON']):
            location_details = html.Div([
                html.H5("Location Information", style={"borderBottom": f"2px solid {nhs_colors['primary']}", "marginTop": "15px"}),
                html.Div([
                    html.Strong("ZIP Code: "),
                    html.Span(patient.get('ZIP', 'N/A'))
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Strong("Coordinates: "),
                    html.Span(f"Lat: {patient.get('LAT', 'N/A'):.4f}, Lon: {patient.get('LON', 'N/A'):.4f}" 
                              if 'LAT' in patient and 'LON' in patient else 'N/A')
                ], style={"marginBottom": "8px"}),
            ])
            details_rows.append(location_details)

        return True, details_rows

    elif prop_id == "close-modal":
        return False, ""

    return is_open, ""

# Update table info based on filters
@app.callback(
    Output("table-info", "children"),
    [Input("patient-data-table", "derived_virtual_data"),
     Input("patient-data-table", "derived_virtual_indices")]
)
def update_table_info(rows, indices):
    if not rows or not indices:
        return "No data matches the current filters"
    
    return f"Showing {len(rows)} out of {len(df_all)} total patients"

# Add a new callback to optimize memory usage when switching sections
@app.callback(
    Output("memory-management", "children"),
    [Input(f"{section}-header", "n_clicks") for section in ["demographics", "health-indices", "geographic", "model-performance", "xai-insights", "patient-data"]]
)
def optimize_memory(*args):
    """Release memory when user switches between sections"""
    release_memory()
    return ""

# Add these utility functions after the existing RAM optimization functions

def sanitize_datatable_values(df):
    """
    Sanitize DataFrame for DataTable use by converting complex data types to strings
    and ensuring all values are one of [string, number, boolean]
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Check if column contains non-scalar values
        if any(isinstance(x, (list, dict, np.ndarray)) for x in df_copy[col].dropna().head(10)):
            # Convert complex types to string representation
            df_copy[col] = df_copy[col].apply(lambda x: str(x) if x is not None else None)
        # Ensure numeric types are converted to Python int/float (not numpy types)
        elif df_copy[col].dtype.kind in 'iuf':
            df_copy[col] = df_copy[col].apply(lambda x: float(x) if pd.notnull(x) else None)
    
    return df_copy

# Modify the create_model_comparison_chart function (or add it if it doesn't exist)
def create_model_comparison_chart(df_model, max_points=3000, color_by='cluster'):
    """Create optimized scatter plot comparing predicted vs actual health index"""
    if df_model is None or df_model.empty:
        return None
    
    if not all(col in df_model.columns for col in ["Health_Index", "PredictedHI_final"]):
        return None
    
    # Apply sampling for large datasets
    if len(df_model) > max_points:
        df_model = smart_sample_dataframe(df_model, max_points=max_points, method='cluster')
    
    # Determine color column based on user selection
    if color_by == 'cluster' and "Cluster_final" in df_model.columns:
        color_column = "Cluster_final"
        color_discrete_sequence = px.colors.qualitative.G10
        title = "Predicted vs Actual Health Index (by Cluster)"
    elif color_by == 'risk' and "Risk_Category" in df_model.columns:
        color_column = "Risk_Category"
        color_discrete_sequence = risk_colors
        title = "Predicted vs Actual Health Index (by Risk Category)"
    else:
        # Default coloring
        color_column = "Cluster_final" if "Cluster_final" in df_model.columns else None
        color_discrete_sequence = None
        title = "Predicted vs Actual Health Index"
    
    fig = px.scatter(
        df_model,
        x="Health_Index",
        y="PredictedHI_final",
        color=color_column,
        color_discrete_sequence=color_discrete_sequence,
        opacity=0.7,
        title=f"{title} (Sampled: {len(df_model)} points)" if len(df_model) > max_points else title,
        labels={"Health_Index": "Actual Health Index", "PredictedHI_final": "Predicted Health Index"},
        hover_data=["Id"] + (["Group"] if "Group" in df_model.columns else [])
    )
    
    # Add y=x reference line
    fig.add_trace(
        go.Scatter(
            x=[df_model["Health_Index"].min(), df_model["Health_Index"].max()],
            y=[df_model["Health_Index"].min(), df_model["Health_Index"].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction',
            hoverinfo='skip'
        )
    )
    
    apply_2decimal_format(fig)
    return fig

# Now modify the model performance callback to use this function and add a toggle for visualization mode
# Replace the existing update_model_performance callback with this updated version

@app.callback(
    [Output("model-metrics-display", "children"),
     Output("model-scatter-plot", "children"),
     Output("model-cluster-display", "children")],
    [Input("model-performance-dropdown", "value"),
     Input("model-viz-toggle", "value")]
)
def update_model_performance(selected_model, viz_mode):
    if not selected_model:
        return html.Div("No model selected."), html.Div(), html.Div()
    
    # Set default visualization mode if not provided
    if viz_mode is None:
        viz_mode = "cluster"  # Default to cluster visualization
    
    # Re-load model data in case it changed
    all_models = load_final_model_outputs()
    mdata = all_models.get(selected_model, {})
    
    metrics = mdata.get("metrics", {})
    df_model = mdata.get("df", pd.DataFrame())
    tsne_src = mdata.get("tsne_img")
    umap_src = mdata.get("umap_img")

    # Metrics cards
    metrics_cards = html.Div([
        html.H5(f"Model Metrics - {selected_model}", style={"marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(kpi_card("Test MSE", f"{metrics.get('test_mse', 'N/A')}"), width=4),
            dbc.Col(kpi_card("Test R²", f"{metrics.get('test_r2', 'N/A')}"), width=4),
            dbc.Col(kpi_card("Silhouette", f"{metrics.get('Silhouette', 'N/A')}"), width=4),
        ]),
        dbc.Row([
            dbc.Col(kpi_card("Calinski-Harabasz", f"{metrics.get('Calinski_Harabasz', 'N/A')}"), width=6),
            dbc.Col(kpi_card("Davies-Bouldin", f"{metrics.get('Davies_Bouldin', 'N/A')}"), width=6),
        ], style={"marginTop": "10px"})
    ])

    # Create the scatter using the new function with sampling
    if "PredictedHI_final" in df_model.columns and "Health_Index" in df_model.columns:
        scatter_fig = create_model_comparison_chart(df_model, color_by=viz_mode)
        scatter_plot = dcc.Graph(figure=scatter_fig)
    else:
        scatter_plot = html.Div("Prediction data not available for scatter plot.")

    # Cluster display with both visualizations
    cluster_display = html.Div([
        html.H5("Clustering Visualizations", style={"marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(html.Img(src=tsne_src, style={"width": "100%"}) 
                    if tsne_src else html.Div("t-SNE plot not available"), width=6),
            dbc.Col(html.Img(src=umap_src, style={"width": "100%"}) 
                    if umap_src else html.Div("UMAP plot not available"), width=6),
        ])
    ])
    
    return metrics_cards, scatter_plot, cluster_display

# In the Model Performance Panel section, add the visualization toggle
# Find the Model Performance Panel in the layout and modify it:

# Model Performance Panel
collapsible_card(
    "Model Performance & Insights", 
    html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Model:"),
                    dcc.Dropdown(
                        id="model-performance-dropdown",
                        options=model_dropdown_options,
                        value=model_dropdown_options[0]["value"] if model_dropdown_options else None,
                        clearable=False,
                    )
                ], width=8),
                dbc.Col([
                    html.Label("Visualization Mode:"),
                    dcc.RadioItems(
                        id="model-viz-toggle",
                        options=[
                            {"label": "By Cluster", "value": "cluster"},
                            {"label": "By Risk Category", "value": "risk"}
                        ],
                        value="cluster",
                        inline=True
                    )
                ], width=4)
            ], style={"marginBottom": "15px"})
        ]),
        html.Div(id="model-metrics-display"),
        dbc.Row([
            dbc.Col(html.Div(id="model-scatter-plot"), width=6),
            dbc.Col(html.Div(id="model-cluster-display"), width=6),
        ])
    ]),
    "model-performance",
    initially_open=True
)

# Modify the patient data table to use sanitized data
# Update the Patient Data Table Panel:

# Patient Data Table Panel
collapsible_card(
    "Patient Data Table", 
    html.Div([
        dash_table.DataTable(
            id="patient-data-table",
            data=sanitize_datatable_values(df_all.head(10)).to_dict("records"),
            columns=[{"name": i, "id": i} for i in df_all.iloc[:, :10].columns],
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': nhs_colors["primary"], 'color': nhs_colors["secondary"]},
        ),
        html.Div(id="table-info", style={"marginTop": "10px"})
    ]),
    "patient-data",
    initially_open=False
)

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run_server(debug=True)

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

import gc
from sklearn.cluster import KMeans
import warnings
import psutil
import math

from flask_caching import Cache
from memory_utils import MemoryMonitor, DataSampler
import time
from dash.dash import no_update
import dask.dataframe as dd

# Add imports for parallel processing
import multiprocessing as mp
from functools import partial
import swifter  # More efficient apply operations
# For optimized plotting with large datasets
from plotly.graph_objects import Scattergl

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
# Improved Data Loading with Type Optimization
# -----------------------------
def optimize_dtypes(df):
    """Aggressively optimize datatypes to reduce memory usage"""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    # Convert object columns to categories when beneficial
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype('category')
    return df

def load_data_in_chunks(path, chunk_size=100000):
    """Load large pickle files in chunks using dask"""
    try:
        # For very large files, use dask for chunked processing
        if os.path.getsize(path) > 1e9:  # 1GB threshold
            dask_df = dd.read_pickle(path)
            return optimize_dtypes(dask_df.compute())
        else:
            # For smaller files, use pandas with dtype optimization
            df = pd.read_pickle(path)
            return optimize_dtypes(df)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()

# -----------------------------
# Load Base Data - Optimized
# -----------------------------
if os.path.exists(PICKLE_ALL):
    logger.info(f"Loading enriched data from {PICKLE_ALL}...")
    
    # Define essential columns to load initially - reduces memory footprint
    essential_cols = ["Id", "BIRTHDATE", "AGE", "GENDER", "INCOME", 
                      "Health_Index", "CharlsonIndex", "ElixhauserIndex",
                      "Risk_Category", "ZIP", "LAT", "LON"]
    
    # Use optimized loading function
    df_all = load_data_in_chunks(PICKLE_ALL)
    
    # Only keep essential columns initially
    if not df_all.empty and len(essential_cols) > 0:
        existing_cols = [col for col in essential_cols if col in df_all.columns]
        df_all = df_all[existing_cols]
        
    logger.info(f"Data loaded: {len(df_all)} rows, {df_all.memory_usage().sum() / 1e6:.2f} MB")
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

# Merge location or income if missing - with optimization
missing_loc_cols = any(col not in df_all.columns for col in ["ZIP", "LAT", "LON"])
if missing_loc_cols and os.path.exists(CSV_PATIENTS):
    logger.info("Loading additional location data...")
    # Only load the columns we need
    df_loc = pd.read_csv(CSV_PATIENTS, usecols=["Id", "BIRTHDATE", "ZIP", "LAT", "LON", "INCOME"])
    df_loc = optimize_dtypes(df_loc)
    
    # Convert dates more efficiently
    if "BIRTHDATE" in df_loc.columns:
        df_loc["BIRTHDATE"] = pd.to_datetime(df_loc["BIRTHDATE"], errors="coerce")
        df_loc["AGE"] = ((pd.Timestamp("today") - df_loc["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    
    # More efficient merge
    df_all = pd.merge(df_all, df_loc, on="Id", how="left", suffixes=("", "_csv"))
    del df_loc  # Release memory
    gc.collect()

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

def apply_2decimal_format(fig):
    """
    Forces x-axis & y-axis ticks to show 2 decimals (ex: 3 -> 3.00).
    """
    fig.update_layout(
        xaxis=dict(tickformat=".2f"),
        yaxis=dict(tickformat=".2f")
    )
    return fig

def smart_sample_dataframe(df, max_points=5000, min_points=500, method='random'):
    """
    Intelligently sample a dataframe to reduce memory usage in visualizations.
    Optimized for performance with faster algorithms.
    """
    if df is None or df.empty:
        return df
    
    # If dataframe is already small enough, return as is
    if len(df) <= max_points:
        return df
        
    sample_size = min(max_points, max(min_points, int(len(df) * 0.1)))
    
    if method == 'random':
        return df.sample(sample_size, random_state=42)
        
    elif method == 'stratified' and 'Risk_Category' in df.columns:
        # More efficient stratified sampling
        categories = df['Risk_Category'].unique()
        result = pd.DataFrame()
        
        # Pre-calculate proportions for faster processing
        category_counts = df['Risk_Category'].value_counts(normalize=True)
        
        for category in categories:
            category_df = df[df['Risk_Category'] == category]
            category_size = max(1, int(sample_size * category_counts[category]))
            result = pd.concat([result, category_df.sample(min(category_size, len(category_df)), random_state=42)])
            
        return result
        
    elif method == 'cluster':
        try:
            # Get only the numeric columns we need
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 5:
                # Select columns with highest variance for better clustering
                variances = df[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = variances.index[:5]
                
            # Fill na values efficiently
            cluster_data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Use fewer clusters for large datasets
            n_clusters = min(int(np.sqrt(sample_size)), 50)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use init='k-means++' for faster convergence
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init='k-means++')
                # Create a view instead of copy where possible
                cluster_labels = kmeans.fit_predict(cluster_data)
                df_copy = df.copy()
                df_copy['cluster'] = cluster_labels
                
                # Pre-calculate cluster proportions
                cluster_counts = df_copy['cluster'].value_counts(normalize=True)
                
                result = pd.DataFrame()
                for c in range(n_clusters):
                    cluster_df = df_copy[df_copy['cluster'] == c]
                    # Proportional sampling
                    csize = max(1, int(sample_size * cluster_counts.get(c, 1/n_clusters)))
                    result = pd.concat([result, cluster_df.sample(min(csize, len(cluster_df)), random_state=42)])
                    
                return result.drop(columns=['cluster'])
        except Exception as e:
            logger.warning(f"Cluster sampling failed: {e}. Falling back to random sampling.")
            return df.sample(sample_size, random_state=42)
            
    return df.sample(sample_size, random_state=42)

# Initialize the app with better performance options
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True, 
    external_stylesheets=external_stylesheets + [
        {'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css', 'rel': 'stylesheet'}
    ],
    # Add Meta tags for better mobile performance
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    # Set long callbacks for heavy operations
    update_title=None,
)
server = app.server

# Enhanced cache configuration with increased timeout for heavy operations
cache = Cache(app.server, config={
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": "cache-directory",
    "CACHE_DEFAULT_TIMEOUT": 7200,  # 2 hours cache
    "CACHE_THRESHOLD": 500  # Store up to 500 items in cache
})

# Now define the cached function after cache is properly initialized
@cache.memoize(timeout=7200)  # Longer cache timeout for expensive operations
def load_final_model_outputs():
    """
    Cached function to load model outputs - won't reload if called again with same args
    Uses optimized loading for better performance
    """
    # We'll keep a small subset of base columns to merge in.
    base_cols = ["Id", "Health_Index", "CharlsonIndex", "ElixhauserIndex"]
    models_data = {}
    
    # Process models in parallel for faster loading
    max_workers = min(len(final_groups), mp.cpu_count() - 1)
    
    def process_model(grp):
        model_name = grp["model"]
        label_name = grp["label"]
        model_dir = os.path.join(FINALS_DIR, model_name)
        preds_csv    = os.path.join(model_dir, f"{model_name}_predictions.csv")
        clusters_csv = os.path.join(model_dir, f"{model_name}_clusters.csv")
        metrics_json = os.path.join(model_dir, f"{model_name}_metrics.json")
        cluster_json = os.path.join(model_dir, f"{model_name}_clusters.json")
        tsne_png = os.path.join(model_dir, f"{model_name}_tsne.png")
        umap_png = os.path.join(model_dir, f"{model_name}_umap.png")
        
        # Merge predictions + clusters - optimized loading
        if os.path.exists(preds_csv) and os.path.exists(clusters_csv):
            # Load with dtype optimization
            df_preds = pd.read_csv(preds_csv, 
                                  dtype={"Id": "str", "Predicted_Health_Index": "float32"})
            df_clusters = pd.read_csv(clusters_csv, 
                                     dtype={"Id": "str", "Cluster": "int8"})
            
            # More efficient merge
            df_model = pd.merge(df_preds, df_clusters, on="Id", how="outer", suffixes=("", "_cluster"))
            df_model.rename(
                columns={
                    "Predicted_Health_Index": "PredictedHI_final",
                    "Cluster": "Cluster_final"
                },
                inplace=True
            )
            
            # Merge with base data efficiently
            if not df_all.empty:
                df_model = pd.merge(
                    df_model,
                    df_all[base_cols],
                    on="Id",
                    how="left"
                )
            
            # Optimize datatypes after merge
            df_model = optimize_dtypes(df_model)
        else:
            df_model = pd.DataFrame()
        
        # Load & combine metrics - no change needed here
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
        
        return label_name, {
            "df": df_model,
            "metrics": metrics_primary,
            "tsne_img": tsne_img,
            "umap_img": umap_img
        }
    
    # Sequential for small number of models, parallel for many
    if len(final_groups) <= 2 or max_workers <= 1:
        for grp in final_groups:
            label_name, data = process_model(grp)
            models_data[label_name] = data
    else:
        with mp.Pool(max_workers) as pool:
            results = pool.map(process_model, final_groups)
            for label_name, data in results:
                models_data[label_name] = data
    
    return models_data

final_models_data = load_final_model_outputs()

# ...existing code...

# Add a periodic memory cleanup callback
@app.callback(
    Output("memory-management", "children"),
    [Input("apply-filters-btn", "n_clicks"),
     Input("reset-filters-btn", "n_clicks")],
    prevent_initial_call=True
)
def cleanup_memory(filter_clicks, reset_clicks):
    """Perform garbage collection and report memory usage after filter operations"""
    # Force garbage collection
    gc.collect()
    
    # Get current memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    logger.info(f"Memory usage: {memory_mb:.2f} MB")
    
    # Return timestamp to avoid caching
    return f"Memory cleaned: {time.time()}"

# Add callback to prefetch data for selected model to improve responsiveness
@app.callback(
    Output("model-dropdown", "options"),
    [Input("global-model-dropdown", "value")],
    prevent_initial_call=True
)
def prefetch_model_data(selected_model):
    """Prefetch data for the selected model to improve responsiveness"""
    # This function will trigger model data loading before it's needed
    _ = load_final_model_outputs()
    
    # Return the unchanged options
    return [{"label": g["label"], "value": g["label"]} for g in final_groups]

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    # Optimize server configuration for better performance
    app.run_server(
        debug=True,
        dev_tools_hot_reload=False,  # Disable hot reload for performance
        threaded=True,
        host='0.0.0.0',
    )
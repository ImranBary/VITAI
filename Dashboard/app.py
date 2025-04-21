import os
import json
import base64
import logging
import numpy as np
import pandas as pd
import math
import gc
import time
import warnings
import subprocess
import threading
import queue
import signal
from datetime import datetime
from typing import Optional, Union  # Add this import for older Python compatibility

# Add import for run_generate_and_predict from parent directory
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback_context, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats

from xai_formatter import format_explanation

import psutil
from flask_caching import Cache
import diskcache

# For parallel processing in sampling routines
from concurrent.futures import ThreadPoolExecutor

# Add these imports for improved system resource detection
import multiprocessing
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
FINALS_DIR = os.path.join(DATA_DIR, "finals")
EXPLAIN_XAI_DIR = os.path.join(DATA_DIR, "explain_xai")

PICKLE_ALL = os.path.join(DATA_DIR, "patient_data_with_all_indices.pkl")
CSV_PATIENTS = os.path.join(DATA_DIR, "patients.csv")

final_groups = [
    {"model": "combined_diabetes_tabnet", "label": "Diabetes"},
    {"model": "combined_all_ckd_tabnet", "label": "CKD"},
    {"model": "combined_none_tabnet", "label": "General"},
]

# ------------------------------------------------------------
# SYSTEM RESOURCE DETECTION & OPTIMIZATION SETTINGS
# ------------------------------------------------------------
def get_system_resources():
    """Detect system resources and optimize settings accordingly"""
    cpu_count = multiprocessing.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

    # Advanced hardware detection
    system_info = {
        "cpu_count": cpu_count,
        "memory_gb": total_memory_gb,
        "platform": platform.system(),
        "processor": platform.processor(),
        "is_high_end": cpu_count >= 8 and total_memory_gb >= 32,
    }

    # Optimize settings based on detected hardware
    is_high_end = system_info["is_high_end"]  # Explicitly extract the value

    optimization_settings = {
        "max_threads": max(int(cpu_count * 0.8), 4),  # Use 80% of available cores
        "memory_limit": int(total_memory_gb * 0.7),  # Use 70% of available memory (GB)
        "batch_size": 10000 if is_high_end else 5000,
        "sampling_threshold": 50000 if is_high_end else 10000,
        "use_advanced_clustering": is_high_end,
        "webgl_threshold": 5000 if is_high_end else 1000,
        "is_high_end": is_high_end,  # Make sure to include this key explicitly
    }

    logger.info(f"System resources detected: {system_info}")
    logger.info(f"Performance settings: {optimization_settings}")

    return optimization_settings


# Get optimized settings for this system
PERF_SETTINGS = get_system_resources()

# ------------------------------------------------------------
# DATA LOADING WITH MEMORY MAPPING & DTYPE OPTIMIZATIONS
# ------------------------------------------------------------
if os.path.exists(PICKLE_ALL):
    # Use parallel reading for large pickle files on high-end systems
    if PERF_SETTINGS["is_high_end"]:
        # Load pickle in a more memory efficient way
        df_all = pd.read_pickle(PICKLE_ALL)

        # Use parallel optimization of datatypes
        def optimize_column_dtypes(col):
            if col[1].dtype == "float64":
                return col[0], col[1].astype("float32")
            elif col[1].dtype == "int64":
                return col[0], col[1].astype("int32")
            return col[0], col[1]

        # Process columns in parallel
        with ThreadPoolExecutor(max_workers=PERF_SETTINGS["max_threads"]) as executor:
            optimized_cols = list(
                executor.map(
                    optimize_column_dtypes,
                    [(col, df_all[col]) for col in df_all.columns],
                )
            )

        for col_name, col_data in optimized_cols:
            df_all[col_name] = col_data

        logger.info(
            f"Loaded and optimized enriched data from {PICKLE_ALL} using parallel processing."
        )
    else:
        # Original loading code for lower-end systems
        df_all = pd.read_pickle(PICKLE_ALL)
        for col in df_all.select_dtypes(include=["float64"]).columns:
            df_all[col] = df_all[col].astype("float32")
        for col in df_all.select_dtypes(include=["int64"]).columns:
            df_all[col] = df_all[col].astype("int32")
        logger.info(f"Loaded enriched data from {PICKLE_ALL}.")

    essential_cols = [
        "Id",
        "BIRTHDATE",
        "AGE",
        "GENDER",
        "INCOME",
        "Health_Index",
        "CharlsonIndex",
        "ElixhauserIndex",
        "Risk_Category",
        "ZIP",
        "LAT",
        "LON",
        "RACE",
        "HEALTHCARE_EXPENSES",
        "ETHNICITY",
        "MARITAL",
        "HEALTHCARE_COVERAGE",
    ]
    df_all = df_all[[col for col in essential_cols if col in df_all.columns]]
else:
    # For CSV loading, use chunksize and parallel processing on high-end systems
    logger.warning("Enriched pickle not found; falling back to optimized CSV loading.")
    if PERF_SETTINGS["is_high_end"]:
        # Calculate optimal chunk size based on memory
        chunk_size = min(
            500000, int(PERF_SETTINGS["memory_limit"] * 1000000 / 50)
        )  # Rough estimate: 50 bytes per row

        chunks = []
        for chunk in pd.read_csv(
            CSV_PATIENTS, memory_map=True, low_memory=True, chunksize=chunk_size
        ):
            # Process each chunk
            chunk["BIRTHDATE"] = pd.to_datetime(chunk["BIRTHDATE"], errors="coerce")
            chunk["AGE"] = (
                ((pd.Timestamp("today") - chunk["BIRTHDATE"]).dt.days / 365.25)
                .fillna(0)
                .astype(int)
            )
            chunks.append(chunk)

        df_all = pd.concat(chunks)
        del chunks  # Free memory
        gc.collect()

        # Generate random fields in parallel
        np.random.seed(42)
        with ThreadPoolExecutor(max_workers=PERF_SETTINGS["max_threads"]) as executor:
            # Generate random data in parallel
            n_rows = len(df_all)
            health_index = executor.submit(
                lambda: np.random.uniform(1, 10, n_rows).round(2)
            )
            charlson_index = executor.submit(
                lambda: np.random.uniform(0, 5, n_rows).round(2)
            )
            elixhauser_index = executor.submit(
                lambda: np.random.uniform(0, 15, n_rows).round(2)
            )
            cluster = executor.submit(lambda: np.random.choice([0, 1, 2], n_rows))

            # Assign results
            df_all["Health_Index"] = health_index.result()
            df_all["CharlsonIndex"] = charlson_index.result()
            df_all["ElixhauserIndex"] = elixhauser_index.result()
            df_all["Cluster"] = cluster.result()

        df_all["Predicted_Health_Index"] = (
            df_all["Health_Index"] + np.random.normal(0, 0.5, len(df_all))
        ).round(2)
        df_all["Actual"] = df_all["Health_Index"]
    else:
        # Original CSV loading code for lower-end systems
        df_all = pd.read_csv(CSV_PATIENTS, memory_map=True, low_memory=True)
        df_all["BIRTHDATE"] = pd.to_datetime(df_all["BIRTHDATE"], errors="coerce")
        df_all["AGE"] = (
            ((pd.Timestamp("today") - df_all["BIRTHDATE"]).dt.days / 365.25)
            .fillna(0)
            .astype(int)
        )
        np.random.seed(42)
        df_all["Health_Index"] = np.random.uniform(1, 10, len(df_all)).round(2)
        df_all["CharlsonIndex"] = np.random.uniform(0, 5, len(df_all)).round(2)
        df_all["ElixhauserIndex"] = np.random.uniform(0, 15, len(df_all)).round(2)
        df_all["Cluster"] = np.random.choice([0, 1, 2], len(df_all))
        df_all["Predicted_Health_Index"] = (
            df_all["Health_Index"] + np.random.normal(0, 0.5, len(df_all))
        ).round(2)
        df_all["Actual"] = df_all["Health_Index"]

missing_loc_cols = any(col not in df_all.columns for col in ["ZIP", "LAT", "LON"])
if missing_loc_cols and os.path.exists(CSV_PATIENTS):
    df_loc = pd.read_csv(
        CSV_PATIENTS,
        usecols=["Id", "BIRTHDATE", "ZIP", "LAT", "LON", "INCOME"],
        memory_map=True,
        low_memory=True,
    )
    df_loc["BIRTHDATE"] = pd.to_datetime(df_loc["BIRTHDATE"], errors="coerce")
    df_loc["AGE"] = (
        ((pd.Timestamp("today") - df_loc["BIRTHDATE"]).dt.days / 365.25)
        .fillna(0)
        .astype(int)
    )
    df_all = pd.merge(df_all, df_loc, on="Id", how="left", suffixes=("", "_csv"))

if "Health_Index" not in df_all.columns:
    df_all["Health_Index"] = np.random.uniform(1, 10, len(df_all)).round(2)

df_all["Age_Group"] = pd.cut(
    df_all["AGE"] if "AGE" in df_all.columns else 0,
    bins=[0, 18, 35, 50, 65, 80, 120],
    labels=["0-18", "19-35", "36-50", "51-65", "66-80", "80+"],
)

df_all["Risk_Category"] = pd.cut(
    df_all["Health_Index"],
    bins=[0, 3, 6, 8, 10],
    labels=["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk"],
)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def encode_image(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return None


# Helper function to switch scatter traces to WebGL for large data - OPTIMIZED
def use_webgl_rendering(fig, threshold=None):
    if threshold is None:
        threshold = PERF_SETTINGS["webgl_threshold"]

    for trace in fig.data:
        if trace.type == "scatter" and hasattr(trace, "x") and len(trace.x) > threshold:
            trace.type = "scattergl"
    return fig


# Aggressive caching can be applied to expensive plot creation if the input data fingerprint is stable.
# (For demonstration, we decorate one such function below.)
def memoize_fig(func):
    cache_key = f"{func.__name__}_cache"
    cache_store = {}

    def wrapper(*args, **kwargs):
        key = (
            args[0].shape[0] if (args and isinstance(args[0], pd.DataFrame)) else None,
            tuple(kwargs.items()),
        )
        if key in cache_store:
            return cache_store[key]
        fig = func(*args, **kwargs)
        cache_store[key] = fig
        return fig

    return wrapper


def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024


def log_memory_usage(label):
    logger.info(f"Memory usage at {label}: {get_memory_usage():.2f} MB")


# ------------------------------------------------------------
# PARALLEL SAMPLING FUNCTION - OPTIMIZED
# ------------------------------------------------------------
def smart_sample_dataframe(df, max_points=None, min_points=500, method="random"):
    # Use system-specific sampling thresholds
    if max_points is None:
        max_points = PERF_SETTINGS["sampling_threshold"]

    if df is None or df.empty:
        return df
    if len(df) <= max_points:
        return df

    sample_size = min(max_points, max(min_points, int(len(df) * 0.1)))

    if method == "random":
        return df.sample(sample_size, random_state=42)
    elif method == "stratified" and "Risk_Category" in df.columns:
        unique_categories = df["Risk_Category"].unique()

        # Use more threads on high-end systems
        with ThreadPoolExecutor(max_workers=PERF_SETTINGS["max_threads"]) as executor:

            def sample_category(category):
                category_df = df[df["Risk_Category"] == category]
                category_size = max(1, int(sample_size * len(category_df) / len(df)))
                return category_df.sample(
                    min(category_size, len(category_df)), random_state=42
                )

            sampled = list(executor.map(sample_category, unique_categories))
        return pd.concat(sampled)
    elif method == "cluster" and df.select_dtypes(include=["number"]).shape[1] >= 2:
        try:
            # On high-end systems, use more sophisticated clustering
            if PERF_SETTINGS["use_advanced_clustering"] and len(df) > 50000:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if (
                    len(numeric_cols) > 10
                ):  # Can handle more columns on high-end systems
                    numeric_cols = numeric_cols[:10]
                cluster_data = df[numeric_cols].fillna(df[numeric_cols].mean())

                # Use more clusters on high-end systems
                n_clusters = min(int(math.sqrt(sample_size) * 1.5), 100)

                # Use MiniBatchKMeans with larger batch size for better performance
                from sklearn.cluster import MiniBatchKMeans

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        batch_size=PERF_SETTINGS["batch_size"],
                        random_state=42,
                    )
                    df_copy = df.copy()
                    df_copy["cluster"] = kmeans.fit_predict(cluster_data)

                # Use parallel processing with more threads
                with ThreadPoolExecutor(
                    max_workers=PERF_SETTINGS["max_threads"]
                ) as executor:

                    def sample_cluster(cluster):
                        cluster_df = df_copy[df_copy["cluster"] == cluster]
                        cluster_size = max(
                            1, int(sample_size * len(cluster_df) / len(df))
                        )
                        return cluster_df.sample(
                            min(cluster_size, len(cluster_df)), random_state=42
                        )

                    sampled = list(executor.map(sample_cluster, range(n_clusters)))
                return pd.concat(sampled).drop(columns=["cluster"])
            else:
                # Fall back to existing implementation for smaller datasets
                return df.sample(sample_size, random_state=42)
        except Exception as e:
            logger.warning(
                f"Advanced cluster sampling failed, falling back to random: {e}"
            )
            return df.sample(sample_size, random_state=42)
    return df.sample(sample_size, random_state=42)


def release_memory():
    gc.collect()


# ------------------------------------------------------------
# DASHBOARD PLOTTING FUNCTIONS (with WebGL enhancements)
# ------------------------------------------------------------
def apply_2decimal_format(fig):
    fig.update_layout(xaxis=dict(tickformat=".2f"), yaxis=dict(tickformat=".2f"))
    return fig


@memoize_fig
def create_demographic_charts(df):
    gender_fig, age_fig, risk_fig = None, None, None
    if "GENDER" in df.columns:
        df_gender = df.dropna(subset=["GENDER"])
        if len(df_gender) > 0:
            gender_counts = df_gender["GENDER"].value_counts().reset_index()
            gender_fig = px.pie(
                gender_counts,
                values="count",
                names="GENDER",
                title="Gender Distribution",
                hole=0.4,
                color_discrete_sequence=["#5A9BD5", "#52A373"],
            )
            gender_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    if "Age_Group" in df.columns or "AGE" in df.columns:
        age_column = "Age_Group" if "Age_Group" in df.columns else "AGE"
        df_age = df.dropna(subset=[age_column])
        if len(df_age) > 0:
            age_fig = px.histogram(
                df_age,
                x=age_column,
                title="Age Distribution",
                color="Risk_Category" if "Risk_Category" in df.columns else None,
                color_discrete_sequence=["#5A9BD5", "#52A373", "#F0A860", "#E66C6C"],
                category_orders={
                    "Age_Group": ["0-18", "19-35", "36-50", "51-65", "66-80", "80+"],
                    "Risk_Category": [
                        "Very Low Risk",
                        "Low Risk",
                        "Moderate Risk",
                        "High Risk",
                    ],
                },
                barmode="group",
                opacity=0.85,
            )
            age_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
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
                    "High Risk": "#E66C6C",
                    "Moderate Risk": "#F0A860",
                    "Low Risk": "#52A373",
                    "Very Low Risk": "#5A9BD5",
                },
            )
            risk_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return gender_fig, age_fig, risk_fig


def create_race_demographics(df, max_points=3000):
    """
    Create visualizations for race demographics and related healthcare metrics
    """
    if "RACE" not in df.columns:
        return None, None

    df_race = df.dropna(subset=["RACE"])
    if len(df_race) == 0:
        return None, None

    # Race distribution pie chart
    race_counts = df_race["RACE"].value_counts().reset_index()
    race_fig = px.pie(
        race_counts,
        values="count",
        names="RACE",
        title="Race Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    race_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))

    # Healthcare expenses by race
    expense_fig = None
    if "HEALTHCARE_EXPENSES" in df_race.columns:
        # Aggregate healthcare expenses by race
        race_expenses = (
            df_race.groupby("RACE")["HEALTHCARE_EXPENSES"]
            .agg(["mean", "count"])
            .reset_index()
        )
        race_expenses.columns = ["RACE", "Average_Expenses", "Count"]

        # Use only races with significant sample size to avoid skew
        min_samples = 10
        race_expenses = race_expenses[race_expenses["Count"] >= min_samples]

        if len(race_expenses) > 0:
            expense_fig = px.bar(
                race_expenses,
                x="RACE",
                y="Average_Expenses",
                title="Average Healthcare Expenses by Race",
                color="RACE",
                text_auto=".2s",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hover_data=["Count"],
            )
            expense_fig.update_layout(
                xaxis_title="Race",
                yaxis_title="Average Healthcare Expenses ($)",
                margin=dict(t=40, b=0, l=0, r=0),
            )

    return race_fig, expense_fig


def create_indices_comparison(df, max_points=3000):
    """
    Enhanced version using hexagonal binning and heatmap instead of sampling
    """
    if all(
        col in df.columns
        for col in ["Health_Index", "CharlsonIndex", "ElixhauserIndex"]
    ):
        df_filtered = df.dropna(
            subset=["Health_Index", "CharlsonIndex", "ElixhauserIndex"]
        )

        # Use hexagonal binning for large datasets, faster and more informative than sampling
        if len(df_filtered) > max_points:
            # Create a density heatmap without attempting to add contours
            fig = px.density_heatmap(
                df_filtered,
                x="Health_Index",
                y="CharlsonIndex",
                z="ElixhauserIndex",
                histfunc="avg",
                nbinsx=50,
                nbinsy=50,
                title="Health Indices Relationship (Density View)",
                color_continuous_scale=["#5A9BD5", "#52A373", "#F0A860", "#E66C6C"],
            )

            # Add a more informative hover template without contours property
            fig.update_traces(
                hovertemplate=(
                    "Health Index: %{x:.2f}<br>"
                    + "Charlson Index: %{y:.2f}<br>"
                    + "Avg Elixhauser: %{z:.2f}"
                )
            )
        else:
            # For smaller datasets, use scatter plot with color indicating Elixhauser Index
            fig = px.scatter(
                df_filtered,
                x="Health_Index",
                y="CharlsonIndex",
                color="ElixhauserIndex",
                color_continuous_scale=["#5A9BD5", "#52A373", "#F0A860", "#E66C6C"],
                opacity=0.6,
                title=f"Health Indices Relationship ({len(df_filtered)} patients)",
            )

        apply_2decimal_format(fig)
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        release_memory()
        return fig
    return None


def create_income_health_chart(df, max_points=3000):
    """
    Enhanced version using hexbin plot instead of sampling
    """
    if "INCOME" in df.columns and "Health_Index" in df.columns:
        df_filtered = df.dropna(subset=["INCOME", "Health_Index"])
        if len(df_filtered) == 0:
            return go.Figure().update_layout(
                title="No valid income-health data available"
            )

        if len(df_filtered) > max_points:
            # Use hexagonal binning for better density visualization
            fig = px.density_heatmap(
                df_filtered,
                x="INCOME",
                y="Health_Index",
                color_continuous_scale=["#5A9BD5", "#52A373", "#F0A860", "#E66C6C"],
                nbinsx=40,
                nbinsy=30,
                title="Income and Health Index Relationship",
            )

            # Add a trendline to show the relationship
            x = df_filtered["INCOME"]
            y = df_filtered["Health_Index"]

            # Calculate trendline
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_pred = slope * x_range + intercept

            # Add trendline to figure
            fig.add_traces(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode="lines",
                    name=f"Trend (R²: {r_value**2:.3f})",
                    line=dict(color="black", width=2),
                )
            )

            # Add a more informative hover template
            fig.update_traces(
                hovertemplate=(
                    "Income: %{x}<br>" + "Health Index: %{y}<br>" + "Count: %{z}"
                )
            )
        else:
            # For smaller datasets, use scatter with risk category coloring
            fig = px.scatter(
                df_filtered,
                x="INCOME",
                y="Health_Index",
                color="Risk_Category"
                if "Risk_Category" in df_filtered.columns
                else None,
                color_discrete_sequence=["#5A9BD5", "#52A373", "#F0A860", "#E66C6C"]
                if "Risk_Category" in df_filtered.columns
                else None,
                category_orders={
                    "Risk_Category": [
                        "Very Low Risk",
                        "Low Risk",
                        "Moderate Risk",
                        "High Risk",
                    ]
                },
                opacity=0.7,
                trendline="ols",
                title=f"Income and Health Index Relationship ({len(df_filtered)} patients)",
            )

        apply_2decimal_format(fig)
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        return fig
    return None


def create_compact_map(df, height=600, max_points=None):
    """
    Creates a heatmap showing the concentration of patients with higher weights for high-risk patients
    """
    if "LAT" not in df.columns or "LON" not in df.columns:
        return None

    df_map = df.dropna(subset=["LAT", "LON"])
    if len(df_map) == 0:
        return go.Figure().update_layout(
            title="No valid location data available", height=height
        )

    # Create a density-enhanced visualization
    # First create the heatmap base with reduced intensity
    fig = px.density_mapbox(
        df_map,
        lat="LAT",
        lon="LON",
        z="Health_Index" if "Health_Index" in df_map.columns else None,
        radius=15,
        center=dict(lat=df_map["LAT"].mean(), lon=df_map["LON"].mean()),
        zoom=7,
        height=height,
        opacity=0.65,
        color_continuous_scale=[
            [0, "#5A9BD5"],
            [0.3, "#52A373"],
            [0.6, "#F0A860"],
            [1, "#E66C6C"],
        ],
        mapbox_style="carto-positron",
        title=f"Patient Geographic Distribution (n={len(df_map)} patients)",
    )

    # Add contour overlay with reduced intensity
    if "Health_Index" in df_map.columns and len(df_map) > 100:
        # Create a subset for the highest risk patients (top 20%)
        high_risk_threshold = df_map["Health_Index"].quantile(0.8)
        high_risk_df = df_map[df_map["Health_Index"] >= high_risk_threshold]

        if len(high_risk_df) > 20:
            fig.add_trace(
                go.Densitymapbox(
                    lat=high_risk_df["LAT"],
                    lon=high_risk_df["LON"],
                    radius=20,
                    opacity=0.4,
                    colorscale=[
                        [0, "rgba(0,0,0,0)"],
                        [0.5, "rgba(255,185,28,0.2)"],
                        [1, "rgba(230,108,108,0.4)"],
                    ],
                    showscale=False,
                    hoverinfo="none",
                    name="High Risk Areas",
                )
            )

    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Health Risk",
            tickvals=[1, 3, 6, 9],
            ticktext=["Very Low", "Low", "Moderate", "High"],
        ),
    )

    return fig


def create_correlation_matrix(df, max_cols=15):
    numeric_df = df.select_dtypes(include=["number"])
    if len(numeric_df.columns) > max_cols:
        priority_cols = [
            col
            for col in [
                "Health_Index",
                "CharlsonIndex",
                "ElixhauserIndex",
                "AGE",
                "INCOME",
                "PredictedHI_final",
            ]
            if col in numeric_df.columns
        ]
        other_cols = [col for col in numeric_df.columns if col not in priority_cols]
        selected_cols = priority_cols + other_cols[: max_cols - len(priority_cols)]
        numeric_df = numeric_df[selected_cols]
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        corr_fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=f"Feature Correlations (Top {len(corr)} features)",
            color_continuous_scale=[[0, "#5A9BD5"], [0.5, "#FFFFFF"], [1, "#E66C6C"]],
            zmin=-1,
            zmax=1,
        )
        apply_2decimal_format(corr_fig)
        corr_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        return corr_fig
    return None


def create_health_trend_chart(df, max_points=3000):
    if "Age_Group" in df.columns and "Health_Index" in df.columns:
        df_filtered = df.dropna(subset=["Age_Group", "Health_Index"])
        if len(df_filtered) == 0:
            return go.Figure().update_layout(
                title="No valid data for health trend chart"
            )
        if len(df_filtered) > max_points:
            health_by_age = (
                df_filtered.groupby("Age_Group")["Health_Index"]
                .agg(["mean", "count", "std"])
                .reset_index()
            )
            health_by_age.rename(columns={"mean": "Health_Index"}, inplace=True)
        else:
            health_by_age = (
                df_filtered.groupby("Age_Group")["Health_Index"].mean().reset_index()
            )
        fig = px.line(
            health_by_age,
            x="Age_Group",
            y="Health_Index",
            markers=True,
            title="Average Health Index by Age Group",
            category_orders={
                "Age_Group": ["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]
            },
            hover_data=["count", "std"] if "count" in health_by_age.columns else None,
        )
        apply_2decimal_format(fig)
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        release_memory()
        return fig
    return None


def sanitize_datatable_values(df):
    # Check if input is a list (from JSON deserialization)
    # and convert it back to DataFrame if needed
    if isinstance(df, list):
        df = pd.DataFrame(df)
        if df.empty:
            return df

    df_copy = df.copy()
    for col in df_copy.select_dtypes(
        include=["int8", "int16", "int32", "int64"]
    ).columns:
        df_copy[col] = df_copy[col].astype("int").replace({pd.NA: None, np.nan: None})
    for col in df_copy.select_dtypes(include=["float16", "float32", "float64"]).columns:
        df_copy[col] = df_copy[col].astype("float").replace({pd.NA: None, np.nan: None})
    for col in df_copy.select_dtypes(include=["datetime"]).columns:
        df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d")
    for col in df_copy.select_dtypes(include=["object"]).columns:
        df_copy[col] = df_copy[col].fillna("").astype(str)
    for col in df_copy.select_dtypes(include=["category"]).columns:
        df_copy[col] = df_copy[col].astype(str).replace({"nan": "", "None": ""})
    df_copy = df_copy.fillna("")
    return df_copy


# ------------------------------------------------------------
# DASH APP INITIALIZATION & LAYOUT
# ------------------------------------------------------------
nhs_colors = {
    "background": "#F7F7F7",
    "text": "#333333",
    "primary": "#005EB8",
    "secondary": "#FFFFFF",
    "accent": "#00843D",
    "highlight": "#FFB81C",
    "risk_high": "#E66C6C",
    "risk_medium": "#F0A860",
    "risk_low": "#52A373",
    "risk_verylow": "#5A9BD5",
}
external_stylesheets = [dbc.themes.FLATLY]

risk_colors = [
    nhs_colors["risk_verylow"],
    nhs_colors["risk_low"],
    nhs_colors["risk_medium"],
    nhs_colors["risk_high"],
]

progress_stage_icons = {
    "Initializing": "fa fa-cog",
    "GeneratingData": "fa fa-database",
    "ProcessingFiles": "fa fa-file-alt",
    "CalculatingIndices": "fa fa-calculator",
    "NormalizingFeatures": "fa fa-chart-bar",
    "RunningPredictions": "fa fa-chart-line",
    "Completed": "fa fa-check-circle",
}


def create_stage_markers(current_stage, overall_progress, eta="calculating..."):
    stages = list(progress_stage_icons.keys())

    stage_markers = []
    cumulative_percentage = 0

    for i, stage in enumerate(stages):
        if stage == "Completed" and current_stage != "Completed":
            continue

        if i == 0:
            position_percent = 0
        elif i == len(stages) - 1 and stage == "Completed":
            position_percent = 100
        else:
            position_percent = (i / (len(stages) - 1)) * 100

        is_active = stage == current_stage
        is_completed = False

        if overall_progress > position_percent:
            is_completed = True

        stage_marker = html.Div(
            className=f"stage-marker {'active' if is_active else ''} {'completed' if is_completed else ''}",
            style={
                "left": f"{position_percent}%",
                "transform": "translateX(-50%)",
                "position": "absolute",
                "bottom": "-25px",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "width": "60px",
                "transition": "all 0.3s ease",
            },
            children=[
                html.Div(
                    className=f"stage-icon-container {'active' if is_active else ''} {'completed' if is_completed else ''}",
                    style={
                        "width": "30px",
                        "height": "30px",
                        "borderRadius": "50%",
                        "backgroundColor": nhs_colors["primary"]
                        if is_active
                        else ("#52A373" if is_completed else "#e0e0e0"),
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "color": "white",
                        "marginBottom": "5px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.2)"
                        if is_active
                        else "none",
                        "transform": "scale(1.2)" if is_active else "scale(1)",
                        "transition": "all 0.3s ease",
                    },
                    children=[html.I(className=progress_stage_icons[stage])],
                ),
                html.Span(
                    stage,
                    style={
                        "fontSize": "10px",
                        "textAlign": "center",
                        "fontWeight": "bold" if is_active else "normal",
                        "color": nhs_colors["primary"] if is_active else "#666",
                        "transition": "all 0.3s ease",
                    },
                ),
            ],
        )
        stage_markers.append(stage_marker)

    eta_display = None
    if current_stage != "Completed" and eta != "calculating...":
        eta_display = html.Div(
            [
                html.Div(
                    [
                        html.I(className="fa fa-clock", style={"marginRight": "5px"}),
                        html.Span("Estimated Time Remaining:"),
                    ],
                    style={"fontWeight": "bold", "marginBottom": "5px"},
                ),
                html.Div(
                    eta, style={"fontSize": "16px", "color": nhs_colors["primary"]}
                ),
            ],
            style={
                "position": "absolute",
                "right": "0",
                "top": "-45px",
                "backgroundColor": "white",
                "padding": "8px 12px",
                "borderRadius": "4px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                "border": f"1px solid {nhs_colors['primary']}",
            },
        )

    return html.Div(
        [
            eta_display,
            html.Div(
                style={
                    "width": "100%",
                    "height": "40px",
                    "position": "relative",
                    "marginBottom": "40px",
                },
                children=[
                    dbc.Progress(
                        value=overall_progress,
                        striped=False,
                        animated=True,
                        style={
                            "height": "10px",
                            "borderRadius": "5px",
                            "backgroundColor": "#e0e0e0",
                        },
                        className="mb-0",
                        color="primary" if current_stage != "Completed" else "success",
                    ),
                    *stage_markers,
                ],
            ),
        ]
    )


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets
    + [
        {
            "href": "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
            "rel": "stylesheet",
        }
    ],
)
server = app.server

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory",
        "CACHE_DEFAULT_TIMEOUT": 3600,
    },
)
app.server.config.update(
    {
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory",
        "CACHE_DEFAULT_TIMEOUT": 3600,
    }
)
cache = Cache(app.server)

# ------------------------------------------------------------
# LONG CALLBACK SETUP - ENHANCED
# ------------------------------------------------------------
cache_dir = os.path.join(BASE_DIR, "Dashboard", "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Scale cache size based on available memory
cache_size_limit = (
    PERF_SETTINGS["memory_limit"] * 1024 * 1024 * 1024 / 2
)  # Half of allocated memory in bytes
long_callback_cache = diskcache.Cache(cache_dir, size_limit=cache_size_limit)
long_callback_manager = DiskcacheLongCallbackManager(long_callback_cache)

# Add these globals after other config globals
progress_queue: queue.Queue[str] = queue.Queue(maxsize=10_000)
process_handle: Optional[subprocess.Popen] = None  # Fixed type annotation
process_lock = threading.Lock()        # guards process_handle

# Helper function to run the executable in a background thread
def _launch_generate_and_predict(cmd: list[str]) -> None:
    """
    Spawn GenerateAndPredict and funnel its stdout into progress_queue.
    Runs as a daemon thread so Dash's main event loop remains free.
    """
    global process_handle

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,                       # line‑buffered
        universal_newlines=True,
        creationflags=(
            subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        ),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # make child Python unbuffered
    ) as proc:
        with process_lock:
            process_handle = proc

        for raw in iter(proc.stdout.readline, ""):
            progress_queue.put_nowait(raw.rstrip("\n"))

        progress_queue.put_nowait(f"[EXITCODE] {proc.returncode}")

    with process_lock:
        process_handle = None

# CLI command builder
from pathlib import Path
def _build_command(population, perf_opts, mem_util, cpu_util, threads):
    exe_root = Path(__file__).resolve().parents[1]
    for cand in (
        exe_root / "GenerateAndPredict.exe",
        exe_root / "GenerateAndPredict",
        Path("./GenerateAndPredict.exe"),
        Path("./GenerateAndPredict"),
    ):
        if cand.exists():
            exe_path = str(cand)
            break
    else:
        raise FileNotFoundError("GenerateAndPredict executable not found")

    flags = [
        f"--population={population}",
        *(opt for opt in (
            "--enable-xai"          if "enable_xai"          in perf_opts else "",
            "--performance-mode"    if "performance_mode"    in perf_opts else "",
            "--extreme-performance" if "extreme_performance" in perf_opts else "",
            "--force-cpu"           if "force_cpu"           in perf_opts else "",
        ) if opt),
        f"--memory-util={mem_util}",
        f"--cpu-util={cpu_util}",
        f"--threads={threads}",
    ]
    return [exe_path, *flags]

# Start/Cancel controller callback
@app.callback(
    Output("progress-interval",    "disabled"),
    Output("run-model-btn",        "disabled"),
    Output("cancel-btn-container", "style"),
    Input("run-model-btn",   "n_clicks"),
    Input("cancel-model-btn","n_clicks"),
    State("population-input",   "value"),
    State("performance-options","value"),
    State("memory-util-slider", "value"),
    State("cpu-util-slider",    "value"),
    State("threads-slider",     "value"),
    prevent_initial_call=True,
)
def start_or_cancel(run_clicks, cancel_clicks,
                    population, perf_opts,
                    mem_util, cpu_util, threads):

    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    # ---- Cancel pressed ----------------------------------------------------
    if trig == "cancel-model-btn":
        with process_lock:
            if process_handle and process_handle.poll() is None:
                try:
                    process_handle.send_signal(signal.SIGINT)
                    process_handle.wait(timeout=5)
                except Exception:
                    process_handle.kill()
        return True, False, {"display": "none"}   # stop polling, enable Run

    # ---- Run pressed -------------------------------------------------------
    if not run_clicks:
        raise PreventUpdate

    cmd = _build_command(population, perf_opts, mem_util, cpu_util, threads)
    threading.Thread(
        target=_launch_generate_and_predict, args=(cmd,), daemon=True
    ).start()

    return False, True, {"display": "block"}      # start polling, disable Run


# Interval poller callback
@app.callback(
    Output("execution-state-store", "data"),
    Input("progress-interval", "n_intervals"),
    State("execution-state-store", "data"),
    prevent_initial_call=True,
)
def stream_progress(_tick, state):
    """Move any new stdout lines into the JSON store expected by the UI."""
    if state is None:
        state = {
            "status":   "running",
            "stage":    "Initializing",
            "progress": 0.0,
            "output":   [],
            "error":    None,
            "remaining":"calculating...",
        }

    changed = False
    while not progress_queue.empty():
        line = progress_queue.get_nowait()
        changed = True

        # --- process finished? ---------------------------------------------
        if line.startswith("[EXITCODE]"):
            rc = int(line.split()[1])
            state["status"]   = "completed" if rc == 0 else "error"
            state["stage"]    = "Completed"
            state["progress"] = 100
            if rc != 0:
                state["error"] = f"Process returned {rc}"
            continue

        # --- normal log / progress line ------------------------------------
        state["output"].append(line)
        if len(state["output"]) > 500:
            state["output"] = state["output"][-500:]

        m = re.search(
            r'\[PROGRESS\] stage="([^"]+)" percent=([\d.]+) remaining="([^"]+)"',
            line,
        )
        if m:
            state["stage"]     = m.group(1)
            state["progress"]  = float(m.group(2))
            state["remaining"] = m.group(3)

    if not changed:
        raise PreventUpdate
    return state

# ------------------------------------------------------------
# DASHBOARD LAYOUT & COMPONENTS FOR MODEL EXECUTION
# ------------------------------------------------------------

model_execution_form = html.Div(
    [
        html.H5(
            "Model Execution Parameters",
            style={"color": nhs_colors["primary"], "marginBottom": "15px"},
        ),
        html.Label("Population Size:"),
        dbc.Input(
            id="population-input",
            type="number",
            min=1,
            max=10000,
            step=10,
            value=100,
            style={"marginBottom": "15px"},
        ),
        html.Label("Performance Options:"),
        dbc.Checklist(
            id="performance-options",
            options=[
                {"label": "Enable XAI", "value": "enable_xai"},
                {"label": "Performance Mode", "value": "performance_mode"},
                {"label": "Extreme Performance", "value": "extreme_performance"},
                {"label": "Force CPU (no GPU)", "value": "force_cpu"},
            ],
            value=[],
            style={"marginBottom": "15px"},
        ),
        html.Label("Memory Utilization (%):"),
        dcc.Slider(
            id="memory-util-slider",
            min=10,
            max=90,
            step=5,
            marks={i: f"{i}%" for i in range(10, 91, 10)},
            value=70,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("CPU Utilization (%):", style={"marginTop": "15px"}),
        dcc.Slider(
            id="cpu-util-slider",
            min=10,
            max=90,
            step=5,
            marks={i: f"{i}%" for i in range(10, 91, 10)},
            value=80,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Number of Threads:", style={"marginTop": "15px"}),
        dcc.Slider(
            id="threads-slider",
            min=1,
            max=16,
            step=1,
            marks={i: f"{i}" for i in range(1, 17, 1)},
            value=4,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(
            [
                dbc.Button(
                    "Run Model",
                    id="run-model-btn",
                    color="primary",
                    className="mr-1",
                    style={"marginTop": "20px", "width": "100%"},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Button(
                    "Cancel Execution",
                    id="cancel-model-btn",
                    color="danger",
                    className="mr-1",
                    style={"marginTop": "10px", "display": "none", "width": "100%"},
                ),
            ],
            id="cancel-btn-container",
        ),
    ],
    style={
        "backgroundColor": nhs_colors["secondary"],
        "padding": "15px",
        "borderRadius": "5px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
    },
)

progress_visualization = html.Div(
    [
        html.Div(
            [
                html.H4("Model Execution Progress", style={"marginBottom": "15px"}),
                html.Div(
                    [
                        html.Label("Current Stage:"),
                        html.Div(
                            id="current-stage",
                            style={
                                "padding": "10px",
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "5px",
                                "marginBottom": "15px",
                                "fontWeight": "bold",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            id="progress-bar-with-stages",
                            style={"marginBottom": "30px", "position": "relative"},
                        )
                    ]
                ),
                html.Label("Execution Log:"),
                dbc.Card(
                    dbc.CardBody(
                        html.Div(
                            id="execution-log",
                            style={
                                "maxHeight": "400px",
                                "overflowY": "auto",
                                "whiteSpace": "pre-line",
                                "fontFamily": "monospace",
                                "fontSize": "12px",
                            },
                        )
                    ),
                    style={"marginBottom": "15px"},
                ),
                html.Div(
                    id="error-message",
                    style={"color": "red", "marginBottom": "15px", "display": "none"},
                ),
                html.Div(
                    id="execution-status",
                    style={"marginTop": "10px", "fontWeight": "bold"},
                ),
            ],
            style={
                "backgroundColor": nhs_colors["secondary"],
                "padding": "15px",
                "borderRadius": "5px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                "marginBottom": "15px",
            },
        ),
        html.Div(id="results-container", style={"display": "none"}),
    ],
    id="progress-container",
    style={"display": "none"},
)

tabs_component = dcc.Tabs(
    [
        dcc.Tab(
            label="Overview",
            children=[
                html.Div(
                    [
                        html.Div(
                            [filter_panel],
                            style={
                                "width": "20%",
                                "minWidth": "250px",
                                "marginRight": "15px",
                            },
                        ),
                        html.Div(
                            [
                                collapsible_card(
                                    "Patient Demographics & Risk Distribution",
                                    html.Div(
                                        id="demographics-container",
                                        children=[
                                            dcc.Loading(id="demographics-loading")
                                        ],
                                    ),
                                    "demographics",
                                    initially_open=True,
                                ),
                            ],
                            style={"width": "80%"},
                        ),
                    ],
                    style={"display": "flex"},
                )
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "borderBottom": "1px solid #d6d6d6",
                "padding": "6px",
            },
        ),
        dcc.Tab(
            label="Health Indices",
            children=[
                html.Div(
                    id="health-indices-tab-content",
                    children=[dcc.Loading(id="health-indices-loading")],
                )
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "borderBottom": "1px solid #d6d6d6",
                "padding": "6px",
            },
        ),
        dcc.Tab(
            label="Geographic Distribution",
            children=[
                html.Div(id="geo-tab-content", children=[dcc.Loading(id="geo-loading")])
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "borderBottom": "1px solid #d6d6d6",
                "padding": "6px",
            },
        ),
        dcc.Tab(
            label="Model Performance",
            children=[
                html.Div(
                    id="model-tab-content", children=[dcc.Loading(id="model-loading")]
                )
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "borderBottom": "1px solid #d6d6d6",
                "padding": "6px",
            },
        ),
        dcc.Tab(
            label="XAI Insights",
            children=[
                html.Div(id="xai-tab-content", children=[dcc.Loading(id="xai-loading")])
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "borderBottom": "1px solid #d6d6d6",
                "padding": "6px",
            },
        ),
        dcc.Tab(
            label="Patient Data",
            children=[
                html.Div(
                    id="patient-tab-content",
                    children=[dcc.Loading(id="patient-loading")],
                )
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "borderBottom": "1px solid #d6d6d6",
                "padding": "6px",
            },
        ),
        dcc.Tab(
            label="Model Execution",
            children=[
                html.Div(
                    [
                        html.H3(
                            "Model Execution and Patient Generation",
                            style={
                                "color": nhs_colors["primary"],
                                "marginBottom": "15px",
                            },
                        ),
                        html.P(
                            "Generate synthetic patients and run prediction models with customizable parameters.",
                            style={"marginBottom": "20px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [model_execution_form],
                                    style={
                                        "width": "30%",
                                        "minWidth": "300px",
                                        "marginRight": "15px",
                                    },
                                ),
                                html.Div(
                                    [progress_visualization], style={"width": "70%"}
                                ),
                            ],
                            style={"display": "flex"},
                        ),
                    ],
                    style={"padding": "15px"},
                )
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "borderBottom": "1px solid #d6d6d6",
                "padding": "6px",
            },
        ),
    ],
    style={"borderBottom": "1px solid #d6d6d6"},
)

# For high-end systems, also create a custom CSS to improve UI responsiveness
if PERF_SETTINGS["is_high_end"]:
    performance_css = """
        .dash-graph {
            will-change: transform;
        }
        .dash-spreadsheet-container {
            will-change: transform;
        }
        .dash-table-container {
            will-change: transform;
        }
        .dash-graph .main-svg {
            transform: translateZ(0);
        }
    """
else:
    performance_css = ""

app.layout = html.Div(
    [
        dcc.Markdown(
            """
        <style>
            .stage-marker .stage-icon-container {
                transition: all 0.3s ease;
            }
            .stage-marker .stage-icon-container.active {
                box-shadow: 0 0 10px rgba(0, 94, 184, 0.7);
            }
            .stage-marker .stage-icon-container.completed {
                background-color: #52A373 !important;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            .stage-marker.active .stage-icon-container {
                animation: pulse 2s infinite;
            }
            """
            + performance_css
            + """
        </style>
    """,
            dangerously_allow_html=True,
        ),
        html.Div(
            [
                html.H1(
                    "VITAI Healthcare Analytics Dashboard",
                    style={"color": nhs_colors["primary"], "marginBottom": "5px"},
                ),
                html.P(
                    "Comprehensive Patient Health Analytics & Predictive Modeling",
                    style={"color": nhs_colors["text"], "marginBottom": "15px"},
                ),
                kpi_row,
            ],
            style={
                "backgroundColor": nhs_colors["secondary"],
                "padding": "15px 20px",
                "borderRadius": "5px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                "marginBottom": "15px",
            },
        ),
        tabs_component,
        dcc.Store(id="filtered-data-store", storage_type="memory"),
        dcc.Store(id="selected-model-store", storage_type="memory"),
        dcc.Store(id="execution-state-store", storage_type="memory"),
        dcc.Interval(
            id="progress-interval", interval=300, disabled=True  # in milliseconds
        ),
        dcc.Interval(id="initial-load-trigger", interval=100, max_intervals=1),
        html.Div(id="memory-management", style={"display": "none"}),
        patient_modal,
    ],
    style={"backgroundColor": nhs_colors["background"], "padding": "15px"},
)


@app.callback(
    Output("filtered-data-store", "data"),
    [
        Input("apply-filters-btn", "n_clicks"),
        Input("initial-load-trigger", "n_intervals"),
    ],
    [
        State("global-model-dropdown", "value"),
        State("global-risk-dropdown", "value"),
        State("age-range-slider", "value"),
        State("income-range-slider", "value"),
        State("health-index-slider", "value"),
    ],
)
def filter_data(
    n_clicks, n_intervals, model, risk, age_range, income_range, health_range
):
    ctx = callback_context
    if not ctx.triggered:
        return df_all.to_dict("records")

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "initial-load-trigger" and n_intervals is not None:
        return df_all.to_dict("records")

    if n_clicks is None and trigger_id == "apply-filters-btn":
        return df_all.to_dict("records")

    filtered_df = df_all.copy()
    if model != "All" and "Group" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Group"] == model]
    if risk != "All" and "Risk_Category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Risk_Category"] == risk]
    if "AGE" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["AGE"] >= age_range[0]) & (filtered_df["AGE"] <= age_range[1])
        ]
    if "INCOME" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["INCOME"] >= income_range[0])
            & (filtered_df["INCOME"] <= income_range[1])
        ]
    if "Health_Index" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["Health_Index"] >= health_range[0])
            & (filtered_df["Health_Index"] <= health_range[1])
        ]
    filtered_df = smart_sample_dataframe(
        filtered_df, max_points=10000, method="stratified"
    )
    return filtered_df.to_dict("records")


@app.callback(
    Output("demographics-container", "children"), [Input("filtered-data-store", "data")]
)
def update_demographics_content(filtered_data):
    if filtered_data is None:
        df_to_use = df_all
    else:
        df_to_use = pd.DataFrame(filtered_data)

    if df_to_use.empty:
        return html.Div(
            "No data available after filtering",
            style={"padding": "20px", "textAlign": "center"},
        )

    gender_chart, age_chart, risk_chart = create_demographic_charts(df_to_use)
    race_chart, healthcare_expense_chart = create_race_demographics(df_to_use)

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(figure=gender_chart)
                        if gender_chart
                        else "No gender data available",
                        width=12,
                        md=4,
                    ),
                    dbc.Col(
                        dcc.Graph(figure=age_chart)
                        if age_chart
                        else "No age data available",
                        width=12,
                        md=4,
                    ),
                    dbc.Col(
                        dcc.Graph(figure=risk_chart)
                        if risk_chart
                        else "No risk data available",
                        width=12,
                        md=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(figure=race_chart)
                        if race_chart
                        else "No race data available",
                        width=12,
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Graph(figure=healthcare_expense_chart)
                        if healthcare_expense_chart
                        else "No healthcare expense data available",
                        width=12,
                        md=6,
                    ),
                ],
                style={"marginTop": "15px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(figure=income_health_fig)
                        if income_health_fig
                        else "No income-health data available",
                        width=12,
                    ),
                ],
                style={"marginTop": "15px"},
            ),
        ]
    )


@app.callback(
    Output("health-indices-tab-content", "children"),
    [Input("filtered-data-store", "data")],
)
def load_health_indices_content(filtered_data):
    if filtered_data is None:
        df_to_use = df_all
    else:
        df_to_use = pd.DataFrame(filtered_data)
    if df_to_use.empty:
        return html.Div(
            "No data available after filtering",
            style={"padding": "20px", "textAlign": "center"},
        )
    indices_fig = create_indices_comparison(df_to_use)
    health_trend = create_health_trend_chart(df_to_use)
    corr_matrix = create_correlation_matrix(df_to_use)
    return html.Div(
        [
            collapsible_card(
                "Health Indices Comparison",
                dcc.Graph(figure=indices_fig)
                if indices_fig
                else "No health indices data available",
                "health-indices",
                initially_open=True,
            ),
            collapsible_card(
                "Health Trends by Age Group",
                dcc.Graph(figure=health_trend)
                if health_trend
                else "No health trend data available",
                "health-trends",
                initially_open=True,
            ),
            collapsible_card(
                "Feature Correlation Matrix",
                dcc.Graph(figure=corr_matrix)
                if corr_matrix
                else "No correlation data available",
                "correlation-matrix",
                initially_open=True,
            ),
        ]
    )


@app.callback(
    Output("geo-tab-content", "children"), [Input("filtered-data-store", "data")]
)
def load_geo_content(filtered_data):
    if filtered_data is None:
        df_to_use = df_all
    else:
        df_to_use = pd.DataFrame(filtered_data)
    if df_to_use.empty:
        return html.Div(
            "No data available after filtering",
            style={"padding": "20px", "textAlign": "center"},
        )
    map_fig = create_compact_map(df_to_use, height=600)
    return html.Div(
        [
            collapsible_card(
                "Geographic Distribution of Patients",
                dcc.Graph(figure=map_fig)
                if map_fig
                else "No geographic data available",
                "geo-map",
                initially_open=True,
            ),
            html.Div(
                [
                    html.H5(
                        "Geographic Health Risk Concentration",
                        style={"marginBottom": "15px"},
                    ),
                    html.P(
                        "This heatmap displays the concentration of patients by location, with higher intensity (red) indicating areas with higher concentration of high-risk patients. Areas in yellow to orange indicate moderate risk concentration, while areas in blue to green indicate lower risk concentration."
                    ),
                ],
                style={
                    "padding": "15px",
                    "backgroundColor": "white",
                    "borderRadius": "5px",
                    "marginTop": "15px",
                },
            ),
        ]
    )


@app.callback(
    Output("model-tab-content", "children"), [Input("filtered-data-store", "data")]
)
def load_model_performance_content(filtered_data):
    model_cards = []
    for label, model_data in final_models_data.items():
        metrics = model_data["metrics"]
        metrics_rows = [
            html.Div(
                [
                    html.Strong("Test MSE: "),
                    html.Span(
                        f"{metrics['test_mse']:.4f}"
                        if isinstance(metrics["test_mse"], (int, float))
                        else metrics["test_mse"]
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Strong("Test R²: "),
                    html.Span(
                        f"{metrics['test_r2']:.4f}"
                        if isinstance(metrics["test_r2"], (int, float))
                        else metrics["test_r2"]
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Strong("Silhouette Score: "),
                    html.Span(
                        f"{metrics['Silhouette']:.4f}"
                        if isinstance(metrics["Silhouette"], (int, float))
                        else metrics["Silhouette"]
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
        ]
        visualizations = []
        if model_data["tsne_img"]:
            visualizations.append(
                dbc.Col(
                    html.Img(src=model_data["tsne_img"], style={"width": "100%"}),
                    width=6,
                )
            )
        if model_data["umap_img"]:
            visualizations.append(
                dbc.Col(
                    html.Img(src=model_data["umap_img"], style={"width": "100%"}),
                    width=6,
                )
            )

        if not model_data["df"].empty:
            df = model_data["df"]
            perf_fig = px.scatter(
                df,
                x="Health_Index",
                y="PredictedHI_final",
                color="Cluster_final" if "Cluster_final" in df.columns else None,
                color_continuous_scale=["#5A9BD5", "#52A373", "#F0A860", "#E66C6C"],
                opacity=0.7,
                title=f"Health Index and Prediction ({label}) - {len(df)} patients",
                hover_data=["Id"],
            )

            x_range = [df["Health_Index"].min(), df["Health_Index"].max()]
            perf_fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=x_range,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name="Perfect Prediction",
                )
            )

            model_plot = dcc.Graph(figure=perf_fig)
        else:
            model_plot = "No prediction data available"

        model_card = collapsible_card(
            f"{label} Model Performance",
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(html.Div(metrics_rows), width=12, md=4),
                            dbc.Col(model_plot, width=12, md=8),
                        ]
                    ),
                    dbc.Row(visualizations, style={"marginTop": "15px"})
                    if visualizations
                    else None,
                ]
            ),
            f"model-{label.lower().replace(' ', '-')}",
            initially_open=True,
        )
        model_cards.append(model_card)
    if not model_cards:
        return html.Div(
            "No model performance data available",
            style={"padding": "20px", "textAlign": "center"},
        )
    return html.Div(model_cards)


@app.callback(
    Output("xai-tab-content", "children"), [Input("filtered-data-store", "data")]
)
def load_xai_content(filtered_data):
    return html.Div(
        [
            html.H4("Explainable AI Insights", style={"marginBottom": "15px"}),
            html.Div(
                [
                    html.Label("Select Model:"),
                    xai_dropdown,
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                id="xai-content-display",
                children=[
                    html.Div(
                        "Select a model to view XAI insights",
                        style={
                            "padding": "20px",
                            "textAlign": "center",
                            "color": nhs_colors["text"],
                        },
                    )
                ],
            ),
        ],
        style={"padding": "15px", "backgroundColor": "white", "borderRadius": "5px"},
    )


@app.callback(
    Output("xai-content-display", "children"), [Input("xai-model-dropdown", "value")]
)
def update_xai_insights(selected_model):
    if not selected_model:
        return html.Div(
            "Please select a model to view XAI insights",
            style={
                "padding": "20px",
                "textAlign": "center",
                "color": nhs_colors["text"],
            },
        )
    model_data = None
    for grp in final_groups:
        if grp["label"] == selected_model:
            model_name = grp["model"]
            model_dir = os.path.join(EXPLAIN_XAI_DIR, model_name)
            break
    shap_summary_png = (
        os.path.join(model_dir, f"{model_name}_shap_summary.png")
        if "model_dir" in locals()
        else None
    )
    feat_imp_png = (
        os.path.join(model_dir, f"{model_name}_feature_importance.png")
        if "model_dir" in locals()
        else None
    )
    shap_img = (
        encode_image(shap_summary_png)
        if shap_summary_png and os.path.exists(shap_summary_png)
        else None
    )
    feat_img = (
        encode_image(feat_imp_png)
        if feat_imp_png and os.path.exists(feat_imp_png)
        else None
    )
    content = []
    if shap_img:
        content.append(
            html.Div(
                [
                    html.H5("SHAP Summary Plot", style={"marginBottom": "10px"}),
                    html.Img(src=shap_img, style={"width": "100%"}),
                ],
                style={"marginBottom": "20px"},
            )
        )
    if feat_img:
        content.append(
            html.Div(
                [
                    html.H5("Feature Importance", style={"marginBottom": "10px"}),
                    html.Img(src=feat_img, style={"width": "100%"}),
                ],
                style={"marginBottom": "20px"},
            )
        )
    try:
        explanation = format_explanation(selected_model)
        if explanation:
            content.append(
                html.Div(
                    [
                        html.H5("Model Explanation", style={"marginBottom": "10px"}),
                        html.Div(explanation),
                    ]
                )
            )
    except Exception as e:
        logger.error(f"Error formatting explanation: {e}")
    if not content:
        content.append(
            html.Div(
                "No XAI insights available for this model",
                style={"padding": "20px", "textAlign": "center"},
            )
        )
    return html.Div(
        content,
        style={"padding": "15px", "backgroundColor": "white", "borderRadius": "5px"},
    )


@app.callback(
    Output("patient-tab-content", "children"), [Input("filtered-data-store", "data")]
)
def load_patient_data_content(filtered_data):
    if filtered_data is None:
        df_to_use = df_all
    else:
        df_to_use = pd.DataFrame(filtered_data)
    if df_to_use.empty:
        return html.Div(
            "No data available after filtering",
            style={"padding": "20px", "textAlign": "center"},
        )
    display_df = df_to_use.head(1000)
    display_cols = [
        col
        for col in display_df.columns
        if col
        in [
            "Id",
            "AGE",
            "GENDER",
            "INCOME",
            "Health_Index",
            "CharlsonIndex",
            "ElixhauserIndex",
            "Risk_Category",
            "Group",
            "PredictedHI_final",
            "Cluster_final",
        ]
    ]
    if not display_cols:
        display_cols = display_df.columns[:8]
    sanitized_data = sanitize_datatable_values(display_df[display_cols])
    return html.Div(
        [
            html.H4(
                f"Patient Data ({len(display_df)} of {len(df_to_use)} patients shown)",
                style={"marginBottom": "15px"},
            ),
            html.Div(
                [
                    dbc.Button(
                        "Export CSV",
                        id="export-csv-btn",
                        color="primary",
                        className="me-1",
                        style={"marginRight": "10px", "marginBottom": "15px"},
                    ),
                    dcc.Download(id="download-dataframe-csv"),
                ]
            ),
            dash_table.DataTable(
                id="patient-data-table",
                columns=[{"name": i, "id": i} for i in display_cols],
                data=sanitized_data.to_dict("records"),
                page_size=15,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                style_table={"overflowX": "auto"},
                style_cell={
                    "minWidth": "100px",
                    "maxWidth": "300px",
                    "whiteSpace": "normal",
                    "textAlign": "left",
                },
                style_header={
                    "backgroundColor": nhs_colors["primary"],
                    "color": nhs_colors["secondary"],
                    "fontWeight": "bold",
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(245, 245, 250)",
                    }
                ],
            ),
            html.Div(id="click-patient-info", style={"marginTop": "20px"}),
        ],
        style={"padding": "15px", "backgroundColor": "white", "borderRadius": "5px"},
    )


@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("export-csv-btn", "n_clicks")],
    [State("filtered-data-store", "data")],
)
def export_csv(n_clicks, filtered_data):
    if n_clicks is None or not n_clicks or filtered_data is None:
        raise PreventUpdate
    df_to_export = pd.DataFrame(filtered_data)
    return dcc.send_data_frame(df_to_export.to_csv, "vitai_patient_data.csv")


@app.callback(
    [Output("patient-modal", "is_open"), Output("patient-detail-body", "children")],
    [Input("patient-data-table", "active_cell")],
    [State("patient-data-table", "data"), State("patient-modal", "is_open")],
)
def display_patient_details(active_cell, table_data, is_open):
    if active_cell is None:
        return is_open, no_update
    row_id = active_cell["row"]
    patient_data = table_data[row_id]
    details = []
    patient_id = patient_data.get("Id", "Unknown")
    for key, value in patient_data.items():
        if key != "Id":
            details.append(
                html.Div(
                    [html.Strong(f"{key}: "), html.Span(f"{value}")],
                    style={"marginBottom": "8px"},
                )
            )
    content = html.Div(
        [
            html.H4(f"Patient ID: {patient_id}", style={"marginBottom": "20px"}),
            html.Div(details),
        ]
    )
    return True, content


@app.callback(Output("close-modal", "n_clicks"), [Input("close-modal", "n_clicks")])
def close_modal(n_clicks):
    if n_clicks:
        return None
    return n_clicks


if __name__ == "__main__":
    app.run_server(debug=True)

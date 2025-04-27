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
import re
from datetime import datetime
from typing import Optional, Union

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

from concurrent.futures import ThreadPoolExecutor

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

risk_colors = [
    nhs_colors["risk_verylow"],
    nhs_colors["risk_low"],
    nhs_colors["risk_medium"],
    nhs_colors["risk_high"],
]

intersectional_colors = ["#E66C6C", "#F0A860", "#52A373"]

def get_system_resources():
    """Detect system resources and optimize settings accordingly"""
    cpu_count = multiprocessing.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

    system_info = {
        "cpu_count": cpu_count,
        "memory_gb": total_memory_gb,
        "platform": platform.system(),
        "processor": platform.processor(),
        "is_high_end": cpu_count >= 8 and total_memory_gb >= 32,
    }

    is_high_end = system_info["is_high_end"]

    optimization_settings = {
        "max_threads": max(int(cpu_count * 0.8), 4),
        "memory_limit": int(total_memory_gb * 0.7),
        "batch_size": 10000 if is_high_end else 5000,
        "sampling_threshold": 50000 if is_high_end else 10000,
        "use_advanced_clustering": is_high_end,
        "webgl_threshold": 5000 if is_high_end else 1000,
        "is_high_end": is_high_end,
    }

    logger.info(f"System resources detected: {system_info}")
    logger.info(f"Performance settings: {optimization_settings}")

    return optimization_settings

PERF_SETTINGS = get_system_resources()

if os.path.exists(PICKLE_ALL):
    if PERF_SETTINGS["is_high_end"]:
        df_all = pd.read_pickle(PICKLE_ALL)

        def optimize_column_dtypes(col):
            if col[1].dtype == "float64":
                return col[0], col[1].astype("float32")
            elif col[1].dtype == "int64":
                return col[0], col[1].astype("int32")
            return col[0], col[1]

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
    logger.warning("Enriched pickle not found; falling back to optimized CSV loading.")
    if PERF_SETTINGS["is_high_end"]:
        chunk_size = min(
            500000, int(PERF_SETTINGS["memory_limit"] * 1000000 / 50)
        )

        chunks = []
        for chunk in pd.read_csv(
            CSV_PATIENTS, memory_map=True, low_memory=True, chunksize=chunk_size
        ):
            chunk["BIRTHDATE"] = pd.to_datetime(chunk["BIRTHDATE"], errors="coerce")
            chunk["AGE"] = (
                ((pd.Timestamp("today") - chunk["BIRTHDATE"]).dt.days / 365.25)
                .fillna(0)
                .astype(int)
            )
            chunks.append(chunk)

        df_all = pd.concat(chunks)
        del chunks

        np.random.seed(42)
        with ThreadPoolExecutor(max_workers=PERF_SETTINGS["max_threads"]) as executor:
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

            df_all["Health_Index"] = health_index.result()
            df_all["CharlsonIndex"] = charlson_index.result()
            df_all["ElixhauserIndex"] = elixhauser_index.result()
            df_all["Cluster"] = cluster.result()

        df_all["Predicted_Health_Index"] = (
            df_all["Health_Index"] + np.random.normal(0, 0.5, len(df_all))
        ).round(2)
        df_all["Actual"] = df_all["Health_Index"]
    else:
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

def collapsible_card(title, content, card_id, initially_open=False):
    return html.Div(
        [
            html.H5(
                [
                    html.Span(title),
                    html.I(
                        className=f"fas {'fa-chevron-down' if initially_open else 'fa-chevron-right'} ml-2",
                        id=f"{card_id}-toggle-icon",
                    ),
                ],
                id=f"{card_id}-header",
                style={
                    "cursor": "pointer",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "padding": "10px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "5px",
                },
            ),
            html.Div(
                content,
                id=f"{card_id}-content",
                style={"display": "block" if initially_open else "none", "padding": "15px"},
            ),
        ],
        className="mb-3",
        style={"backgroundColor": "white", "borderRadius": "5px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
    )

def encode_image(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return None

def use_webgl_rendering(fig, threshold=None):
    if threshold is None:
        threshold = PERF_SETTINGS["webgl_threshold"]

    for trace in fig.data:
        if trace.type == "scatter" and hasattr(trace, "x") and len(trace.x) > threshold:
            trace.type = "scattergl"
    return fig

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

def smart_sample_dataframe(df, max_points=None, min_points=500, method="random"):
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
            if PERF_SETTINGS["use_advanced_clustering"] and len(df) > 50000:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 10:
                    numeric_cols = numeric_cols[:10]
                cluster_data = df[numeric_cols].fillna(df[numeric_cols].mean())

                n_clusters = min(int(math.sqrt(sample_size) * 1.5), 100)

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
                return df.sample(sample_size, random_state=42)
        except Exception as e:
            logger.warning(
                f"Advanced cluster sampling failed, falling back to random: {e}"
            )
            return df.sample(sample_size, random_state=42)
    return df.sample(sample_size, random_state=42)

def release_memory():
    pass

def create_compact_map(df, height=600, health_index_range=None, center=None, zoom=7):
    logger.info(f"Creating compact map. Input df shape: {df.shape if df is not None else 'None'}")
    if df is None or df.empty:
         logger.warning("Input dataframe for map is None or empty.")
         return go.Figure().update_layout(
             title="No data provided for map", height=height,
             xaxis={"visible": False}, yaxis={"visible": False},
             annotations=[{"text": "No data available", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
         )

    if "LAT" not in df.columns or "LON" not in df.columns:
        logger.warning("LAT or LON columns missing from dataframe.")
        return go.Figure().update_layout(
            title="Geographic data columns (LAT, LON) not found", height=height,
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{"text": "LAT/LON columns missing", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
        )

    df_map = df.dropna(subset=["LAT", "LON"])
    if health_index_range and "Health_Index" in df_map.columns:
        df_map = df_map[(df_map["Health_Index"] >= health_index_range[0]) & (df_map["Health_Index"] <= health_index_range[1])]

    logger.info(f"Dataframe shape after dropping NaN LAT/LON: {df_map.shape}")

    if len(df_map) == 0:
        logger.warning("No valid location data (non-NaN LAT/LON) available after filtering/dropping NaNs.")
        return go.Figure(layout={
            "title": "No Valid Geographic Data Points",
            "height": height,
            "mapbox": {
                "style": "carto-positron",
                "center": center if center else {"lat": 39.8283, "lon": -98.5795},
                "zoom": 3
            },
            "annotations": [{
                "text": "No patients with valid location data found in the current selection.",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "grey"}
            }]
        })

    if not center and not df_map.empty:
        center = dict(lat=df_map["LAT"].mean(), lon=df_map["LON"].mean())
    elif not center:
        center = dict(lat=39.8283, lon=-98.5795)

    try:
        fig = px.density_mapbox(
            df_map,
            lat="LAT",
            lon="LON",
            z="Health_Index" if "Health_Index" in df_map.columns else None,
            radius=15,
            center=center,
            zoom=zoom,
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

        if "Health_Index" in df_map.columns and len(df_map) > 100:
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
            mapbox_center=center,
            mapbox_zoom=zoom,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title="Health Risk",
                tickvals=[1, 3, 6, 9],
                ticktext=["Very Low", "Low", "Moderate", "High"],
            ),
        )
        logger.info("Successfully created geographic map.")
        return fig
    except Exception as e:
        logger.error(f"Error creating geographic map: {e}", exc_info=True)
        return go.Figure(layout={
            "title": "Error Generating Map",
            "height": height,
            "annotations": [{
                "text": f"An error occurred: {e}",
                "xref": "paper", "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "red"}
            }]
        })

def create_enhanced_geo_disparities_map(df, height=600, health_index_range=None, center=None, zoom=7):
    """Create a visualization of geographic health disparities by demographic factors."""
    logger.info(f"Creating enhanced geo disparities map. Input df shape: {df.shape if df is not None else 'None'}")
    
    if df is None or df.empty:
         logger.warning("Input dataframe for disparities map is None or empty.")
         return go.Figure().update_layout(
             title="No data provided for disparities map", height=height,
             xaxis={"visible": False}, yaxis={"visible": False},
             annotations=[{"text": "No data available", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
         )

    if "LAT" not in df.columns or "LON" not in df.columns:
        logger.warning("LAT or LON columns missing from dataframe.")
        return go.Figure().update_layout(
            title="Geographic data columns (LAT, LON) not found", height=height,
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{"text": "LAT/LON columns missing", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
        )
    
    # Check available demographic columns
    has_income = "INCOME" in df.columns
    has_race = "RACE" in df.columns
    has_health_index = "Health_Index" in df.columns
    
    df_map = df.dropna(subset=["LAT", "LON"])
    
    if health_index_range and has_health_index:
        df_map = df_map[(df_map["Health_Index"] >= health_index_range[0]) & (df_map["Health_Index"] <= health_index_range[1])]
    
    if len(df_map) == 0:
        logger.warning("No valid location data available after filtering/dropping NaNs.")
        return go.Figure(layout={
            "title": "No Valid Geographic Data Points for Disparities Analysis",
            "height": height,
            "mapbox": {
                "style": "carto-positron",
                "center": center if center else {"lat": 39.8283, "lon": -98.5795},
                "zoom": 3
            },
            "annotations": [{
                "text": "No patients with valid location data found in the current selection.",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "grey"}
            }]
        })

    if not center and not df_map.empty:
        center = dict(lat=df_map["LAT"].mean(), lon=df_map["LON"].mean())
    elif not center:
        center = dict(lat=39.8283, lon=-98.5795)
    
    # Create sampled dataset for visualization (avoid too many points)
    df_sample = smart_sample_dataframe(df_map, max_points=5000, method="stratified")
    
    # Create visualization based on available demographics
    if has_income and has_health_index:
        # Create income quartiles for coloring
        df_sample['Income_Quartile'] = pd.qcut(
            df_sample['INCOME'], 
            q=4, 
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        ).astype(str)
        
        # Create scatter map showing health index by income quartile
        fig = px.scatter_mapbox(
            df_sample,
            lat="LAT",
            lon="LON",
            color="Income_Quartile",
            color_discrete_map={
                "Q1 (Low)": "#E66C6C",    # Red for low income
                "Q2": "#F0A860",          # Orange for lower-middle income
                "Q3": "#52A373",          # Green for upper-middle income
                "Q4 (High)": "#5A9BD5",   # Blue for high income
            },
            size="Health_Index",
            size_max=15,
            opacity=0.7,
            title="Income-Based Health Disparities Map",
            mapbox_style="carto-positron",
            height=height,
            hover_data=["INCOME", "Health_Index"]
        )
        
        # Add a legend title
        fig.update_layout(
            legend_title="Income Quartile",
            mapbox=dict(
                center=center,
                zoom=zoom
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
        
    elif has_race and has_health_index:
        # Get the top races to avoid too many categories
        top_races = df_map['RACE'].value_counts().nlargest(5).index.tolist()
        df_filtered = df_sample[df_sample['RACE'].isin(top_races)]
        
        if not df_filtered.empty:
            # Create scatter map with racial health disparities
            fig = px.scatter_mapbox(
                df_filtered,
                lat="LAT",
                lon="LON",
                color="RACE",
                size="Health_Index",
                size_max=15,
                opacity=0.7,
                title="Racial Health Disparities Map",
                mapbox_style="carto-positron",
                height=height,
                hover_data=["RACE", "Health_Index"]
            )
            
            # Add a legend title
            fig.update_layout(
                legend_title="Race",
                mapbox=dict(
                    center=center,
                    zoom=zoom
                ),
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
            )
        else:
            fig = go.Figure(layout={
                "title": "Insufficient race data for disparities analysis",
                "height": height,
                "mapbox": {
                    "style": "carto-positron",
                    "center": center,
                    "zoom": zoom
                },
                "margin": {"r": 0, "t": 40, "l": 0, "b": 0},
            })
    else:
        # Create a fallback map if we don't have income or race data with health index
        if has_health_index:
            # Create scatter map of health index
            fig = px.scatter_mapbox(
                df_sample,
                lat="LAT",
                lon="LON",
                color="Health_Index",
                color_continuous_scale=[
                    [0, "#5A9BD5"],    # Very Low Risk - blue
                    [0.3, "#52A373"],   # Low Risk - green
                    [0.6, "#F0A860"],   # Moderate Risk - orange
                    [1, "#E66C6C"],    # High Risk - red
                ],
                opacity=0.7,
                title="Geographic Health Index Distribution (Limited demographic data)",
                mapbox_style="carto-positron",
                height=height,
                zoom=zoom,
                center=center,
            )
            
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="Health Index",
                    tickvals=[1, 3, 6, 9],
                    ticktext=["Very Low", "Low", "Moderate", "High"],
                ),
                mapbox=dict(
                    center=center,
                    zoom=zoom
                ),
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
            )
        else:
            fig = go.Figure(layout={
                "title": "Insufficient data for health disparities analysis",
                "height": height,
                "mapbox": {
                    "style": "carto-positron",
                    "center": center,
                    "zoom": zoom
                },
                "margin": {"r": 0, "t": 40, "l": 0, "b": 0},
            })
            
    # Add a note about health disparities
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text="Health disparities are differences in health outcomes between different demographic groups.",
        showarrow=False,
        font=dict(size=10, color="black"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    logger.info("Successfully created enhanced geographic disparities map.")
    return fig

def create_health_inequality_chart(df, max_points=3000):
    if df is None or df.empty:
        return None
        
    required_cols = ["Health_Index", "INCOME", "RACE", "AGE"]
    if not all(col in df.columns for col in required_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required data for health inequality analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=400, title="Health Inequality Analysis - Missing Data")
        return fig
        
    if len(df) > max_points:
        df_sample = df.sample(max_points, random_state=42)
    else:
        df_sample = df.copy()
        
    df_sample['Income_Quartile'] = pd.qcut(
        df_sample['INCOME'], 
        q=4, 
        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    )
    
    income_health = df_sample.groupby('Income_Quartile')['Health_Index'].agg(['mean', 'count']).reset_index()
    income_health.columns = ['Income_Quartile', 'Avg_Health_Index', 'Count']
    
    total_count = income_health['Count'].sum()
    income_health['Percentage'] = (income_health['Count'] / total_count * 100).round(1)
    
    if len(income_health) >= 4:
        highest_health = income_health.iloc[-1]['Avg_Health_Index']
        lowest_health = income_health.iloc[0]['Avg_Health_Index']
        inequality_ratio = highest_health / lowest_health if lowest_health > 0 else float('nan')
    else:
        inequality_ratio = float('nan')
    
    fig = px.bar(
        income_health,
        x='Income_Quartile',
        y='Avg_Health_Index',
        color='Income_Quartile',
        color_discrete_sequence=['#5A9BD5', '#52A373', '#F0A860', '#E66C6C'],
        text='Percentage',
        labels={
            'Income_Quartile': 'Income Quartile',
            'Avg_Health_Index': 'Average Health Index'
        },
        title='Health Inequality Across Income Groups'
    )
    
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    
    if not pd.isna(inequality_ratio):
        fig.add_annotation(
            text=f"Health Inequality Ratio: {inequality_ratio:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=12, color="red" if inequality_ratio > 1.5 else "black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
    
    fig.update_layout(
        xaxis_title="Income Quartile",
        yaxis_title="Average Health Index",
        xaxis={'categoryorder': 'array', 'categoryarray': ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']},
        height=400
    )
    
    return fig

def sanitize_datatable_values(df):
    """Sanitize values for display in DataTable"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df_copy = df.copy()
    
    # Convert numeric values to formatted strings
    for col in df_copy.select_dtypes(include=['float']).columns:
        df_copy[col] = df_copy[col].round(2).astype(str)
    
    # Convert date columns to formatted strings
    date_cols = [col for col in df_copy.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        if col in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
    
    # Replace NaN values with empty strings
    df_copy = df_copy.fillna('')
    
    return df_copy

def create_demographic_charts(df):
    """Create charts for demographic data"""
    if df is None or df.empty:
        return None, None, None
    
    # Gender distribution
    if "GENDER" in df.columns:
        gender_counts = df["GENDER"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        gender_fig = px.pie(
            gender_counts,
            values="Count",
            names="Gender",
            title="Gender Distribution",
            color_discrete_sequence=["#5A9BD5", "#F0A860", "#52A373"],
        )
        gender_fig.update_layout(height=300)
    else:
        gender_fig = None
    
    # Age distribution
    if "Age_Group" in df.columns:
        age_counts = df["Age_Group"].value_counts().reset_index()
        age_counts.columns = ["Age Group", "Count"]
        age_fig = px.bar(
            age_counts,
            x="Age Group",
            y="Count",
            title="Age Distribution",
            color_discrete_sequence=["#5A9BD5"],
        )
        age_fig.update_layout(height=300)
    else:
        age_fig = None
    
    # Risk distribution
    if "Risk_Category" in df.columns:
        risk_counts = df["Risk_Category"].value_counts().reset_index()
        risk_counts.columns = ["Risk Category", "Count"]
        risk_fig = px.bar(
            risk_counts,
            x="Risk Category",
            y="Count",
            title="Risk Distribution",
            color="Risk Category",
            color_discrete_map={
                "Very Low Risk": nhs_colors["risk_verylow"],
                "Low Risk": nhs_colors["risk_low"],
                "Moderate Risk": nhs_colors["risk_medium"],
                "High Risk": nhs_colors["risk_high"],
            },
        )
        risk_fig.update_layout(height=300)
    else:
        risk_fig = None
    
    return gender_fig, age_fig, risk_fig

def create_race_demographics(df):
    """Create race distribution and healthcare expense charts"""
    if df is None or df.empty:
        return None, None
    
    # Race distribution
    if "RACE" in df.columns:
        race_counts = df["RACE"].value_counts().reset_index()
        race_counts.columns = ["Race", "Count"]
        race_fig = px.pie(
            race_counts,
            values="Count",
            names="Race",
            title="Race Distribution",
            hole=0.4,
        )
        race_fig.update_layout(height=400)
    else:
        race_fig = None
    
    # Healthcare expenses
    if "HEALTHCARE_EXPENSES" in df.columns and "Risk_Category" in df.columns:
        expenses_by_risk = df.groupby("Risk_Category")["HEALTHCARE_EXPENSES"].mean().reset_index()
        expense_fig = px.bar(
            expenses_by_risk,
            x="Risk_Category",
            y="HEALTHCARE_EXPENSES",
            title="Average Healthcare Expenses by Risk Category",
            color="Risk_Category",
            color_discrete_map={
                "Very Low Risk": nhs_colors["risk_verylow"],
                "Low Risk": nhs_colors["risk_low"],
                "Moderate Risk": nhs_colors["risk_medium"],
                "High Risk": nhs_colors["risk_high"],
            },
        )
        expense_fig.update_layout(height=400)
        expense_fig.update_yaxes(title="Average Expenses ($)")
    else:
        expense_fig = None
    
    return race_fig, expense_fig

def create_income_health_chart(df):
    """Create a chart showing the relationship between income and health index"""
    if df is None or df.empty or "INCOME" not in df.columns or "Health_Index" not in df.columns:
        return None
    
    # Sample data if it's too large
    if len(df) > 5000:
        df_sample = df.sample(5000, random_state=42)
    else:
        df_sample = df
    
    # Create scatter plot
    fig = px.scatter(
        df_sample,
        x="INCOME",
        y="Health_Index",
        color="Risk_Category" if "Risk_Category" in df_sample.columns else None,
        color_discrete_map={
            "Very Low Risk": nhs_colors["risk_verylow"],
            "Low Risk": nhs_colors["risk_low"],
            "Moderate Risk": nhs_colors["risk_medium"],
            "High Risk": nhs_colors["risk_high"],
        },
        title="Income vs. Health Index",
        opacity=0.7,
    )
    
    # Add trend line
    if len(df_sample) > 10:
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_sample["INCOME"], df_sample["Health_Index"]
            )
            x_range = [df_sample["INCOME"].min(), df_sample["INCOME"].max()]
            y_range = [slope * x + intercept for x in x_range]
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name=f"Trend Line (r={r_value:.2f})",
                )
            )
        except Exception as e:
            logger.warning(f"Could not add trend line: {e}")
    
    fig.update_layout(height=500)
    return fig

def create_intersectional_analysis(df):
    """Create charts for intersectional analysis"""
    if df is None or df.empty:
        return None
    
    charts = []
    
    # Gender and Race analysis
    if "GENDER" in df.columns and "RACE" in df.columns and "Health_Index" in df.columns:
        # Filter to include only the most frequent racial categories for clarity
        top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
        filtered_df = df[df["RACE"].isin(top_races)].copy()
        
        if not filtered_df.empty:
            # Calculate mean health index by gender and race
            grouped = filtered_df.groupby(["GENDER", "RACE"])["Health_Index"].mean().reset_index()
            
            # Create grouped bar chart
            fig1 = px.bar(
                grouped,
                x="RACE",
                y="Health_Index",
                color="GENDER",
                title="Health Index by Race and Gender",
                barmode="group",
                color_discrete_sequence=["#5A9BD5", "#F0A860"],
            )
            fig1.update_layout(height=400)
            charts.append(fig1)
    
    # Income and Race analysis
    if "RACE" in df.columns and "INCOME" in df.columns:
        # Filter to include only the most frequent racial categories
        top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
        filtered_df = df[df["RACE"].isin(top_races)].copy()
        
        if not filtered_df.empty:
            # Calculate mean income by race
            grouped = filtered_df.groupby("RACE")["INCOME"].mean().reset_index()
            
            # Create bar chart
            fig2 = px.bar(
                grouped,
                x="RACE",
                y="INCOME",
                title="Average Income by Race",
                color="RACE",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(height=400, showlegend=False)
            charts.append(fig2)
    
    return charts if charts else None

def create_indices_comparison(df):
    """Create a comparison chart for health indices"""
    if df is None or df.empty:
        return None
    
    required_cols = ["Health_Index", "CharlsonIndex", "ElixhauserIndex"]
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Sample data if too large
    if len(df) > 3000:
        df_sample = df.sample(3000, random_state=42)
    else:
        df_sample = df
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df_sample["CharlsonIndex"],
            y=df_sample["Health_Index"],
            mode="markers",
            name="Charlson vs Health",
            marker=dict(color="#5A9BD5", opacity=0.6),
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_sample["ElixhauserIndex"],
            y=df_sample["Health_Index"],
            mode="markers",
            name="Elixhauser vs Health",
            marker=dict(color="#F0A860", opacity=0.6),
        ),
        secondary_y=False,
    )
    
    # Add figure title, axis labels
    fig.update_layout(
        title_text="Health Indices Comparison",
        height=500,
    )
    
    # Set axis titles
    fig.update_xaxes(title_text="Index Value")
    fig.update_yaxes(title_text="Health Index", secondary_y=False)
    
    return fig

def create_health_trend_chart(df):
    """Create a chart showing health trends by age group"""
    if df is None or df.empty or "Age_Group" not in df.columns:
        return None
    
    health_metrics = [col for col in ["Health_Index", "CharlsonIndex", "ElixhauserIndex"] if col in df.columns]
    
    if not health_metrics:
        return None
    
    # Aggregate data by age group
    agg_dict = {metric: "mean" for metric in health_metrics}
    grouped = df.groupby("Age_Group").agg(agg_dict).reset_index()
    
    # Ensure age groups are in correct order
    age_order = ["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]
    grouped["Age_Group"] = pd.Categorical(grouped["Age_Group"], categories=age_order, ordered=True)
    grouped = grouped.sort_values("Age_Group")
    
    # Create line chart
    fig = go.Figure()
    
    for metric in health_metrics:
        fig.add_trace(
            go.Scatter(
                x=grouped["Age_Group"],
                y=grouped[metric],
                mode="lines+markers",
                name=metric,
            )
        )
    
    fig.update_layout(
        title="Health Metrics by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Index Value",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )
    
    return fig

def create_correlation_matrix(df):
    """Create a correlation matrix for numeric health data"""
    if df is None or df.empty:
        return None
    
    # Select relevant numerical columns
    numeric_cols = [
        col for col in ["Health_Index", "CharlsonIndex", "ElixhauserIndex", 
                       "AGE", "INCOME", "HEALTHCARE_EXPENSES"] 
        if col in df.columns
    ]
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix of Health Metrics",
        height=500,
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
    )
    
    return fig

def create_clinical_risk_clusters(df, model_name="Unknown"):
    """Create visualization of clinical risk clusters"""
    if df is None or df.empty:
        return None
    
    required_cols = ["Health_Index", "CharlsonIndex", "ElixhauserIndex"]
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Sample data if too large
    if len(df) > 3000:
        df_sample = df.sample(3000, random_state=42)
    else:
        df_sample = df
    
    # Add cluster if available, otherwise use Risk_Category
    color_col = "Cluster" if "Cluster" in df_sample.columns else "Risk_Category"
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df_sample,
        x="Health_Index",
        y="CharlsonIndex",
        z="ElixhauserIndex",
        color=color_col if color_col in df_sample.columns else None,
        title=f"Clinical Risk Clustering for {model_name} Model",
        opacity=0.7,
        color_discrete_sequence=risk_colors if "Cluster" in df_sample.columns else None,
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Health Index",
            yaxis_title="Charlson Index",
            zaxis_title="Elixhauser Index",
        ),
        height=700,
    )
    
    return fig

external_stylesheets = [dbc.themes.FLATLY]

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
    stages = [
        "Initializing",
        "GeneratingData",
        "ProcessingFiles", 
        "CalculatingIndices",
        "NormalizingFeatures",
        "RunningPredictions",
        "Completed"
    ]
    
    stage_percentages = {
        "Initializing": 5.0,
        "GeneratingData": 15.0,
        "ProcessingFiles": 30.0,
        "CalculatingIndices": 25.0,
        "NormalizingFeatures": 10.0,
        "RunningPredictions": 15.0,
        "Completed": 0.0
    }
    
    stage_markers = []
    cumulative_percentage = 0

    for i, stage in enumerate(stages):
        if stage == "Completed" and current_stage != "Completed":
            continue

        if i == 0:
            position_percent = 0
        elif stage == "Completed":
            position_percent = 100
        else:
            position_percent = sum([stage_percentages[s] for s in stages[:i]])

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
                "bottom": "-30px",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "width": "65px",
                "transition": "all 0.3s ease",
            },
            children=[
                html.Div(
                    className=f"stage-icon-container {'active' if is_active else ''} {'completed' if is_completed else ''}",
                    style={
                        "width": "35px",
                        "height": "35px",
                        "borderRadius": "50%",
                        "backgroundColor": nhs_colors["primary"]
                        if is_active
                        else ("#52A373" if is_completed else "#e0e0e0"),
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "color": "white",
                        "marginBottom": "8px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.3)"
                        if is_active
                        else "none",
                        "transform": "scale(1.3)" if is_active else "scale(1)",
                        "transition": "all 0.3s ease",
                        "zIndex": "10" if is_active else "1",
                        "animation": "pulse 2s infinite" if is_active else "none",
                    },
                    children=[html.I(className=progress_stage_icons[stage])],
                ),
                html.Span(
                    stage,
                    style={
                        "fontSize": "11px",
                        "textAlign": "center",
                        "fontWeight": "bold" if is_active else "normal",
                        "color": nhs_colors["primary"] if is_active else "#666",
                        "transition": "all 0.3s ease",
                        "maxWidth": "65px",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                    },
                ),
                html.Span(
                    f"{stage_percentages[stage]}%",
                    style={
                        "fontSize": "9px",
                        "color": "#888",
                        "marginTop": "2px",
                    },
                ) if stage_percentages[stage] > 0 else None,
            ],
        )
        stage_markers.append(stage_marker)

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
                eta, 
                style={
                    "fontSize": "18px",
                    "color": nhs_colors["primary"],
                    "fontWeight": "bold",
                }
            ),
        ],
        style={
            "position": "absolute",
            "right": "0",
            "top": "-50px",
            "backgroundColor": "white",
            "padding": "10px 15px",
            "borderRadius": "6px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
            "border": f"1px solid {nhs_colors['primary']}",
            "zIndex": "5",
        },
    ) if current_stage != "Completed" else None

    return html.Div(
        [
            eta_display,
            html.Div(
                style={
                    "width": "100%",
                    "height": "50px",
                    "position": "relative",
                    "marginBottom": "50px",
                },
                children=[
                    dbc.Progress(
                        value=overall_progress,
                        striped=True,
                        animated=True,
                        style={
                            "height": "12px",
                            "borderRadius": "6px",
                            "backgroundColor": "#e0e0e0",
                            "marginTop": "10px",
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
    },
)
cache = Cache(app.server)

cache_dir = os.path.join(BASE_DIR, "Dashboard", "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

cache_size_limit = (
    PERF_SETTINGS["memory_limit"] * 1024 * 1024 * 1024 / 2
)
long_callback_cache = diskcache.Cache(cache_dir, size_limit=cache_size_limit)
long_callback_manager = DiskcacheLongCallbackManager(long_callback_cache)

progress_queue: queue.Queue[str] = queue.Queue(maxsize=10_000)
process_handle: Optional[subprocess.Popen] = None
process_lock = threading.Lock()

def _launch_generate_and_predict(cmd: list[str]) -> None:
    global process_handle

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        creationflags=(
            subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        ),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ) as proc:
        with process_lock:
            process_handle = proc

        while True:
            raw = proc.stdout.readline()
            if not raw:
                break
            progress_queue.put_nowait(raw.rstrip("\n"))
            
            if "[PROGRESS]" in raw:
                sys.stdout.flush()

        progress_queue.put_nowait(f"[EXITCODE] {proc.returncode}")

    with process_lock:
        process_handle = None

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

    if trig == "cancel-model-btn":
        with process_lock:
            if process_handle and process_handle.poll() is None:
                try:
                    process_handle.send_signal(signal.SIGINT)
                    process_handle.wait(timeout=5)
                except Exception:
                    process_handle.kill()
        return True, False, {"display": "none"}

    if not run_clicks:
        raise PreventUpdate

    cmd = _build_command(population, perf_opts, mem_util, cpu_util, threads)
    threading.Thread(
        target=_launch_generate_and_predict, args=(cmd,), daemon=True
    ).start()

    return False, True, {"display": "block"}

@app.callback(
    Output("execution-state-store", "data"),
    Input("progress-interval", "n_intervals"),
    State("execution-state-store", "data"),
    prevent_initial_call=True,
)
def stream_progress(_tick, state):
    if state is None:
        state = {
            "status":   "running",
            "stage":    "Initializing",
            "progress": 0.0,
            "output":   [],
            "error":    None,
            "remaining":"calculating...",
            "last_update": time.time(),
        }

    changed = False
    while not progress_queue.empty():
        line = progress_queue.get_nowait()
        changed = True
        state["last_update"] = time.time()

        state["output"].append(line)
        if len(state["output"]) > 500:
            state["output"] = state["output"][-500:]

        if "[INFO] GenerateAndPredict completed successfully" in line:
            state["status"] = "completed"
            state["stage"] = "Completed"
            state["progress"] = 100
            continue

        if line.startswith("[EXITCODE]"):
            try:
                rc = int(line.split()[1])
                if state["status"] != "completed":
                    state["status"] = "completed" if rc == 0 else "error"
                    if rc != 0:
                        state["error"] = f"Process returned {rc}"
            except (ValueError, IndexError):
                if state["status"] != "completed":
                    state["status"] = "error"
                    state["error"] = "Process terminated with unknown exit code"
            
            if state["stage"] != "Completed":
                state["stage"] = "Completed"
                state["progress"] = 100
            continue

        m = re.search(
            r'\[PROGRESS\] stage="([^"]+)" percent=([\d.]+) remaining="([^"]+)"',
            line,
        )
        if m:
            state["stage"]     = m.group(1)
            state["progress"]  = float(m.group(2))
            state["remaining"] = m.group(3)

    if state["status"] == "running" and time.time() - state["last_update"] > 10:
        if state["remaining"] == "calculating...":
            state["remaining"] = "calculating... (awaiting updates)"
        elif "awaiting updates" not in state["remaining"]:
            state["remaining"] += " (awaiting updates)"

    if not changed:
        raise PreventUpdate
    return state

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

# Define default ranges for sliders, handle case where df_all might be empty initially
default_age_min = df_all["AGE"].min() if "AGE" in df_all.columns and not df_all.empty else 0
default_age_max = df_all["AGE"].max() if "AGE" in df_all.columns and not df_all.empty else 100
default_income_min = df_all["INCOME"].min() if "INCOME" in df_all.columns and not df_all.empty else 0
default_income_max = df_all["INCOME"].max() if "INCOME" in df_all.columns and not df_all.empty else 100000
default_health_min = df_all["Health_Index"].min() if "Health_Index" in df_all.columns and not df_all.empty else 0
default_health_max = df_all["Health_Index"].max() if "Health_Index" in df_all.columns and not df_all.empty else 10

# Define filter panel component
filter_panel = html.Div(
    [
        html.H5(
            "Dashboard Filters",
            style={"color": nhs_colors["primary"], "marginBottom": "15px"},
        ),
        html.Label("Model Group:"),
        dcc.Dropdown(
            id="global-model-dropdown",
            options=[{"label": "All", "value": "All"}] + [
                {"label": group["label"], "value": group["label"]} for group in final_groups
            ],
            value="All",
            clearable=False,
        ),
        html.Label("Risk Category:", style={"marginTop": "15px"}),
        dcc.Dropdown(
            id="global-risk-dropdown",
            options=[
                {"label": "All", "value": "All"},
                {"label": "High Risk", "value": "High Risk"},
                {"label": "Moderate Risk", "value": "Moderate Risk"},
                {"label": "Low Risk", "value": "Low Risk"},
                {"label": "Very Low Risk", "value": "Very Low Risk"},
            ],
            value="All",
            clearable=False,
        ),
        html.Label("Age Range:", style={"marginTop": "15px"}),
        dcc.RangeSlider(
            id="age-range-slider",
            min=default_age_min,
            max=default_age_max,
            step=1,
            marks={i: str(i) for i in range(int(default_age_min), int(default_age_max)+1, 20)},
            value=[default_age_min, default_age_max],
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Income Range:", style={"marginTop": "15px"}),
        dcc.RangeSlider(
            id="income-range-slider",
            min=default_income_min,
            max=default_income_max,
            step=1000,
            marks={i: f"£{i:,}" for i in range(int(default_income_min), int(default_income_max)+1, 25000)},
            value=[default_income_min, default_income_max],
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Health Index Range:", style={"marginTop": "15px"}),
        dcc.RangeSlider(
            id="health-index-slider",
            min=default_health_min,
            max=default_health_max,
            step=0.1,
            marks={i: str(i) for i in range(0, 11, 2)},
            value=[default_health_min, default_health_max],
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(
            [
                dbc.Button(
                    "Apply Filters",
                    id="apply-filters-btn",
                    color="primary",
                    className="mr-1",
                    style={"marginRight": "5px"},
                ),
                dbc.Button(
                    "Reset Filters",
                    id="reset-filters-btn",
                    color="secondary",
                    className="mr-1",
                    style={"marginRight": "5px"},
                ),
            ],
            style={
                "marginTop": "20px",
                "display": "flex",
                "justifyContent": "space-between",
            },
        ),
    ],
    style={
        "backgroundColor": nhs_colors["secondary"],
        "padding": "15px",
        "borderRadius": "5px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
    },
)

# Define patient modal component
patient_modal = dbc.Modal(
    [
        dbc.ModalHeader("Patient Details"),
        dbc.ModalBody(id="patient-detail-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ml-auto", n_clicks=0)
        ),
    ],
    id="patient-modal",
    size="lg",
)

# Define KPI row component - moved before app.layout
kpi_row = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H5("Total Patients", className="kpi-title"),
                            html.H3(f"{len(df_all):,}", className="kpi-value"),
                        ],
                        className="kpi-card",
                    ),
                    width=12, lg=3,
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H5("High Risk Patients", className="kpi-title"),
                            html.H3(
                                f"{len(df_all[df_all['Risk_Category'] == 'High Risk']) if 'Risk_Category' in df_all.columns else 0:,}", 
                                className="kpi-value"
                            ),
                        ],
                        className="kpi-card",
                        style={"backgroundColor": "#FADBD8"},
                    ),
                    width=12, lg=3,
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H5("Average Health Index", className="kpi-title"),
                            html.H3(
                                f"{df_all['Health_Index'].mean():.2f}" if 'Health_Index' in df_all.columns else "N/A", 
                                className="kpi-value"
                            ),
                        ],
                        className="kpi-card",
                        style={"backgroundColor": "#D5F5E3"},
                    ),
                    width=12, lg=3,
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H5("Avg. Charlson Index", className="kpi-title"),
                            html.H3(
                                f"{df_all['CharlsonIndex'].mean():.2f}" if 'CharlsonIndex' in df_all.columns else "N/A", 
                                className="kpi-value"
                            ),
                        ],
                        className="kpi-card",
                        style={"backgroundColor": "#D6EAF8"},
                    ),
                    width=12, lg=3,
                ),
            ],
            className="g-2",
        ),
    ],
    style={"marginBottom": "15px"},
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
                                        children=[dcc.Loading(id="demographics-loading")],
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
            id="progress-interval", interval=100, disabled=True
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
    Output("geo-tab-content", "children"), 
    [Input("filtered-data-store", "data")],
)
def load_geo_content(filtered_data):
    if filtered_data is None:
        df_to_use = df_all
        logger.info("Geo tab: Using initial full data for layout.")
    else:
        df_to_use = pd.DataFrame(filtered_data)
        logger.info(f"Geo tab: Using filtered data for layout. Shape: {df_to_use.shape}")
    
    if df_to_use.empty:
        return html.Div(
            "No data available after filtering",
            style={"padding": "20px", "textAlign": "center"},
        )
    
    min_hi = df_to_use["Health_Index"].min() if "Health_Index" in df_to_use.columns and not df_to_use.empty else 0
    max_hi = df_to_use["Health_Index"].max() if "Health_Index" in df_to_use.columns and not df_to_use.empty else 10
    
    geo_filter_panel = html.Div(
        [
            html.H5("Geographic Filters", style={"marginBottom": "15px"}),
            html.Label("Health Index Range:"),
            dcc.RangeSlider(
                id="geo-health-index-slider",
                min=min_hi,
                max=max_hi,
                step=0.1,
                marks={
                    round(min_hi): str(round(min_hi)),
                    round((min_hi + max_hi) / 2): str(round((min_hi + max_hi) / 2)),
                    round(max_hi): str(round(max_hi))},
                value=[min_hi, max_hi],
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div(
                [
                    dbc.Button(
                        "Apply Geo Filters",
                        id="apply-geo-filters-btn",
                        color="primary",
                        className="mt-2",
                        style={"width": "100%"}
                    ),
                    dbc.Button(
                        "Reset Geo Filters",
                        id="reset-geo-filters-btn",
                        color="secondary",
                        className="mt-2",
                        style={"width": "100%"}
                    ),
                ],
                style={"marginTop": "15px"}
            ),
        ],
        style={
            "backgroundColor": nhs_colors["secondary"],
            "padding": "15px",
            "borderRadius": "5px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "marginBottom": "15px",
        }
    )
    
    map_fig = dcc.Loading(id="geo-map-loading", children=[html.Div(id="geo-map-container")])
    enhanced_geo_disparities_fig = dcc.Loading(id="geo-disparities-map-loading", children=[html.Div(id="geo-disparities-map-container")])
    
    return html.Div(
        [
            geo_filter_panel,
            dcc.Store(id="geo-health-filter-store", data={"range": [min_hi, max_hi]}),
            dcc.Store(id="map-view-store", data={"center": None, "zoom": 7}),
            collapsible_card(
                "Geographic Distribution of Patients",
                map_fig,
                "geo-map",
                initially_open=True,
            ),
            collapsible_card(
                "Geographic Health Disparities",
                enhanced_geo_disparities_fig,
                "geo-disparities-map",
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
    Output("geo-health-filter-store", "data"),
    [
        Input("apply-geo-filters-btn", "n_clicks"),
        Input("reset-geo-filters-btn", "n_clicks")
    ],
    [
        State("geo-health-index-slider", "value"),
        State("filtered-data-store", "data"),
        State("geo-health-filter-store", "data")
    ]
)
def update_geo_filter(apply_clicks, reset_clicks, health_range, filtered_data, current_filter):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "reset-geo-filters-btn":
        if filtered_data:
            df = pd.DataFrame(filtered_data)
            min_hi = df["Health_Index"].min() if "Health_Index" in df.columns else 0
            max_hi = df["Health_Index"].max() if "Health_Index" in df.columns else 10
            return {"range": [min_hi, max_hi]}
    
    elif trigger_id == "apply-geo-filters-btn":
        return {"range": health_range}
    
    return current_filter

@app.callback(
    [
        Output("geo-map-container", "children"),
        Output("map-view-store", "data")
    ],
    [
        Input("filtered-data-store", "data"),
        Input("geo-health-filter-store", "data"),
        Input("map-view-store", "data")
    ]
)
def update_patient_map(filtered_data, health_filter, map_view):
    if filtered_data is None:
        return html.Div("No data available"), map_view
    
    df_to_use = pd.DataFrame(filtered_data)
    if df_to_use.empty:
        return html.Div("No data available after filtering"), map_view
    
    center = map_view.get("center") if map_view else None
    zoom = map_view.get("zoom") if map_view else 7
    
    map_fig = create_compact_map(
        df_to_use,
        height=600,
        health_index_range=health_filter.get("range") if health_filter else None,
        center=center,
        zoom=zoom
    )
    
    if map_fig is None:
        return html.Div("No geographic data available"), map_view
    
    new_center = map_fig.layout.mapbox.center if map_fig.layout.mapbox else center
    new_zoom = map_fig.layout.mapbox.zoom if map_fig.layout.mapbox else zoom
    updated_view = {"center": new_center, "zoom": new_zoom}
    
    return dcc.Graph(figure=map_fig, id="patient-geo-map"), updated_view

@app.callback(
    Output("geo-disparities-map-container", "children"),
    [
        Input("filtered-data-store", "data"),
        Input("geo-health-filter-store", "data"),
        Input("map-view-store", "data")
    ]
)
def update_disparities_map(filtered_data, health_filter, map_view):
    if filtered_data is None:
        return html.Div("No data available")
    
    df_to_use = pd.DataFrame(filtered_data)
    if df_to_use.empty:
        return html.Div("No data available after filtering")
    
    center = map_view.get("center") if map_view else None
    zoom = map_view.get("zoom") if map_view else 7
    
    enhanced_geo_disparities_fig = create_enhanced_geo_disparities_map(
        df_to_use,
        height=600,
        health_index_range=health_filter.get("range") if health_filter else None,
        center=center,
        zoom=zoom
    )
    
    if enhanced_geo_disparities_fig is None:
        return html.Div("No geographic disparities data available")
    
    return dcc.Graph(figure=enhanced_geo_disparities_fig, id="disparities-geo-map")

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
    income_health_fig = create_income_health_chart(df_to_use)
    inequality_chart = create_health_inequality_chart(df_to_use)
    intersectional_charts = create_intersectional_analysis(df_to_use)

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
                        dcc.Graph(figure=inequality_chart)
                        if inequality_chart
                        else "No health inequality data available",
                        width=12,
                    ),
                ],
                style={"marginTop": "15px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.H5("Intersectional Analysis of Health Disparities", className="mt-4 mb-2"),
                        width=12,
                    ),
                ],
                style={"marginTop": "15px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(figure=intersectional_charts[0])
                        if isinstance(intersectional_charts, list) and len(intersectional_charts) > 0
                        else (dcc.Graph(figure=intersectional_charts) if intersectional_charts is not None
                              else "No intersectional data available"),
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        dcc.Graph(figure=intersectional_charts[1]) 
                        if isinstance(intersectional_charts, list) and len(intersectional_charts) > 1
                        else "",
                        width=12,
                        lg=6,
                    ),
                ],
                style={"marginTop": "5px"},
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
            html.Div(
                [
                    html.H5("Health Indices Comparison"),
                    dcc.Graph(figure=indices_fig) if indices_fig else 
                    html.Div("No health indices data available")
                ],
                style={
                    "backgroundColor": "white", 
                    "borderRadius": "5px", 
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "marginBottom": "15px",
                    "padding": "15px"
                }
            ),
            html.Div(
                [
                    html.H5("Health Trends by Age Group"),
                    dcc.Graph(figure=health_trend) if health_trend else 
                    html.Div("No health trend data available")
                ],
                style={
                    "backgroundColor": "white", 
                    "borderRadius": "5px", 
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "marginBottom": "15px",
                    "padding": "15px"
                }
            ),
            html.Div(
                [
                    html.H5("Feature Correlation Matrix"),
                    dcc.Graph(figure=corr_matrix) if corr_matrix else 
                    html.Div("No correlation data available")
                ],
                style={
                    "backgroundColor": "white", 
                    "borderRadius": "5px", 
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "marginBottom": "15px",
                    "padding": "15px"
                }
            )
        ]
    )

@app.callback(
    Output("model-tab-content", "children"), [Input("filtered-data-store", "data")]
)
def load_model_performance_content(filtered_data):
    model_cards = []
    
    for group in final_groups:
        model_id = group["model"]
        model_label = group["label"]
        
        model_dir = os.path.join(FINALS_DIR, model_id)
        
        if not os.path.exists(model_dir):
            continue
            
        metrics = {}
        visualization_images = {"tsne": None, "umap": None}
        cluster_info = {}
        model_df = pd.DataFrame()
        
        metrics_path = os.path.join(model_dir, f"{model_id}_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                logger.info(f"Loaded metrics for {model_id}: {metrics}")
            except Exception as e:
                logger.error(f"Error loading metrics for {model_id}: {e}")
        
        cluster_metrics_path = os.path.join(model_dir, f"{model_id}_clusters.json")
        if os.path.exists(cluster_metrics_path):
            try:
                with open(cluster_metrics_path, 'r') as f:
                    cluster_info = json.load(f)
                logger.info(f"Loaded cluster metrics for {model_id}")
            except Exception as e:
                logger.error(f"Error loading cluster metrics for {model_id}: {e}")
        
        pred_path = os.path.join(model_dir, f"{model_id}_predictions.csv")
        cluster_path = os.path.join(model_dir, f"{model_id}_clusters.csv")
        
        if os.path.exists(pred_path):
            try:
                pred_df = pd.read_csv(pred_path)
                if os.path.exists(cluster_path):
                    cluster_df = pd.read_csv(cluster_path)
                    model_df = pd.merge(pred_df, cluster_df, on="Id", how="left", suffixes=('', '_cluster'))
                else:
                    model_df = pred_df
                
                if filtered_data is not None:
                    df_filtered = pd.DataFrame(filtered_data)
                    if "Id" in df_filtered.columns and not df_filtered.empty:
                        required_cols = ["Id", "Health_Index", "AGE", "CharlsonIndex", 
                                         "ElixhauserIndex", "HEALTHCARE_EXPENSES"]
                        cols_to_merge = [col for col in required_cols if col in df_filtered.columns]
                        
                        missing_cols = [col for col in required_cols if col not in df_filtered.columns]
                        if missing_cols and 'Id' in df_all.columns:
                            logger.info(f"Attempting to get missing columns {missing_cols} from df_all")
                            additional_cols = ["Id"] + [col for col in missing_cols if col in df_all.columns]
                            if len(additional_cols) > 1:
                                supplement_df = df_all[additional_cols]
                                if not df_filtered.empty:
                                    df_filtered = pd.merge(df_filtered, supplement_df, on="Id", how="left",
                                                         suffixes=('', '_orig'))
                                    cols_to_merge = list(set(cols_to_merge) | set(additional_cols))
                        
                        if cols_to_merge:
                            model_df = pd.merge(model_df, df_filtered[cols_to_merge], 
                                              on="Id", how="inner", suffixes=('', '_patient'))
                
                logger.info(f"Loaded prediction data for {model_id}: {len(model_df)} rows")
                logger.info(f"Available columns in model_df: {model_df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Error loading prediction data for {model_id}: {e}")
        
        tsne_path = os.path.join(model_dir, f"{model_id}_tsne.png")
        umap_path = os.path.join(model_dir, f"{model_id}_umap.png")
        
        if os.path.exists(tsne_path):
            visualization_images["tsne"] = encode_image(tsne_path)
        if os.path.exists(umap_path):
            visualization_images["umap"] = encode_image(umap_path)
        
        metrics_rows = []
        
        if metrics:
            for metric_name, metric_value in metrics.items():
                formatted_value = "N/A"
                if isinstance(metric_value, (int, float)):
                    formatted_value = f"{metric_value:.4f}"
                else:
                    formatted_value = str(metric_value)
                
                metrics_rows.append(
                    html.Div(
                        [
                            html.Strong(f"{metric_name.replace('_', ' ').title()}: "),
                            html.Span(formatted_value)
                        ],
                        style={"marginBottom": "8px"}
                    )
                )
        
        if cluster_info:
            metrics_rows.append(html.Hr(style={"marginTop": "15px", "marginBottom": "15px"}))
            metrics_rows.append(html.H6("Clustering Information"))
            
            if "chosen_k" in cluster_info:
                metrics_rows.append(
                    html.Div(
                        [
                            html.Strong("Number of Clusters (K): "),
                            html.Span(f"{cluster_info['chosen_k']}")
                        ],
                        style={"marginBottom": "8px"}
                    )
                )
            
            if "silhouette" in cluster_info:
                silhouette = cluster_info["silhouette"]
                silhouette_color = "black"
                silhouette_alert = ""
                if isinstance(silhouette, (int, float)):
                    formatted_silhouette = f"{silhouette:.4f}"
                    if silhouette < 0.25:
                        silhouette_color = nhs_colors["risk_high"]
                        silhouette_alert = " (Poor)"
                    elif silhouette < 0.5:
                        silhouette_color = nhs_colors["risk_medium"]
                        silhouette_alert = " (Fair)"
                    else:
                        silhouette_color = nhs_colors["risk_low"]
                        silhouette_alert = " (Good)"
                else:
                    formatted_silhouette = "N/A"
            
            for metric_name, metric_value in cluster_info.items():
                if metric_name not in ["chosen_k", "silhouette"] and isinstance(metric_value, (int, float)):
                    formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
        
        if not model_df.empty and "Health_Index" in model_df.columns and "Predicted_Health_Index" in model_df.columns:
            plot_df = smart_sample_dataframe(model_df, max_points=3000, method="stratified")
            
            if "Cluster" in plot_df.columns:
                perf_fig = px.scatter(
                    plot_df, x="Health_Index", y="Predicted_Health_Index",
                    color="Cluster",
                    color_discrete_sequence=risk_colors,
                    hover_data=["Id"],
                    title=f"{model_label}: Actual vs. Predicted Health Index"
                )
            else:
                perf_fig = px.scatter(
                    plot_df, x="Health_Index", y="Predicted_Health_Index",
                    hover_data=["Id"],
                    title=f"{model_label}: Actual vs. Predicted Health Index"
                )

            x_range = [plot_df["Health_Index"].min(), plot_df["Health_Index"].max()]
            perf_fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=x_range,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name="Perfect Prediction"
                )
            )

            perf_fig.update_layout(
                xaxis_title="Actual Health Index",
                yaxis_title="Predicted Health Index",
                legend_title="Cluster",
                height=500
            )
            
            model_plot = dcc.Graph(figure=perf_fig)
        else:
            model_plot = html.Div("No prediction data available", style={"padding": "20px", "textAlign": "center"})
        
        visualizations = []
        if visualization_images["tsne"]:
            visualizations.append(
                dbc.Col(
                    html.Div([
                        html.H6("t-SNE Visualization", style={"textAlign": "center", "marginBottom": "10px"}),
                        html.Img(
                            src=visualization_images["tsne"], 
                            style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}
                        )
                    ]),
                    width=12, md=6
                )
            )
        
        if visualization_images["umap"]:
            visualizations.append(
                dbc.Col(
                    html.Div([
                        html.H6("UMAP Visualization", style={"textAlign": "center", "marginBottom": "10px"}),
                        html.Img(
                            src=visualization_images["umap"], 
                            style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}
                        )
                    ]),
                    width=12, md=6
                )
            )
        
        risk_cluster_fig = None
        try:
            if all(col in model_df.columns for col in ["Health_Index", "CharlsonIndex", "ElixhauserIndex"]):
                risk_cluster_fig = create_clinical_risk_clusters(model_df, model_name=model_label)
        except Exception as e:
            logger.warning(f"Could not create risk cluster plot for {model_label}: {e}")
        
        model_card = collapsible_card(
            f"{model_label} Model Performance",
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div([
                                    html.H6("Model Metrics", style={"marginBottom": "10px"}),
                                    html.Div(metrics_rows)
                                ]), 
                                width=12, md=4, style={"backgroundColor": "#f9f9f9", "padding": "15px", "borderRadius": "5px"}
                            ),
                            dbc.Col(model_plot, width=12, md=8),
                        ],
                        className="mb-4"
                    ),
                    dbc.Row(visualizations, style={"marginTop": "15px"}) if visualizations else None,
                    html.Div(
                        [
                            html.H6("Model Information", style={"marginBottom": "10px"}),
                            html.Div(
                                [
                                    html.P([
                                        html.Strong("Model ID: "),
                                        html.Span(model_id)
                                    ]),
                                    html.P([
                                        html.Strong("Description: "),
                                        html.Span(f"TabNet model trained on {model_label.lower()} patient population")
                                    ]),
                                    html.P([
                                        html.Strong("Clustering Algorithm: "),
                                        html.Span(f"K-Means with K={cluster_info.get('chosen_k', 'N/A')} clusters")
                                    ]),
                                ]
                            )
                        ],
                        style={"marginTop": "20px", "backgroundColor": "#f9f9f9", "padding": "15px", "borderRadius": "5px"}
                    ) if cluster_info else None,
                    html.Div(
                        [
                            html.H6("Cluster Distribution", style={"marginBottom": "10px", "marginTop": "20px"}),
                            dcc.Graph(
                                figure=px.histogram(
                                    model_df, 
                                    x="Cluster",
                                    title=f"Distribution of Patients Across Clusters ({model_label})",
                                    color="Cluster",
                                    color_discrete_sequence=risk_colors,
                                    labels={"Cluster": "Cluster ID"}
                                ).update_layout(showlegend=False)
                            )
                        ]
                    ) if "Cluster" in model_df.columns and not model_df.empty else None,
                    html.Div(
                        [
                            html.H6("Clinical Risk Stratification", style={"marginBottom": "10px", "marginTop": "20px"}),
                            dcc.Graph(figure=risk_cluster_fig)
                        ]
                    ) if risk_cluster_fig else None,
                ]
            ),
            f"model-{model_label.lower().replace(' ', '-')}",
            initially_open=True,
        )
        
        model_cards.append(model_card)
    
    if not model_cards:
        return html.Div(
            [
                html.H4("Model Performance", style={"marginBottom": "15px"}),
                html.Div(
                    "No model performance data available. Please run the models using the Model Execution tab.",
                    style={"padding": "20px", "textAlign": "center", "backgroundColor": "white", "borderRadius": "5px"}
                )
            ],
            style={"padding": "15px"}
        )
    
    return html.Div(
        [
            html.H4("Model Performance Analysis", style={"marginBottom": "15px"}),
            html.P(
                "The following models were trained to predict Health Index values for different patient populations. "
                "Each model includes performance metrics, visualization of predictions, and clustering analysis.",
                style={"marginBottom": "20px"}
            ),
            html.Div(model_cards)
        ],
        style={"padding": "15px"}
    )

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
                    dcc.Dropdown(
                        id="xai-model-dropdown",
                        options=[{"label": group["label"], "value": group["label"]} for group in final_groups],
                        value=final_groups[0]["label"] if final_groups else None,
                        clearable=False,
                    ),
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
        return html.Div("Please select a model.", style={"padding": "20px", "textAlign": "center"})
    
    model_data = None
    model_name = None
    model_dir = None
    for grp in final_groups:
        if grp["label"] == selected_model:
            model_name = grp["model"]
            model_dir = os.path.join(FINALS_DIR, model_name)
            break
    
    if not model_dir or not os.path.exists(model_dir):
        return html.Div(f"Data directory not found for model: {selected_model}", style={"padding": "20px", "textAlign": "center", "color": "red"})
    
    shap_summary_png = os.path.join(model_dir, f"{model_name}_shap_summary.png")
    feat_imp_png = os.path.join(model_dir, f"{model_name}_feature_importance.png")
    
    shap_img = encode_image(shap_summary_png) if os.path.exists(shap_summary_png) else None
    feat_img = encode_image(feat_imp_png) if os.path.exists(feat_imp_png) else None
    
    content = []
    content.append(html.H5(f"XAI Insights for {selected_model} Model"))
    
    if shap_img:
        content.append(html.Div([
            html.H6("SHAP Summary Plot"),
            html.Img(src=shap_img, style={"maxWidth": "100%", "height": "auto", "marginBottom": "15px"})
        ]))
    if feat_img:
        content.append(html.Div([
            html.H6("Feature Importance Plot"),
            html.Img(src=feat_img, style={"maxWidth": "100%", "height": "auto", "marginBottom": "15px"})
        ]))
    
    try:
        explanation_html = format_explanation(model_name)
        if explanation_html:
            content.append(html.Div(dcc.Markdown(explanation_html), style={"marginTop": "15px"}))
    except Exception as e:
        logger.error(f"Error formatting explanation for {selected_model}: {e}")
        content.append(html.P(f"Could not load formatted explanation: {e}", style={"color": "orange"}))
    
    if not content:
        content.append(html.Div("No XAI insights available for this model", style={"padding": "20px", "textAlign": "center"}))
    
    return html.Div(content, style={"padding": "15px", "backgroundColor": "white", "borderRadius": "5px"})

@app.callback(
    Output("patient-tab-content", "children"), [Input("filtered-data-store", "data")]
)
def load_patient_data_content(filtered_data):
    if filtered_data is None:
        df_to_use = df_all
        logger.info("Patient data tab: Using initial full data.")
    else:
        df_to_use = pd.DataFrame(filtered_data)
        logger.info(f"Patient data tab: Using filtered data. Shape: {df_to_use.shape}")
    
    if df_to_use.empty:
        return html.Div("No patient data available after filtering.", style={"padding": "20px", "textAlign": "center"})
    
    display_cols = [col for col in ["Id", "AGE", "GENDER", "RACE", "INCOME", "Health_Index", "Risk_Category", "Cluster"] if col in df_to_use.columns]
    df_display = sanitize_datatable_values(df_to_use[display_cols])
    
    return html.Div([
        html.H4("Patient Data Table", style={"marginBottom": "15px"}),
        dbc.Button("Export to CSV", id="export-csv-btn", color="secondary", className="mb-3"),
        dcc.Download(id="download-dataframe-csv"),
        dash_table.DataTable(
            id='patient-data-table',
            columns=[{"name": i, "id": i} for i in df_display.columns],
            data=df_display.to_dict('records'),
            page_size=15,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'fontWeight': 'bold'},
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable="single",
        )
    ], style={"padding": "15px", "backgroundColor": "white", "borderRadius": "5px"})

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("export-csv-btn", "n_clicks")],
    [State("filtered-data-store", "data")],
)
def export_csv(n_clicks, filtered_data):
    if n_clicks is None or filtered_data is None:
        raise PreventUpdate
    
    df_export = pd.DataFrame(filtered_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_data_frame(df_export.to_csv, f"patient_data_{timestamp}.csv", index=False)

@app.callback(
    [Output("patient-modal", "is_open"), Output("patient-detail-body", "children")],
    [Input("patient-data-table", "active_cell")],
    [State("patient-data-table", "data"), State("patient-modal", "is_open")],
)
def display_patient_details(active_cell, table_data, is_open):
    if not active_cell or not table_data:
        return is_open, no_update
    
    row_index = active_cell['row']
    patient_data = table_data[row_index]
    
    details = []
    for key, value in patient_data.items():
        details.append(html.Div([
            html.Strong(f"{key.replace('_', ' ').title()}: "),
            html.Span(str(value))
        ]))
    
    return True, html.Div(details)

@app.callback(Output("close-modal", "n_clicks"), [Input("close-modal", "n_clicks")])
def close_modal(n_clicks):
    return 0

@app.callback(
    [
        Output("current-stage", "children"),
        Output("progress-bar-with-stages", "children"),
        Output("execution-log", "children"),
        Output("error-message", "style"),
        Output("error-message", "children"),
        Output("progress-container", "style"),
        Output("execution-status", "children")
    ],
    [Input("execution-state-store", "data")],
    prevent_initial_call=True,
)
def update_progress_ui(state):
    if state is None:
        return "Initializing", [], "", {"display": "none"}, "", {"display": "none"}, "Not Started"
    
    stage = state.get("stage", "Unknown")
    progress = state.get("progress", 0)
    output_log = "\n".join(state.get("output", []))
    error = state.get("error")
    status = state.get("status", "idle")
    remaining = state.get("remaining", "N/A")
    
    error_style = {"display": "block", "color": "red", "marginBottom": "15px"} if error else {"display": "none"}
    progress_container_style = {"display": "block"} if status in ["running", "completed", "error"] else {"display": "none"}
    
    status_message = f"Status: {status.capitalize()}"
    if status == "error":
        status_message += f" - {error}"
    elif status == "completed":
        status_message = "Status: Completed Successfully"
    
    progress_bar_component = create_stage_markers(stage, progress, remaining)
    
    return stage, progress_bar_component, output_log, error_style, error or "", progress_container_style, status_message

@app.callback(
    [
        Output("results-container", "children"),
        Output("results-container", "style")
    ],
    [Input("execution-state-store", "data")],
    prevent_initial_call=True
)
def update_results_display(state):
    if state is None or state.get("status") != "completed" or state.get("error"):
        return "", {"display": "none"}
    
    results_content = html.Div([
        html.H5("Execution Completed Successfully", style={"color": "green"}),
        html.P("Model execution finished. Results may be available in the 'Data/finals' or 'Data/explain_xai' directories."),
    ], style={"padding": "15px", "backgroundColor": "#e9f7ef", "borderRadius": "5px", "border": "1px solid green"})
    
    return results_content, {"display": "block", "marginTop": "15px"}

collapsible_card_ids = ["demographics"]

@app.callback(
    [Output(f"{card_id}-content", "style") for card_id in collapsible_card_ids] +
    [Output(f"{card_id}-toggle-icon", "className") for card_id in collapsible_card_ids],
    [Input(f"{card_id}-header", "n_clicks") for card_id in collapsible_card_ids],
    [State(f"{card_id}-content", "style") for card_id in collapsible_card_ids],
    prevent_initial_call=True
)
def toggle_collapsible_cards(*args):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0].replace("-header", "")
    num_cards = len(collapsible_card_ids)
    n_clicks_list = args[:num_cards]
    style_list = args[num_cards:]
    
    outputs = []
    icon_outputs = []
    
    for i, card_id in enumerate(collapsible_card_ids):
        current_style = style_list[i] or {"display": "none"}
        is_open = current_style.get("display") == "block"
        new_style = {"display": "none"}
        icon_class = "fas fa-chevron-right ml-2"
        
        if card_id == trigger_id:
            if is_open:
                new_style = {"display": "none"}
                icon_class = "fas fa-chevron-right ml-2"
            else:
                new_style = {"display": "block", "padding": "15px"}
                icon_class = "fas fa-chevron-down ml-2"
        else:
            new_style = current_style
            icon_class = "fas fa-chevron-down ml-2" if is_open else "fas fa-chevron-right ml-2"
        
        outputs.append(new_style)
        icon_outputs.append(icon_class)
    
    return outputs + icon_outputs

if __name__ == "__main__":
    app.run_server(debug=True)
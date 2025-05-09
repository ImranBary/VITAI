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
import shlex  # Add this import
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
import paramiko

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

# --- Remote execution -------------------------------------------------
SSH_ENABLED_DEFAULT = True                  # default option
SSH_HOST         = "imran@192.168.137.87"   # Direct ethernet IP
SSH_KEY_PATH     = os.path.expanduser("~/.ssh/id_ed25519")   # private key
REMOTE_WORKDIR   = "/home/imran/Desktop/VITAI"

# Direct ethernet connection settings
DIRECT_ETHERNET_IP = "192.168.137.87"  # Your Pi's IP from direct connection
DIRECT_ETHERNET_USER = "imran"         # Your Pi's username

def get_raspberry_pi_ip():
    """Try to resolve Raspberry Pi IP address using multiple methods"""
    try:
        # For direct ethernet connection, try the known IP first
        import subprocess
        import platform
        
        def ping(host):
            """
            Returns True if host responds to a ping request
            """
            # Ping command parameters as function of OS
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', host]
            
            try:
                return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
            except:
                return False

        # Try direct ethernet IP first
        if ping(DIRECT_ETHERNET_IP):
            return DIRECT_ETHERNET_IP

        # If direct connection fails, try other methods
        import socket
        try:
            socket.gethostbyname("raspberrypi.local")
            return "raspberrypi.local"
        except socket.gaierror:
            pass

        return None
    except Exception as e:
        logger.error(f"Error resolving Raspberry Pi IP: {str(e)}")
        return None

def get_system_resources():
    """Detect system resources and optimize settings accordingly"""
    cpu_count = multiprocessing.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used_gb = memory.used / (1024 * 1024 * 1024)
    memory_total_gb = memory.total / (1024 * 1024 * 1024)

    system_info = {
        "cpu_count": cpu_count,
        "cpu_percent": cpu_percent,
        "memory_total_gb": memory_total_gb,
        "memory_used_gb": memory_used_gb,
        "memory_percent": memory_percent,
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

    return system_info, optimization_settings

def get_remote_system_resources():
    """Get system resources from Raspberry Pi via SSH"""
    try:
        # For direct ethernet connection, use the known IP
        hostname = DIRECT_ETHERNET_IP
        username = DIRECT_ETHERNET_USER

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect with timeout and retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting SSH connection to {hostname} as {username}")
                ssh.connect(
                    hostname,
                    username=username,
                    key_filename=SSH_KEY_PATH,
                    timeout=5,
                    banner_timeout=5
                )
                logger.info("SSH connection successful")
                break
            except paramiko.AuthenticationException as e:
                logger.error(f"SSH Authentication failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
            except paramiko.SSHException as e:
                logger.error(f"SSH error: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        
        # Get CPU info using mpstat for more accurate readings
        stdin, stdout, stderr = ssh.exec_command("mpstat 1 1 | tail -n 1 | awk '{print 100-$NF}'")
        cpu_percent = float(stdout.read().decode().strip())
        
        # Get memory info using free command
        stdin, stdout, stderr = ssh.exec_command("free -m | grep Mem")
        mem_info = stdout.read().decode().strip().split()
        memory_total_mb = int(mem_info[1])
        memory_used_mb = int(mem_info[2])
        memory_percent = (memory_used_mb / memory_total_mb) * 100
        
        # Get CPU count
        stdin, stdout, stderr = ssh.exec_command("nproc")
        cpu_count = int(stdout.read().decode().strip())
        
        # Get temperature with error handling
        try:
            stdin, stdout, stderr = ssh.exec_command("vcgencmd measure_temp")
            temp = stdout.read().decode().strip().replace("temp=", "").replace("'C", "")
            temperature = float(temp)
        except:
            temperature = None
        
        # Get uptime
        stdin, stdout, stderr = ssh.exec_command("uptime -p")
        uptime = stdout.read().decode().strip()
        
        # Get load average
        stdin, stdout, stderr = ssh.exec_command("cat /proc/loadavg | awk '{print $1}'")
        load_avg = float(stdout.read().decode().strip())
        
        system_info = {
            "cpu_count": cpu_count,
            "cpu_percent": cpu_percent,
            "memory_total_gb": memory_total_mb / 1024,
            "memory_used_gb": memory_used_mb / 1024,
            "memory_percent": memory_percent,
            "temperature": temperature,
            "platform": "Raspberry Pi",
            "processor": "ARM",
            "is_high_end": False,
            "uptime": uptime,
            "load_average": load_avg,
            "hostname": hostname
        }
        
        ssh.close()
        return system_info
        
    except paramiko.AuthenticationException:
        logger.error("SSH Authentication failed")
        return None
    except paramiko.SSHException as e:
        logger.error(f"SSH error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to get remote system resources: {str(e)}")
        return None

def create_system_monitor_card():
    """Create a card to display system resources"""
    return dbc.Card(
        [
            dbc.CardHeader("System Resources", className="text-center"),
            dbc.CardBody(
                [
                    html.Div(id="system-resources-content"),
                    dcc.Interval(
                        id="system-resources-interval",
                        interval=2000,  # Update every 2 seconds
                        n_intervals=0
                    )
                ]
            )
        ],
        className="mb-4"
    )

SYSTEM_INFO, PERF_SETTINGS = get_system_resources()

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
            margin={"r": 20, "t": 40, "l": 20, "b": 20},
            autosize=True,
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
                margin={"r": 20, "t": 40, "l": 20, "b": 20},
                autosize=True,
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
                margin={"r": 20, "t": 40, "l": 20, "b": 20},
                autosize=True,
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

def create_health_inequalities_chart(df):
    """Create visualizations showing health inequalities across different demographic groups"""
    if df is None or df.empty:
        return None
    
    charts = []
    
    # Check for required columns
    demo_cols = ["RACE", "GENDER", "ETHNICITY", "INCOME", "Health_Index"]
    if not all(col in df.columns for col in demo_cols):
        return None
    
    # 1. Health Index by Race and Gender (Intersectional analysis)
    try:
        # Filter to include only the most frequent racial categories
        top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
        filtered_df = df[df["RACE"].isin(top_races)].copy()
        
        if not filtered_df.empty:
            # Calculate mean health index by race and gender
            race_gender_health = filtered_df.groupby(["RACE", "GENDER"])["Health_Index"].mean().reset_index()
            
            fig1 = px.bar(
                race_gender_health,
                x="RACE",
                y="Health_Index",
                color="GENDER",
                barmode="group",
                title="Health Index by Race and Gender",
                color_discrete_sequence=["#005EB8", "#F0A860"],
                labels={"RACE": "Race", "Health_Index": "Average Health Index", "GENDER": "Gender"}
            )
            
            fig1.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
            charts.append(fig1)
    except Exception as e:
        logger.warning(f"Could not create race/gender health inequality chart: {e}")
    
    # 2. Health Index by Income Quartile
    try:
        # Create income quartiles
        df_with_quartiles = df.copy()
        df_with_quartiles['Income_Quartile'] = pd.qcut(
            df_with_quartiles['INCOME'], 
            q=4, 
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        )
        
        # Calculate mean health metrics by income quartile
        income_health = df_with_quartiles.groupby('Income_Quartile').agg({
            'Health_Index': 'mean',
            'CharlsonIndex': 'mean',
            'ElixhauserIndex': 'mean',
            'HEALTHCARE_EXPENSES': 'mean'
        }).reset_index()
        
        fig2 = px.bar(
            income_health,
            x='Income_Quartile',
            y=['Health_Index', 'CharlsonIndex', 'ElixhauserIndex'],
            barmode='group',
            title='Health Indices by Income Quartile',
            labels={
                'Income_Quartile': 'Income Quartile',
                'value': 'Average Index Value',
                'variable': 'Health Metric'
            },
            color_discrete_sequence=["#005EB8", "#00843D", "#E66C6C"]
        )
        
        fig2.update_layout(
            height=500, 
            margin=dict(l=20, r=20, t=40, b=20), 
            autosize=True,
            xaxis={'categoryorder': 'array', 'categoryarray': ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']}
        )
        charts.append(fig2)
        
        # Also create healthcare expenses by income quartile
        fig3 = px.bar(
            income_health,
            x='Income_Quartile',
            y='HEALTHCARE_EXPENSES',
            title='Healthcare Expenses by Income Quartile',
            color='Income_Quartile',
            color_discrete_sequence=["#5A9BD5", "#52A373", "#F0A860", "#E66C6C"],
            labels={
                'Income_Quartile': 'Income Quartile',
                'HEALTHCARE_EXPENSES': 'Average Healthcare Expenses ($)'
            }
        )
        
        fig3.update_layout(
            height=500, 
            margin=dict(l=20, r=20, t=40, b=20), 
            autosize=True,
            xaxis={'categoryorder': 'array', 'categoryarray': ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']}
        )
        charts.append(fig3)
    except Exception as e:
        logger.warning(f"Could not create income quartile health inequality chart: {e}")
    
    # 3. Healthcare Utilization Metrics by Race
    try:
        if all(col in df.columns for col in ["RACE", "Hospitalizations_Count", "Medications_Count", "Abnormal_Observations_Count"]):
            # Filter to top races for clarity
            top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
            filtered_df = df[df["RACE"].isin(top_races)].copy()
            
            if not filtered_df.empty:
                # Calculate mean healthcare utilization metrics by race
                utilization_by_race = filtered_df.groupby("RACE").agg({
                    "Hospitalizations_Count": "mean",
                    "Medications_Count": "mean",
                    "Abnormal_Observations_Count": "mean"
                }).reset_index()
                
                # Reshape data for grouped bar chart
                util_long = pd.melt(
                    utilization_by_race, 
                    id_vars=["RACE"], 
                    value_vars=["Hospitalizations_Count", "Medications_Count", "Abnormal_Observations_Count"],
                    var_name="Metric", 
                    value_name="Average Count"
                )
                
                fig4 = px.bar(
                    util_long,
                    x="RACE",
                    y="Average Count",
                    color="Metric",
                    barmode="group",
                    title="Healthcare Utilization Metrics by Race",
                    color_discrete_sequence=["#005EB8", "#00843D", "#FFB81C"],
                    labels={"RACE": "Race", "Average Count": "Average Count", "Metric": "Healthcare Metric"}
                )
                
                fig4.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
                charts.append(fig4)
    except Exception as e:
        logger.warning(f"Could not create healthcare utilization by race chart: {e}")
    
    return charts if charts else None

def create_health_boxplots(df):
    """Create boxplots showing distribution of health indices across demographic groups"""
    if df is None or df.empty:
        return None
    
    # Check for required columns
    if not all(col in df.columns for col in ["Health_Index", "RACE", "GENDER"]):
        return None
    
    try:
        # Sample data if too large
        if len(df) > 5000:
            df_sample = df.sample(5000, random_state=42)
        else:
            df_sample = df.copy()
        
        # Filter to top races for clarity
        top_races = df_sample["RACE"].value_counts().nlargest(5).index.tolist()
        filtered_df = df_sample[df_sample["RACE"].isin(top_races)].copy()
        
        if filtered_df.empty:
            return None
        
        # Create boxplot
        fig = px.box(
            filtered_df,
            x="RACE",
            y="Health_Index",
            color="GENDER",
            title="Health Index Distribution by Race and Gender",
            points="outliers",
            labels={"RACE": "Race", "Health_Index": "Health Index", "GENDER": "Gender"},
            color_discrete_sequence=["#005EB8", "#F0A860"]
        )
        
        fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        return fig
    except Exception as e:
        logger.warning(f"Could not create health boxplots: {e}")
        return None

def create_comorbidity_analysis(df):
    """Create visualization showing relationship between health factors"""
    if df is None or df.empty:
        return None
    
    # Check for required columns
    health_cols = ["Health_Index", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"]
    if not all(col in df.columns for col in health_cols):
        return None
    
    try:
        # Sample data if too large
        if len(df) > 5000:
            df_sample = df.sample(5000, random_state=42)
        else:
            df_sample = df.copy()
        
        # Create scatter plot of healthcare expenses vs. health index with healthcare coverage as size
        fig = px.scatter(
            df_sample,
            x="HEALTHCARE_EXPENSES",
            y="Health_Index",
            size="HEALTHCARE_COVERAGE" if "HEALTHCARE_COVERAGE" in df_sample.columns else None,
            color="Risk_Category" if "Risk_Category" in df_sample.columns else None,
            hover_data=["GENDER", "RACE", "AGE"],
            title="Healthcare Expenses vs. Health Index",
            labels={
                "HEALTHCARE_EXPENSES": "Healthcare Expenses ($)",
                "Health_Index": "Health Index",
                "HEALTHCARE_COVERAGE": "Healthcare Coverage",
                "Risk_Category": "Risk Category"
            },
            color_discrete_map={
                "Very Low Risk": nhs_colors["risk_verylow"],
                "Low Risk": nhs_colors["risk_low"],
                "Moderate Risk": nhs_colors["risk_medium"],
                "High Risk": nhs_colors["risk_high"],
            } if "Risk_Category" in df_sample.columns else None
        )
        
        fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        return fig
    except Exception as e:
        logger.warning(f"Could not create comorbidity analysis: {e}")
        return None

# Define external_stylesheets before app initialization
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=MuseoModerno:wght@900&display=swap",
    dbc.themes.FLATLY,
    {
        "href": "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
        "rel": "stylesheet",
    }
]

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
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

@app.callback(
    [Output("progress-interval",    "disabled"),
     Output("run-model-btn",        "disabled"),
     Output("cancel-btn-container", "style"),
     Output("reset-btn-container",  "style"),
     Output("model-run-state-store", "data")],
    [Input("run-model-btn",   "n_clicks"),
     Input("cancel-model-btn","n_clicks"),
     Input("reset-model-btn", "n_clicks"),
     Input("execution-state-store", "data")],
    [State("population-input",   "value"),
     State("performance-options","value"),
     State("memory-util-slider", "value"),
     State("cpu-util-slider",    "value"),
     State("threads-slider",     "value"),
     State("execution-mode-toggle", "value"),
     State("model-run-state-store", "data")],
    prevent_initial_call=True,
)
def start_or_cancel(run_clicks, cancel_clicks, reset_clicks, execution_state,
                    population, perf_opts,
                    mem_util, cpu_util, threads,
                    execution_mode, run_state):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Show reset button if execution completed or errored
    if trigger_id == "execution-state-store" and execution_state:
        if execution_state.get("status") in ["completed", "error"]:
            return no_update, no_update, no_update, {"display": "block", "marginTop": "10px"}, run_state
        return no_update, no_update, no_update, {"display": "none"}, run_state

    if trigger_id == "cancel-model-btn":
        with process_lock:
            if process_handle and process_handle.poll() is None:
                try:
                    process_handle.send_signal(signal.SIGINT)
                    process_handle.wait(timeout=5)
                except Exception:
                    process_handle.kill()
        return True, False, {"display": "none"}, {"display": "none"}, {"running": False}

    if trigger_id == "reset-model-btn":
        # Fully reset the UI state to allow running the model again
        return True, False, {"display": "none"}, {"display": "none"}, {"running": False}

    # Robust run logic: allow run if not currently running
    if trigger_id == "run-model-btn" and (not run_state or not run_state.get("running", False)):
        use_remote = (execution_mode == "remote")
        cmd = _build_command(population, perf_opts, mem_util, cpu_util, threads, use_remote)
        threading.Thread(
            target=_launch_generate_and_predict, args=(cmd,), daemon=True
        ).start()
        # Enable progress interval, disable run button, show cancel, hide reset, set running state
        return False, True, {"display": "block"}, {"display": "none"}, {"running": True}

    # Default: do not update
    raise PreventUpdate

@app.callback(
    Output("execution-state-store", "data"),
    Input("progress-interval", "n_intervals"),
    Input("reset-model-btn", "n_clicks"),    # Add reset button as input
    State("execution-state-store", "data"),
    prevent_initial_call=True,
)
def stream_progress(_tick, reset_clicks, state):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # If reset button clicked, return a fresh state
    if trigger_id == "reset-model-btn":
        return {
            "status":   "idle",
            "stage":    "Initializing",
            "progress": 0.0,
            "output":   [],
            "error":    None,
            "remaining":"calculating...",
            "last_update": time.time(),
        }
    
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
        # Add remote execution toggle
        html.Div(
            [
                html.Label("Execution Mode:", style={"marginBottom": "5px"}),
                dbc.RadioItems(
                    id="execution-mode-toggle",
                    options=[
                        {"label": "Local Machine", "value": "local"},
                        {"label": "Raspberry Pi (Remote)", "value": "remote"},
                    ],
                    value="remote" if SSH_ENABLED_DEFAULT else "local",
                    inline=True,
                    style={"marginBottom": "15px"},
                ),
                html.Div(
                    [
                        html.I(className="fas fa-info-circle me-2"),
                        html.Span("Remote execution runs the model on Raspberry Pi via SSH.")
                    ],
                    className="text-muted small",
                    style={"marginTop": "5px"}
                )
            ],
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
        html.Div(
            [
                dbc.Button(
                    "Reset & Run Again",
                    id="reset-model-btn",
                    color="success",
                    className="mr-1",
                    style={"marginTop": "10px", "width": "100%"},
                ),
            ],
            id="reset-btn-container",
            style={"display": "none"},
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
                        create_system_monitor_card(),  # Add system monitor card
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5("Current Stage", className="text-center"),
                                        html.Div(id="current-stage", className="text-center"),
                                    ],
                                    className="col-12 col-md-4",
                                ),
                                html.Div(
                                    [
                                        html.H5("Overall Progress", className="text-center"),
                                        html.Div(id="progress-bar-with-stages"),
                                    ],
                                    className="col-12 col-md-8",
                                ),
                            ],
                            className="row mb-4",
                        ),
                        html.Div(
                            [
                                html.H5("Execution Log", className="text-center"),
                                html.Div(
                                    id="execution-log",
                                    style={
                                        "maxHeight": "200px",
                                        "overflowY": "auto",
                                        "backgroundColor": "#f8f9fa",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                    },
                                ),
                            ],
                            className="mb-4",
                        ),
                        html.Div(
                            id="error-message",
                            style={"display": "none"},
                            className="alert alert-danger",
                        ),
                        html.Div(
                            id="execution-status",
                            className="mt-3 text-center",
                            style={"fontWeight": "bold"}
                        ),
                    ],
                    style={
                        "backgroundColor": nhs_colors["secondary"],
                        "padding": "20px",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
            ],
            className="col-12",
        ),
    ],
    className="row",
    id="progress-container",
    style={"display": "none"},
)

# Add results container
results_container = html.Div(
    id="results-container",
    style={"display": "none"},
    className="mt-4"
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
            .vitai-logo {
                font-family: 'MuseoModerno', cursive, sans-serif !important;
                font-weight: 900 !important;
                text-transform: uppercase !important;
                color: #005EB8 !important;
                font-size: 2.6rem !important;
                line-height: 1.1;
                margin-bottom: 0.2em;
                display: inline-block;
                letter-spacing: 0.08em;
            }
        </style>
    """,
            dangerously_allow_html=True,
        ),
        html.Div(
            [
                html.H1(
                    [html.Span("VITAI", className="vitai-logo"), " Healthcare Analytics Dashboard"],
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
        results_container,
        dcc.Store(id="model-run-state-store", storage_type="memory"),
    ],
    style={"backgroundColor": nhs_colors["background"], "padding": "15px"},
)

@app.callback(
    Output("filtered-data-store", "data"),
    [
        Input("apply-filters-btn", "n_clicks"),
        Input("reset-filters-btn", "n_clicks"),
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
    apply_clicks, reset_clicks, n_intervals, model, risk, age_range, income_range, health_range
):
    ctx = callback_context
    if not ctx.triggered:
        return df_all.to_dict("records")

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "initial-load-trigger" and n_intervals is not None:
        return df_all.to_dict("records")

    if trigger_id == "reset-filters-btn":
        # For reset, return the entire dataset
        return df_all.to_dict("records")

    if apply_clicks is None and trigger_id == "apply-filters-btn":
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
    [
        Output("global-model-dropdown", "value"),
        Output("global-risk-dropdown", "value"),
        Output("age-range-slider", "value"),
        Output("income-range-slider", "value"),
        Output("health-index-slider", "value")
    ],
    [Input("reset-filters-btn", "n_clicks")],
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    # Get default values for sliders
    default_age_min = df_all["AGE"].min() if "AGE" in df_all.columns and not df_all.empty else 0
    default_age_max = df_all["AGE"].max() if "AGE" in df_all.columns and not df_all.empty else 100
    default_income_min = df_all["INCOME"].min() if "INCOME" in df_all.columns and not df_all.empty else 0
    default_income_max = df_all["INCOME"].max() if "INCOME" in df_all.columns and not df_all.empty else 100000
    default_health_min = df_all["Health_Index"].min() if "Health_Index" in df_all.columns and not df_all.empty else 0
    default_health_max = df_all["Health_Index"].max() if "Health_Index" in df_all.columns and not df_all.empty else 10
    
    # Return default values for all filter components
    return (
        "All",  # global-model-dropdown
        "All",  # global-risk-dropdown
        [default_age_min, default_age_max],  # age-range-slider
        [default_income_min, default_income_max],  # income-range-slider
        [default_health_min, default_health_max]  # health-index-slider
    )

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
    [Input("filtered-data-store", "data")]
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
    
    # Create visualization components for health indices
    health_inequalities_charts = create_health_inequalities_visualizations(df_to_use)
    
    return html.Div([
        html.H4("Health Indices Analysis", style={"marginBottom": "15px"}),
        html.P(
            "This analysis explores health inequalities across different demographic factors including income, race, gender, and age groups.",
            style={"marginBottom": "20px"}
        ),
        html.Div(health_inequalities_charts)
    ], style={"padding": "15px", "backgroundColor": "white", "borderRadius": "5px"})

def create_health_inequalities_visualizations(df):
    """Create comprehensive visualizations showing health inequalities"""
    if df is None or df.empty:
        return html.Div("No data available for health indices analysis.")
    
    visualizations = []
    
    # 1. Health indices distribution by demographic groups
    health_distribution_card = create_health_distribution_card(df)
    if health_distribution_card:
        visualizations.append(health_distribution_card)
    
    # 2. Income vs health index with trend analysis
    income_health_card = create_income_health_trends_card(df)
    if income_health_card:
        visualizations.append(income_health_card)
    
    # 3. Multiple health indices comparison across groups
    multi_indices_card = create_multi_indices_comparison(df)
    if multi_indices_card:
        visualizations.append(multi_indices_card)
    
    # 4. Health indices radar chart by demographic group
    radar_card = create_health_indices_radar(df)
    if radar_card:
        visualizations.append(radar_card)
    
    # 5. Healthcare access and utilization inequality card
    healthcare_access_card = create_healthcare_access_card(df)
    if healthcare_access_card:
        visualizations.append(healthcare_access_card)
    
    return html.Div(visualizations)

def create_health_distribution_card(df):
    """Create visualization showing distribution of health indices across demographic groups"""
    if df is None or df.empty:
        return None
    
    try:
        # Create the boxplots showing health index distribution by race and gender
        boxplot_fig = create_health_boxplots(df)
        
        # Create violin plots for more detailed health index distribution by income quartile
        violin_fig = None
        if all(col in df.columns for col in ["Health_Index", "INCOME"]):
            # Create income quartiles
            df_with_quartiles = df.copy()
            df_with_quartiles['Income_Quartile'] = pd.qcut(
                df_with_quartiles['INCOME'], 
                q=4, 
                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
            )
            
            # Sample if dataset is too large for violin plot
            if len(df_with_quartiles) > 5000:
                df_sample = smart_sample_dataframe(df_with_quartiles, max_points=5000, method="stratified")
            else:
                df_sample = df_with_quartiles
                
            # Create violin plot
            violin_fig = px.violin(
                df_sample, 
                x="Income_Quartile", 
                y="Health_Index",
                color="Income_Quartile",
                box=True, 
                points="all", 
                title="Health Index Distribution by Income Quartile",
                color_discrete_map={
                    "Q1 (Low)": nhs_colors["risk_high"],
                    "Q2": nhs_colors["risk_medium"],
                    "Q3": nhs_colors["risk_low"],
                    "Q4 (High)": nhs_colors["risk_verylow"]
                },
                labels={"Income_Quartile": "Income Quartile", "Health_Index": "Health Index"}
            )
            
            violin_fig.update_layout(
                height=600, 
                margin=dict(l=20, r=20, t=40, b=20), 
                autosize=True,
                xaxis={'categoryorder': 'array', 'categoryarray': ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']}
            )
        
        return collapsible_card(
            "Health Index Distribution by Demographic Groups",
            html.Div([
                html.P("These visualizations show how Health Index values are distributed across different demographic groups, highlighting potential inequalities in health outcomes.", 
                       style={"marginBottom": "15px"}),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(figure=boxplot_fig) if boxplot_fig else "No data available for boxplot visualization",
                        width=12,
                        lg=6
                    ),
                    dbc.Col(
                        dcc.Graph(figure=violin_fig) if violin_fig else "No data available for violin plot visualization",
                        width=12,
                        lg=6
                    ),
                ]),
            ]),
            "health-distribution-card",
            initially_open=True
        )
    except Exception as e:
        logger.warning(f"Could not create health distribution card: {e}")
        return None

def create_income_health_trends_card(df):
    """Create visualization showing income vs health trends with inequalities"""
    if df is None or df.empty:
        return None
    
    try:
        # Income vs Health Index Trend
        income_health_fig = create_income_health_chart(df)
        
        # Create inequality curve showing healthcare expenses vs income
        inequality_fig = None
        if all(col in df.columns for col in ["INCOME", "HEALTHCARE_EXPENSES"]):
            # Create income deciles
            df_with_deciles = df.copy()
            df_with_deciles['Income_Decile'] = pd.qcut(
                df_with_deciles['INCOME'], 
                q=10, 
                labels=[f'D{i}' for i in range(1, 11)]
            )
            
            # Calculate ratio of healthcare expenses to income by income decile
            expense_ratio = df_with_deciles.groupby('Income_Decile').apply(
                lambda x: (x['HEALTHCARE_EXPENSES'].mean() / x['INCOME'].mean()) * 100
            ).reset_index()
            expense_ratio.columns = ['Income_Decile', 'Expense_Percentage']
            
            inequality_fig = px.line(
                expense_ratio,
                x='Income_Decile',
                y='Expense_Percentage',
                markers=True,
                title="Healthcare Expenses as % of Income (by Income Decile)",
                labels={
                    'Income_Decile': 'Income Decile (D1=Lowest, D10=Highest)',
                    'Expense_Percentage': 'Healthcare Expenses (% of Income)'
                }
            )
            
            inequality_fig.update_traces(
                line=dict(color=nhs_colors["primary"], width=3),
                marker=dict(size=10, color=nhs_colors["primary"])
            )
            
            # Add annotation for inequality interpretation
            inequality_fig.add_annotation(
                text="Higher percentages for lower income groups indicate<br>disproportionate financial burden of healthcare",
                xref="paper", yref="paper",
                x=0.02, y=0.95,
                showarrow=False,
                font=dict(size=12, color="#333"),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="#005EB8",
                borderwidth=1,
                borderpad=4,
                align="left"
            )
            
            inequality_fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        
        return collapsible_card(
            "Income-Related Health Inequalities",
            html.Div([
                html.P("These visualizations explore the relationship between income and health outcomes, highlighting socioeconomic inequalities in health.", 
                       style={"marginBottom": "15px"}),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(figure=income_health_fig) if income_health_fig else "No data available for income-health visualization",
                        width=12,
                        lg=6
                    ),
                    dbc.Col(
                        dcc.Graph(figure=inequality_fig) if inequality_fig else "No data available for healthcare expense inequality visualization",
                        width=12,
                        lg=6
                    ),
                ]),
            ]),
            "income-health-trends-card",
            initially_open=True
        )
    except Exception as e:
        logger.warning(f"Could not create income health trends card: {e}")
        return None

def create_multi_indices_comparison(df):
    """Create visualization comparing multiple health indices across demographic groups"""
    if df is None or df.empty:
        return None
    
    try:
        # First visualization: Compare health indices across racial groups
        race_indices_fig = None
        if all(col in df.columns for col in ["RACE", "Health_Index", "CharlsonIndex", "ElixhauserIndex"]):
            top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
            filtered_df = df[df["RACE"].isin(top_races)].copy()
            
            if not filtered_df.empty:
                # Calculate mean indices by race
                race_indices = filtered_df.groupby("RACE").agg({
                    "Health_Index": "mean",
                    "CharlsonIndex": "mean",
                    "ElixhauserIndex": "mean"
                }).reset_index()
                
                # Normalize values between 0-1 for comparison (higher is better for Health_Index, lower is better for others)
                for col in ["CharlsonIndex", "ElixhauserIndex"]:
                    max_val = race_indices[col].max()
                    if max_val > 0:  # Avoid division by zero
                        race_indices[f"{col}_Normalized"] = 1 - (race_indices[col] / max_val)
                
                if race_indices["Health_Index"].max() > 0:
                    race_indices["Health_Index_Normalized"] = race_indices["Health_Index"] / race_indices["Health_Index"].max()
                
                # Create the radar chart data
                categories = ["Health", "Comorbidity (inverse)", "Disease Burden (inverse)"]
                
                fig = go.Figure()
                
                for i, race in enumerate(race_indices["RACE"]):
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            race_indices.loc[race_indices["RACE"] == race, "Health_Index_Normalized"].iloc[0],
                            race_indices.loc[race_indices["RACE"] == race, "CharlsonIndex_Normalized"].iloc[0],
                            race_indices.loc[race_indices["RACE"] == race, "ElixhauserIndex_Normalized"].iloc[0],
                        ],
                        theta=categories,
                        fill='toself',
                        name=race
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Comparative Health Indices by Race",
                    height=600,
                    margin=dict(l=50, r=50, t=50, b=50),
                    showlegend=True
                )
                
                race_indices_fig = fig
        
        # Second visualization: Compare indices across income groups
        income_indices_fig = None
        if all(col in df.columns for col in ["INCOME", "Health_Index", "CharlsonIndex", "ElixhauserIndex"]):
            # Create income quartiles
            df_with_quartiles = df.copy()
            df_with_quartiles['Income_Quartile'] = pd.qcut(
                df_with_quartiles['INCOME'], 
                q=4, 
                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
            )
            
            # Calculate mean indices by income quartile
            income_indices = df_with_quartiles.groupby("Income_Quartile").agg({
                "Health_Index": "mean",
                "CharlsonIndex": "mean",
                "ElixhauserIndex": "mean"
            }).reset_index()
            
            # Create comparative bar chart
            income_indices_fig = px.bar(
                income_indices,
                x="Income_Quartile",
                y=["Health_Index", "CharlsonIndex", "ElixhauserIndex"],
                barmode="group",
                title="Health Indices by Income Quartile",
                color_discrete_sequence=[nhs_colors["primary"], nhs_colors["risk_medium"], nhs_colors["risk_high"]],
                labels={
                    "Income_Quartile": "Income Quartile",
                    "value": "Index Value",
                    "variable": "Health Metric"
                }
            )
            
            income_indices_fig.update_layout(
                height=600, 
                margin=dict(l=20, r=20, t=40, b=20), 
                autosize=True,
                xaxis={'categoryorder': 'array', 'categoryarray': ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']}
            )
            
            # Add annotation explaining the metrics
            income_indices_fig.add_annotation(
                text="Health Index: higher is better<br>Charlson & Elixhauser: lower is better",
                xref="paper", yref="paper",
                x=0.01, y=0.98,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="#005EB8",
                borderwidth=1,
                borderpad=4
            )
        
        return collapsible_card(
            "Comparative Health Indices Analysis",
            html.Div([
                html.P(
                    "These visualizations compare multiple health indices across different demographic groups, revealing patterns of health inequality.",
                    style={"marginBottom": "15px"}
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(figure=race_indices_fig) if race_indices_fig else "No data available for racial health indices comparison",
                        width=12,
                        lg=6
                    ),
                    dbc.Col(
                        dcc.Graph(figure=income_indices_fig) if income_indices_fig else "No data available for income-based health indices comparison",
                        width=12,
                        lg=6
                    ),
                ]),
            ]),
            "multi-indices-comparison-card",
            initially_open=True
        )
    except Exception as e:
        logger.warning(f"Could not create multi-indices comparison card: {e}")
        return None

def create_health_indices_radar(df):
    """Create radar chart visualization showing health indices by demographic group"""
    if df is None or df.empty:
        return None
    
    try:
        # Create intersectional demographic radar chart for Health Index
        intersectional_fig = None
        if all(col in df.columns for col in ["GENDER", "Age_Group", "RACE", "Health_Index"]):
            # Focus on the most common race categories
            top_races = df["RACE"].value_counts().nlargest(3).index.tolist()
            
            # Create intersectional groups (gender + age group + race)
            intersect_data = []
            for gender in df["GENDER"].unique():
                for age_group in ["19-35", "36-50", "51-65", "66-80"]:  # Select key age groups
                    for race in top_races:
                        subset = df[(df["GENDER"] == gender) & 
                                   (df["Age_Group"] == age_group) & 
                                   (df["RACE"] == race)]
                        
                        if len(subset) >= 30:  # Only include if we have enough data
                            group_name = f"{gender}, {age_group}, {race}"
                            health_index = subset["Health_Index"].mean()
                            intersect_data.append({
                                "Group": group_name,
                                "Gender": gender,
                                "Age_Group": age_group,
                                "Race": race,
                                "Health_Index": health_index
                            })
            
            if intersect_data:
                intersect_df = pd.DataFrame(intersect_data)
                
                # Create pivot table for radar chart
                pivot_table = pd.pivot_table(
                    intersect_df,
                    values="Health_Index",
                    index=["Gender", "Race"],
                    columns=["Age_Group"],
                    aggfunc="mean"
                ).reset_index()
                
                # Filter to ensure we have complete data across age groups
                complete_rows = pivot_table.dropna()
                
                if not complete_rows.empty:
                    # Create radar chart
                    fig = go.Figure()
                    
                    for i, row in complete_rows.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=row[["19-35", "36-50", "51-65", "66-80"]].values,
                            theta=["19-35", "36-50", "51-65", "66-80"],
                            fill='toself',
                            name=f"{row['Gender']}, {row['Race']}"
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10]
                            )
                        ),
                        title="Health Index Across Age Groups by Gender and Race",
                        height=600,
                        margin=dict(l=50, r=50, t=50, b=50),
                        showlegend=True,
                        legend=dict(
                            title="Demographic Group",
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    # Add annotation explaining the visualization
                    fig.add_annotation(
                        text="This radar chart shows health inequalities across intersectional demographic groups.<br>Each line represents a combined gender and race group, with values across age groups.",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.3,
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="#005EB8",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                    
                    intersectional_fig = fig
        
        # Create a complementary visualization - Health Indices by age and gender
        age_gender_fig = None
        if all(col in df.columns for col in ["GENDER", "Age_Group", "Health_Index", "CharlsonIndex"]):
            # Calculate mean indices by age group and gender
            age_gender_indices = df.groupby(["Age_Group", "GENDER"]).agg({
                "Health_Index": "mean",
                "CharlsonIndex": "mean"
            }).reset_index()
            
            # Create a proper ordering for age groups
            age_order = ["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]
            age_gender_indices["Age_Group"] = pd.Categorical(
                age_gender_indices["Age_Group"], 
                categories=age_order, 
                ordered=True
            )
            age_gender_indices = age_gender_indices.sort_values("Age_Group")
            
            # Create line chart
            age_gender_fig = px.line(
                age_gender_indices,
                x="Age_Group",
                y="Health_Index",
                color="GENDER",
                markers=True,
                title="Health Index Trajectory Across Lifespan by Gender",
                color_discrete_sequence=[nhs_colors["primary"], nhs_colors["risk_medium"]],
                labels={"Age_Group": "Age Group", "Health_Index": "Health Index", "GENDER": "Gender"}
            )
            
            # Add Charlson Index as second y-axis
            age_gender_fig.add_trace(
                go.Scatter(
                    x=age_gender_indices[age_gender_indices["GENDER"] == df["GENDER"].iloc[0]]["Age_Group"],
                    y=age_gender_indices[age_gender_indices["GENDER"] == df["GENDER"].iloc[0]]["CharlsonIndex"],
                    mode="lines+markers",
                    name="Charlson Index",
                    line=dict(color="#FFB81C", width=2, dash="dot"),
                    marker=dict(size=8, color="#FFB81C"),
                    yaxis="y2"
                )
            )
            
            # Update layout for dual y-axis
            age_gender_fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True,
                yaxis=dict(title="Health Index"),
                yaxis2=dict(
                    title="Charlson Index",
                    overlaying="y",
                    side="right",
                    range=[0, age_gender_indices["CharlsonIndex"].max() * 1.2]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Add annotation explaining the two metrics
            age_gender_fig.add_annotation(
                text="Health Index (left axis): higher is better<br>Charlson Index (right axis): lower is better",
                xref="paper", yref="paper",
                x=0.01, y=0.98,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="#005EB8",
                borderwidth=1,
                borderpad=4
            )
        
        return collapsible_card(
            "Intersectional Health Inequalities Analysis",
            html.Div([
                html.P(
                    "These visualizations analyze health indices across multiple demographic factors simultaneously, highlighting intersectional health inequalities.",
                    style={"marginBottom": "15px"}
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(figure=intersectional_fig) if intersectional_fig else "Insufficient data for intersectional health analysis",
                        width=12,
                        lg=6
                    ),
                    dbc.Col(
                        dcc.Graph(figure=age_gender_fig) if age_gender_fig else "No data available for age-gender health indices analysis",
                        width=12,
                        lg=6
                    ),
                ]),
            ]),
            "health-indices-radar-card",
            initially_open=True
        )
    except Exception as e:
        logger.warning(f"Could not create health indices radar card: {e}")
        return None

def create_healthcare_access_card(df):
    """Create visualization showing healthcare access and utilization inequalities"""
    if df is None or df.empty:
        return None
    
    try:
        # Healthcare expenses vs. coverage visualization
        expenses_coverage_fig = None
        if all(col in df.columns for col in ["HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "Risk_Category"]):
            # Sample data for scatter plot
            if len(df) > 3000:
                df_sample = df.sample(3000, random_state=42)
            else:
                df_sample = df.copy()
            
            # Create scatter plot
            expenses_coverage_fig = px.scatter(
                df_sample,
                x="HEALTHCARE_EXPENSES",
                y="HEALTHCARE_COVERAGE",
                color="Risk_Category",
                size="Health_Index",
                hover_data=["GENDER", "RACE", "AGE"],
                title="Healthcare Coverage vs. Expenses by Risk Category",
                color_discrete_map={
                    "Very Low Risk": nhs_colors["risk_verylow"],
                    "Low Risk": nhs_colors["risk_low"],
                    "Moderate Risk": nhs_colors["risk_medium"],
                    "High Risk": nhs_colors["risk_high"],
                },
                labels={
                    "HEALTHCARE_EXPENSES": "Healthcare Expenses ($)",
                    "HEALTHCARE_COVERAGE": "Healthcare Coverage",
                    "Risk_Category": "Risk Category",
                    "Health_Index": "Health Index"
                }
            )
            
            # Add trend line for the entire dataset
            expenses_coverage_fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )
        
        # Healthcare coverage by demographic group visualization
        coverage_demo_fig = None
        if all(col in df.columns for col in ["HEALTHCARE_COVERAGE", "RACE", "INCOME"]):
            # Create income quartiles
            df_with_quartiles = df.copy()
            df_with_quartiles['Income_Quartile'] = pd.qcut(
                df_with_quartiles['INCOME'], 
                q=4, 
                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
            )
            
            # Get top races
            top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
            filtered_df = df_with_quartiles[df_with_quartiles["RACE"].isin(top_races)].copy()
            
            # Calculate mean healthcare coverage by race and income quartile
            coverage_by_demo = filtered_df.groupby(["RACE", "Income_Quartile"])["HEALTHCARE_COVERAGE"].mean().reset_index()
            
            # Create grouped bar chart
            coverage_demo_fig = px.bar(
                coverage_by_demo,
                x="RACE",
                y="HEALTHCARE_COVERAGE",
                color="Income_Quartile",
                barmode="group",
                title="Healthcare Coverage by Race and Income",
                color_discrete_map={
                    "Q1 (Low)": nhs_colors["risk_high"],
                    "Q2": nhs_colors["risk_medium"],
                    "Q3": nhs_colors["risk_low"],
                    "Q4 (High)": nhs_colors["risk_verylow"]
                },
                labels={
                    "RACE": "Race",
                    "HEALTHCARE_COVERAGE": "Average Healthcare Coverage",
                    "Income_Quartile": "Income Quartile"
                }
            )
            
            coverage_demo_fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True,
                yaxis=dict(title="Average Healthcare Coverage")
            )
            
            # Add annotation about healthcare coverage
            coverage_demo_fig.add_annotation(
                text="Higher healthcare coverage values indicate better insurance<br>and access to healthcare services",
                xref="paper", yref="paper",
                x=0.01, y=0.98,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="#005EB8",
                borderwidth=1,
                borderpad=4
            )
        
        return collapsible_card(
            "Healthcare Access and Utilization Inequalities",
            html.Div([
                html.P(
                    "These visualizations explore inequalities in healthcare access, coverage, and utilization across different demographic groups.",
                    style={"marginBottom": "15px"}
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(figure=expenses_coverage_fig) if expenses_coverage_fig else "No data available for healthcare expenses vs. coverage visualization",
                        width=12,
                        lg=6
                    ),
                    dbc.Col(
                        dcc.Graph(figure=coverage_demo_fig) if coverage_demo_fig else "No data available for healthcare coverage by demographic group visualization",
                        width=12,
                        lg=6
                    ),
                ]),
            ]),
            "healthcare-access-card",
            initially_open=True
        )
    except Exception as e:
        logger.warning(f"Could not create healthcare access card: {e}")
        return None

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
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True,
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
                                ).update_layout(showlegend=False, height=600, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
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
        html.H4("Model Execution Results", className="mb-4"),
        html.Div([
            html.H5("Execution Summary", className="mb-3"),
            html.P([
                html.Strong("Status: "),
                html.Span("Completed Successfully", className="text-success")
            ]),
            html.P([
                html.Strong("Total Time: "),
                html.Span(state.get("remaining", "N/A"))
            ]),
            html.P([
                html.Strong("Final Stage: "),
                html.Span(state.get("stage", "Unknown"))
            ])
        ], className="mb-4"),
        html.Div([
            html.H5("Output Log", className="mb-3"),
            html.Div(
                "\n".join(state.get("output", [])),
                style={
                    "maxHeight": "300px",
                    "overflowY": "auto",
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "fontFamily": "monospace",
                    "fontSize": "12px"
                }
            )
        ])
    ])
    
    return results_content, {"display": "block"}

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

def sanitize_datatable_values(df):
    """Prepare dataframe values for presentation in a data table"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    display_df = df.copy()
    
    # Convert floats to 2 decimal places
    for col in display_df.select_dtypes(include=['float']).columns:
        display_df[col] = display_df[col].round(2)
    
    # Format currency values
    if 'INCOME' in display_df.columns:
        display_df['INCOME'] = display_df['INCOME'].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else "")
    if 'HEALTHCARE_EXPENSES' in display_df.columns:
        display_df['HEALTHCARE_EXPENSES'] = display_df['HEALTHCARE_EXPENSES'].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else "")
    
    # Convert date columns to string format
    for col in display_df.select_dtypes(include=['datetime']).columns:
        display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
    
    # Fill NaN values with an empty string for display
    display_df = display_df.fillna("")
    
    return display_df

def create_demographic_charts(df):
    """Create charts showing gender, age, and risk distribution"""
    if df is None or df.empty:
        return None, None, None
    
    # 1. Gender distribution
    gender_chart = None
    if "GENDER" in df.columns:
        gender_counts = df["GENDER"].value_counts().reset_index()
        gender_counts.columns = ["GENDER", "Count"]
        
        gender_chart = px.pie(
            gender_counts, 
            values="Count", 
            names="GENDER",
            title="Gender Distribution",
            color_discrete_sequence=[nhs_colors["primary"], nhs_colors["risk_medium"]],
            hole=0.4
        )
        gender_chart.update_traces(textposition='inside', textinfo='percent+label')
        gender_chart.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
    
    # 2. Age distribution
    age_chart = None
    if "Age_Group" in df.columns:
        age_counts = df["Age_Group"].value_counts().reset_index()
        age_counts.columns = ["Age_Group", "Count"]
        
        # Sort age groups in correct order
        age_order = ["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]
        age_counts["Age_Group"] = pd.Categorical(age_counts["Age_Group"], categories=age_order, ordered=True)
        age_counts = age_counts.sort_values("Age_Group")
        
        age_chart = px.bar(
            age_counts,
            x="Age_Group",
            y="Count",
            title="Age Distribution",
            color_discrete_sequence=[nhs_colors["primary"]],
            labels={"Age_Group": "Age Group", "Count": "Number of Patients"}
        )
        age_chart.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
    
    # 3. Risk category distribution
    risk_chart = None
    if "Risk_Category" in df.columns:
        risk_counts = df["Risk_Category"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Category", "Count"]
        
        # Sort risk categories in correct order
        risk_order = ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk"]
        risk_counts["Risk_Category"] = pd.Categorical(risk_counts["Risk_Category"], categories=risk_order, ordered=True)
        risk_counts = risk_counts.sort_values("Risk_Category")
        
        risk_chart = px.bar(
            risk_counts,
            x="Risk_Category",
            y="Count",
            title="Risk Distribution",
            color="Risk_Category",
            color_discrete_map={
                "Very Low Risk": nhs_colors["risk_verylow"],
                "Low Risk": nhs_colors["risk_low"],
                "Moderate Risk": nhs_colors["risk_medium"],
                "High Risk": nhs_colors["risk_high"],
            },
            labels={"Risk_Category": "Risk Category", "Count": "Number of Patients"}
        )
        risk_chart.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300, showlegend=False)
    
    return gender_chart, age_chart, risk_chart

def create_race_demographics(df):
    """Create charts showing race distribution and healthcare expenses by race"""
    if df is None or df.empty:
        return None, None
    
    # 1. Race distribution
    race_chart = None
    if "RACE" in df.columns:
        # Get top N races to keep chart readable
        top_races = df["RACE"].value_counts().nlargest(7).index.tolist()
        df_top_races = df[df["RACE"].isin(top_races)].copy()
        other_count = len(df) - len(df_top_races)
        
        race_counts = df_top_races["RACE"].value_counts().reset_index()
        race_counts.columns = ["RACE", "Count"]
        
        # Add "Other" category if needed
        if other_count > 0:
            race_counts = pd.concat([
                race_counts,
                pd.DataFrame({"RACE": ["Other"], "Count": [other_count]})
            ])
        
        race_chart = px.bar(
            race_counts,
            x="RACE",
            y="Count",
            title="Racial Distribution",
            color_discrete_sequence=[nhs_colors["primary"]],
            labels={"RACE": "Race", "Count": "Number of Patients"}
        )
        race_chart.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
    
    # 2. Healthcare expenses by race
    healthcare_expense_chart = None
    if all(col in df.columns for col in ["RACE", "HEALTHCARE_EXPENSES"]):
        # Get top N races to keep chart readable
        top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
        df_top_races = df[df["RACE"].isin(top_races)].copy()
        
        # Calculate mean healthcare expenses by race
        expenses_by_race = df_top_races.groupby("RACE")["HEALTHCARE_EXPENSES"].mean().reset_index()
        expenses_by_race = expenses_by_race.sort_values("HEALTHCARE_EXPENSES", ascending=False)
        
        healthcare_expense_chart = px.bar(
            expenses_by_race,
            x="RACE",
            y="HEALTHCARE_EXPENSES",
            title="Average Healthcare Expenses by Race",
            color_discrete_sequence=[nhs_colors["accent"]],
            labels={"RACE": "Race", "HEALTHCARE_EXPENSES": "Average Healthcare Expenses ($)"}
        )
        healthcare_expense_chart.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
    
    return race_chart, healthcare_expense_chart

def create_income_health_chart(df):
    """Create visualization showing the relationship between income and health indices"""
    if df is None or df.empty:
        return None
    
    if not all(col in df.columns for col in ["INCOME", "Health_Index"]):
        return None
    
    try:
        # Sample data if too large
        if len(df) > 5000:
            df_sample = df.sample(5000, random_state=42)
        else:
            df_sample = df.copy()
        
        # Create income quartiles
        df_sample['Income_Quartile'] = pd.qcut(
            df_sample['INCOME'], 
            q=4, 
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        )
        
        fig = px.scatter(
            df_sample,
            x="INCOME",
            y="Health_Index",
            color="Income_Quartile",
            color_discrete_map={
                "Q1 (Low)": nhs_colors["risk_high"],
                "Q2": nhs_colors["risk_medium"],
                "Q3": nhs_colors["risk_low"],
                "Q4 (High)": nhs_colors["risk_verylow"],
            },
            opacity=0.7,
            title="Income vs. Health Index",
            labels={
                "INCOME": "Income ($)",
                "Health_Index": "Health Index",
                "Income_Quartile": "Income Quartile"
            },
            trendline="ols",
            trendline_color_override="#333333"
        )
        
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        return fig
    except Exception as e:
        logger.warning(f"Could not create income vs health chart: {e}")
        return None

def create_health_inequality_chart(df):
    """Create visualization showing health inequality across income levels"""
    if df is None or df.empty:
        return None
    
    if not all(col in df.columns for col in ["INCOME", "Health_Index", "HEALTHCARE_EXPENSES"]):
        return None
    
    try:
        # Create income deciles for more granular analysis
        df_with_deciles = df.copy()
        df_with_deciles['Income_Decile'] = pd.qcut(
            df_with_deciles['INCOME'], 
            q=10, 
            labels=[f'D{i}' for i in range(1, 11)]
        )
        
        # Calculate mean health index and healthcare expenses by income decile
        inequality_data = df_with_deciles.groupby('Income_Decile').agg({
            'Health_Index': 'mean',
            'HEALTHCARE_EXPENSES': 'mean'
        }).reset_index()
        
        # Create a dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add health index line
        fig.add_trace(
            go.Scatter(
                x=inequality_data['Income_Decile'], 
                y=inequality_data['Health_Index'],
                name="Health Index",
                line=dict(color=nhs_colors["primary"], width=3),
                mode="lines+markers"
            ),
            secondary_y=False
        )
        
        # Add healthcare expenses bars
        fig.add_trace(
            go.Bar(
                x=inequality_data['Income_Decile'], 
                y=inequality_data['HEALTHCARE_EXPENSES'],
                name="Healthcare Expenses",
                marker_color=nhs_colors["accent"],
                opacity=0.7
            ),
            secondary_y=True
        )
        
        # Customize the layout
        fig.update_layout(
            title="Health Inequality Across Income Levels",
            xaxis_title="Income Decile (D1=Lowest, D10=Highest)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified"
        )
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Average Health Index", secondary_y=False, color=nhs_colors["primary"])
        fig.update_yaxes(title_text="Average Healthcare Expenses ($)", secondary_y=True, color=nhs_colors["accent"])
        
        return fig
    except Exception as e:
        logger.warning(f"Could not create health inequality chart: {e}")
        return None

def create_intersectional_analysis(df):
    """Create visualizations showing intersectional analysis of health outcomes"""
    if df is None or df.empty:
        return None
    
    # Check for required columns
    req_cols = ["GENDER", "RACE", "INCOME", "Health_Index", "AGE"]
    if not all(col in df.columns for col in req_cols):
        return None
    
    try:
        # 1. Gender and Race Intersectional Analysis
        # Get the top 5 most common racial categories
        top_races = df["RACE"].value_counts().nlargest(5).index.tolist()
        filtered_df = df[df["RACE"].isin(top_races)].copy()
        
        # Calculate mean health index by gender and race
        intersect_data = filtered_df.groupby(["GENDER", "RACE"])["Health_Index"].mean().reset_index()
        
        # Create the first intersectional chart
        fig1 = px.bar(
            intersect_data,
            x="RACE",
            y="Health_Index",
            color="GENDER",
            barmode="group",
            title="Intersectional Analysis: Gender and Race",
            color_discrete_sequence=[nhs_colors["primary"], nhs_colors["risk_medium"]],
            labels={
                "RACE": "Race",
                "Health_Index": "Average Health Index",
                "GENDER": "Gender"
            }
        )
        
        fig1.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        
        # 2. Age, Gender, and Health Outcomes
        # Create age groups for analysis
        if "Age_Group" not in filtered_df.columns:
            filtered_df["Age_Group"] = pd.cut(
                filtered_df["AGE"],
                bins=[0, 18, 35, 50, 65, 80, 120],
                labels=["0-18", "19-35", "36-50", "51-65", "66-80", "80+"]
            )
        
        # Calculate mean health index by age group and gender
        age_gender_data = filtered_df.groupby(["Age_Group", "GENDER"])["Health_Index"].mean().reset_index()
        
        # Create the second intersectional chart
        fig2 = px.line(
            age_gender_data,
            x="Age_Group",
            y="Health_Index",
            color="GENDER",
            markers=True,
            title="Intersectional Analysis: Age and Gender",
            color_discrete_sequence=[nhs_colors["primary"], nhs_colors["risk_medium"]],
            labels={
                "Age_Group": "Age Group",
                "Health_Index": "Average Health Index",
                "GENDER": "Gender"
            }
        )
        
        fig2.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        
        return [fig1, fig2]
    except Exception as e:
        logger.warning(f"Could not create intersectional analysis charts: {e}")
        return None

def create_clinical_risk_clusters(df, model_name=""):
    """Create visualization showing clinical risk clusters"""
    if df is None or df.empty:
        return None
    
    # Check for required columns
    if not all(col in df.columns for col in ["Health_Index", "CharlsonIndex", "ElixhauserIndex"]):
        return None
    
    try:
        # Sample data if too large
        if len(df) > 3000:
            df_sample = df.sample(3000, random_state=42)
        else:
            df_sample = df.copy()
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df_sample,
            x="Health_Index",
            y="CharlsonIndex",
            z="ElixhauserIndex",
            color="Cluster" if "Cluster" in df_sample.columns else "Risk_Category",
            title=f"Clinical Risk Stratification - {model_name}",
            labels={
                "Health_Index": "Health Index",
                "CharlsonIndex": "Charlson Index",
                "ElixhauserIndex": "Elixhauser Index",
                "Cluster": "Risk Cluster",
                "Risk_Category": "Risk Category"
            },
            color_discrete_sequence=risk_colors
        )
        
        fig.update_layout(height=800, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
        return fig
    except Exception as e:
        logger.warning(f"Could not create clinical risk clusters visualization: {e}")
        return None

# Progress queue and process handle initialization
progress_queue = queue.Queue(maxsize=10_000)
process_handle = None
process_lock = threading.Lock()

from pathlib import Path
def _build_command(population, perf_opts, mem_util, cpu_util, threads, use_remote=False):
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
    
    if use_remote:
        # Build a single shell string that runs in the Pi's working dir
        remote_cmd = " ".join(["./GenerateAndPredict", *map(shlex.quote, flags)])
        return [
            "ssh",
            "-i", SSH_KEY_PATH,
            "-o", "StrictHostKeyChecking=no",  # Don't ask for host key verification
            "-o", "UserKnownHostsFile=/dev/null",  # Don't store host key
            "-o", "LogLevel=ERROR",  # Reduce SSH noise
            SSH_HOST,
            f"cd {shlex.quote(REMOTE_WORKDIR)} && {remote_cmd}"
        ]
    else:
        # Legacy local execution keeps working
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
        return [exe_path, *flags]

def _launch_generate_and_predict(cmd: list[str]) -> None:
    global process_handle

    logger.info(f"Launching command: {' '.join(cmd)}")
    
    # Determine if this is a remote SSH execution
    is_remote = cmd[0] == "ssh" if cmd else False
    
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

        # Log the start of execution
        if is_remote:
            progress_queue.put_nowait("[INFO] Started remote execution on Raspberry Pi")
            progress_queue.put_nowait("[PROGRESS] Initializing")
        else:
            progress_queue.put_nowait("[INFO] Started local execution")

        while True:
            raw = proc.stdout.readline()
            if not raw:
                break
                
            line = raw.rstrip("\n")
            
            # For remote execution, we need to parse the output differently
            if is_remote:
                # Look for progress indicators in the output
                if "[INFO]" in line:
                    progress_queue.put_nowait(line)
                elif "[PROGRESS]" in line:
                    progress_queue.put_nowait(line)
                elif "[ERROR]" in line:
                    progress_queue.put_nowait(line)
                elif "[WARNING]" in line:
                    progress_queue.put_nowait(line)
                elif "connection closed" in line.lower():
                    progress_queue.put_nowait("[WARNING] SSH connection closed unexpectedly")
                else:
                    # For other output, prefix with [INFO] to ensure it's captured
                    progress_queue.put_nowait(f"[INFO] {line}")
            else:
                progress_queue.put_nowait(line)
            
            # Make sure important progress messages are flushed immediately
            if "[PROGRESS]" in line:
                sys.stdout.flush()

        # Check return code and put appropriate message in queue
        return_code = proc.returncode
        if return_code == 0:
            if is_remote:
                progress_queue.put_nowait("[INFO] Remote execution completed successfully")
                progress_queue.put_nowait("[PROGRESS] Completed")
            else:
                progress_queue.put_nowait("[INFO] Local execution completed successfully")
        else:
            if is_remote:
                progress_queue.put_nowait(f"[ERROR] Remote execution failed with code {return_code}")
            else:
                progress_queue.put_nowait(f"[ERROR] Local execution failed with code {return_code}")

        progress_queue.put_nowait(f"[EXITCODE] {return_code}")

    with process_lock:
        process_handle = None

@app.callback(
    Output("system-resources-content", "children"),
    [Input("system-resources-interval", "n_intervals"),
     Input("execution-mode-toggle", "value")],
    prevent_initial_call=True
)
def update_system_resources(n_intervals, execution_mode):
    """Update system resources display with modern grid layout and gauges"""
    try:
        if execution_mode == "remote":
            system_info = get_remote_system_resources()
            if system_info is None:
                return html.Div([
                    html.H6("Remote System Status", className="mt-3"),
                    html.P("Failed to connect to Raspberry Pi", className="text-danger"),
                    html.P("Please check:", className="mt-2"),
                    html.Ul([
                        html.Li("SSH connection settings"),
                        html.Li("Raspberry Pi is powered on"),
                        html.Li("Network connectivity"),
                        html.Li("SSH key permissions"),
                        html.Li("Hostname resolution (try using IP address instead of hostname)")
                    ]),
                    html.P([
                        "Current SSH settings: ",
                        html.Code(f"Host: {SSH_HOST}, Key: {SSH_KEY_PATH}")
                    ], className="mt-3")
                ])
        else:
            system_info, _ = get_system_resources()
        
        # Validate system_info before using
        required_keys = [
            "cpu_percent", "memory_percent", "processor", "cpu_count", "memory_total_gb", "platform"
        ]
        missing_or_none = [k for k in required_keys if k not in system_info or system_info[k] is None]
        if missing_or_none:
            logger.error(f"System info missing or None for keys: {missing_or_none}. system_info: {system_info}")
            return html.Div([
                html.H6("System Monitor Error", className="mt-3"),
                html.P("Could not retrieve system resource information.", className="text-danger"),
                html.P(f"Missing or invalid keys: {', '.join(missing_or_none)}", className="text-muted small"),
                html.P("Try refreshing the page or check your system monitoring dependencies.", className="text-muted small")
            ])
        
        # Create circular gauge for CPU (match Memory gauge layout)
        cpu_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=system_info["cpu_percent"],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#333"},
                'bar': {'color': "rgb(0, 123, 255)", 'thickness': 0.25},
                'steps': [
                    {'range': [0, 50], 'color': "rgb(40, 167, 69)"},
                    {'range': [50, 80], 'color': "rgb(255, 193, 7)"},
                    {'range': [80, 100], 'color': "rgb(220, 53, 69)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            },
            number={
                'font': {'size': 32, 'color': '#223A5E'},
                'suffix': '%',
            },
            title={'text': "CPU Usage (%)", 'font': {'size': 18, 'color': '#223A5E'}},
        ))
        cpu_gauge.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Create circular gauge for Memory (remove 'shape': 'semi')
        memory_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=system_info["memory_percent"],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#333"},
                'bar': {'color': "rgb(23, 162, 184)", 'thickness': 0.25},
                'steps': [
                    {'range': [0, 50], 'color': "rgb(40, 167, 69)"},
                    {'range': [50, 80], 'color': "rgb(255, 193, 7)"},
                    {'range': [80, 100], 'color': "rgb(220, 53, 69)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
                # 'shape': 'angular' is default, so omit
            },
            number={
                'font': {'size': 32, 'color': '#223A5E'},
                'suffix': '%',
            },
            title={'text': "Memory Usage (%)", 'font': {'size': 18, 'color': '#223A5E'}},
        ))
        memory_gauge.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Create system info cards
        system_info_display = [
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("System Information", className="text-center mb-3"),
                        html.Div([
                            html.P([
                                html.I(className="fas fa-microchip me-2"),
                                f"Processor: {system_info['processor']}"
                            ], className="mb-2"),
                            html.P([
                                html.I(className="fas fa-server me-2"),
                                f"CPU Cores: {system_info['cpu_count']}"
                            ], className="mb-2"),
                            html.P([
                                html.I(className="fas fa-memory me-2"),
                                f"Total Memory: {system_info['memory_total_gb']:.1f}GB"
                            ], className="mb-2"),
                            html.P([
                                html.I(className="fas fa-desktop me-2"),
                                f"Platform: {system_info['platform']}"
                            ], className="mb-2"),
                        ], className="p-3 bg-light rounded")
                    ], className="h-100")
                ], width=12, md=4),
                dbc.Col([
                    html.Div([
                        html.H6("CPU Usage", className="text-center mb-3"),
                        dcc.Graph(figure=cpu_gauge, config={'displayModeBar': False})
                    ], className="h-100")
                ], width=12, md=4),
                dbc.Col([
                    html.Div([
                        html.H6("Memory Usage", className="text-center mb-3"),
                        dcc.Graph(figure=memory_gauge, config={'displayModeBar': False})
                    ], className="h-100")
                ], width=12, md=4)
            ], className="mb-4")
        ]
        
        # Add Raspberry Pi specific information if available
        if system_info.get("temperature") is not None:
            temp_color = "success" if system_info["temperature"] < 70 else "danger"
            system_info_display.append(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Raspberry Pi Status", className="text-center mb-3"),
                            html.Div([
                                html.P([
                                    html.I(className="fas fa-thermometer-half me-2"),
                                    "Temperature: ",
                                    html.Span(
                                        f"{system_info['temperature']}°C",
                                        className=f"text-{temp_color}"
                                    )
                                ], className="mb-2"),
                                html.P([
                                    html.I(className="fas fa-clock me-2"),
                                    f"Uptime: {system_info.get('uptime', 'N/A')}"
                                ], className="mb-2"),
                                html.P([
                                    html.I(className="fas fa-tachometer-alt me-2"),
                                    f"Load Average: {system_info.get('load_average', 'N/A'):.2f}"
                                ], className="mb-2"),
                                html.P([
                                    html.I(className="fas fa-network-wired me-2"),
                                    f"Connected to: {system_info.get('hostname', 'N/A')}"
                                ], className="mb-2"),
                            ], className="p-3 bg-light rounded")
                        ], className="h-100")
                    ], width=12)
                ])
            )
        
        return html.Div(system_info_display)
        
    except Exception as e:
        logger.error(f"Error updating system resources: {str(e)}")
        return html.Div("Error updating system resources", className="text-danger")

if __name__ == "__main__":
    app.run_server(debug=True)
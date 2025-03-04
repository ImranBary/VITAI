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
import plotly.express as px
import plotly.graph_objects as go

from xai_formatter import format_explanation

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
    df_all = pd.read_pickle(PICKLE_ALL)
    logger.info(f"Loaded enriched data from {PICKLE_ALL}.")
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

def load_final_model_outputs():
    """
    Load each model’s predictions and clusters, then also merge in
    `Health_Index` (and optionally more columns) from df_all
    so we can build a scatter plot of (Health_Index vs. PredictedHI_final).
    """
    models_data = {}

    # We'll keep a small subset of base columns to merge in.  
    # Adjust as you need: e.g., "CharlsonIndex", "ElixhauserIndex".
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
            # Combine them
            # (Renaming keys as needed to match your code's usage)
            metrics_primary["Silhouette"]         = metrics_cluster.get("silhouette", "N/A")
            metrics_primary["Calinski_Harabasz"]  = metrics_cluster.get("calinski", "N/A")
            metrics_primary["Davies_Bouldin"]     = metrics_cluster.get("davies_bouldin", "N/A")
        else:
            metrics_primary.setdefault("Silhouette", "N/A")
            metrics_primary.setdefault("Calinski_Harabasz", "N/A")
            metrics_primary.setdefault("Davies_Bouldin", "N/A")

        # Encode t-SNE / UMAP
        tsne_img = encode_image(tsne_png)
        umap_img = encode_image(umap_png)

        models_data[label_name] = {
            "df": df_model,
            "metrics": metrics_primary,
            "tsne_img": tsne_img,
            "umap_img": umap_img
        }

    return models_data

# -----------------------------
# Now that we have df_all loaded, we also do a "global" merge so that
# df_all has all "PredictedHI_final" columns. You can do that if you want
# a single data source. But we can skip that if you rely on the
# per-model merges above.

# For now, let's keep it simple: the "df_model" used in the callback
# will have "Health_Index", "PredictedHI_final", "Cluster_final".

logger.info("Merging final model outputs with base data if needed.")
final_models_data = load_final_model_outputs()

# If you want to also unify them in df_all, do so:
list_models = []
for label, mdata in final_models_data.items():
    if not mdata["df"].empty:
        mdata["df"]["Group"] = label
        list_models.append(mdata["df"])
if list_models:
    df_models = pd.concat(list_models, ignore_index=True)
    # If you want a master df with everything:
    df_all = pd.merge(df_all, df_models.drop(columns=["Health_Index","CharlsonIndex","ElixhauserIndex"], errors="ignore"),
                      on="Id", how="left")
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

# -------------- [ Dashboard Layout Below, Same as Before ] --------------

nhs_colors = {
    "background": "#F7F7F7",
    "text": "#333333",
    "primary": "#005EB8",
    "secondary": "#FFFFFF",
    "accent": "#00843D"
}
external_stylesheets = [dbc.themes.FLATLY]

def kpi_card(title, value):
    return html.Div(
        style={
            "padding": "20px",
            "margin": "10px",
            "backgroundColor": nhs_colors["secondary"],
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "borderRadius": "5px",
            "textAlign": "center"
        },
        children=[
            html.H4(title, style={"color": nhs_colors["primary"]}),
            html.H2(value, style={"color": nhs_colors["text"], "fontWeight": "bold"})
        ]
    )

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

def correlation_figure(df):
    relevant_cols = ["AGE", "INCOME", "Health_Index", "CharlsonIndex", "ElixhauserIndex"]
    existing = [c for c in relevant_cols if c in df.columns]
    if len(existing) < 2:
        return go.Figure().update_layout(title="Not enough numeric columns for correlation")

    corr = df[existing].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Among Key Features")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    return fig

overview_tab = dbc.Container([
    html.H2("Overview & KPIs", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    dbc.Row([
        dbc.Col(kpi_card("Total Patients", TOTAL_PATIENTS), width=2),
        dbc.Col(kpi_card("Avg Age",       AVG_AGE), width=2),
        dbc.Col(kpi_card("Avg Income",    f"£{AVG_INCOME}"), width=2),
        dbc.Col(kpi_card("Avg Health",    AVG_HEALTH_INDEX), width=2),
        dbc.Col(kpi_card("Avg Charlson",  AVG_CHARLSON), width=2),
        dbc.Col(kpi_card("Avg Elixhauser",AVG_ELIXHAUSER), width=2),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id="hist-healthindex",
                figure=px.histogram(
                    df_all, x="Health_Index", nbins=30,
                    title="Distribution of Health Index",
                    template="plotly_white"
                )
            ),
            width=6
        ),
        dbc.Col(
            dcc.Graph(
                id="hist-charlson",
                figure=px.histogram(
                    df_all, x="CharlsonIndex", nbins=30,
                    title="Distribution of Charlson Index",
                    template="plotly_white"
                )
            ),
            width=6
        )
    ]),
    html.Br(),
    dcc.Graph(id="correlation-heatmap", figure=correlation_figure(df_all)),
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

model_details_tab = dbc.Container([
    html.H2("Model Details & Clustering", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    html.Div([
        html.Label("Select Final Model Group:"),
        dcc.Dropdown(
            id="model-group-dropdown",
            options=[{"label": g["label"], "value": g["label"]} for g in final_groups],
            value="General",
            clearable=False,
            style={"width": "300px", "marginBottom": "20px"}
        )
    ]),
    html.Div(id="model-details-content")
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

xai_tab = dbc.Container([
    html.H2("Explainable AI (XAI) Insights", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    html.Div([
        html.Label("Select Model Group for XAI:"),
        dcc.Dropdown(
            id="xai-model-group-dropdown",
            options=[{"label": g["label"], "value": g["label"]} for g in final_groups],
            value="General",
            clearable=False,
            style={"width": "300px", "marginBottom": "20px"}
        )
    ]),
    html.Div(id="xai-content-area")
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

map_tab = dbc.Container([
    html.H2("Patient Map & Inequalities", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Model Group:"),
            dcc.Dropdown(
                id="map-model-dropdown",
                options=[{"label": "All", "value": "All"}] +
                        [{"label": g["label"], "value": g["label"]} for g in final_groups],
                value="All",
                clearable=False
            ),
            html.Br(),
            html.Label("Map Visualization Mode:"),
            dcc.RadioItems(
                id="map-visual-mode",
                options=[
                    {"label": "Scatter", "value": "scatter"},
                    {"label": "Heatmap", "value": "heatmap"}
                ],
                value="scatter"
            ),
            html.Br(),
            html.Label("Income Range:"),
            dcc.RangeSlider(
                id="income-range-slider",
                min=df_all["INCOME"].min() if "INCOME" in df_all.columns else 0,
                max=df_all["INCOME"].max() if "INCOME" in df_all.columns else 100000,
                step=1000,
                value=[
                    df_all["INCOME"].min() if "INCOME" in df_all.columns else 0,
                    df_all["INCOME"].max() if "INCOME" in df_all.columns else 100000
                ],
                tooltip={"always_visible": True}
            ),
            html.Br(),
            html.Label("Health Index Range:"),
            dcc.RangeSlider(
                id="healthindex-range-slider",
                min=df_all["Health_Index"].min(),
                max=df_all["Health_Index"].max(),
                value=[df_all["Health_Index"].min(), df_all["Health_Index"].max()],
                tooltip={"always_visible": True}
            ),
        ], width=3),
        dbc.Col(dcc.Graph(id="inequalities-map"), width=9),
    ])
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

df_raw_display = df_all.head(50).copy()
if "SEQUENCE" in df_raw_display.columns:
    df_raw_display["SEQUENCE"] = df_raw_display["SEQUENCE"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

raw_data_tab = dbc.Container([
    html.H2("Raw Data (First 50 Rows)", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    dash_table.DataTable(
        id="raw-data-table",
        data=df_raw_display.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df_raw_display.columns],
        page_size=10,
        filter_action="native",
        sort_action="native",
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': nhs_colors["primary"], 'color': nhs_colors["secondary"]},
    ),
    html.Br(),
    html.H4("Click on a Row to View Patient Details"),
    patient_modal,
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("VITAI Dashboard", style={"color": nhs_colors["primary"], "fontWeight": "bold"}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Overview", href="#")),
            dbc.NavItem(dbc.NavLink("Model Details", href="#")),
            dbc.NavItem(dbc.NavLink("XAI Insights", href="#")),
            dbc.NavItem(dbc.NavLink("Map & Inequalities", href="#")),
            dbc.NavItem(dbc.NavLink("Raw Data", href="#")),
        ], navbar=True)
    ]),
    color=nhs_colors["secondary"],
    light=True,
    sticky="top"
)

app.layout = html.Div([
    navbar,
    dcc.Tabs(id="main-tabs", value="overview-tab", children=[
        dcc.Tab(label="Overview",           value="overview-tab"),
        dcc.Tab(label="Model Details",      value="model-details-tab"),
        dcc.Tab(label="XAI Insights",       value="xai-tab"),
        dcc.Tab(label="Map & Inequalities", value="map-tab"),
        dcc.Tab(label="Raw Data",           value="raw-data-tab")
    ]),
    html.Div(id="tabs-content")
], style={"backgroundColor": nhs_colors["background"]})

# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    Output("tabs-content", "children"),
    Input("main-tabs", "value")
)
def render_tabs(tab):
    if tab == "overview-tab":
        return overview_tab
    elif tab == "model-details-tab":
        return model_details_tab
    elif tab == "xai-tab":
        return xai_tab
    elif tab == "map-tab":
        return map_tab
    elif tab == "raw-data-tab":
        return raw_data_tab
    return html.Div("Tab not found")

@app.callback(
    Output("model-details-content", "children"),
    Input("model-group-dropdown", "value")
)
def update_model_details(selected_group):
    if not selected_group:
        return html.Div("No model group selected.")

    # Re-load model data in case it changed
    # (Typically it won't, but safe to re-check)
    all_models = load_final_model_outputs()
    mdata      = all_models.get(selected_group, {})
    metrics    = mdata.get("metrics", {})
    df_model   = mdata.get("df", pd.DataFrame())
    tsne_src   = mdata.get("tsne_img")
    umap_src   = mdata.get("umap_img")

    if "PredictedHI_final" in df_model.columns and "Health_Index" in df_model.columns:
        scatter_fig = px.scatter(
            df_model,
            x="Health_Index",
            y="PredictedHI_final",
            color="Cluster_final",
            title=f"Predicted vs Actual Health Index ({selected_group})",
            template="plotly_white"
        )
    else:
        scatter_fig = go.Figure().update_layout(
            title="Data Not Available",
            template="plotly_white"
        )

    metrics_div = html.Div([
        html.P(f"Test MSE: {metrics.get('test_mse', 'N/A')}"),
        html.P(f"Test R2: {metrics.get('test_r2', 'N/A')}"),
        html.P(f"Silhouette: {metrics.get('Silhouette', 'N/A')}"),
        html.P(f"Calinski-Harabasz: {metrics.get('Calinski_Harabasz', 'N/A')}"),
        html.P(f"Davies-Bouldin: {metrics.get('Davies_Bouldin', 'N/A')}"),
    ], style={
        "padding": "10px",
        "backgroundColor": nhs_colors["secondary"],
        "borderRadius": "5px",
        "marginBottom": "20px"
    })

    tsne_img = html.Img(src=tsne_src, style={"width": "50%", "height": "auto"}) if tsne_src else html.Div("t-SNE plot not available")#changed height to 50%, see how this affect db
    umap_img = html.Img(src=umap_src, style={"width": "50%", "height": "auto"}) if umap_src else html.Div("UMAP plot not available")

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=scatter_fig), width=6),
            dbc.Col(metrics_div, width=6),
        ]),
        html.Hr(),
        html.H4("t-SNE Visualization"),
        tsne_img,
        html.Br(),
        html.Br(),
        html.H4("UMAP Visualization"),
        umap_img
    ])

@app.callback(
    Output("xai-content-area", "children"),
    Input("xai-model-group-dropdown", "value")
)
def update_xai_insights(selected_group):
    if not selected_group:
        return html.Div("No group selected for XAI.")
    
    found = [fg for fg in final_groups if fg["label"] == selected_group]
    if not found:
        return html.Div(f"Cannot find model group: {selected_group}")
    model_dir_name = found[0]["model"]
    model_xai_dir  = os.path.join(EXPLAIN_XAI_DIR, model_dir_name)

    shap_path         = os.path.join(model_xai_dir, f"{model_dir_name}_shap_values.npy")
    ig_path           = os.path.join(model_xai_dir, f"{model_dir_name}_ig_values.npy")
    anchors_path      = os.path.join(model_xai_dir, f"{model_dir_name}_anchors_local.csv")
    deeplift_path     = os.path.join(model_xai_dir, f"{model_dir_name}_deeplift_values.npy")
    cluster_lime_path = os.path.join(model_xai_dir, f"{model_dir_name}_cluster_lime_explanations.csv")
    
    content = []

    # SHAP
    if os.path.exists(shap_path):
        shap_vals = np.load(shap_path)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        features = [f"Feat_{i}" for i in range(len(mean_abs))]
        df_shap = pd.DataFrame({"Feature": features, "Importance": mean_abs}).sort_values("Importance", ascending=False)
        shap_fig = px.bar(df_shap, x="Importance", y="Feature", orientation="h",
                          title="Global SHAP Summary", template="plotly_white")
        content.append(html.Div([
            html.H4("SHAP Values", style={"color": nhs_colors["primary"]}),
            dcc.Graph(figure=shap_fig)
        ]))
    else:
        content.append(html.Div("[SHAP] No shap_values.npy found"))

    # IG
    if os.path.exists(ig_path):
        ig_vals = np.load(ig_path)
        mean_abs_ig = np.mean(np.abs(ig_vals), axis=0)
        feats_ig = [f"Feat_{i}" for i in range(len(mean_abs_ig))]
        df_ig = pd.DataFrame({"Feature": feats_ig, "IG_Importance": mean_abs_ig})
        df_ig.sort_values("IG_Importance", ascending=False, inplace=True)
        ig_fig = px.bar(df_ig, x="IG_Importance", y="Feature", orientation="h",
                        title="Integrated Gradients Summary", template="plotly_white")
        content.append(html.Div([
            html.H4("Integrated Gradients", style={"color": nhs_colors["primary"]}),
            dcc.Graph(figure=ig_fig)
        ]))
    else:
        content.append(html.Div("[IG] No ig_values.npy found."))

    # Anchors
    if os.path.exists(anchors_path):
        df_anchors = pd.read_csv(anchors_path)
        if "Anchors" in df_anchors.columns:
            df_anchors["Formatted Explanation"] = df_anchors["Anchors"].apply(format_explanation)
        anchors_table = dash_table.DataTable(
            data=df_anchors.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_anchors.columns],
            page_size=10,
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': nhs_colors["primary"], 'color': nhs_colors["secondary"]}
        )
        content.append(html.Div([
            html.H4("Anchors (Critical Cases)", style={"color": nhs_colors["primary"]}),
            anchors_table
        ]))
    else:
        content.append(html.Div("[Anchors] No anchors_local.csv found."))

    # DeepLIFT
    if os.path.exists(deeplift_path):
        dl_vals = np.load(deeplift_path)
        content.append(html.Div([
            html.P(f"[DeepLIFT] Found {dl_vals.shape[0]} outlier attributions.")
        ]))
    else:
        content.append(html.Div("[DeepLIFT] Not found."))

    # Cluster-based LIME
    if os.path.exists(cluster_lime_path):
        df_lime = pd.read_csv(cluster_lime_path)
        if "LIME_Explanation" in df_lime.columns:
            df_lime["Formatted Explanation"] = df_lime["LIME_Explanation"].apply(format_explanation)
        lime_table = dash_table.DataTable(
            data=df_lime.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_lime.columns],
            page_size=5,
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': nhs_colors["primary"], 'color': nhs_colors["secondary"]}
        )
        content.append(html.Div([
            html.H4("Cluster-based LIME Explanations", style={"color": nhs_colors["primary"]}),
            lime_table
        ]))
    else:
        content.append(html.Div("[LIME] No cluster_lime_explanations.csv found."))

    return html.Div(content)

@app.callback(
    Output("inequalities-map", "figure"),
    [
        Input("map-model-dropdown", "value"),
        Input("map-visual-mode", "value"),
        Input("income-range-slider", "value"),
        Input("healthindex-range-slider", "value")
    ]
)
def update_inequalities_map(selected_model_group, vis_mode, income_range, healthindex_range):
    dff = df_all.copy()
    if selected_model_group != "All" and "Group" in dff.columns:
        dff = dff[dff["Group"] == selected_model_group]

    if "INCOME" in dff.columns:
        dff = dff[(dff["INCOME"] >= income_range[0]) & (dff["INCOME"] <= income_range[1])]

    dff = dff[(dff["Health_Index"] >= healthindex_range[0]) &
              (dff["Health_Index"] <= healthindex_range[1])]

    if "LAT" not in dff.columns or "LON" not in dff.columns:
        return go.Figure().update_layout(title="No LAT/LON data available")

    if vis_mode == "scatter":
        fig = px.scatter_mapbox(
            dff,
            lat="LAT",
            lon="LON",
            hover_name="Id",
            hover_data=["AGE", "INCOME", "Health_Index", "CharlsonIndex", "ElixhauserIndex", "Group"]
                      if "Group" in dff.columns else ["AGE", "INCOME", "Health_Index"],
            color="INCOME" if "INCOME" in dff.columns else None,
            size="Health_Index" if "Health_Index" in dff.columns else None,
            color_continuous_scale="Turbo",
            zoom=5,
            height=600,
            title="Patient Distribution & Socio-Economic Filter (Scatter)"
        )
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0, "t":50, "l":0, "b":0})
    else:
        fig = px.density_mapbox(
            dff, lat="LAT", lon="LON",
            radius=10,
            center={"lat": 42.3, "lon": -71.0},
            zoom=6,
            height=600,
            title="Patient Distribution & Socio-Economic Filter (Heatmap)"
        )
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0, "t":50, "l":0, "b":0})

    return fig

@app.callback(
    [
        Output("patient-modal", "is_open"),
        Output("patient-detail-body", "children")
    ],
    [
        Input("raw-data-table", "active_cell"),
        Input("close-modal", "n_clicks")
    ],
    [
        State("patient-modal", "is_open"),
        State("raw-data-table", "data")
    ]
)
def toggle_patient_modal(active_cell, close_click, is_open, table_data):
    ctx = callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "raw-data-table" and active_cell:
            row_data = table_data[active_cell["row"]]
            patient_id = row_data.get("Id")

            patient_records = df_all[df_all["Id"] == patient_id]
            if patient_records.empty:
                return True, [html.P("Patient not found.")]

            patient = patient_records.iloc[0]
            details = [
                html.P(f"ID: {patient.get('Id', 'N/A')}"),
                html.P(f"Age: {patient.get('AGE', 'N/A')}"),
                html.P(f"Income: £{patient.get('INCOME', 'N/A')}"),
                html.P(f"Health Index: {patient.get('Health_Index', 'N/A')}"),
                html.P(f"Charlson Index: {patient.get('CharlsonIndex', 'N/A')}"),
                html.P(f"Elixhauser Index: {patient.get('ElixhauserIndex', 'N/A')}"),
                html.P(f"Group: {patient.get('Group', 'N/A')}"),
                html.P(f"PredictedHI_final: {patient.get('PredictedHI_final', 'N/A')}"),
                html.P(f"Cluster_final: {patient.get('Cluster_final', 'N/A')}")
            ]
            return True, details
        elif prop_id == "close-modal":
            return False, ""
    return is_open, ""

if __name__ == "__main__":
    app.run_server(debug=True)

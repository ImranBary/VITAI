import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import json
import glob

# --------------------------------------------------
# Determine the base directory (project root)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
PICKLE_PATH = os.path.join(DATA_DIR, "patient_data_with_all_indices.pkl")
CSV_PATH = os.path.join(DATA_DIR, "patients.csv")
FINALS_DIR = os.path.join(DATA_DIR, "finals")
EXPLAIN_XAI_DIR = os.path.join(DATA_DIR, "explain_xai")

# --------------------------------------------------
# Define the three final model groups and their labels
# (You can add 'All' to combine them if you want.)
# --------------------------------------------------
final_groups = [
    {"model": "combined_diabetes_tabnet", "label": "Diabetes"},
    {"model": "combined_all_ckd_tabnet", "label": "CKD"},
    {"model": "combined_none_tabnet", "label": "None"}
]

# --------------------------------------------------
# Data Loading & Preprocessing (base data)
# --------------------------------------------------
if os.path.exists(PICKLE_PATH):
    df_all = pd.read_pickle(PICKLE_PATH)
else:
    # Fallback for demonstration if user doesn't have the pickle
    df_all = pd.read_csv(CSV_PATH)
    df_all["BIRTHDATE"] = pd.to_datetime(df_all["BIRTHDATE"], errors="coerce")
    df_all["AGE"] = ((pd.Timestamp("today") - df_all["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    np.random.seed(42)
    # Synthetic values for Health_Index, CharlsonIndex, etc.
    df_all["Health_Index"] = np.random.uniform(1, 10, size=len(df_all)).round(2)
    df_all["CharlsonIndex"] = np.random.uniform(0, 5, size=len(df_all)).round(2)
    df_all["ElixhauserIndex"] = np.random.uniform(0, 15, size=len(df_all)).round(2)
    df_all["Cluster"] = np.random.choice([0, 1, 2], len(df_all))
    df_all["Predicted_Health_Index"] = (
        df_all["Health_Index"] + np.random.normal(0, 0.5, len(df_all))
    ).round(2)
    df_all["Actual"] = df_all["Health_Index"]

# Merge location columns from CSV if missing
if "ZIP" not in df_all.columns:
    df_csv = pd.read_csv(CSV_PATH, usecols=["Id", "BIRTHDATE", "ZIP", "LAT", "LON", "INCOME"])
    df_csv["BIRTHDATE"] = pd.to_datetime(df_csv["BIRTHDATE"], errors="coerce")
    df_csv["AGE"] = ((pd.Timestamp("today") - df_csv["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    df_all = pd.merge(df_all, df_csv, on="Id", how="left", suffixes=("", "_csv"))

# --------------------------------------------------
# Load final model outputs for each group and combine
# --------------------------------------------------
list_models = []
for grp in final_groups:
    model_dir = os.path.join(FINALS_DIR, grp["model"])
    predictions_csv = os.path.join(model_dir, f"{grp['model']}_predictions.csv")
    clusters_csv = os.path.join(model_dir, f"{grp['model']}_clusters.csv")
    if os.path.exists(predictions_csv) and os.path.exists(clusters_csv):
        df_preds = pd.read_csv(predictions_csv)    # Id, Predicted_Health_Index
        df_clusters = pd.read_csv(clusters_csv)    # Id, Predicted_Health_Index, Cluster
        # Merge predictions and clusters
        df_model = pd.merge(df_preds, df_clusters, on="Id", how="outer", suffixes=("", "_cluster"))
        df_model.rename(columns={
            "Predicted_Health_Index": "PredictedHI_final",
            "Cluster": "Cluster_final"
        }, inplace=True)
        df_model["Group"] = grp["label"]
        list_models.append(df_model)
    else:
        print(f"Outputs not found for model {grp['model']}")

if list_models:
    df_models = pd.concat(list_models, ignore_index=True)
else:
    df_models = pd.DataFrame()

# Merge final model outputs with base data by Id
df_all = pd.merge(df_all, df_models, on="Id", how="left")

# --------------------------------------------------
# For demonstration, store metrics for each final model in a dict
# The script attempts to load them from <model>_metrics.json
# --------------------------------------------------
model_metrics_map = {}
for grp in final_groups:
    mpath = os.path.join(FINALS_DIR, grp["model"], f"{grp['model']}_metrics.json")
    if os.path.exists(mpath):
        with open(mpath, "r") as f:
            mm = json.load(f)
        model_metrics_map[grp["label"]] = mm
    else:
        model_metrics_map[grp["label"]] = {
            "test_mse": "N/A",
            "test_r2": "N/A",
            "Silhouette": "N/A",
            "Calinski_Harabasz": "N/A",
            "Davies_Bouldin": "N/A"
        }

# --------------------------------------------------
# Pre-compute a few global KPI values (all patients)
# --------------------------------------------------
TOTAL_PATIENTS = len(df_all)
AVG_AGE = round(df_all["AGE"].mean(), 1) if "AGE" in df_all.columns else "N/A"
AVG_INCOME = round(df_all["INCOME"].mean(), 0) if "INCOME" in df_all.columns else "N/A"
AVG_HEALTH_INDEX = round(df_all["Health_Index"].mean(), 2) if "Health_Index" in df_all.columns else "N/A"
AVG_CHARLSON = round(df_all["CharlsonIndex"].mean(), 2) if "CharlsonIndex" in df_all.columns else "N/A"
AVG_ELIXHAUSER = round(df_all["ElixhauserIndex"].mean(), 2) if "ElixhauserIndex" in df_all.columns else "N/A"

# --------------------------------------------------
# NHS-Inspired Color Scheme & Theme
# --------------------------------------------------
nhs_colors = {
    "background": "#F7F7F7",
    "text": "#333333",
    "primary": "#005EB8",  # NHS Blue
    "secondary": "#FFFFFF",
    "accent": "#00843D"
}

external_stylesheets = [dbc.themes.FLATLY]

# --------------------------------------------------
# KPI Card Component
# --------------------------------------------------
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

# --------------------------------------------------
# Patient Details Modal (for per-patient info)
# --------------------------------------------------
patient_modal = html.Div(
    [
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
    ]
)

# --------------------------------------------------
# Initialize Dash App
# --------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

# --------------------------------------------------
# Navbar
# --------------------------------------------------
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("VITAI Dashboard", style={"color": nhs_colors["primary"], "fontWeight": "bold"}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Overview", href="#")),
            dbc.NavItem(dbc.NavLink("Model & Clustering", href="#")),
            dbc.NavItem(dbc.NavLink("XAI Insights", href="#")),
            dbc.NavItem(dbc.NavLink("Map & Inequalities", href="#")),
            dbc.NavItem(dbc.NavLink("Raw Data", href="#"))
        ], navbar=True)
    ]),
    color=nhs_colors["secondary"],
    light=True,
    sticky="top"
)

# --------------------------------------------------
# Overview Tab (Global Stats & KPI)
# --------------------------------------------------
overview_tab = dbc.Container([
    html.H2("Overview & KPIs", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    dbc.Row([
        dbc.Col(kpi_card("Total Patients", TOTAL_PATIENTS), width=2),
        dbc.Col(kpi_card("Avg Age", AVG_AGE), width=2),
        dbc.Col(kpi_card("Avg Income", f"£{AVG_INCOME}"), width=2),
        dbc.Col(kpi_card("Avg Health Index", AVG_HEALTH_INDEX), width=2),
        dbc.Col(kpi_card("Avg Charlson", AVG_CHARLSON), width=2),
        dbc.Col(kpi_card("Avg Elixhauser", AVG_ELIXHAUSER), width=2)
    ], className="mb-4"),
    # Simple distribution of actual Health_Index
    dbc.Row([
        dbc.Col(dcc.Graph(
            id="hist-healthindex",
            figure=px.histogram(
                df_all, x="Health_Index", nbins=30,
                title="Distribution of Actual (Composite) Health Index",
                template="plotly_white"
            )
        ), width=6),
        dbc.Col(dcc.Graph(
            id="hist-charlson",
            figure=px.histogram(
                df_all, x="CharlsonIndex", nbins=30,
                title="Distribution of Charlson Index",
                template="plotly_white"
            )
        ), width=6)
    ])
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

# --------------------------------------------------
# Model & Clustering Tab
# --------------------------------------------------
model_tab = dbc.Container([
    html.H2("Model & Clustering", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    
    # Model Selection
    html.Div([
        html.Label("Select Final Model Group:"),
        dcc.Dropdown(
            id="model-group-dropdown",
            options=[{"label": grp["label"], "value": grp["label"]} for grp in final_groups],
            value="None",
            clearable=False,
            style={"width": "300px", "marginBottom": "20px"}
        )
    ]),
    
    html.Div(id="model-cluster-content")
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

# --------------------------------------------------
# XAI Insights Tab
# --------------------------------------------------
xai_tab = dbc.Container([
    html.H2("XAI Insights", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    html.Div([
        html.Label("Select Final Model Group (for XAI):"),
        dcc.Dropdown(
            id="xai-model-group-dropdown",
            options=[{"label": grp["label"], "value": grp["label"]} for grp in final_groups],
            value="None",
            clearable=False,
            style={"width": "300px", "marginBottom": "20px"}
        )
    ]),
    html.Div(id="xai-content-area")
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

# --------------------------------------------------
# Map & Inequalities Tab
# --------------------------------------------------
map_tab = dbc.Container([
    html.H2("Patient Map & Inequalities", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Model Group:"),
            dcc.Dropdown(
                id="map-model-dropdown",
                options=[{"label": "All", "value": "All"}] + 
                        [{"label": grp["label"], "value": grp["label"]} for grp in final_groups],
                value="All",
                clearable=False
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
                marks=None,  # or create custom marks
                tooltip={"always_visible": True}
            ),
            html.Br(),
            html.Label("Health Index Range:"),
            dcc.RangeSlider(
                id="healthindex-range-slider",
                min=df_all["Health_Index"].min(),
                max=df_all["Health_Index"].max(),
                value=[df_all["Health_Index"].min(), df_all["Health_Index"].max()],
                marks=None,
                tooltip={"always_visible": True}
            )
        ], width=3),
        dbc.Col(dcc.Graph(id="inequalities-map"), width=9)
    ])
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

# --------------------------------------------------
# Raw Data Tab
# --------------------------------------------------
raw_data_tab = dbc.Container([
    html.H2("Raw Data", style={"color": nhs_colors["primary"]}),
    html.Hr(),
    html.H4("First 50 Rows"),
    dash_table.DataTable(
        id="raw-data-table",
        data=df_all.head(50).to_dict("records"),
        columns=[{"name": i, "id": i} for i in df_all.columns],
        page_size=10,
        filter_action="native",
        sort_action="native",
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': nhs_colors["primary"], 'color': nhs_colors["secondary"]}
    ),
    html.Br(),
    html.H4("Click on a Row to View Patient Details"),
    
    patient_modal
], fluid=True, style={"backgroundColor": nhs_colors["background"], "padding": "20px"})

# --------------------------------------------------
# Main Layout with Tabs
# --------------------------------------------------
app.layout = html.Div([
    navbar,
    dcc.Tabs(id="main-tabs", value="overview-tab", children=[
        dcc.Tab(label="Overview", value="overview-tab"),
        dcc.Tab(label="Model & Clustering", value="model-tab"),
        dcc.Tab(label="XAI Insights", value="xai-tab"),
        dcc.Tab(label="Map & Inequalities", value="map-tab"),
        dcc.Tab(label="Raw Data", value="raw-data-tab")
    ]),
    html.Div(id="tabs-content"),
], style={"backgroundColor": nhs_colors["background"]})

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

# 1) Tabs Callback
@app.callback(
    Output("tabs-content", "children"),
    Input("main-tabs", "value")
)
def render_tabs(tab):
    if tab == "overview-tab":
        return overview_tab
    elif tab == "model-tab":
        return model_tab
    elif tab == "xai-tab":
        return xai_tab
    elif tab == "map-tab":
        return map_tab
    elif tab == "raw-data-tab":
        return raw_data_tab
    return html.Div("Tab not found")

# 2) Model & Clustering Content
@app.callback(
    Output("model-cluster-content", "children"),
    Input("model-group-dropdown", "value")
)
def update_model_cluster_content(selected_group):
    """
    Dynamically show scatter of Predicted vs Actual for the chosen final model group,
    plus clustering metrics, etc.
    """
    # Filter df_all to that group
    if selected_group is None:
        return html.Div("No group selected.")
    
    # If "Group" is not in columns or "All" is chosen, use entire dataset
    if "Group" not in df_all.columns or selected_group not in df_all["Group"].unique():
        dff = df_all.copy()
    else:
        dff = df_all[df_all["Group"] == selected_group].copy()
    
    # Fetch model metrics
    mmetrics = model_metrics_map.get(selected_group, {})
    
    # Scatter Predicted vs Actual
    if "PredictedHI_final" in dff.columns and "Health_Index" in dff.columns:
        scatter_fig = px.scatter(
            dff, x="Health_Index", y="PredictedHI_final",
            color="Cluster_final",
            title=f"Predicted vs Actual Health Index ({selected_group})",
            template="plotly_white"
        )
    else:
        scatter_fig = go.Figure()
        scatter_fig.update_layout(title="Data Not Available", template="plotly_white")
    
    # t-SNE / UMAP placeholders if columns exist
    if ("tSNE_x" in dff.columns) and ("tSNE_y" in dff.columns):
        tsne_fig = px.scatter(
            dff, x="tSNE_x", y="tSNE_y", color="Cluster_final",
            title=f"{selected_group} t-SNE Visualization", template="plotly_white"
        )
    else:
        # fallback
        tsne_fig = px.scatter(
            x=np.random.normal(0,1,len(dff)), 
            y=np.random.normal(0,1,len(dff)), 
            color=dff.get("Cluster_final", None),
            title="t-SNE (placeholder)", template="plotly_white"
        )
    if ("UMAP_x" in dff.columns) and ("UMAP_y" in dff.columns):
        umap_fig = px.scatter(
            dff, x="UMAP_x", y="UMAP_y", color="Cluster_final",
            title=f"{selected_group} UMAP Visualization", template="plotly_white"
        )
    else:
        umap_fig = px.scatter(
            x=np.random.normal(0,1,len(dff)), 
            y=np.random.normal(0,1,len(dff)), 
            color=dff.get("Cluster_final", None),
            title="UMAP (placeholder)", template="plotly_white"
        )
    
    # Display cluster metrics
    cluster_metrics_div = html.Div([
        html.P(f"Test MSE: {mmetrics.get('test_mse', 'N/A')}"),
        html.P(f"Test R2: {mmetrics.get('test_r2', 'N/A')}"),
        html.P(f"Silhouette: {mmetrics.get('Silhouette', 'N/A')}"),
        html.P(f"Calinski-Harabasz: {mmetrics.get('Calinski_Harabasz', 'N/A')}"),
        html.P(f"Davies-Bouldin: {mmetrics.get('Davies_Bouldin', 'N/A')}")
    ], style={"padding": "10px", "backgroundColor": nhs_colors["secondary"], "borderRadius": "5px"})
    
    layout = html.Div([
        dcc.Graph(figure=scatter_fig),
        cluster_metrics_div,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=tsne_fig), width=6),
            dbc.Col(dcc.Graph(figure=umap_fig), width=6)
        ])
    ])
    return layout

# 3) XAI Insights Content
@app.callback(
    Output("xai-content-area", "children"),
    Input("xai-model-group-dropdown", "value")
)
def update_xai_insights(selected_group):
    """
    Display SHAP, IG, local anchors, LIME cluster explanations, etc. 
    by reading from Data/explain_xai/<model_id>/ 
    """
    if not selected_group:
        return html.Div("No group selected for XAI.")
    
    # Map the label -> actual model directory name
    # e.g., "Diabetes" -> "combined_diabetes_tabnet"
    found = [fg for fg in final_groups if fg["label"] == selected_group]
    if not found:
        return html.Div(f"Cannot find model group: {selected_group}")
    model_dir_name = found[0]["model"]
    model_xai_dir = os.path.join(EXPLAIN_XAI_DIR, model_dir_name)
    
    # We'll try to read shap, ig, etc.
    shap_path = os.path.join(model_xai_dir, f"{model_dir_name}_shap_values.npy")
    ig_path = os.path.join(model_xai_dir, f"{model_dir_name}_ig_values.npy")
    anchors_path = os.path.join(model_xai_dir, f"{model_dir_name}_anchors_local.csv")
    deeplift_path = os.path.join(model_xai_dir, f"{model_dir_name}_deeplift_values.npy")
    cluster_lime_path = os.path.join(model_xai_dir, f"{model_dir_name}_cluster_lime_explanations.csv")
    
    content = []
    
    # (A) Global SHAP
    if os.path.exists(shap_path):
        shap_vals = np.load(shap_path)  # shape: (n_samples, n_features)
        # Summarize (mean(|shap|) across features)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        # We'll guess feature cols from the original df or just label them "Feature 1" etc. for now
        # In reality, you'd pass them properly from the feature pipeline
        features = [f"Feat_{i}" for i in range(len(mean_abs))]
        df_shap = pd.DataFrame({
            "Feature": features,
            "Importance": mean_abs
        }).sort_values("Importance", ascending=False)
        shap_fig = px.bar(
            df_shap, x="Importance", y="Feature", orientation="h",
            title="Global SHAP Summary", template="plotly_white"
        )
        content.append(html.Div([
            html.H4("SHAP Values", style={"color": nhs_colors["primary"]}),
            dcc.Graph(figure=shap_fig)
        ]))
    else:
        content.append(html.Div("[SHAP] No shap_values.npy found"))
    
    # (B) Integrated Gradients
    if os.path.exists(ig_path):
        ig_vals = np.load(ig_path)
        mean_abs_ig = np.mean(np.abs(ig_vals), axis=0)
        feats_ig = [f"Feat_{i}" for i in range(len(mean_abs_ig))]
        df_ig = pd.DataFrame({"Feature": feats_ig, "IG_Importance": mean_abs_ig})
        df_ig.sort_values("IG_Importance", ascending=False, inplace=True)
        ig_fig = px.bar(
            df_ig, x="IG_Importance", y="Feature", orientation="h",
            title="Integrated Gradients Summary", template="plotly_white"
        )
        content.append(html.Div([
            html.H4("Integrated Gradients", style={"color": nhs_colors["primary"]}),
            dcc.Graph(figure=ig_fig)
        ]))
    else:
        content.append(html.Div("[IG] No ig_values.npy found."))
    
    # (C) Anchors (Local Explanations for critical cases)
    if os.path.exists(anchors_path):
        df_anchors = pd.read_csv(anchors_path)
        # columns: RowIndex, Precision, Coverage, Anchors
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
    
    # (D) DeepLIFT for outliers
    if os.path.exists(deeplift_path):
        dl_vals = np.load(deeplift_path)
        # We won't do a fancy chart, but you could do similarly to SHAP
        content.append(html.Div([
            html.P(f"[DeepLIFT] Found {dl_vals.shape[0]} outlier attributions.")
        ]))
    else:
        content.append(html.Div("[DeepLIFT] Not found."))
    
    # (E) Cluster-based LIME
    if os.path.exists(cluster_lime_path):
        df_lime = pd.read_csv(cluster_lime_path)
        # columns: [Cluster, RepresentativeIndex, LIME_Explanation]
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

# 4) Map & Inequalities
@app.callback(
    Output("inequalities-map", "figure"),
    [
        Input("map-model-dropdown", "value"),
        Input("income-range-slider", "value"),
        Input("healthindex-range-slider", "value")
    ]
)
def update_inequalities_map(selected_model_group, income_range, healthindex_range):
    dff = df_all.copy()
    # Filter by group if not 'All'
    if selected_model_group != "All" and "Group" in dff.columns:
        dff = dff[dff["Group"] == selected_model_group]
    # Filter by income
    if "INCOME" in dff.columns:
        dff = dff[
            (dff["INCOME"] >= income_range[0]) &
            (dff["INCOME"] <= income_range[1])
        ]
    # Filter by actual Health_Index
    dff = dff[
        (dff["Health_Index"] >= healthindex_range[0]) &
        (dff["Health_Index"] <= healthindex_range[1])
    ]
    if "LAT" not in dff.columns or "LON" not in dff.columns:
        fig = go.Figure()
        fig.update_layout(title="No LAT/LON data available")
        return fig
    # color by INCOME or Health_Index
    fig = px.scatter_mapbox(
        dff, lat="LAT", lon="LON",
        hover_name="Id",
        hover_data=["AGE", "INCOME", "Health_Index", "CharlsonIndex", "ElixhauserIndex", "Group"],
        color="INCOME" if "INCOME" in dff.columns else None,
        size="Health_Index",
        color_continuous_scale="Turbo",
        zoom=5,
        height=600,
        title="Patient Distribution & Socio-Economic Filter"
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0, "t":50, "l":0, "b":0})
    return fig

# 5) Patient Modal: open/close on raw-data table row click
@app.callback(
    [Output("patient-modal", "is_open"),
     Output("patient-detail-body", "children")],
    [Input("raw-data-table", "active_cell"),
     Input("close-modal", "n_clicks")],
    [State("patient-modal", "is_open"),
     State("raw-data-table", "data")]
)
def toggle_patient_modal(active_cell, close_click, is_open, table_data):
    ctx = callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "raw-data-table" and active_cell:
            row_data = table_data[active_cell["row"]]
            patient_id = row_data["Id"]
            patient = df_all[df_all["Id"] == patient_id].iloc[0]
            details = [
                html.P(f"ID: {patient['Id']}"),
                html.P(f"Age: {patient['AGE']}"),
                html.P(f"Income: £{patient.get('INCOME', 'N/A')}"),
                html.P(f"Health Index: {patient.get('Health_Index', 'N/A')}"),
                html.P(f"Charlson Index: {patient.get('CharlsonIndex', 'N/A')}"),
                html.P(f"Elixhauser Index: {patient.get('ElixhauserIndex', 'N/A')}"),
                html.P(f"Group (final model): {patient.get('Group', 'N/A')}"),
                html.P(f"PredictedHI_final: {patient.get('PredictedHI_final', 'N/A')}"),
                html.P(f"Cluster_final: {patient.get('Cluster_final', 'N/A')}")
            ]
            return True, details
        elif prop_id == "close-modal":
            return False, ""
    return is_open, ""

# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)

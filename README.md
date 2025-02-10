# VITAI: An Interpretable Health Severity Index Using Advanced Deep Learning and Explainable AI on Structured EHR Data

**Author:** Imran Feisal  
**Student ID:** w1843601  
**Supervisor:** Dr Alexandra Psarrou  
**Date:** 14/11/2024 (Project Proposal) / 10/02/2025 (Current Version)  

## Introduction
Electronic Health Records (EHRs) contain high-dimensional, structured patient data that can be leveraged to assess patient health status. Traditional comorbidity indices such as the Charlson and Elixhauser indices, while useful, are often static and lack the capacity for deep feature analysis. VITAI is an innovative project that combines advanced deep learning techniques (Variational Autoencoders and TabNet) with Explainable AI (XAI) methods (SHAP, LIME, Integrated Gradients, DeepLIFT, and Anchors) to develop a real-time, interpretable health severity index ranging from 1 (healthy) to 10 (most severe). This index is designed to assist healthcare professionals with transparent, data-driven clinical decision-making.

## Project Overview
VITAI aims to:
- Preprocess and integrate structured EHR data (synthetic data generated via Synthea) by handling missing values, outliers, and encoding categorical variables.
- Compute standard clinical indices such as the Charlson and Elixhauser Comorbidity Indices.
- Develop advanced models (primarily TabNet with a segmented approach for diabetes, CKD, and a general catch-all model) to predict a composite health severity index.
- Integrate Explainable AI methods to provide both global and local explanations using SHAP, Integrated Gradients, Anchors, and cluster-based LIME.
- Validate and visualize the results through clustering, statistical tests, and dashboard visualizations.

## Directory Structure
```plaintext
VITAI/
├── Data/
├── Explain_Xai/
│   └── final_explain_xai_clustered_lime.py
├── Finals/
│   └── final_three_tabnet.py
├── Validations/
│   ├── combine_model_comparisons.py
│   └── validate_final_tabnet_models.py
├── vitai_scripts/
│   ├── cluster_utils.py
│   ├── data_prep.py
│   ├── feature_utils.py
│   ├── model_utils.py
│   ├── run_vitai_tests_main.py
│   └── subset_utils.py
├── charlson_comorbidity.py
├── data_exploration.py
├── data_preprocessing.py
├── elixhauser_comorbidity.py
├── health_index.py
├── tabnet_model.py
└── visualise_vitai_results.py
```

## Scripts Overview
### Root-Level Scripts
- **charlson_comorbidity.py** - Computes Charlson Comorbidity Index (CCI) using SNOMED codes.
- **data_exploration.py** - Provides functions for initial data exploration.
- **data_preprocessing.py** - Prepares synthetic EHR data for modeling.
- **elixhauser_comorbidity.py** - Computes Elixhauser Comorbidity Index (ECI).
- **health_index.py** - Computes a composite health index integrating multiple indicators.
- **tabnet_model.py** - Trains a TabNet model with hyperparameter tuning.
- **visualise_vitai_results.py** - Generates visualizations for model outputs.

### Explain_Xai Folder
- **final_explain_xai_clustered_lime.py** - Uses SHAP, Integrated Gradients, and a cluster-based LIME approach for explainability.

### Finals Folder
- **final_three_tabnet.py** - Trains and evaluates three TabNet models (Diabetes, CKD, Catch-All).

### Validations Folder
- **combine_model_comparisons.py** - Aggregates and compares performance metrics.
- **validate_final_tabnet_models.py** - Validates model outputs against clinical indices.

### vitai_scripts Folder
- **cluster_utils.py** - Functions for clustering and visualization.
- **data_prep.py** - Prepares final dataset.
- **feature_utils.py** - Extracts relevant features for modeling.
- **model_utils.py** - Runs models and gathers training metrics.
- **run_vitai_tests_main.py** - Executes full VITAI pipeline.
- **subset_utils.py** - Filters dataset into condition-specific subsets.

## Methodology
- **Data Preprocessing:** Missing value handling, outlier treatment, encoding, and sequence building.
- **Composite Health Index Calculation:** PCA and scaling techniques to compute final index.
- **Model Development:** Variational Autoencoders (VAEs) and TabNet models with hyperparameter tuning.
- **Explainable AI Integration:** SHAP, Integrated Gradients, Anchors, and LIME.
- **Validation & Visualization:** Clustering, statistical tests, and dimensionality reduction techniques.

## Tools and Technologies
- **Programming:** Python 3.x
- **Deep Learning:** PyTorch, TensorFlow/Keras
- **Explainability:** SHAP, LIME, Captum, Alibi
- **Data Processing & Visualization:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, UMAP, t-SNE
- **Hyperparameter Tuning:** Optuna
- **Development:** Jupyter Notebook, VS Code / PyCharm
- **Hardware:** NVIDIA CUDA-enabled GPU


## Acknowledgements and References
This project was developed as part of my placement year at the NHS with contributions from healthcare professionals, academic supervisors, and industry experts.

"""
explainability.py
Author: Imran Feisal
Date: 21/01/2025

Description:
A utility module providing comprehensive model explainability approaches
for your final TabNet model. Features include:
  - SHAP: Feature-attribution-based global & local explanations.
  - LIME: Local explanations for individual predictions.
  - TabNet Intrinsic Feature Masks (attention).

Typical Usage (example):

    from explainability import TabNetExplainability

    # Suppose your final model is saved here:
    model_path = "Data/finals/final_composite_none_tabnet_model.zip"

    # Create an instance
    xai = TabNetExplainability(
        model_path=model_path,
        column_names=["AGE","DECEASED","GENDER", ...], # your feature columns
        is_regression=True
    )

    xai.load_model()

    # Generate SHAP values:
    shap_values = xai.shap_explain(X_train, X_sample=X_test[:100], auto_sample=True, sample_size=2000)
    xai.plot_shap_summary(shap_values, max_display=15)

    # LIME for a single instance:
    lime_exp = xai.lime_explainer(X_train)
    instance_explanation = xai.lime_explain_instance(lime_exp, X_test[0], num_features=6)
    instance_explanation.show_in_notebook()

    # TabNet attention masks:
    masks = xai.get_feature_masks(X_test[:50])
    xai.plot_feature_mask_heatmap(masks, step_index=0)

"""

import os
import logging
import numpy as np
import pandas as pd

# SHAP
import shap

# LIME
import lime
import lime.lime_tabular

# PyTorch / TabNet
import torch
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TabNetExplainability:
    """
    A utility class for post-hoc and intrinsic explainability on TabNet models.

    Methods Provided:
      - load_model()
      - shap_explain(), plot_shap_summary(), plot_shap_waterfall()
      - lime_explainer(), lime_explain_instance()
      - get_feature_masks(), plot_feature_mask_heatmap()
    """

    def __init__(
        self,
        model_path: str,
        column_names: list,
        class_names: list = None,
        is_regression: bool = True,
        device_name: str = None,
    ):
        """
        Args:
            model_path (str): Path to the saved TabNet model, e.g. "Data/finals/final_model.zip".
            column_names (list): List of features in the order the model expects.
            class_names (list): If classification, provide class labels. If regression, can be None or ["HealthIndex"].
            is_regression (bool): True if it's a regression model (e.g. Health Index), else False for classification.
            device_name (str): "cuda" or "cpu". If None, uses cuda if available, otherwise cpu.
        """
        self.model_path = model_path
        self.column_names = column_names
        self.class_names = class_names if class_names is not None else []
        self.is_regression = is_regression
        self.device_name = device_name or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        logger.info(f"[INIT] TabNetExplainability created for model={model_path}")

    def load_model(self):
        """
        Loads the TabNet model from disk using TabNetRegressor or TabNetClassifier.
        """
        logger.info(f"[LOAD] Loading TabNet model from {self.model_path} ...")

        if self.is_regression:
            self.model = TabNetRegressor()
        else:
            self.model = TabNetClassifier()

        self.model.load_model(self.model_path)
        self.model.device_name = self.device_name
        logger.info(f"[LOAD] Model loaded. Using device={self.device_name}")

    # -----------------------------------------------------------------------
    # 1) SHAP EXPLANATIONS
    # -----------------------------------------------------------------------
    def shap_explain(
        self,
        X_train: np.ndarray,
        X_sample: np.ndarray = None,
        auto_sample: bool = False,
        sample_size: int = 2000
    ):
        """
        Computes SHAP values for a TabNet model using the shap.Explainer approach.

        Args:
            X_train (np.ndarray): Training data for background distribution (n_samples x n_features).
            X_sample (np.ndarray): Data to explain. If None, we use X_train for the explanation as well.
            auto_sample (bool): Whether to randomly sample X_train to reduce size.
            sample_size (int): Max number of samples if auto_sample=True.

        Returns:
            shap_values: A shap.Explanation object with the computed SHAP values.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call 'load_model()' first.")

        if X_sample is None:
            X_sample = X_train

        # Optionally reduce background if data is large
        if auto_sample and len(X_train) > sample_size:
            idx = np.random.choice(len(X_train), size=sample_size, replace=False)
            background = X_train[idx]
        else:
            background = X_train

        # TabNet's "predict" for regression returns shape [N], classification returns shape [N] or [N, #classes]
        def model_predict(data: np.ndarray):
            return self.model.predict(data)

        logger.info(f"[SHAP] Building explainer with background size={len(background)} ...")
        explainer = shap.Explainer(
            model_predict,
            background,
            feature_names=self.column_names
        )

        logger.info(f"[SHAP] Computing SHAP values for sample shape={X_sample.shape} ...")
        shap_values = explainer(X_sample)
        return shap_values

    def plot_shap_summary(self, shap_values, max_display=20):
        """
        Draws a SHAP summary beeswarm plot.

        Args:
            shap_values: shap.Explanation object from shap_explain().
            max_display (int): Max number of features to display.
        """
        shap.plots.beeswarm(shap_values, max_display=max_display)
        plt.title("SHAP Summary Plot")
        plt.show()

    def plot_shap_waterfall(self, shap_values, row_index=0):
        """
        Shows a waterfall plot for a single instance's SHAP values.

        Args:
            shap_values: The shap.Explanation object from shap_explain().
            row_index (int): Which row in shap_values to show.
        """
        shap.plots.waterfall(shap_values[row_index])
        plt.title(f"SHAP Waterfall - Sample {row_index}")
        plt.show()

    # -----------------------------------------------------------------------
    # 2) LIME EXPLANATIONS
    # -----------------------------------------------------------------------
    def lime_explainer(self, X_train: np.ndarray, discretize_continuous: bool = False):
        """
        Constructs a LIME TabularExplainer for local interpretability.

        Args:
            X_train (np.ndarray): Training data (n_samples x n_features).
            discretize_continuous (bool): If True, LIME bins continuous features.

        Returns:
            A lime.lime_tabular.LimeTabularExplainer object.
        """
        logger.info("[LIME] Creating LimeTabularExplainer ...")
        return lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.column_names,
            class_names=self.class_names if not self.is_regression else None,
            discretize_continuous=discretize_continuous,
            mode='regression' if self.is_regression else 'classification'
        )

    def lime_explain_instance(
        self,
        explainer: lime.lime_tabular.LimeTabularExplainer,
        instance: np.ndarray,
        label_index: int = 0,
        num_features: int = 5
    ):
        """
        Runs a LIME explanation on a single sample.

        Args:
            explainer: The LimeTabularExplainer from lime_explainer().
            instance (np.ndarray): 1D array with shape (n_features,).
            label_index (int): If classification, which class label to explain.
                               If regression, typically 0.
            num_features (int): Number of features to show in the explanation.

        Returns:
            A lime explanation object which you can show or print.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call 'load_model()' first.")

        def predict_fn(data: np.ndarray):
            return self.model.predict(data)

        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            labels=(label_index,),
            num_features=num_features
        )
        return explanation

    # -----------------------------------------------------------------------
    # 3) TABNET FEATURE MASK ATTENTION
    # -----------------------------------------------------------------------
    def get_feature_masks(self, X_batch: np.ndarray):
        """
        Retrieves TabNet's step-wise feature masks (attention) for X_batch.

        Args:
            X_batch (np.ndarray): shape [batch_size, n_features]

        Returns:
            A list of length n_steps, each an array [batch_size, n_features].
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call 'load_model()' first.")

        step_masks, _ = self.model.forward_masks(X_batch)
        feature_masks = []
        for mask_tensor in step_masks:
            feature_masks.append(mask_tensor.detach().cpu().numpy())

        logger.info(f"[ATTENTION] Extracted {len(feature_masks)} step masks. Each shape => (batch_size, n_features).")
        return feature_masks

    def plot_feature_mask_heatmap(self, feature_masks: list, step_index: int = 0):
        """
        Displays a heatmap of the TabNet feature mask at a given step.

        Args:
            feature_masks (list): Output from get_feature_masks(...).
            step_index (int): Which step to visualise (0-based).
        """
        if step_index >= len(feature_masks):
            logger.error("[ATTENTION] Invalid step_index=%d, max=%d", step_index, len(feature_masks) - 1)
            return

        mask = feature_masks[step_index]
        plt.figure(figsize=(10, 6))
        sns.heatmap(mask, cmap="viridis")
        plt.xlabel("Features")
        plt.ylabel("Samples")
        plt.title(f"TabNet Feature Mask - Step {step_index}")
        plt.show()

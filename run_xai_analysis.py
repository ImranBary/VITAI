#!/usr/bin/env python
"""
run_xai_analysis.py

Script to run explainability (XAI) analyses on the TabNet model predictions.
Uses SHAP, Integrated Gradients, LIME, and Anchors to explain predictions.

This script is called by GenerateAndPredict.exe when the --enable-xai flag is used.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import from existing XAI module
from Explain_Xai import final_explain_xai_clustered_lime as xai

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run XAI analysis on TabNet predictions")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU for XAI computations")
    args = parser.parse_args()
    
    logger.info("Starting XAI analysis...")
    
    # Create output directory for XAI results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xai_dir = os.path.join("Data", "explain_xai", timestamp)
    os.makedirs(xai_dir, exist_ok=True)
    
    # Configure XAI module to use this directory
    xai.EXPLAIN_DIR = xai_dir
    
    # Set the patient data pickle path
    pickle_path = os.path.join("Data", "patient_data_with_all_indices.pkl")
    if os.path.exists(pickle_path):
        xai.FULL_DATA_PKL = pickle_path
    else:
        logger.warning(f"Patient data pickle not found at {pickle_path}")
    
    # Force CPU if requested
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Forcing CPU usage for XAI computations")
    
    try:
        # Run XAI analysis
        xai.main()
        logger.info(f"XAI analysis completed - results saved to {xai_dir}")
        return 0
    except Exception as e:
        logger.error(f"XAI analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

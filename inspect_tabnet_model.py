#!/usr/bin/env python
"""
inspect_tabnet_model.py

Utility script to inspect TabNet model metadata and understand its expected inputs.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s][%(levelname)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    logger.error("PyTorch TabNet is not installed. Please install with: pip install pytorch-tabnet")
    sys.exit(1)

def inspect_model_file(model_path):
    """Inspect a TabNet model file and extract metadata about expected inputs."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
        
    logger.info(f"Inspecting model: {model_path}")
    
    # Load model
    try:
        model = TabNetRegressor()
        model.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Extract model architecture information
    input_dim = getattr(model.network, 'input_dim', 'Unknown')
    output_dim = getattr(model.network, 'output_dim', 'Unknown')
    
    # Get categorical indices
    cat_idxs = getattr(model, 'cat_idxs', [])
    cat_dims = getattr(model, 'cat_dims', [])
    
    logger.info(f"Model expects input dimension: {input_dim}")
    logger.info(f"Model output dimension: {output_dim}")
    logger.info(f"Categorical indices: {cat_idxs}")
    logger.info(f"Categorical dimensions: {cat_dims}")
    
    # Check for related metadata files
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).replace('.zip', '').replace('_model', '')
    
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Found metadata file: {metadata_path}")
            logger.info(f"Feature columns: {metadata.get('feature_columns', [])}")
            logger.info(f"Feature config: {metadata.get('feature_config', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to read metadata file: {e}")
    
    # Look for sample input data
    sample_path = os.path.join(model_dir, f"{model_name}_sample_input.csv")
    if os.path.exists(sample_path):
        logger.info(f"Found sample input file: {sample_path}")
        logger.info(f"Use this file to see the expected features.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Inspect TabNet model files")
    parser.add_argument("model_path", help="Path to the TabNet model file (.zip)")
    args = parser.parse_args()
    
    inspect_model_file(args.model_path)

if __name__ == "__main__":
    main()

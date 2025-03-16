#!/usr/bin/env python
"""
Simple script to verify that TabNet models can be loaded.
"""

import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def check_packages():
    """Check if required packages are installed."""
    try:
        import numpy
        import pandas
        import torch
        import sklearn
        from pytorch_tabnet.tab_model import TabNetRegressor
        import joblib
        
        logger.info("All required packages are available:")
        logger.info(f"Python: {sys.version.split()[0]}")
        logger.info(f"NumPy: {numpy.__version__}")
        logger.info(f"Pandas: {pandas.__version__}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"scikit-learn: {sklearn.__version__}")
        return True
    except ImportError as e:
        logger.error(f"Missing package: {str(e)}")
        return False

def check_models():
    """Try to load each TabNet model."""
    model_ids = [
        "combined_diabetes_tabnet",
        "combined_all_ckd_tabnet",
        "combined_none_tabnet"
    ]
    
    potential_dirs = [
        ".",
        "Data/finals",
        "Data/models",
    ]
    
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    models_ok = True
    
    for model_id in model_ids:
        model_found = False
        for base_dir in potential_dirs:
            # Check for model directory
            model_dir = os.path.join(base_dir, model_id)
            model_path = os.path.join(model_dir, f"{model_id}_model")
            
            if os.path.exists(f"{model_path}.zip") or os.path.exists(model_path):
                model_found = True
                logger.info(f"Found model: {model_path}")
                
                try:
                    # Try to load the model
                    regressor = TabNetRegressor(device_name=device)
                    regressor.load_model(model_path)
                    logger.info(f"Successfully loaded model: {model_id}")
                    
                    # Check if scaler exists
                    scaler_path = os.path.join(model_dir, f"{model_id}_scaler.joblib")
                    if os.path.exists(scaler_path):
                        import joblib
                        scaler = joblib.load(scaler_path)
                        logger.info(f"Successfully loaded scaler for model: {model_id}")
                    else:
                        logger.warning(f"No scaler found for model: {model_id}")
                    
                    # Check if metadata exists
                    metadata_path = os.path.join(model_dir, f"{model_id}_metadata.json")
                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        logger.info(f"Successfully loaded metadata for model: {model_id}")
                    else:
                        logger.warning(f"No metadata found for model: {model_id}")
                    
                    break
                except Exception as e:
                    logger.error(f"Error loading model {model_id}: {str(e)}")
                    models_ok = False
        
        if not model_found:
            logger.error(f"Model not found: {model_id}")
            models_ok = False
    
    return models_ok

if __name__ == "__main__":
    try:
        packages_ok = check_packages()
        if not packages_ok:
            sys.exit(1)
        
        models_ok = check_models()
        if not models_ok:
            logger.error("Some models could not be loaded.")
            sys.exit(2)
        
        logger.info("All models verified successfully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(3)

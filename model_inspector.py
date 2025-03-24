import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib

def inspect_model(model_path, force_cpu=True):
    """
    Inspect a TabNet model to determine embedding dimensions and expected values
    """
    print(f"Loading model from {model_path}")
    
    device = "cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    model = TabNetRegressor(device_name=device)
    
    if os.path.isfile(model_path):
        model_zip = model_path
    elif os.path.isfile(model_path + ".zip"):
        model_zip = model_path + ".zip"
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
    
    print(f"Loading model from {model_zip}")
    model.load_model(model_zip)
    
    # Get embedding info
    network = model.network
    embedder = network.embedder
    
    print("\nEmbedding Information:")
    for i, embedding in enumerate(embedder.embeddings):
        print(f"Embedding {i}: input_dim={embedding.num_embeddings}, output_dim={embedding.embedding_dim}")
    
    # Get categorical column indices
    cat_idxs = network.cat_idxs
    cat_dims = network.cat_dims
    
    print("\nCategorical Columns:")
    for i, (idx, dim) in enumerate(zip(cat_idxs, cat_dims)):
        print(f"Categorical column {i}: index={idx}, max_dim={dim}")
    
    return {
        'cat_idxs': cat_idxs,
        'cat_dims': cat_dims,
        'total_columns': network.input_dim
    }

def main():
    # Find models in the Data directory
    base_dir = "Data/finals"
    model_dirs = [
        "combined_diabetes_tabnet", 
        "combined_all_ckd_tabnet",
        "combined_none_tabnet"
    ]
    
    model_info = {}
    
    for model_dir in model_dirs:
        model_id = model_dir
        model_path = os.path.join(base_dir, model_dir, f"{model_id}_model")
        
        try:
            print(f"\n--- Inspecting {model_id} ---")
            model_info[model_id] = inspect_model(model_path)
            
            # Also check scaler information
            scaler_path = os.path.join(base_dir, model_dir, f"{model_id}_scaler.joblib")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"\nScaler information:")
                print(f"  Feature names: {scaler.feature_names_in_}")
                print(f"  Number of features: {len(scaler.feature_names_in_)}")
                
        except Exception as e:
            print(f"Error inspecting {model_id}: {e}")
            
    with open("model_info.txt", "w") as f:
        for model_id, info in model_info.items():
            f.write(f"{model_id}:\n")
            f.write(f"  Total columns: {info['total_columns']}\n")
            f.write(f"  Categorical indices: {info['cat_idxs']}\n")
            f.write(f"  Category dimensions: {info['cat_dims']}\n\n")
    
    print("\nModel information saved to model_info.txt")

if __name__ == "__main__":
    main()

# tabnet_inference.py
import sys
import os
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

def main():
    if len(sys.argv) < 3:
        print("Usage: python tabnet_inference.py <model_id> <csv_for_inference>")
        sys.exit(1)

    model_id = sys.argv[1]
    csv_path = sys.argv[2]

    model_dir = os.path.join("Data","finals",model_id)
    model_file= os.path.join(model_dir, f"{model_id}_model.zip")
    if not os.path.exists(model_file):
        print(f"[ERROR] Model file not found: {model_file}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # For demonstration, let's do minimal preprocessing.
    # In your actual workflow, replicate the label encoding, scaling, etc.

    feat_df = df.drop(columns=["Id"], errors="ignore")
    X = feat_df.values

    regressor = TabNetRegressor()
    regressor.load_model(model_file)
    preds = regressor.predict(X).flatten()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("Data","new_predictions",model_id)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_id}_predictions_{timestamp}.csv")

    out_df = pd.DataFrame({
        "Id": df["Id"],
        "Predicted_Health_Index": preds
    })
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Predictions saved => {out_csv}")

if __name__ == "__main__":
    main()

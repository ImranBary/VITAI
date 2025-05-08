import pandas as pd

# Replace 'your_file.pkl' with the path to your .pkl file
pkl_file_path = 'Data/patient_data_with_all_indices.pkl'

try:
    # Load the .pkl file
    data = pd.read_pickle(pkl_file_path)
    
    # Check if the data is a DataFrame
    if isinstance(data, pd.DataFrame):
        print("Columns in the .pkl file:")
        print(data.columns)
    else:
        print("The .pkl file does not contain a DataFrame.")
except Exception as e:
    print(f"An error occurred: {e}")
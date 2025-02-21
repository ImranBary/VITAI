import os
import pickle
import pandas as pd

def read_and_print_pandas_dataframes_from_pkl(directory):
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith(".pkl"):
            file_path = os.path.join(directory, entry.name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                if isinstance(data, pd.DataFrame):
                    print(f"Filename: {entry.name}")
                    print("Column and First Row Value Pairs:")
                    first_row = data.iloc[0]
                    for column in data.columns:
                        print(f"{column}: {first_row[column]}")
                    print("\n")
                else:
                    print(f"The file {entry.name} does not contain a pandas DataFrame.")
                    print("\n")

read_and_print_pandas_dataframes_from_pkl(r'C:\Users\imran\Documents\VITAI\Data')

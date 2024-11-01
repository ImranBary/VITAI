import pandas as pd
import os
import sys

def pkl_to_csv(pkl_file, csv_file):
    # Read the pickle file
    df = pd.read_pickle(pkl_file)
    
    # Write the dataframe to a CSV file
    df.to_csv(csv_file, index=False)
    print(f"Successfully converted {pkl_file} to {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_pkl_files.py <input_directory>")
    else:
        input_directory = sys.argv[1]
        
        if not os.path.isdir(input_directory):
            print(f"The directory {input_directory} does not exist.")
            sys.exit(1)
        
        for filename in os.listdir(input_directory):
            if filename.endswith(".pkl"):
                pkl_file = os.path.join(input_directory, filename)
                csv_file = os.path.splitext(pkl_file)[0] + ".csv"
                pkl_to_csv(pkl_file, csv_file)
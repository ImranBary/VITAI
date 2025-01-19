import pandas as pd

# Replace 'your_file.pkl' with the path to your .pkl file
file_path = 'Data/temp_composite_none_tabnet_20250113_001634.pkl'

# Read the .pkl file into a DataFrame
df = pd.read_pickle(file_path)


#save top 5 rows of the dataframe to a csv file named temp_composite_none_tabnet_20250113_001634.csv
df.head().to_csv('Data/temp_composite_none_tabnet_20250113_001634.csv')

import pandas as pd

# Replace 'your_file.pkl' with the path to your .pkl file
file_path = 'Data/patient_data_with_health_index_cci_diabetes.pkl'

# Read the .pkl file into a DataFrame
df = pd.read_pickle(file_path)

# Display the DataFrame
print(df[df["CharlsonIndex"] > 0])
import pandas as pd

# Read the CSV file
df = pd.read_csv('Data/patients.csv')

# Get the first 5 rows
sample_df = df.head(5)

# Save the sample data to a new CSV file
sample_df.to_csv('sample_data.csv', index=False)
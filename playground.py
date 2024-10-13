import pandas as pd

# Function to read and filter CSV
def read_and_filter_csv(file_path, id1, id2):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Filter the DataFrame based on the specified IDs
    filtered_df = df[(df['Id'] == id1) | (df['Id'] == id2)]
    
    return filtered_df

# Example usage
file_path = 'aggregated_patient_data.csv'
id1 = 'dcdd5200-6453-4158-f01d-890cc4b7b248'
id2 = '94d463a0-c7cc-76e9-72fe-5e17d083f507'

filtered_data = read_and_filter_csv(file_path, id1, id2)
filtered_data.to_csv('filtered.csv',index=False)

import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect(r'E:\synthea.db')
cursor = conn.cursor()

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('analysis_results.csv')

# Save the DataFrame to the SQLite database
df.to_sql('analysis_results', conn, if_exists='replace', index=False)

# Commit the transaction and close the connection
conn.commit()
conn.close()
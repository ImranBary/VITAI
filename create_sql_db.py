import pandas as pd
import sqlite3
import os

# Define the data directory
data_dir = '/Users/ImranBary/DataGen/synthea/output/csv/'

# Connect to SQLite database
conn = sqlite3.connect('synthea.db')

# Function to load CSV files into SQLite in chunks
def load_csv_to_sqlite(csv_file, table_name, conn, chunksize=10000):
    csv_path = os.path.join(data_dir, csv_file)
    print(f"Loading {csv_file} into {table_name} table...")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk.to_sql(table_name, conn, if_exists='append', index=False)
    print(f"Finished loading {csv_file}.")

# List of CSV files and corresponding table names
csv_tables = {
    'patients.csv': 'patients',
    'encounters.csv': 'encounters',
    'medications.csv': 'medications',
    'allergies.csv': 'allergies',
    'procedures.csv': 'procedures',
    'careplans.csv': 'careplans',
    'conditions.csv': 'conditions',
    'immunizations.csv': 'immunizations',
    'observations.csv': 'observations'
}

# Load each CSV into the SQLite database
for csv_file, table_name in csv_tables.items():
    load_csv_to_sqlite(csv_file, table_name, conn)

# Function to create indexes
def create_index(conn, table_name, column_name):
    index_name = f'idx_{table_name}_{column_name}'
    print(f"Creating index {index_name} on {table_name}({column_name})...")
    conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name});')
    print(f"Index {index_name} created.")

# Create indexes on key columns
create_index(conn, 'patients', 'Id')
create_index(conn, 'encounters', 'Id')
create_index(conn, 'encounters', 'PATIENT')
create_index(conn, 'medications', 'PATIENT')
create_index(conn, 'allergies', 'PATIENT')
create_index(conn, 'procedures', 'PATIENT')
create_index(conn, 'careplans', 'PATIENT')
create_index(conn, 'conditions', 'PATIENT')
create_index(conn, 'immunizations', 'PATIENT')
create_index(conn, 'observations', 'PATIENT')

conn.close()
 
import pandas as pd
import sqlite3

def query_database(sql_query, db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Execute the query and store the result in a DataFrame
    df = pd.read_sql_query(sql_query, conn)
    
    # Close the connection
    conn.close()
    
    return df

sql_query = '''
SELECT * 
From patients
Where patients.Id like 'dcdd5200-6453-4158-f01d-890cc4b7b248'  ;'''

db_path = "synthea.db"

result_df = query_database(sql_query,db_path)
result_df.to_csv('test.csv',)

print(result_df)

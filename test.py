import pandas as pd

# Load the CSV file
df = pd.read_csv('/Users/ImranBary/VITAI/combined_data.csv')


# Give me a frequency number of how many patients are deceased
deceased_count = df[df['DECEASED'] == 1]['DECEASED'].count()
print(f"Deceased patients: {deceased_count}")



print(f"Total patinets: {df['Id'].count()}")




# Give me a frequency number of how many patients are above the age of 100 against the total patients in the dataset
print(f"Patients over 100: {(df[df['AGE'] > 100]['AGE'].count() / df['AGE'].count())*100:.2f}%")

# Print the patients with the most number of conditions and their age
most_conditions_patient = df['num_conditions'].value_counts().head(1).index[0]
patient_age = df[df['num_conditions'] == most_conditions_patient]['AGE'].values[0]
print(f"Patient with the most conditions: {most_conditions_patient}, Age: {patient_age}")

# Print the first 2 entries
#df.head(100).to_csv('test.csv',index=False)
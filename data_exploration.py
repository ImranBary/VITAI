# data_exploration.py

import pandas as pd
import os

def load_data(data_dir):
    """Load data for exploration."""
    conditions = pd.read_csv(os.path.join(data_dir, 'conditions.csv'))
    encounters = pd.read_csv(os.path.join(data_dir, 'encounters.csv'))
    medications = pd.read_csv(os.path.join(data_dir, 'medications.csv'))
    observations = pd.read_csv(os.path.join(data_dir, 'observations.csv'))
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    procedures = pd.read_csv(os.path.join(data_dir, 'procedures.csv'))
    return conditions, encounters, medications, observations, patients, procedures

def explore_conditions(conditions):
    """Explore conditions data."""
    print("Conditions Data Columns:")
    print(conditions.columns)
    print("\nConditions Data Sample (first row):")
    print(conditions.iloc[0])

def explore_encounters(encounters):
    """Explore encounters data."""
    print("encounters Data Columns:")
    print(encounters.columns)
    print("\nencounters Data Sample (first row):")
    print(encounters.iloc[0])

def explore_medications(medications):
    """Explore medications data."""
    print("medications Data Columns:")
    print(medications.columns)
    print("\nmedications Data Sample (first row):")
    print(medications.iloc[0])

def explore_observations(observations):
    """Explore observations data."""
    print("observations Data Columns:")
    print(observations.columns)
    print("\nobservations Data Sample (first row):")
    print(observations.iloc[0])
    
def explore_patients(patients):
    """Explore patients data."""
    print("patients Data Columns:")
    print(patients.columns)
    print("\npatients Data Sample (first row):")
    print(patients.iloc[0])
    
    
def explore_procedures(procedures):
    """Explore procedures data."""
    print("procedures Data Columns:")
    print(procedures.columns)
    print("\nprocedures Data Sample (first row):")
    print(procedures.iloc[0])
    
    
def main():
    data_dir =  'Data'  
    conditions, encounters, medications, observations, patients, procedures = load_data(data_dir)



    explore_conditions(conditions)
    explore_encounters(encounters)
    explore_medications(medications)
    explore_observations(observations)
    explore_patients(patients)
    explore_procedures(procedures)

if __name__ == '__main__':
    main()

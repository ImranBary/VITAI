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
    print("Conditions Data Sample:")
    print(conditions.head())
    #print("\nUnique Condition Descriptions:")
    #print(conditions['DESCRIPTION'].unique())

def explore_encounters(encounters):
    """Explore encounters data."""
    print("Encounters Data Sample:")
    print(encounters.head())
    #print("\nEncounter Classes:")
    #print(encounters['ENCOUNTERCLASS'].unique())

def explore_medications(medications):
    """Explore medications data."""
    print("Medications Data Sample:")
    print(medications.head())
    #print("\nUnique Medication Descriptions:")
    #print(medications['DESCRIPTION'].unique())

def explore_observations(observations):
    """Explore observations data."""
    print("Observations Data Sample:")
    print(observations.head())
    #print("\nUnique Observation Descriptions:")
    #print(observations['DESCRIPTION'].unique())
    
def explore_patients(patients):
    """Explore patients data."""
    print("Patients Data Sample:")
    print(patients.head())
    
    
def explore_procedures(procedures):
    """Explore procedures data."""
    print("Procedures Data Sample:")
    print(procedures.head())

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

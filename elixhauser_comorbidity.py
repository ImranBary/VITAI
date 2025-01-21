# elixhauser_comorbidity.py
# Author: Imran Feisal
# Date: 21/01/2025
# Description:
# This script computes the Elixhauser Comorbidity Index (ECI) from SNOMED codes.
# It expects a conditions dataframe with SNOMED-CT codes and patient IDs.
# The output is a DataFrame with patient IDs and their computed ECI scores.

import pandas as pd

def assign_eci_weights(ElixhauserCategory):
    """
    Assign weights for the Elixhauser Comorbidity Index based on the Van Walraven method.
    """
    category_to_weight = {
        'Congestive heart failure': 7,
        'Cardiac arrhythmias': 5,
        'Valvular disease': 4,
        'Pulmonary circulation disorders': 6,
        'Peripheral vascular disorders': 2,
        'Hypertension, uncomplicated': -1,
        'Hypertension, complicated': 0,
        'Paralysis': 7,
        'Other neurological disorders': 6,
        'Chronic pulmonary disease': 3,
        'Diabetes, uncomplicated': 0,
        'Diabetes, complicated': 7,
        'Hypothyroidism': 0,
        'Renal failure': 5,
        'Liver disease': 11,
        'Peptic ulcer disease': 0,
        'AIDS/HIV': 0,
        'Lymphoma': 9,
        'Metastatic cancer': 14,
        'Solid tumour without metastasis': 8,
        'Rheumatoid arthritis/collagen vascular diseases': 4,
        'Coagulopathy': 11,
        'Obesity': 0,
        'Weight loss': 6,
        'Fluid and electrolyte disorders': 5,
        'Blood loss anaemia': 3,
        'Deficiency anaemias': 0,
        'Alcohol abuse': 0,
        'Drug abuse': 0,
        'Psychoses': 0,
        'Depression': -3
    }
    return category_to_weight.get(ElixhauserCategory, 0)

def compute_eci(conditions):
    """
    Compute the Elixhauser Comorbidity Index for each patient from a DataFrame of SNOMED-CT codes.
    
    Args:
        conditions (pd.DataFrame): Must include ['PATIENT', 'CODE'] columns where
                                   CODE is a SNOMED code (int or str).
    
    Returns:
        pd.DataFrame: A DataFrame with columns ['PATIENT', 'ElixhauserIndex'].
                      If a patient has no mapped comorbidities, ElixhauserIndex = 0.
    """

    # Define the SNOMED to Elixhauser category mapping dictionary
    SNOMED_TO_ELIXHAUSER = {
        # Congestive Heart Failure
        88805009: "Congestive heart failure",
        84114007: "Congestive heart failure",

        # Cardiac Arrhythmias
        49436004: "Cardiac arrhythmias",

        # Valvular Disease
        48724000: "Valvular disease",
        91434003: "Pulmonic valve regurgitation",
        79619009: "Mitral valve stenosis",
        111287006: "Tricuspid valve regurgitation",
        49915006: "Tricuspid valve stenosis",
        60573004: "Aortic valve stenosis",
        60234000: "Aortic valve regurgitation",
        
        # Pulmonary Circulation Disorders
        65710008: "Pulmonary circulation disorders",
        706870000: "Acute pulmonary embolism",
        67782005: "Acute respiratory distress syndrome",

        # Peripheral Vascular Disorders
        698754002: "Peripheral vascular disorders",

        # Hypertension
        59621000: "Hypertension, uncomplicated",

        # Paralysis
        698754002: "Paralysis",
        128188000: "Paralysis",

        # Other Neurological Disorders
        69896004: "Other neurological disorders",
        128613002: "Seizure disorder",

        # Chronic Pulmonary Disease
        195967001: "Chronic pulmonary disease",
        233678006: "Chronic pulmonary disease",

        # Diabetes, Complicated
        368581000119106: "Diabetes, complicated",
        422034002: "Diabetes, complicated",
        90781000119102: "Diabetes, complicated",

        # Diabetes, Uncomplicated
        44054006: "Diabetes, uncomplicated",

        # Renal Failure
        129721000119106: "Renal failure",
        433144002: "Renal failure",

        # Liver Disease
        128302006: "Liver disease",
        61977001: "Liver disease",

        # Peptic Ulcer Disease
        # (Not identified in the dataset)

        # AIDS/HIV
        62479008: "AIDS/HIV",
        86406008: "AIDS/HIV",

        # Lymphoma
        93143009: "Lymphoma",

        # Metastatic Cancer
        94503003: "Metastatic cancer",
        94260004: "Metastatic cancer",

        # Solid Tumour Without Metastasis
        126906006: "Solid tumour without metastasis",
        254637007: "Solid tumour without metastasis",

        # Rheumatoid Arthritis / Collagen Vascular Diseases
        69896004: "Rheumatoid arthritis/collagen vascular diseases",
        200936003: "Rheumatoid arthritis/collagen vascular diseases",

        # Coagulopathy
        234466008: "Coagulopathy",

        # Obesity
        408512008: "Obesity",
        162864005: "Obesity",

        # Weight Loss
        278860009: "Weight loss",

        # Fluid and Electrolyte Disorders
        389087006: "Fluid and electrolyte disorders",

        # Blood Loss Anaemia
        # (Not identified in the dataset)

        # Deficiency Anaemias
        271737000: "Deficiency anaemias",

        # Alcohol Abuse
        7200002: "Alcohol abuse",

        # Drug Abuse
        6525002: "Drug abuse",

        # Psychoses
        47505003: "Psychoses",

        # Depression
        370143000: "Depression",
        36923009: "Depression",
    }


    # Map SNOMED codes to Elixhauser categories
    conditions['ElixhauserCategory'] = conditions['CODE'].map(SNOMED_TO_ELIXHAUSER)

    # Assign weights based on categories
    conditions['ECI_Weight'] = conditions['ElixhauserCategory'].apply(assign_eci_weights)

    # For each patient, sum the unique weights for each category
    patient_eci = (
        conditions
        .groupby(['PATIENT', 'ElixhauserCategory'])['ECI_Weight']
        .max()
        .reset_index()
    )

    patient_eci_sum = (
        patient_eci
        .groupby('PATIENT')['ECI_Weight']
        .sum()
        .reset_index()
    )

    # Rename the result column for clarity
    patient_eci_sum.rename(columns={'ECI_Weight': 'ElixhauserIndex'}, inplace=True)

    return patient_eci_sum

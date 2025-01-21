# charlson_comorbidity.py
# Author: Imran Feisal  
# Date: 22/15/2024
# Description:
# This script computes the Charlson Comorbidity Index (CCI) from SNOMED codes.
# It expects a CSV mapping file with columns:
# code, coding_system, description, entity, list_name, upload_date, medcode, snomedctdescriptionid, CharlsonCategory
# It also expects a conditions dataframe with SNOMED-CT codes.
#
# The output is a DataFrame with patient IDs and their computed CCI scores.

import pandas as pd
import os

def load_cci_mapping(data_dir):
    """
    Load the Charlson mapping file.
    Expects a CSV with columns: ['code', 'CharlsonCategory', ...].
    If no CSV or the file is missing, this function might raise FileNotFoundError
    or produce an empty DataFrame (depending on your needs).
    """
    cci_filepath = os.path.join(data_dir, 'res195-comorbidity-cci-snomed.csv')
    cci_df = pd.read_csv(cci_filepath)
    # Basic cleanup or renaming if needed
    # e.g. rename columns to something standard
    # cci_df.rename(columns={'code': 'SNOMED', 'CharlsonCategory': 'CharlsonCategory'}, inplace=True)
    return cci_df

def assign_cci_weights(CharlsonCategory):
    """
    Assign Charlson weights based on category.
    As per the original Charlson Comorbidity Index:
      - Myocardial infarction, Congestive heart failure, Peripheral vascular disease,
        Cerebrovascular disease, Dementia, Chronic pulmonary disease, Connective tissue disease,
        Ulcer disease, Mild liver disease, Diabetes without end-organ damage => weight 1
      - Hemiplegia, Moderate/severe kidney disease, Diabetes with end-organ damage,
        Any tumour (solid tumor), leukemia, lymphoma => weight 2
      - Moderate or severe liver disease => weight 3
      - Metastatic solid tumour, AIDS => weight 6
    """
    category_to_weight = {
        'Myocardial infarction': 1,
        'Congestive heart failure': 1,
        'Peripheral vascular disease': 1,
        'Cerebrovascular disease': 1,
        'Dementia': 1,
        'Chronic pulmonary disease': 1,
        'Connective tissue disease': 1,
        'Ulcer disease': 1,
        'Mild liver disease': 1,
        'Diabetes without end-organ damage': 1,
        'Hemiplegia': 2,
        'Moderate or severe kidney disease': 2,
        'Diabetes with end-organ damage': 2,
        'Any tumour, leukaemia, lymphoma': 2,
        'Moderate or severe liver disease': 3,
        'Metastatic solid tumour': 6,
        'AIDS/HIV': 6
    }
    return category_to_weight.get(CharlsonCategory, 0)

def compute_cci(conditions, cci_mapping):
    """
    Compute the Charlson Comorbidity Index for each patient from a DataFrame of SNOMED-CT codes.
    
    Args:
        conditions (pd.DataFrame): Must include ['PATIENT', 'CODE'] columns where
                                   CODE is a SNOMED code (int or str).
        cci_mapping (pd.DataFrame): A CSV-based lookup with at least:
                                   ['code', 'CharlsonCategory']
                                   (Loaded from load_cci_mapping).
                                   Some codes may be missing from the CSV.
    
    Returns:
        pd.DataFrame: A DataFrame with columns ['PATIENT', 'CharlsonIndex'].
                      If a patient has no mapped comorbidities, CharlsonIndex = 0.
    """

    # 1. Define a fallback dictionary for SNOMED -> CharlsonCategory
    SNOMED_TO_CHARLSON = {
        #
        # MYOCARDIAL INFARCTION (weight 1)
        #
        22298006: "Myocardial infarction",      # "Myocardial infarction (disorder)"
        401303003: "Myocardial infarction",     # "Acute ST segment elevation myocardial infarction"
        401314000: "Myocardial infarction",     # "Acute non-ST segment elevation myocardial infarction"
        129574000: "Myocardial infarction",     # "Postoperative myocardial infarction (disorder)"
        #
        # CONGESTIVE HEART FAILURE (weight 1)
        #
        88805009: "Congestive heart failure",   # "Chronic congestive heart failure (disorder)"
        84114007: "Congestive heart failure",   # "Heart failure (disorder)"
        #
        # PERIPHERAL VASCULAR DISEASE (weight 1)
        #
        # -- None in your list match typical “peripheral vascular disease” codes --
        #
        # CEREBROVASCULAR DISEASE (weight 1)
        #
        230690007: "Cerebrovascular disease",   # "Cerebrovascular accident (disorder)"
        #
        # DEMENTIA (weight 1)
        #
        26929004: "Dementia",                   # "Alzheimer's disease (disorder)"
        230265002: "Dementia",                  # "Familial Alzheimer's disease of early onset (disorder)"
        #
        # CHRONIC PULMONARY DISEASE (weight 1)
        #
        185086009: "Chronic pulmonary disease", # "Chronic obstructive bronchitis (disorder)"
        87433001:  "Chronic pulmonary disease", # "Pulmonary emphysema (disorder)"
        195967001: "Chronic pulmonary disease", # "Asthma (disorder)" – (Some include chronic asthma under COPD)
        233678006: "Chronic pulmonary disease", # "Childhood asthma (disorder)"
        #
        # CONNECTIVE TISSUE DISEASE (weight 1)
        #
        69896004: "Connective tissue disease",  # "Rheumatoid arthritis (disorder)"
        200936003: "Connective tissue disease", # "Lupus erythematosus (disorder)"
        #
        # ULCER DISEASE (weight 1)
        #
        # -- None in your list specifically match “peptic ulcer disease” --
        #
        # MILD LIVER DISEASE (weight 1)
        #
        128302006: "Mild liver disease",        # "Chronic hepatitis C (disorder)" 
        61977001:  "Mild liver disease",        # "Chronic type B viral hepatitis (disorder)"
        #
        # DIABETES WITHOUT END-ORGAN DAMAGE (weight 1)
        #
        44054006: "Diabetes without end-organ damage",  # "Diabetes mellitus type 2 (disorder)"
        #
        # DIABETES WITH END-ORGAN DAMAGE (weight 2)
        #
        368581000119106: "Diabetes with end-organ damage",  # "Neuropathy due to type 2 diabetes mellitus"
        422034002:        "Diabetes with end-organ damage",  # "Retinopathy due to type 2 diabetes mellitus"
        127013003:        "Diabetes with end-organ damage",  # "Disorder of kidney due to diabetes mellitus"
        90781000119102:   "Diabetes with end-organ damage",  # "Microalbuminuria due to type 2 diabetes mellitus"
        157141000119108:  "Diabetes with end-organ damage",  # "Proteinuria due to type 2 diabetes mellitus"
        60951000119105:   "Diabetes with end-organ damage",  # "Blindness due to type 2 diabetes mellitus"
        97331000119101:   "Diabetes with end-organ damage",  # "Macular edema & retinopathy due to T2DM"
        1501000119109:    "Diabetes with end-organ damage",  # "Proliferative retinopathy due to T2DM"
        1551000119108:    "Diabetes with end-organ damage",  # "Nonproliferative retinopathy due to T2DM"
        #
        # HEMIPLEGIA or PARAPLEGIA (weight 2)
        #
        # -- None in your list appear to indicate hemiplegia or paraplegia, 
        #    e.g. “cerebral palsy” is not typically counted as hemiplegia. 
        #
        # MODERATE OR SEVERE KIDNEY DISEASE (weight 2)
        #
        # Some references only count CKD stage 3 or worse. 
        # The user had stage 1 & 2 included, so we’ll keep that approach consistent:
        431855005: "Moderate or severe kidney disease",  # "CKD stage 1 (disorder)"
        431856006: "Moderate or severe kidney disease",  # "CKD stage 2 (disorder)"
        433144002: "Moderate or severe kidney disease",  # "CKD stage 3 (disorder)"
        431857002: "Moderate or severe kidney disease",  # "CKD stage 4 (disorder)"
        46177005:  "Moderate or severe kidney disease",  # "End-stage renal disease (disorder)"
        129721000119106: "Moderate or severe kidney disease",  # "Acute renal failure on dialysis (disorder)"
        #
        # ANY TUMOUR (solid tumor), LEUKEMIA, LYMPHOMA (weight 2)
        #
        254637007: "Any tumour, leukaemia, lymphoma",  # "Non-small cell lung cancer (disorder)"
        254632001: "Any tumour, leukaemia, lymphoma",  # "Small cell carcinoma of lung (disorder)"
        93761005:  "Any tumour, leukaemia, lymphoma",  # "Primary malignant neoplasm of colon"
        363406005: "Any tumour, leukaemia, lymphoma",  # "Malignant neoplasm of colon"
        109838007: "Any tumour, leukaemia, lymphoma",  # "Overlapping malignant neoplasm of colon"
        126906006: "Any tumour, leukaemia, lymphoma",  # "Neoplasm of prostate (disorder)"
        92691004:  "Any tumour, leukaemia, lymphoma",  # "Carcinoma in situ of prostate"
        254837009: "Any tumour, leukaemia, lymphoma",  # "Malignant neoplasm of breast"
        109989006: "Any tumour, leukaemia, lymphoma",  # "Multiple myeloma (disorder)"
        93143009:  "Any tumour, leukaemia, lymphoma",  # "Leukemia disease (disorder)"
        91861009:  "Any tumour, leukaemia, lymphoma",  # "Acute myeloid leukemia (disorder)"
        #
        # MODERATE OR SEVERE LIVER DISEASE (weight 3)
        #
        # -- None in your list mention cirrhosis or advanced hepatic failure 
        #    that we'd classify as 'moderate/severe liver disease'.
        #
        # METASTATIC SOLID TUMOUR (weight 6)
        #
        94503003: "Metastatic solid tumour",    # "Metastatic malignant neoplasm to prostate"
        94260004: "Metastatic solid tumour",    # "Metastatic malignant neoplasm to colon"
        #
        # AIDS/HIV (weight 6)
        #
        62479008: "AIDS/HIV",   # "Acquired immune deficiency syndrome (disorder)"
        86406008: "AIDS/HIV",   # "Human immunodeficiency virus infection (disorder)"
    }


    # 2. Merge conditions with cci_mapping on SNOMED code (left_on='CODE', right_on='code')
    #    This way if a code is present in the CSV, it overrides or supplies CharlsonCategory
    merged = conditions.merge(
        cci_mapping[['code', 'CharlsonCategory']],
        how='left',
        left_on='CODE',
        right_on='code'
    )

    # 3. Fallback: For rows where the CSV didn't provide a CharlsonCategory, try the SNOMED_TO_CHARLSON dict
    #    If neither the CSV nor the dict have it, it remains None/NaN
    def fallback_category(row):
        if pd.notna(row['CharlsonCategory']):
            return row['CharlsonCategory']
        else:
            # Attempt dictionary lookup
            return SNOMED_TO_CHARLSON.get(row['CODE'], None)

    merged['CharlsonCategory'] = merged.apply(fallback_category, axis=1)

    # 4. Compute the Charlson weight for each row
    merged['CCI_Weight'] = merged['CharlsonCategory'].apply(assign_cci_weights)

    # 5. For each patient, sum the unique categories.
    #    i.e. if a patient has multiple codes in the same category, only count once.
    #    We do this by grouping on (PATIENT, CharlsonCategory) and taking the max weight
    #    Then summing across categories for each patient
    patient_cci = (
        merged
        .groupby(['PATIENT', 'CharlsonCategory'])['CCI_Weight']
        .max()
        .reset_index()
    )

    patient_cci_sum = (
        patient_cci
        .groupby('PATIENT')['CCI_Weight']
        .sum()
        .reset_index()
    )

    # Rename column to match the expected return type
    patient_cci_sum.rename(columns={'CCI_Weight': 'CharlsonIndex'}, inplace=True)

    return patient_cci_sum




'''
import pandas as pd
import os

def load_cci_mapping(data_dir):
    """
    Load the Charlson mapping file.
    """
    cci_df = pd.read_csv(os.path.join(data_dir, 'res195-comorbidity-cci-snomed.csv'))
    # Clean and prepare mapping if needed
    return cci_df

def assign_cci_weights(CharlsonCategory):
    """
    Assign Charlson weights based on category.
    As per the original Charlson Comorbidity Index:
    - Myocardial infarction, Congestive heart failure, Peripheral vascular disease, Cerebrovascular disease, Dementia,
      Chronic pulmonary disease, Connective tissue disease, Peptic ulcer disease, Mild liver disease, Diabetes (without end-organ damage)
      all have weight 1.
    - Hemiplegia, Moderate or severe kidney disease, Diabetes with end-organ damage, Any tumor (solid tumor), Leukemia,
      Lymphoma have weight 2.
    - Moderate or severe liver disease has weight 3.
    - Metastatic solid tumor, AIDS have weight 6.
    
    """
    category_to_weight = {
        'Myocardial infarction': 1,
        'Congestive heart failure': 1,
        'Peripheral vascular disease': 1,
        'Cerebrovascular disease': 1,
        'Dementia': 1,
        'Chronic pulmonary disease': 1,
        'Connective tissue disease': 1,
        'Ulcer disease': 1,
        'Mild liver disease': 1,
        'Diabetes without end-organ damage': 1,
        'Hemiplegia': 2,
        'Moderate or severe kidney disease': 2,
        'Diabetes with end-organ damage': 2,
        'Any tumour, leukaemia, lymphoma': 2,
        'Moderate or severe liver disease': 3,
        'Metastatic solid tumour': 6,
        'AIDS/HIV': 6
    }
    return category_to_weight.get(CharlsonCategory, 0)

def compute_cci(conditions, cci_mapping):
    """
    Compute the Charlson Comorbidity Index for each patient.
    
    Args:
        conditions (pd.DataFrame): Conditions data with SNOMED codes.
        cci_mapping (pd.DataFrame): Mapping of SNOMED codes to Charlson categories.

    Returns:
        pd.DataFrame: A DataFrame with ['PATIENT', 'CharlsonIndex'].
    """
    # Merge conditions with cci_mapping on SNOMED code
    merged = conditions.merge(cci_mapping, left_on='CODE', right_on='code', how='left')
    merged['CCI_Weight'] = merged['CharlsonCategory'].apply(assign_cci_weights)

    # For each patient, sum the unique categories (Charlson CCI accounts for comorbidity presence, not frequency)
    # One patient could have multiple codes in the same category, but the category is only counted once.
    # So, we group by patient and category, take max weight for that category, and sum.
    patient_cci = merged.groupby(['PATIENT', 'CharlsonCategory'])['CCI_Weight'].max().reset_index()
    patient_cci_sum = patient_cci.groupby('PATIENT')['CCI_Weight'].sum().reset_index()
    patient_cci_sum.rename(columns={'CCI_Weight':'CharlsonIndex'}, inplace=True)
    return patient_cci_sum
'''
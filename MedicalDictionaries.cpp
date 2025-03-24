#include "MedicalDictionaries.h"
#include <unordered_map>
#include <string>
#include <algorithm>
#include <iostream>
#include <cctype>

// Global lookup tables
std::unordered_map<std::string, float> CHARLSON_CODE_TO_WEIGHT;
std::unordered_map<std::string, float> ELIXHAUSER_CODE_TO_WEIGHT;
std::unordered_map<std::string, float> COMORBIDITY_CODE_TO_WEIGHT;

// For abnormal observation quick lookup
std::unordered_map<std::string, std::pair<double, double>> OBS_NORMAL_RANGES;

// Initialize lookup tables that match Python implementation from charlson_comorbidity.py
void initializeDirectLookups() {
    std::cout << "[INFO] Initializing Charlson lookup tables with SNOMED CT codes\n";
    
    // Charlson comorbidity SNOMED CT codes and weights - directly from charlson_comorbidity.py
    
    // MYOCARDIAL INFARCTION (weight 1)
    CHARLSON_CODE_TO_WEIGHT["22298006"] = 1.0f;    // Myocardial infarction (disorder)
    CHARLSON_CODE_TO_WEIGHT["401303003"] = 1.0f;   // Acute ST segment elevation myocardial infarction
    CHARLSON_CODE_TO_WEIGHT["401314000"] = 1.0f;   // Acute non-ST segment elevation myocardial infarction
    CHARLSON_CODE_TO_WEIGHT["129574000"] = 1.0f;   // Postoperative myocardial infarction (disorder)
    
    // CONGESTIVE HEART FAILURE (weight 1)
    CHARLSON_CODE_TO_WEIGHT["88805009"] = 1.0f;    // Chronic congestive heart failure (disorder)
    CHARLSON_CODE_TO_WEIGHT["84114007"] = 1.0f;    // Heart failure (disorder)
    
    // CEREBROVASCULAR DISEASE (weight 1)
    CHARLSON_CODE_TO_WEIGHT["230690007"] = 1.0f;   // Cerebrovascular accident (disorder)
    
    // DEMENTIA (weight 1)
    CHARLSON_CODE_TO_WEIGHT["26929004"] = 1.0f;    // Alzheimer's disease (disorder)
    CHARLSON_CODE_TO_WEIGHT["230265002"] = 1.0f;   // Familial Alzheimer's disease of early onset (disorder)
    
    // CHRONIC PULMONARY DISEASE (weight 1)
    CHARLSON_CODE_TO_WEIGHT["185086009"] = 1.0f;   // Chronic obstructive bronchitis (disorder)
    CHARLSON_CODE_TO_WEIGHT["87433001"] = 1.0f;    // Pulmonary emphysema (disorder)
    CHARLSON_CODE_TO_WEIGHT["195967001"] = 1.0f;   // Asthma (disorder)
    CHARLSON_CODE_TO_WEIGHT["233678006"] = 1.0f;   // Childhood asthma (disorder)
    
    // CONNECTIVE TISSUE DISEASE (weight 1)
    CHARLSON_CODE_TO_WEIGHT["69896004"] = 1.0f;    // Rheumatoid arthritis (disorder)
    CHARLSON_CODE_TO_WEIGHT["200936003"] = 1.0f;   // Lupus erythematosus (disorder)
    
    // MILD LIVER DISEASE (weight 1)
    CHARLSON_CODE_TO_WEIGHT["128302006"] = 1.0f;   // Chronic hepatitis C (disorder)
    CHARLSON_CODE_TO_WEIGHT["61977001"] = 1.0f;    // Chronic type B viral hepatitis (disorder)
    
    // DIABETES WITHOUT END-ORGAN DAMAGE (weight 1)
    CHARLSON_CODE_TO_WEIGHT["44054006"] = 1.0f;    // Diabetes mellitus type 2 (disorder)
    
    // DIABETES WITH END-ORGAN DAMAGE (weight 2)
    CHARLSON_CODE_TO_WEIGHT["368581000119106"] = 2.0f;  // Neuropathy due to type 2 diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT["422034002"] = 2.0f;        // Retinopathy due to type 2 diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT["127013003"] = 2.0f;        // Disorder of kidney due to diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT["90781000119102"] = 2.0f;   // Microalbuminuria due to type 2 diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT["157141000119108"] = 2.0f;  // Proteinuria due to type 2 diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT["60951000119105"] = 2.0f;   // Blindness due to type 2 diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT["97331000119101"] = 2.0f;   // Macular edema & retinopathy due to T2DM
    CHARLSON_CODE_TO_WEIGHT["1501000119109"] = 2.0f;    // Proliferative retinopathy due to T2DM
    CHARLSON_CODE_TO_WEIGHT["1551000119108"] = 2.0f;    // Nonproliferative retinopathy due to T2DM
    
    // MODERATE OR SEVERE KIDNEY DISEASE (weight 2)
    CHARLSON_CODE_TO_WEIGHT["431855005"] = 2.0f;        // CKD stage 1 (disorder)
    CHARLSON_CODE_TO_WEIGHT["431856006"] = 2.0f;        // CKD stage 2 (disorder)
    CHARLSON_CODE_TO_WEIGHT["433144002"] = 2.0f;        // CKD stage 3 (disorder)
    CHARLSON_CODE_TO_WEIGHT["431857002"] = 2.0f;        // CKD stage 4 (disorder)
    CHARLSON_CODE_TO_WEIGHT["46177005"] = 2.0f;         // End-stage renal disease (disorder)
    CHARLSON_CODE_TO_WEIGHT["129721000119106"] = 2.0f;  // Acute renal failure on dialysis (disorder)
    
    // ANY TUMOUR (solid tumor), LEUKEMIA, LYMPHOMA (weight 2)
    CHARLSON_CODE_TO_WEIGHT["254637007"] = 2.0f;   // Non-small cell lung cancer (disorder)
    CHARLSON_CODE_TO_WEIGHT["254632001"] = 2.0f;   // Small cell carcinoma of lung (disorder)
    CHARLSON_CODE_TO_WEIGHT["93761005"] = 2.0f;    // Primary malignant neoplasm of colon
    CHARLSON_CODE_TO_WEIGHT["363406005"] = 2.0f;   // Malignant neoplasm of colon
    CHARLSON_CODE_TO_WEIGHT["109838007"] = 2.0f;   // Overlapping malignant neoplasm of colon
    CHARLSON_CODE_TO_WEIGHT["126906006"] = 2.0f;   // Neoplasm of prostate (disorder)
    CHARLSON_CODE_TO_WEIGHT["92691004"] = 2.0f;    // Carcinoma in situ of prostate
    CHARLSON_CODE_TO_WEIGHT["254837009"] = 2.0f;   // Malignant neoplasm of breast
    CHARLSON_CODE_TO_WEIGHT["109989006"] = 2.0f;   // Multiple myeloma (disorder)
    CHARLSON_CODE_TO_WEIGHT["93143009"] = 2.0f;    // Leukemia disease (disorder)
    CHARLSON_CODE_TO_WEIGHT["91861009"] = 2.0f;    // Acute myeloid leukemia (disorder)
    
    // METASTATIC SOLID TUMOUR (weight 6)
    CHARLSON_CODE_TO_WEIGHT["94503003"] = 6.0f;    // Metastatic malignant neoplasm to prostate
    CHARLSON_CODE_TO_WEIGHT["94260004"] = 6.0f;    // Metastatic malignant neoplasm to colon
    
    // AIDS/HIV (weight 6)
    CHARLSON_CODE_TO_WEIGHT["62479008"] = 6.0f;   // Acquired immune deficiency syndrome (disorder)
    CHARLSON_CODE_TO_WEIGHT["86406008"] = 6.0f;   // Human immunodeficiency virus infection (disorder)

    // Add string versions of common diabetes and CKD codes for subset identification
    // (no need to convert all numeric codes to strings since the lookup will handle it)
    CHARLSON_CODE_TO_WEIGHT["73211009"] = 1.0f;   // Diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT["46635009"] = 1.0f;   // Diabetes mellitus type 1
    CHARLSON_CODE_TO_WEIGHT["190330002"] = 1.0f;  // Type 2 diabetes mellitus

    std::cout << "[INFO] Added " << CHARLSON_CODE_TO_WEIGHT.size() << " SNOMED CT codes to Charlson dictionary\n";
}

void initializeElixhauserLookups() {
    std::cout << "[INFO] Initializing Elixhauser lookup tables with SNOMED CT codes\n";
    
    // Elixhauser comorbidity codes and weights - directly from elixhauser_comorbidity.py

    // Congestive Heart Failure (weight 7)
    ELIXHAUSER_CODE_TO_WEIGHT["88805009"] = 7.0f;   // Chronic congestive heart failure
    ELIXHAUSER_CODE_TO_WEIGHT["84114007"] = 7.0f;   // Heart failure

    // Cardiac Arrhythmias (weight 5)
    ELIXHAUSER_CODE_TO_WEIGHT["49436004"] = 5.0f;   // Cardiac arrhythmias

    // Valvular Disease (weight 4)
    ELIXHAUSER_CODE_TO_WEIGHT["48724000"] = 4.0f;   // Valvular disease
    ELIXHAUSER_CODE_TO_WEIGHT["91434003"] = 4.0f;   // Pulmonic valve regurgitation
    ELIXHAUSER_CODE_TO_WEIGHT["79619009"] = 4.0f;   // Mitral valve stenosis
    ELIXHAUSER_CODE_TO_WEIGHT["111287006"] = 4.0f;  // Tricuspid valve regurgitation
    ELIXHAUSER_CODE_TO_WEIGHT["49915006"] = 4.0f;   // Tricuspid valve stenosis
    ELIXHAUSER_CODE_TO_WEIGHT["60573004"] = 4.0f;   // Aortic valve stenosis
    ELIXHAUSER_CODE_TO_WEIGHT["60234000"] = 4.0f;   // Aortic valve regurgitation

    // Pulmonary Circulation Disorders (weight 6)
    ELIXHAUSER_CODE_TO_WEIGHT["65710008"] = 6.0f;   // Pulmonary circulation disorders
    ELIXHAUSER_CODE_TO_WEIGHT["706870000"] = 6.0f;  // Acute pulmonary embolism
    ELIXHAUSER_CODE_TO_WEIGHT["67782005"] = 6.0f;   // Acute respiratory distress syndrome

    // Peripheral Vascular Disorders (weight 2)
    ELIXHAUSER_CODE_TO_WEIGHT["698754002"] = 2.0f;  // Peripheral vascular disorders

    // Hypertension, uncomplicated (weight -1)
    ELIXHAUSER_CODE_TO_WEIGHT["59621000"] = -1.0f;  // Hypertension, uncomplicated

    // Paralysis (weight 7)
    ELIXHAUSER_CODE_TO_WEIGHT["698754002"] = 7.0f;  // Paralysis (note: duplicate with peripheral vascular disorders)
    ELIXHAUSER_CODE_TO_WEIGHT["128188000"] = 7.0f;  // Paralysis

    // Other Neurological Disorders (weight 6)
    ELIXHAUSER_CODE_TO_WEIGHT["69896004"] = 6.0f;   // Other neurological disorders
    ELIXHAUSER_CODE_TO_WEIGHT["128613002"] = 6.0f;  // Seizure disorder

    // Chronic Pulmonary Disease (weight 3)
    ELIXHAUSER_CODE_TO_WEIGHT["195967001"] = 3.0f;  // Chronic pulmonary disease
    ELIXHAUSER_CODE_TO_WEIGHT["233678006"] = 3.0f;  // Chronic pulmonary disease

    // Diabetes, Complicated (weight 7)
    ELIXHAUSER_CODE_TO_WEIGHT["368581000119106"] = 7.0f;  // Diabetes, complicated
    ELIXHAUSER_CODE_TO_WEIGHT["422034002"] = 7.0f;        // Diabetes, complicated
    ELIXHAUSER_CODE_TO_WEIGHT["90781000119102"] = 7.0f;   // Diabetes, complicated

    // Diabetes, Uncomplicated (weight 0)
    ELIXHAUSER_CODE_TO_WEIGHT["44054006"] = 0.0f;   // Diabetes, uncomplicated

    // Renal Failure (weight 5)
    ELIXHAUSER_CODE_TO_WEIGHT["129721000119106"] = 5.0f;  // Renal failure
    ELIXHAUSER_CODE_TO_WEIGHT["433144002"] = 5.0f;        // Renal failure

    // Liver Disease (weight 11)
    ELIXHAUSER_CODE_TO_WEIGHT["128302006"] = 11.0f;  // Liver disease
    ELIXHAUSER_CODE_TO_WEIGHT["61977001"] = 11.0f;   // Liver disease

    // AIDS/HIV (weight 0)
    ELIXHAUSER_CODE_TO_WEIGHT["62479008"] = 0.0f;   // AIDS/HIV
    ELIXHAUSER_CODE_TO_WEIGHT["86406008"] = 0.0f;   // AIDS/HIV

    // Lymphoma (weight 9)
    ELIXHAUSER_CODE_TO_WEIGHT["93143009"] = 9.0f;   // Lymphoma

    // Metastatic Cancer (weight 14)
    ELIXHAUSER_CODE_TO_WEIGHT["94503003"] = 14.0f;  // Metastatic cancer
    ELIXHAUSER_CODE_TO_WEIGHT["94260004"] = 14.0f;  // Metastatic cancer

    // Solid Tumour Without Metastasis (weight 8)
    ELIXHAUSER_CODE_TO_WEIGHT["126906006"] = 8.0f;  // Solid tumour without metastasis
    ELIXHAUSER_CODE_TO_WEIGHT["254637007"] = 8.0f;  // Solid tumour without metastasis

    // Rheumatoid Arthritis / Collagen Vascular Diseases (weight 4)
    ELIXHAUSER_CODE_TO_WEIGHT["69896004"] = 4.0f;    // Rheumatoid arthritis/collagen vascular diseases
    ELIXHAUSER_CODE_TO_WEIGHT["200936003"] = 4.0f;   // Rheumatoid arthritis/collagen vascular diseases

    // Coagulopathy (weight 11)
    ELIXHAUSER_CODE_TO_WEIGHT["234466008"] = 11.0f;  // Coagulopathy

    // Obesity (weight 0)
    ELIXHAUSER_CODE_TO_WEIGHT["408512008"] = 0.0f;  // Obesity
    ELIXHAUSER_CODE_TO_WEIGHT["162864005"] = 0.0f;  // Obesity

    // Weight Loss (weight 6)
    ELIXHAUSER_CODE_TO_WEIGHT["278860009"] = 6.0f;  // Weight loss

    // Fluid and Electrolyte Disorders (weight 5)
    ELIXHAUSER_CODE_TO_WEIGHT["389087006"] = 5.0f;  // Fluid and electrolyte disorders

    // Deficiency Anaemias (weight 0)
    ELIXHAUSER_CODE_TO_WEIGHT["271737000"] = 0.0f;  // Deficiency anaemias

    // Alcohol Abuse (weight 0)
    ELIXHAUSER_CODE_TO_WEIGHT["7200002"] = 0.0f;    // Alcohol abuse

    // Drug Abuse (weight 0)
    ELIXHAUSER_CODE_TO_WEIGHT["6525002"] = 0.0f;    // Drug abuse

    // Psychoses (weight 0)
    ELIXHAUSER_CODE_TO_WEIGHT["47505003"] = 0.0f;   // Psychoses

    // Depression (weight -3)
    ELIXHAUSER_CODE_TO_WEIGHT["370143000"] = -3.0f;  // Depression
    ELIXHAUSER_CODE_TO_WEIGHT["36923009"] = -3.0f;   // Depression

    // Add additional codes for CKD and diabetes used in subset identification
    ELIXHAUSER_CODE_TO_WEIGHT["431855005"] = 5.0f;  // Chronic kidney disease
    ELIXHAUSER_CODE_TO_WEIGHT["431856006"] = 5.0f;  // Chronic kidney disease stage 1
    ELIXHAUSER_CODE_TO_WEIGHT["433144002"] = 5.0f;  // Chronic kidney disease stage 2
    ELIXHAUSER_CODE_TO_WEIGHT["431857002"] = 5.0f;  // Chronic kidney disease stage 3
    ELIXHAUSER_CODE_TO_WEIGHT["46177005"] = 5.0f;   // End stage renal disease
    
    ELIXHAUSER_CODE_TO_WEIGHT["73211009"] = 0.5f;   // Diabetes mellitus
    ELIXHAUSER_CODE_TO_WEIGHT["46635009"] = 0.5f;   // Diabetes mellitus type 1
    ELIXHAUSER_CODE_TO_WEIGHT["190330002"] = 0.5f;  // Type 2 diabetes mellitus

    std::cout << "[INFO] Added " << ELIXHAUSER_CODE_TO_WEIGHT.size() << " SNOMED CT codes to Elixhauser dictionary\n";
}

void initializeObsAbnormalDirect() {
    std::cout << "[INFO] Initializing observation normal range lookup\n";
    
    // Normal ranges for common observations - these match health_index.py
    OBS_NORMAL_RANGES = {
        {"Systolic Blood Pressure", {90.0, 120.0}},
        {"Diastolic Blood Pressure", {60.0, 80.0}},
        {"Body Mass Index", {18.5, 24.9}},
        {"Blood Glucose Level", {70.0, 99.0}},
        {"Heart Rate", {60.0, 100.0}}
    };
}

// Helper function to check if observation value is abnormal
bool isAbnormalObsFast(const std::string &obsDescription, double value) {
    // Fast path using direct lookup table
    auto it = OBS_NORMAL_RANGES.find(obsDescription);
    if (it != OBS_NORMAL_RANGES.end()) {
        auto [min_val, max_val] = it->second;
        return value < min_val || value > max_val;
    }
    
    // Try alternate common descriptions
    if (obsDescription.find("systolic") != std::string::npos || 
        obsDescription.find("SYSTOLIC") != std::string::npos) {
        return value < 90.0 || value > 120.0;
    }
    
    if (obsDescription.find("diastolic") != std::string::npos || 
        obsDescription.find("DIASTOLIC") != std::string::npos) {
        return value < 60.0 || value > 80.0;
    }
    
    if (obsDescription.find("BMI") != std::string::npos || 
        obsDescription.find("body mass index") != std::string::npos) {
        return value < 18.5 || value > 24.9;
    }
    
    if (obsDescription.find("glucose") != std::string::npos || 
        obsDescription.find("GLUCOSE") != std::string::npos) {
        return value < 70.0 || value > 99.0;
    }
    
    if (obsDescription.find("heart rate") != std::string::npos || 
        obsDescription.find("HEART RATE") != std::string::npos) {
        return value < 60.0 || value > 100.0;
    }
    
    // For unrecognized observations, fall back to domain-specific thresholds
    if (value > 500.0) return true; // Large values are usually abnormal
    
    return false;  // Don't mark as abnormal if we don't recognize it
}

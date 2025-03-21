#include "MedicalDictionaries.h"
#include <iostream>

// Global variables - defined here
std::unordered_map<std::string, float> CHARLSON_CODE_TO_WEIGHT;
std::unordered_map<std::string, float> ELIXHAUSER_CODE_TO_WEIGHT;
std::unordered_map<std::string, std::pair<double, double>> OBS_ABNORMAL_THRESHOLDS;

// Implementation of the functions declared in the header
void initializeDirectLookups() {
    // Clear existing values first
    CHARLSON_CODE_TO_WEIGHT.clear();
    
    // Add common ICD-10 codes for Charlson Index with expanded coverage
    // Diabetes codes - expanded with more specific codes
    CHARLSON_CODE_TO_WEIGHT.insert({"E08", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E09", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E10", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E11", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E11.9", 1.0f}); // Type 2 diabetes without complications
    CHARLSON_CODE_TO_WEIGHT.insert({"E13", 1.0f});
    
    // Also add full Synthea condition codes (not just the prefix)
    CHARLSON_CODE_TO_WEIGHT.insert({"44054006", 1.0f});    // Type 2 diabetes 
    CHARLSON_CODE_TO_WEIGHT.insert({"73211009", 1.0f});    // Diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT.insert({"46635009", 1.0f});    // Type 1 diabetes
    CHARLSON_CODE_TO_WEIGHT.insert({"15777000", 1.0f});    // Prediabetes
    CHARLSON_CODE_TO_WEIGHT.insert({"237599002", 1.0f});   // Insulin-treated type 2 diabetes
    
    // Heart failure 
    CHARLSON_CODE_TO_WEIGHT.insert({"I50", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"I50.0", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"I50.1", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"I50.9", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"84114007", 1.0f});    // Heart failure
    CHARLSON_CODE_TO_WEIGHT.insert({"88805009", 1.0f});    // Chronic heart failure
    
    // COPD and respiratory diseases
    CHARLSON_CODE_TO_WEIGHT.insert({"J44", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"J44.9", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"J45", 1.0f});         // Asthma
    CHARLSON_CODE_TO_WEIGHT.insert({"J45.9", 1.0f});       // Asthma, unspecified
    CHARLSON_CODE_TO_WEIGHT.insert({"13645005", 1.0f});    // COPD
    CHARLSON_CODE_TO_WEIGHT.insert({"195967001", 1.0f});   // Asthma
    
    // Hypertension
    CHARLSON_CODE_TO_WEIGHT.insert({"I10", 0.5f}); 
    CHARLSON_CODE_TO_WEIGHT.insert({"59621000", 0.5f});    // Essential hypertension
    CHARLSON_CODE_TO_WEIGHT.insert({"38341003", 0.5f});    // Hypertensive disorder
    
    // Cardiovascular diseases
    CHARLSON_CODE_TO_WEIGHT.insert({"I25.9", 1.0f});       // Chronic ischemic heart disease
    CHARLSON_CODE_TO_WEIGHT.insert({"53741008", 1.0f});    // Coronary arteriosclerosis
    CHARLSON_CODE_TO_WEIGHT.insert({"22298006", 1.0f});    // Myocardial infarction
    CHARLSON_CODE_TO_WEIGHT.insert({"399211009", 1.0f});   // History of myocardial infarction
    
    // Renal/Kidney disease
    CHARLSON_CODE_TO_WEIGHT.insert({"N18", 2.0f});         // Chronic kidney disease
    CHARLSON_CODE_TO_WEIGHT.insert({"N18.9", 2.0f});       // Chronic kidney disease, unspecified
    CHARLSON_CODE_TO_WEIGHT.insert({"433144002", 2.0f});   // Chronic kidney disease stage 2
    CHARLSON_CODE_TO_WEIGHT.insert({"431855005", 2.0f});   // Chronic kidney disease stage 1
    
    // Cancer
    CHARLSON_CODE_TO_WEIGHT.insert({"C50", 2.0f});         // Breast cancer
    CHARLSON_CODE_TO_WEIGHT.insert({"C61", 2.0f});         // Prostate cancer
    CHARLSON_CODE_TO_WEIGHT.insert({"C34", 2.0f});         // Lung cancer
    CHARLSON_CODE_TO_WEIGHT.insert({"C18", 2.0f});         // Colon cancer
    CHARLSON_CODE_TO_WEIGHT.insert({"254837009", 2.0f});   // Malignant tumor of breast
    CHARLSON_CODE_TO_WEIGHT.insert({"254901000", 2.0f});   // Tumor of prostate
    
    // Text descriptions (case-insensitive)
    CHARLSON_CODE_TO_WEIGHT.insert({"diabetes", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"heart failure", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"copd", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"hypertension", 0.5f});
    CHARLSON_CODE_TO_WEIGHT.insert({"renal disease", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"kidney disease", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"asthma", 1.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"cancer", 2.0f});
    
    // ===== IMPORTANT: Match the Python snomed_groups with appropriate weights =====
    
    // Cardiovascular Diseases (weight: 3 in Python)
    CHARLSON_CODE_TO_WEIGHT.insert({"53741008", 3.0f});  // Coronary arteriosclerosis
    CHARLSON_CODE_TO_WEIGHT.insert({"445118002", 3.0f}); // Pulmonary embolism
    CHARLSON_CODE_TO_WEIGHT.insert({"59621000", 3.0f});  // Essential hypertension
    CHARLSON_CODE_TO_WEIGHT.insert({"22298006", 3.0f});  // Myocardial infarction
    CHARLSON_CODE_TO_WEIGHT.insert({"56265001", 3.0f});  // Heart disease
    
    // Respiratory Diseases (weight: 2 in Python)
    CHARLSON_CODE_TO_WEIGHT.insert({"19829001", 2.0f});  // Disorders of lung
    CHARLSON_CODE_TO_WEIGHT.insert({"233604007", 2.0f}); // Pneumonia
    CHARLSON_CODE_TO_WEIGHT.insert({"118940003", 2.0f}); // Disorder of respiratory system
    CHARLSON_CODE_TO_WEIGHT.insert({"409622000", 2.0f}); // Respiratory hypersensitivity
    CHARLSON_CODE_TO_WEIGHT.insert({"13645005", 2.0f});  // COPD
    
    // Diabetes (weight: 2 in Python)
    CHARLSON_CODE_TO_WEIGHT.insert({"44054006", 2.0f});  // Type 2 diabetes
    CHARLSON_CODE_TO_WEIGHT.insert({"73211009", 2.0f});  // Diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT.insert({"46635009", 2.0f});  // Type 1 diabetes
    CHARLSON_CODE_TO_WEIGHT.insert({"190330002", 2.0f}); // Type 1 diabetes mellitus
    CHARLSON_CODE_TO_WEIGHT.insert({"15777000", 2.0f});  // Prediabetes
    CHARLSON_CODE_TO_WEIGHT.insert({"237599002", 2.0f}); // Insulin-treated diabetes
    
    // Also add ICD-10 codes for diabetes
    CHARLSON_CODE_TO_WEIGHT.insert({"E08", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E09", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E10", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E11", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"E13", 2.0f});
    
    // Add the Python case-insensitive text matches
    CHARLSON_CODE_TO_WEIGHT.insert({"diabetes", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"heart failure", 3.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"copd", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"hypertension", 3.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"renal disease", 2.0f});
    CHARLSON_CODE_TO_WEIGHT.insert({"kidney disease", 2.0f});
    
    std::cout << "[INFO] Initialized Charlson dictionary with " 
              << CHARLSON_CODE_TO_WEIGHT.size() << " entries\n";
}

// Initialize Elixhauser indices
void initializeElixhauserLookups() {
    ELIXHAUSER_CODE_TO_WEIGHT.clear();
    
    // Diabetes codes with expanded coverage
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"E08", 0.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"E09", 0.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"E10", 0.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"E11", 0.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"E11.9", 0.7f}); // Type 2 diabetes without complications
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"E13", 0.5f});
    
    // SNOMED codes for diabetes
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"73211009", 0.5f}); // Diabetes mellitus
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"44054006", 0.5f}); // Type 2 diabetes
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"46635009", 0.5f}); // Type 1 diabetes
    
    // Heart failure
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"I50", 1.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"I50.0", 1.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"I50.1", 1.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"I50.9", 1.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"84114007", 1.5f});  // Heart failure
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"88805009", 1.5f});  // Chronic heart failure
    
    // Hypertension
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"I10", 0.3f}); // Essential (primary) hypertension
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"59621000", 0.3f}); // Essential hypertension
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"38341003", 0.3f}); // Hypertensive disorder
    
    // COPD
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"J44", 0.9f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"J44.9", 0.9f}); // COPD, unspecified
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"13645005", 0.9f}); // COPD
    
    // Common text descriptions
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"diabetes", 0.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"heart failure", 1.5f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"hypertension", 0.7f});
    ELIXHAUSER_CODE_TO_WEIGHT.insert({"copd", 0.9f});
    
    std::cout << "[INFO] Initialized Elixhauser dictionary with " 
              << ELIXHAUSER_CODE_TO_WEIGHT.size() << " entries\n";
}

// Initialize abnormal observation thresholds
void initializeObsAbnormalDirect() {
    OBS_ABNORMAL_THRESHOLDS.clear();
    
    // Format: {"description", {min_normal, max_normal}}
    OBS_ABNORMAL_THRESHOLDS.insert({"Systolic Blood Pressure", {90.0, 140.0}});
    OBS_ABNORMAL_THRESHOLDS.insert({"Diastolic Blood Pressure", {60.0, 90.0}});
    OBS_ABNORMAL_THRESHOLDS.insert({"Heart Rate", {60.0, 100.0}});
    OBS_ABNORMAL_THRESHOLDS.insert({"Respiratory Rate", {12.0, 20.0}});
    OBS_ABNORMAL_THRESHOLDS.insert({"Body Mass Index", {18.5, 24.9}});
    OBS_ABNORMAL_THRESHOLDS.insert({"Body Temperature", {36.5, 37.5}}); // Celsius
    OBS_ABNORMAL_THRESHOLDS.insert({"Glucose", {70.0, 126.0}});
    OBS_ABNORMAL_THRESHOLDS.insert({"A1C", {4.0, 6.5}});
    OBS_ABNORMAL_THRESHOLDS.insert({"Oxygen Saturation", {94.0, 100.0}});
    
    std::cout << "[INFO] Initialized abnormal observation thresholds with "
              << OBS_ABNORMAL_THRESHOLDS.size() << " entries\n";
}

bool isAbnormalObsFast(const std::string& description, double value) {
    // First try exact match
    auto it = OBS_ABNORMAL_THRESHOLDS.find(description);
    if (it != OBS_ABNORMAL_THRESHOLDS.end()) {
        auto [minNormal, maxNormal] = it->second;
        return value < minNormal || value > maxNormal;
    }
    
    // Try normalized description
    std::string lowerDesc = description;
    std::transform(lowerDesc.begin(), lowerDesc.end(), lowerDesc.begin(), 
                   [](unsigned char c){ return std::tolower(c); });
    
    // Use the same mapping as in the Python code
    if (lowerDesc.find("systolic") != std::string::npos || 
        lowerDesc.find("blood pressure") != std::string::npos) {
        return value < 90.0 || value > 140.0;  // Match Python value ranges
    }
    
    if (lowerDesc.find("diastolic") != std::string::npos) {
        return value < 60.0 || value > 90.0;
    }
    
    if (lowerDesc.find("bmi") != std::string::npos || lowerDesc.find("body mass index") != std::string::npos) {
        return value < 18.5 || value > 24.9;
    }
    
    if (lowerDesc.find("glucose") != std::string::npos) {
        return value < 70.0 || value > 99.0;
    }
    
    if (lowerDesc.find("heart rate") != std::string::npos) {
        return value < 60.0 || value > 100.0;
    }
    
    // Enhanced matching based on Python's observation_mappings
    if (lowerDesc.find("oxygen saturation") != std::string::npos) {
        return value < 94.0 || value > 100.0;
    }
    
    // Default
    return false;
}

double findGroupWeightFast(const std::string& code) {
    // First check if code is directly in Charlson dictionary
    auto charlsonIt = CHARLSON_CODE_TO_WEIGHT.find(code);
    if (charlsonIt != CHARLSON_CODE_TO_WEIGHT.end()) {
        return static_cast<double>(charlsonIt->second);
    }
    
    // Check Elixhauser if not in Charlson
    auto elixIt = ELIXHAUSER_CODE_TO_WEIGHT.find(code);
    if (elixIt != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
        return static_cast<double>(elixIt->second);
    }
    
    // Try prefix matching for both dictionaries
    if (code.length() >= 3) {
        std::string prefix3 = code.substr(0, 3);
        
        charlsonIt = CHARLSON_CODE_TO_WEIGHT.find(prefix3);
        if (charlsonIt != CHARLSON_CODE_TO_WEIGHT.end()) {
            return static_cast<double>(charlsonIt->second);
        }
        
        elixIt = ELIXHAUSER_CODE_TO_WEIGHT.find(prefix3);
        if (elixIt != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
            return static_cast<double>(elixIt->second);
        }
    }
    
    // Not found
    return 0.0;
}

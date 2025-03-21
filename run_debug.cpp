#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include "DataStructures.h"
#include "FileProcessing.h"
#include "MedicalDictionaries.h"
#include "HealthIndex.h"

namespace fs = std::filesystem;

// Test function to validate the medical dictionaries
void testMedicalDictionaries() {
    std::cout << "===== Testing Medical Dictionaries =====\n";
    
    // Initialize dictionaries
    initializeDirectLookups();
    initializeElixhauserLookups();
    initializeObsAbnormalDirect();
    
    // Test common codes for diabetes
    std::vector<std::string> diabetesCodes = {
        "E11.9", "E10", "E11", "44054006", "73211009", "46635009", "diabetes"
    };
    
    std::cout << "\nTesting diabetes codes in Charlson dictionary:\n";
    for (const auto& code : diabetesCodes) {
        auto it = CHARLSON_CODE_TO_WEIGHT.find(code);
        if (it != CHARLSON_CODE_TO_WEIGHT.end()) {
            std::cout << "  FOUND: " << code << " -> " << it->second << "\n";
        } else {
            std::cout << "  NOT FOUND: " << code << "\n";
        }
    }
    
    std::cout << "\nTesting diabetes codes in Elixhauser dictionary:\n";
    for (const auto& code : diabetesCodes) {
        auto it = ELIXHAUSER_CODE_TO_WEIGHT.find(code);
        if (it != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
            std::cout << "  FOUND: " << code << " -> " << it->second << "\n";
        } else {
            std::cout << "  NOT FOUND: " << code << "\n";
        }
    }
    
    // Test common observation descriptions
    std::vector<std::string> obsDescs = {
        "Systolic Blood Pressure", "Diastolic Blood Pressure", 
        "Heart Rate", "Body Mass Index", "Glucose"
    };
    
    std::cout << "\nTesting observation thresholds:\n";
    for (const auto& desc : obsDescs) {
        auto it = OBS_ABNORMAL_THRESHOLDS.find(desc);
        if (it != OBS_ABNORMAL_THRESHOLDS.end()) {
            std::cout << "  FOUND: " << desc << " -> [" << it->second.first 
                      << ", " << it->second.second << "]\n";
        } else {
            std::cout << "  NOT FOUND: " << desc << "\n";
        }
    }
}

// Test function to process a condition file and show matching rates
void testConditionMatching(const std::string& conditionFile) {
    if (!fs::exists(conditionFile)) {
        std::cerr << "Condition file not found: " << conditionFile << "\n";
        return;
    }
    
    std::cout << "\n===== Testing Condition Code Matching with " << conditionFile << " =====\n";
    
    int totalConditions = 0;
    int matchedCharlson = 0;
    int matchedElixhauser = 0;
    int matchedDescription = 0;
    
    ThreadSafeCounter charlsonCounter;
    ThreadSafeCounter elixhauserCounter;
    
    processConditionsInBatches(conditionFile, [&](const ConditionRow &cRow) {
        totalConditions++;
        
        // Try exact code match for Charlson
        auto charlsonIt = CHARLSON_CODE_TO_WEIGHT.find(cRow.CODE);
        if (charlsonIt != CHARLSON_CODE_TO_WEIGHT.end()) {
            matchedCharlson++;
            charlsonCounter.addFloat(cRow.PATIENT, charlsonIt->second);
        }
        
        // Try exact code match for Elixhauser
        auto elixhauserIt = ELIXHAUSER_CODE_TO_WEIGHT.find(cRow.CODE);
        if (elixhauserIt != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
            matchedElixhauser++;
            elixhauserCounter.addFloat(cRow.PATIENT, elixhauserIt->second);
        }
        
        // Try description-based matching
        std::string lowerDesc = cRow.DESCRIPTION;
        std::transform(lowerDesc.begin(), lowerDesc.end(), lowerDesc.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        
        if (lowerDesc.find("diabetes") != std::string::npos) {
            matchedDescription++;
            charlsonCounter.addFloat(cRow.PATIENT, 2.0f);
            elixhauserCounter.addFloat(cRow.PATIENT, 0.5f);
        }
        else if (lowerDesc.find("heart failure") != std::string::npos) {
            matchedDescription++;
            charlsonCounter.addFloat(cRow.PATIENT, 3.0f);
            elixhauserCounter.addFloat(cRow.PATIENT, 1.5f);
        }
        else if (lowerDesc.find("copd") != std::string::npos ||
                 lowerDesc.find("chronic obstructive pulmonary") != std::string::npos) {
            matchedDescription++;
            charlsonCounter.addFloat(cRow.PATIENT, 2.0f);
            elixhauserCounter.addFloat(cRow.PATIENT, 0.9f);
        }
    });
    
    // Print matching statistics
    std::cout << "Total conditions processed: " << totalConditions << "\n";
    std::cout << "Matched Charlson codes: " << matchedCharlson 
              << " (" << (totalConditions > 0 ? (matchedCharlson * 100.0 / totalConditions) : 0) << "%)\n";
    std::cout << "Matched Elixhauser codes: " << matchedElixhauser 
              << " (" << (totalConditions > 0 ? (matchedElixhauser * 100.0 / totalConditions) : 0) << "%)\n";
    std::cout << "Matched by description: " << matchedDescription 
              << " (" << (totalConditions > 0 ? (matchedDescription * 100.0 / totalConditions) : 0) << "%)\n";
    
    // Show some patient scores
    std::cout << "\nSample patient scores:\n";
    
    std::set<std::string> uniquePatients;
    for (const auto& entry : charlsonCounter.internalMap()) {
        uniquePatients.insert(entry.first);
    }
    
    int count = 0;
    for (const auto& patientId : uniquePatients) {
        if (count++ < 10) { // Show first 10 patients
            float charlsonScore = charlsonCounter.getFloat(patientId);
            float elixhauserScore = elixhauserCounter.getFloat(patientId);
            std::cout << "Patient " << patientId 
                      << ": Charlson=" << charlsonScore 
                      << ", Elixhauser=" << elixhauserScore << "\n";
        }
    }
}

int main() {
    testMedicalDictionaries();
    
    // Look for condition files in Data directory
    std::vector<std::string> condFiles;
    fs::path dataDir = "Data";
    
    if (fs::exists(dataDir) && fs::is_directory(dataDir)) {
        for (const auto& entry : fs::directory_iterator(dataDir)) {
            if (entry.is_regular_file() && 
                entry.path().filename().string().find("conditions") != std::string::npos) {
                condFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (!condFiles.empty()) {
        testConditionMatching(condFiles[0]); // Test with the first condition file found
    } else {
        std::cout << "No condition files found in Data directory.\n";
    }
    
    std::cout << "\nDone!\n";
    return 0;
}

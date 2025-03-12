#include "FeatureUtils.h"
#include "DataStructures.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

// Retrieves feature column names based on feature configuration
std::vector<std::string> getFeatureCols(const std::string& featureConfig) {
    std::cout << "[INFO] Getting feature columns for configuration: " << featureConfig << std::endl;
    
    std::vector<std::string> features;
    
    // Define features based on configuration
    if (featureConfig == "combined_all") {
        features = {
            "Id", "Age", "Gender", "CharlsonIndex", "ElixhauserIndex", 
            "Comorbidity_Score", "Hospitalizations_Count", "Medications_Count",
            "Abnormal_Observations_Count", "Health_Index"
        };
    } else if (featureConfig == "basic") {
        features = {"Id", "Age", "Gender", "CharlsonIndex"};
    } else {
        std::cout << "[WARNING] Unknown feature configuration: " << featureConfig << ". Using default features." << std::endl;
        features = {"Id", "Age", "Gender", "Health_Index"};
    }
    
    std::cout << "[INFO] Selected " << features.size() << " features." << std::endl;
    return features;
}

// Writes patient features to a CSV file based on specified feature columns
void writeFeaturesCSV(const std::vector<PatientRecord>& patients, 
                      const std::string& filename, 
                      const std::vector<std::string>& features) {
    std::cout << "[INFO] Writing features to CSV file: " << filename << std::endl;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    for (size_t i = 0; i < features.size(); ++i) {
        file << features[i];
        if (i < features.size() - 1) file << ",";
    }
    file << "\n";
    
    // Write data
    for (const auto& patient : patients) {
        for (size_t i = 0; i < features.size(); ++i) {
            const std::string& feature = features[i];
            
            if (feature == "Id") {
                file << patient.Id;
            } else if (feature == "Age") {
                file << patient.Age;
            } else if (feature == "Gender") {
                file << patient.Gender;
            } else if (feature == "CharlsonIndex") {
                file << patient.CharlsonIndex;
            } else if (feature == "ElixhauserIndex") {
                file << patient.ElixhauserIndex;
            } else if (feature == "Comorbidity_Score") {
                file << patient.Comorbidity_Score;
            } else if (feature == "Hospitalizations_Count") {
                file << patient.Hospitalizations_Count;
            } else if (feature == "Medications_Count") {
                file << patient.Medications_Count;
            } else if (feature == "Abnormal_Observations_Count") {
                file << patient.Abnormal_Observations_Count;
            } else if (feature == "Health_Index") {
                file << patient.Health_Index;
            } else {
                file << "N/A";
            }
            
            if (i < features.size() - 1) file << ",";
        }
        file << "\n";
    }
    
    std::cout << "[INFO] Wrote " << patients.size() << " patient records to " << filename << std::endl;
}

// Saves comprehensive patient data to a CSV file
void saveFinalDataCSV(const std::vector<PatientRecord>& patients, const std::string& filename) {
    std::cout << "[INFO] Saving comprehensive data to CSV file: " << filename << std::endl;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header with all fields
    file << "Id,Birthdate,Gender,Age,CharlsonIndex,ElixhauserIndex,Comorbidity_Score,"
         << "Hospitalizations_Count,Medications_Count,Abnormal_Observations_Count,Health_Index\n";
    
    // Write data
    for (const auto& patient : patients) {
        file << patient.Id << ","
             << patient.Birthdate << ","
             << patient.Gender << ","
             << patient.Age << ","
             << patient.CharlsonIndex << ","
             << patient.ElixhauserIndex << ","
             << patient.Comorbidity_Score << ","
             << patient.Hospitalizations_Count << ","
             << patient.Medications_Count << ","
             << patient.Abnormal_Observations_Count << ","
             << patient.Health_Index << "\n";
    }
    
    std::cout << "[INFO] Wrote all data for " << patients.size() << " patient records to " << filename << std::endl;
}

void cleanupFiles(const std::vector<std::string> &files) {
    for (const auto &f : files) {
        std::remove(f.c_str());
    }
}

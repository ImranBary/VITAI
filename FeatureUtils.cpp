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
                      const std::vector<std::string>& columns) {
    std::cout << "[INFO] Writing features to CSV file: " << filename << "\n";
    
    // Convert column names to lowercase for model compatibility
    std::vector<std::string> lowercaseColumns = convertFeatureNamesToLowercase(columns);
    
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "[ERROR] Could not open file for writing: " << filename << "\n";
        return;
    }
    
    // Write header with lowercase column names
    outFile << "Id";
    for (const auto& col : lowercaseColumns) {
        outFile << "," << col;
    }
    outFile << "\n";
    
    // Write data rows
    for (const auto& p : patients) {
        outFile << p.Id;
        for (const auto& col : columns) { // Use original columns for field lookup
            // Use the original capitalized column name to access PatientRecord fields
            // but the lowercase version was written to the header
            outFile << "," << getPatientFieldByName(p, col);
        }
        outFile << "\n";
    }
    
    outFile.close();
    std::cout << "[INFO] Wrote " << patients.size() << " patient records to " << filename << "\n";
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

// Add a helper function to access PatientRecord fields by name
std::string getPatientFieldByName(const PatientRecord& patient, const std::string& fieldName) {
    // Map field names to their values in the PatientRecord
    if (fieldName == "Age") return std::to_string(patient.Age);
    if (fieldName == "Gender") return patient.Gender;
    if (fieldName == "CharlsonIndex") return std::to_string(patient.CharlsonIndex);
    if (fieldName == "ElixhauserIndex") return std::to_string(patient.ElixhauserIndex);
    if (fieldName == "Comorbidity_Score") return std::to_string(patient.Comorbidity_Score);
    if (fieldName == "Hospitalizations_Count") return std::to_string(patient.Hospitalizations_Count);
    if (fieldName == "Medications_Count") return std::to_string(patient.Medications_Count);
    if (fieldName == "Abnormal_Observations_Count") return std::to_string(patient.Abnormal_Observations_Count);
    if (fieldName == "Health_Index") return std::to_string(patient.Health_Index);
    // Add other fields as needed
    
    // Return empty string for unknown fields
    std::cerr << "[WARNING] Unknown field name: " << fieldName << "\n";
    return "";
}

#include "FeatureUtils.h"
#include "DataStructures.h"
#include "FileProcessing.h"  // Add this include for normalizeFieldName
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

// Define a function to get the exact feature columns expected by the model
std::vector<std::string> getModelExpectedFeatures() {
    // Add 'Id' as the first feature in the list
    return {
        "Id",   // Add the Id column as the first column
        "AGE", 
        "DECEASED", 
        "GENDER", 
        "RACE", 
        "ETHNICITY", 
        "MARITAL", 
        "HEALTHCARE_EXPENSES", 
        "HEALTHCARE_COVERAGE", 
        "INCOME", 
        "Hospitalizations_Count", 
        "Medications_Count", 
        "Abnormal_Observations_Count"
    };
}

// Implementation of getFeatureCols
std::vector<std::string> getFeatureCols(const std::string& featureConfig) {
    // For inference, we need to make sure we include all expected model features
    if (featureConfig == "combined_all" || featureConfig == "combined") {
        return getModelExpectedFeatures();
    }
    
    // Handle other feature configurations as needed
    // ...

    // Default to model expected features
    return getModelExpectedFeatures();
}

// Helper function to get patient field value based on the model's expected column names
std::string getPatientFieldValue(const PatientRecord& p, const std::string& fieldName) {
    // Map the model's expected column names to actual PatientRecord field names/values
    if (fieldName == "Id") return p.Id;  // Add this line to handle the Id field
    if (fieldName == "AGE") return std::to_string(p.Age);
    if (fieldName == "DECEASED") return p.IsDeceased ? "1" : "0";  // Assuming the field is actually IsDeceased
    if (fieldName == "GENDER") return p.Gender;
    if (fieldName == "RACE") return p.RaceCategory;    // Assuming the field is RaceCategory
    if (fieldName == "ETHNICITY") return p.EthnicityCategory;  // Assuming the field is EthnicityCategory
    if (fieldName == "MARITAL") return p.MaritalStatus;  // Assuming the field is MaritalStatus
    if (fieldName == "HEALTHCARE_EXPENSES") return std::to_string(p.HealthcareExpenses);  // Adjusted field name
    if (fieldName == "HEALTHCARE_COVERAGE") return std::to_string(p.HealthcareCoverage);  // Adjusted field name
    if (fieldName == "INCOME") return std::to_string(p.Income);
    if (fieldName == "Hospitalizations_Count") return std::to_string(p.Hospitalizations_Count);
    if (fieldName == "Medications_Count") return std::to_string(p.Medications_Count);
    if (fieldName == "Abnormal_Observations_Count") return std::to_string(p.Abnormal_Observations_Count);
    
    // Default case - unknown field
    std::cerr << "[WARNING] Unknown field requested: " << fieldName << "\n";
    return "";
}

// Enhanced function to write features to CSV ensuring all required columns are included
void writeFeaturesCSV(const std::vector<PatientRecord>& patients, 
                     const std::string& filename,
                     const std::vector<std::string>& features) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "[ERROR] Could not open " << filename << " for writing.\n";
        return;
    }

    // Write header with exact column names
    bool first = true;
    for (const auto& feature : features) {
        if (!first) outFile << ",";
        outFile << feature;
        first = false;
    }
    outFile << "\n";

    // Write patient data
    for (const auto& patient : patients) {
        first = true;
        for (const auto& feature : features) {
            if (!first) outFile << ",";
            outFile << getPatientFieldValue(patient, feature);
            first = false;
        }
        outFile << "\n";
    }
    
    outFile.close();
    std::cout << "[INFO] Successfully wrote " << patients.size() << " patients to " << filename 
              << " with " << features.size() << " features.\n";
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

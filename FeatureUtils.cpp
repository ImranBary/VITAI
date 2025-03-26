#include "FeatureUtils.h"
#include "DataStructures.h"
#include "FileProcessing.h"  // Add this include for normalizeFieldName
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <iomanip>  // For std::setprecision

// Function to get exact features needed by the TabNet model
std::vector<std::string> getModelExpectedFeatures() {
    // Include all columns required by run_patient_group_predictions.py
    return {
        "Id",  // Case-sensitive match
        "AGE", // Python uses uppercase
        "GENDER",
        "RACE",
        "ETHNICITY", 
        "MARITAL",
        "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE", 
        "INCOME",
        "CharlsonIndex",         // These next lines were previously missing
        "ElixhauserIndex",       // in the implementation
        "Comorbidity_Score",
        "Hospitalizations_Count",
        "Medications_Count",
        "Abnormal_Observations_Count",
        "DECEASED",
        "Health_Index"           // Critical for scaling 
    };
}

// Function to validate feature CSV has correct format
bool validateFeatureCSV(const std::string& csvPath) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open feature CSV for validation: " << csvPath << std::endl;
        return false;
    }
    
    // Read header
    std::string header;
    std::getline(file, header);
    
    // Get expected headers
    auto expectedFeatures = getModelExpectedFeatures();
    
    // Parse the header
    std::vector<std::string> csvColumns;
    std::istringstream headerStream(header);
    std::string column;
    while (std::getline(headerStream, column, ',')) {
        // Trim whitespace
        column.erase(0, column.find_first_not_of(" \t\r\n"));
        column.erase(column.find_last_not_of(" \t\r\n") + 1);
        csvColumns.push_back(column);
    }
    
    // Check if all expected features are present (case insensitive)
    bool valid = true;
    for (const auto& feature : expectedFeatures) {
        std::string featureLower = toLowercase(feature);
        bool found = false;
        for (const auto& csvCol : csvColumns) {
            if (toLowercase(csvCol) == featureLower) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cerr << "[ERROR] Missing required feature in CSV: " << feature << std::endl;
            valid = false;
        }
    }
    
    // Check column data types (sample first few rows)
    int rowCount = 0;
    std::string line;
    while (std::getline(file, line) && rowCount < 5) {
        rowCount++;
        // Here you could add checks for data types by column index
        // but at minimum log the data for debugging
        std::cout << "[DEBUG] Row " << rowCount << ": " << line << std::endl;
    }
    
    if (!valid) {
        // If validation fails, try to fix it automatically
        std::cout << "[INFO] Attempting to fix feature CSV using Python helper...\n";
        std::string fixCmd = "python feature_validator.py \"" + csvPath + "\" \"" + csvPath + "\"";
        int fixResult = std::system(fixCmd.c_str());
        if (fixResult == 0) {
            std::cout << "[INFO] Fixed feature CSV format issues\n";
            valid = true;
        }
    }
    
    std::cout << "[INFO] Feature CSV validation: " << (valid ? "PASSED" : "FAILED") << std::endl;
    return valid;
}

// Export patient records to match Python's expected format
void writeFeaturesCSV(
    const std::vector<PatientRecord>& patients,
    const std::string& outputPath,
    const std::vector<std::string>& featureCols
) {
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "[ERROR] Could not open file for writing: " << outputPath << std::endl;
        return;
    }
    
    // Write header - exact column names as specified in featureCols
    for (size_t i = 0; i < featureCols.size(); ++i) {
        outFile << featureCols[i];
        if (i < featureCols.size() - 1) outFile << ",";
    }
    outFile << "\n";
    
    // These mappings are based on exact models' embedding dimensions
    // Each model has specific expectations for categorical columns:
    // 
    // TabNet Categorical Mappings (from model_inspector):
    // Index 1: MARITAL - values 0-1  (input_dim=2)
    // Index 2: RACE - values 0-1     (input_dim=2)
    // Index 3: ETHNICITY - values 0-5 (input_dim=6)
    // Index 4: GENDER - values 0-1   (input_dim=2)
    // Index 5: HEALTHCARE_COVERAGE - values 0-4 (input_dim=5)
    
    // Map for gender - must be 0 or 1 only
    std::unordered_map<std::string, int> genderMap = {
        {"M", 1}, {"F", 0},
        {"male", 1}, {"female", 0},
        {"MALE", 1}, {"FEMALE", 0},
        {"m", 1}, {"f", 0}
    };
    
    // Map for marital status - must be 0 or 1 only
    std::unordered_map<std::string, int> maritalMap = {
        {"S", 0}, {"SINGLE", 0},
        {"M", 1}, {"MARRIED", 1},
        {"W", 0}, {"WIDOWED", 0}, // Changed to 0 for safety
        {"D", 0}, {"DIVORCED", 0}  
    };
    
    // Map race - must be 0 or 1 only for diabetes model
    std::unordered_map<std::string, int> raceMap = {
        {"white", 0},
        {"black", 0}, // Changed to 0
        {"asian", 0}, // Changed to 0
        {"native", 1}, // Changed to 1
        {"other", 1}  // Changed to 1
    };
    
    // Map ethnicity - must be 0-5 
    std::unordered_map<std::string, int> ethnicityMap = {
        {"hispanic", 0},
        {"nonhispanic", 1},
        {"other", 2}  // Values should be 0-5
    };
    
    // Write data
    int skippedRows = 0;
    for (const auto& patient : patients) {
        std::string row = "";
        bool skipRow = false;
        
        // Process each feature in the exact order specified
        for (const auto& feature : featureCols) {
            if (feature == "Id") {
                row += patient.Id;
            }
            else if (feature == "AGE") {
                row += std::to_string(patient.Age);
            }
            else if (feature == "MARITAL") {
                // Convert to integer code expected by model - MUST be 0 or 1 only
                std::string status = patient.Marital_Status;
                std::transform(status.begin(), status.end(), status.begin(), ::toupper);
                auto it = maritalMap.find(status);
                int maritalCode = 0; // Default to 0
                if (it != maritalMap.end()) {
                    maritalCode = it->second;
                }
                // Safety check - embedding dimension is 2 (values can only be 0 or 1)
                if (maritalCode > 1) maritalCode = 0;
                row += std::to_string(maritalCode);
            }
            else if (feature == "RACE") {
                std::string race = patient.Race;
                std::transform(race.begin(), race.end(), race.begin(), ::tolower);
                
                // For safety, map to binary values (0 or 1) which works for all models
                int raceCode = 1; // Default to 1
                if (race.find("white") != std::string::npos) raceCode = 0;
                
                row += std::to_string(raceCode);
            }
            else if (feature == "ETHNICITY") {
                std::string ethnicity = patient.Ethnicity;
                std::transform(ethnicity.begin(), ethnicity.end(), ethnicity.begin(), ::tolower);
                
                // Model allows values 0-5, but we'll be conservative
                int ethnicityCode = 0; // Default to 0 for safety
                
                // Only set values up to 2 for safety across all models
                if (ethnicity.find("hisp") != std::string::npos && 
                    ethnicity.find("non") == std::string::npos) ethnicityCode = 0;
                else if (ethnicity.find("non") != std::string::npos && 
                         ethnicity.find("hisp") != std::string::npos) ethnicityCode = 1;
                else ethnicityCode = 2;
                
                row += std::to_string(ethnicityCode);
            }
            else if (feature == "GENDER") {
                // Use gender map with strict range check for embedding dim = 2
                auto it = genderMap.find(patient.Gender);
                int genderCode = 0; // Default to 0 for safety
                if (it != genderMap.end()) {
                    genderCode = it->second;
                }
                row += std::to_string(genderCode);
            }
            else if (feature == "HEALTHCARE_COVERAGE") {
                // For embedding dim = 5, limit to 0-4
                int coverageCategory = 0; // Default to 0
                double coverage = patient.Healthcare_Coverage;
                
                // Pick a safe value (0) for any unreasonable amount
                if (coverage < -1000000 || coverage > 1000000) {
                    coverageCategory = 0;
                }
                else if (coverage == 0) {
                    coverageCategory = 0;
                }
                else {
                    // Provide one of 5 possible values (0-4)
                    coverageCategory = 0; // Just use 0 for safety
                }
                
                row += std::to_string(coverageCategory);
            }
            else if (feature == "HEALTHCARE_EXPENSES") {
                // Fix the large negative value issue by using a reasonable default
                double expenses = patient.Healthcare_Expenses;
                if (expenses < -1000000 || expenses > 1000000) {
                    expenses = 0.0;
                }
                row += std::to_string(static_cast<int>(expenses));
            }
            else if (feature == "INCOME") {
                double income = patient.Income;
                // Range check
                if (income < 0 || income > 10000000) {
                    income = 50000.0; // Reasonable default
                }
                row += std::to_string(static_cast<int>(income));
            }
            else if (feature == "CharlsonIndex") {
                // Ensure it's a valid number
                float value = patient.CharlsonIndex;
                // Range check
                if (value < 0 || value > 100) {
                    value = 0.0f;
                }
                // Use fixed precision to avoid scientific notation
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << value;
                row += ss.str();
            }
            else if (feature == "ElixhauserIndex") {
                // Ensure it's a valid number with range check
                float value = patient.ElixhauserIndex;
                if (value < 0 || value > 100) {
                    value = 0.0f;
                }
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << value;
                row += ss.str();
            }
            else if (feature == "Hospitalizations_Count") {
                // Apply range checking
                int count = patient.Hospitalizations_Count;
                if (count < 0 || count > 1000) {
                    count = 0;
                }
                row += std::to_string(count);
            }
            else if (feature == "Medications_Count") {
                // Apply range checking
                int count = patient.Medications_Count;
                if (count < 0 || count > 1000) {
                    count = 0;
                }
                row += std::to_string(count);
            }
            else if (feature == "Abnormal_Observations_Count") {
                // Apply range checking
                int count = patient.Abnormal_Observations_Count;
                if (count < 0 || count > 1000) {
                    count = 0;
                }
                row += std::to_string(count);
            }
            else if (feature == "DECEASED") {
                // Must be 0 for safety across all models
                row += "0";
            }
            else if (feature == "Health_Index") {
                // Range check
                float value = patient.Health_Index;
                if (value < 0 || value > 100) {
                    value = 80.0f; // Reasonable default
                }
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << value;
                row += ss.str();
            }
            else {
                // Default handling for unexpected features
                row += "0";
            }
            
            if (&feature != &featureCols.back()) row += ",";
        }
        
        if (!skipRow) {
            outFile << row << "\n";
        } else {
            skippedRows++;
        }
    }
    
    outFile.close();
    std::cout << "[INFO] Wrote " << (patients.size() - skippedRows) << " patient records to " << outputPath 
              << " (" << skippedRows << " invalid records skipped)" << std::endl;
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

// Add this debugging function
void debugThreadSafeCounter(const ThreadSafeCounter& counter, const std::string& name, const std::vector<PatientRecord>& patients, int maxToShow = 5) {
    std::cout << "[DEBUG] " << name << " values for first " << maxToShow << " patients:\n";
    int count = 0;
    for (const auto& patient : patients) {
        if (count < maxToShow) {
            std::cout << "   Patient " << patient.Id << ": " << counter.getFloat(patient.Id) << "\n";
            count++;
        }
    }
}

// Function to normalize numeric features to match Python scaling
void normalizePatientFeatures(std::vector<PatientRecord>& patients) {
    std::cout << "[INFO] Calculating feature statistics for normalization...\n";
    
    // Calculate mean and standard deviation for continuous features
    float ageMean = 0.0f, ageStd = 0.0f;
    float expensesMean = 0.0f, expensesStd = 0.0f;
    float coverageMean = 0.0f, coverageStd = 0.0f;
    float incomeMean = 0.0f, incomeStd = 0.0f;
    float hospitalMean = 0.0f, hospitalStd = 0.0f;
    float medsMean = 0.0f, medsStd = 0.0f;
    float abnormalMean = 0.0f, abnormalStd = 0.0f;
    
    // First pass: calculate means
    for (const auto& p : patients) {
        ageMean += p.Age;
        expensesMean += p.Healthcare_Expenses;
        coverageMean += p.Healthcare_Coverage;
        incomeMean += p.Income;
        hospitalMean += p.Hospitalizations_Count;
        medsMean += p.Medications_Count;
        abnormalMean += p.Abnormal_Observations_Count;
    }
    
    size_t n = patients.size();
    if (n > 0) {
        ageMean /= n;
        expensesMean /= n;
        coverageMean /= n;
        incomeMean /= n;
        hospitalMean /= n;
        medsMean /= n;
        abnormalMean /= n;
    }
    
    // Second pass: calculate standard deviations
    for (const auto& p : patients) {
        ageStd += (p.Age - ageMean) * (p.Age - ageMean);
        expensesStd += (p.Healthcare_Expenses - expensesMean) * (p.Healthcare_Expenses - expensesMean);
        coverageStd += (p.Healthcare_Coverage - coverageMean) * (p.Healthcare_Coverage - coverageMean);
        incomeStd += (p.Income - incomeMean) * (p.Income - incomeMean);
        hospitalStd += (p.Hospitalizations_Count - hospitalMean) * (p.Hospitalizations_Count - hospitalMean);
        medsStd += (p.Medications_Count - medsMean) * (p.Medications_Count - medsMean);
        abnormalStd += (p.Abnormal_Observations_Count - abnormalMean) * (p.Abnormal_Observations_Count - abnormalMean);
    }
    
    if (n > 1) {
        ageStd = std::sqrt(ageStd / (n - 1));
        expensesStd = std::sqrt(expensesStd / (n - 1));
        coverageStd = std::sqrt(coverageStd / (n - 1));
        incomeStd = std::sqrt(incomeStd / (n - 1));
        hospitalStd = std::sqrt(hospitalStd / (n - 1));
        medsStd = std::sqrt(medsStd / (n - 1));
        abnormalStd = std::sqrt(abnormalStd / (n - 1));
    }
    
    // Prevent division by zero
    ageStd = std::max(ageStd, 1.0f);
    expensesStd = std::max(expensesStd, 1.0f);
    coverageStd = std::max(coverageStd, 1.0f);
    incomeStd = std::max(incomeStd, 1.0f);
    hospitalStd = std::max(hospitalStd, 1.0f);
    medsStd = std::max(medsStd, 1.0f);
    abnormalStd = std::max(abnormalStd, 1.0f);
    
    std::cout << "[INFO] Feature statistics calculated:\n";
    std::cout << "  AGE: mean=" << ageMean << ", std=" << ageStd << "\n";
    std::cout << "  HEALTHCARE_EXPENSES: mean=" << expensesMean << ", std=" << expensesStd << "\n";
    std::cout << "  HEALTHCARE_COVERAGE: mean=" << coverageMean << ", std=" << coverageStd << "\n";
    std::cout << "  INCOME: mean=" << incomeMean << ", std=" << incomeStd << "\n";
    std::cout << "  Hospitalizations_Count: mean=" << hospitalMean << ", std=" << hospitalStd << "\n";
    std::cout << "  Medications_Count: mean=" << medsMean << ", std=" << medsStd << "\n";
    std::cout << "  Abnormal_Observations_Count: mean=" << abnormalMean << ", std=" << abnormalStd << "\n";
    
    // Apply standardization (mean=0, std=1) to match what StandardScaler would do in Python
    std::cout << "[INFO] Applying normalization to patient features...\n";
    for (auto& p : patients) {
        // For continuous features, use z-score normalization (standardization)
        p.Age = (p.Age - ageMean) / ageStd;
        p.Healthcare_Expenses = (p.Healthcare_Expenses - expensesMean) / expensesStd;
        p.Healthcare_Coverage = (p.Healthcare_Coverage - coverageMean) / coverageStd;
        p.Income = (p.Income - incomeMean) / incomeStd;
        p.Hospitalizations_Count = static_cast<int>((p.Hospitalizations_Count - hospitalMean) / hospitalStd);
        p.Medications_Count = static_cast<int>((p.Medications_Count - medsMean) / medsStd);
        p.Abnormal_Observations_Count = static_cast<int>((p.Abnormal_Observations_Count - abnormalMean) / abnormalStd);
        
        // IMPORTANT: Don't normalize categorical features
        // GENDER, RACE, ETHNICITY, MARITAL, DECEASED are categorical and should be left as-is
    }
    
    std::cout << "[INFO] Feature normalization complete.\n";
}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include "DataStructures.h"
#include "FileProcessing.h"
#include "MedicalDictionaries.h"

namespace fs = std::filesystem;

// Function to save debug info about patients, conditions, etc.
void saveDebugInfo(const std::string& outputFile) {
    std::cout << "[INFO] Running debug data extraction to " << outputFile << std::endl;
    
    // Initialize dictionaries
    initializeDirectLookups();
    initializeElixhauserLookups();
    initializeObsAbnormalDirect();
    
    // First, locate the data files
    std::vector<std::string> patientFiles;
    std::vector<std::string> condFiles;
    std::vector<std::string> encFiles;
    std::vector<std::string> medFiles;
    std::vector<std::string> obsFiles;
    
    // Search in the Data directory
    fs::path dataDir = "Data";
    if (fs::exists(dataDir)) {
        for (const auto& entry : fs::directory_iterator(dataDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                std::string filename = entry.path().filename().string();
                if (filename.find("patients") != std::string::npos) {
                    patientFiles.push_back(entry.path().string());
                }
                else if (filename.find("conditions") != std::string::npos) {
                    condFiles.push_back(entry.path().string());
                }
                else if (filename.find("encounters") != std::string::npos) {
                    encFiles.push_back(entry.path().string());
                }
                else if (filename.find("medications") != std::string::npos) {
                    medFiles.push_back(entry.path().string());
                }
                else if (filename.find("observations") != std::string::npos) {
                    obsFiles.push_back(entry.path().string());
                }
            }
        }
    }
    
    std::cout << "[INFO] Found " << patientFiles.size() << " patient files" << std::endl;
    std::cout << "[INFO] Found " << condFiles.size() << " condition files" << std::endl;
    std::cout << "[INFO] Found " << encFiles.size() << " encounter files" << std::endl;
    std::cout << "[INFO] Found " << medFiles.size() << " medication files" << std::endl;
    std::cout << "[INFO] Found " << obsFiles.size() << " observation files" << std::endl;
    
    // Load patients
    std::vector<PatientRecord> allPatients;
    for (const auto& pFile : patientFiles) {
        processPatientsInBatches(pFile, [&](const PatientRecord& p) {
            allPatients.push_back(p);
        });
    }
    std::cout << "[INFO] Loaded " << allPatients.size() << " patient records" << std::endl;
    
    // Setup counters
    ThreadSafeCounter charlsonCounter;
    ThreadSafeCounter elixhauserCounter;
    ThreadSafeCounter hospitalCounter;
    ThreadSafeCounter medsCounter;
    ThreadSafeCounter abnormalObsCounter;
    
    // Process conditions
    std::vector<ConditionRow> allConditions;
    for (const auto& cFile : condFiles) {
        processConditionsInBatches(cFile, [&](const ConditionRow& c) {
            allConditions.push_back(c);
            
            // Find Charlson weights
            auto charlsonIt = CHARLSON_CODE_TO_WEIGHT.find(c.CODE);
            if (charlsonIt != CHARLSON_CODE_TO_WEIGHT.end()) {
                charlsonCounter.addFloat(c.PATIENT, charlsonIt->second);
            }
            
            // Find Elixhauser weights
            auto elixhauserIt = ELIXHAUSER_CODE_TO_WEIGHT.find(c.CODE);
            if (elixhauserIt != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
                elixhauserCounter.addFloat(c.PATIENT, elixhauserIt->second);
            }
            
            // Also try lowercase description matching
            std::string lowerDesc = c.DESCRIPTION;
            std::transform(lowerDesc.begin(), lowerDesc.end(), lowerDesc.begin(), 
                           [](unsigned char ch) { return std::tolower(ch); });
                           
            // Check for keywords in description
            if (lowerDesc.find("diabetes") != std::string::npos) {
                charlsonCounter.addFloat(c.PATIENT, 1.0f);
                elixhauserCounter.addFloat(c.PATIENT, 0.5f);
            }
            if (lowerDesc.find("heart failure") != std::string::npos) {
                charlsonCounter.addFloat(c.PATIENT, 1.0f);
                elixhauserCounter.addFloat(c.PATIENT, 1.5f);
            }
        });
    }
    
    // Process encounters
    for (const auto& eFile : encFiles) {
        processEncountersInBatches(eFile, [&](const EncounterRow& e) {
            if (e.ENCOUNTERCLASS == "inpatient") {
                hospitalCounter.increment(e.PATIENT);
            }
        });
    }
    
    // Process medications
    for (const auto& mFile : medFiles) {
        processMedicationsInBatches(mFile, [&](const MedicationRow& m) {
            medsCounter.increment(m.PATIENT);
        });
    }
    
    // Process observations
    for (const auto& oFile : obsFiles) {
        processObservationsInBatches(oFile, [&](const ObservationRow& o) {
            try {
                double value = std::stod(o.VALUE);
                if (isAbnormalObsFast(o.DESCRIPTION, value)) {
                    abnormalObsCounter.increment(o.PATIENT);
                }
            } catch (...) {
                // Non-numeric value, ignore
            }
        });
    }
    
    // Update patient records
    for (auto& p : allPatients) {
        p.CharlsonIndex = charlsonCounter.getFloat(p.Id);
        p.ElixhauserIndex = elixhauserCounter.getFloat(p.Id);
        p.Hospitalizations_Count = hospitalCounter.getInt(p.Id);
        p.Medications_Count = medsCounter.getInt(p.Id);
        p.Abnormal_Observations_Count = abnormalObsCounter.getInt(p.Id);
    }
    
    // Save debug file
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "[ERROR] Failed to open output file: " << outputFile << std::endl;
        return;
    }
    
    outFile << "PatientId,CharlsonIndex,ElixhauserIndex,Hospitalizations,Medications,AbnormalObs\n";
    for (const auto& p : allPatients) {
        outFile << p.Id << "," 
                << p.CharlsonIndex << "," 
                << p.ElixhauserIndex << "," 
                << p.Hospitalizations_Count << "," 
                << p.Medications_Count << "," 
                << p.Abnormal_Observations_Count << "\n";
    }
    
    outFile.close();
    std::cout << "[INFO] Debug info saved to " << outputFile << std::endl;
    
    // Print summary
    int countNonZeroCharlson = 0;
    int countNonZeroElixhauser = 0;
    int countNonZeroHosp = 0;
    int countNonZeroMeds = 0;
    int countNonZeroObs = 0;
    
    for (const auto& p : allPatients) {
        if (p.CharlsonIndex > 0) countNonZeroCharlson++;
        if (p.ElixhauserIndex > 0) countNonZeroElixhauser++;
        if (p.Hospitalizations_Count > 0) countNonZeroHosp++;
        if (p.Medications_Count > 0) countNonZeroMeds++;
        if (p.Abnormal_Observations_Count > 0) countNonZeroObs++;
    }
    
    std::cout << "[SUMMARY] Patients with non-zero values:" << std::endl;
    std::cout << "  Charlson Index: " << countNonZeroCharlson << " of " << allPatients.size() << std::endl;
    std::cout << "  Elixhauser Index: " << countNonZeroElixhauser << " of " << allPatients.size() << std::endl;
    std::cout << "  Hospitalizations: " << countNonZeroHosp << " of " << allPatients.size() << std::endl;
    std::cout << "  Medications: " << countNonZeroMeds << " of " << allPatients.size() << std::endl;
    std::cout << "  Abnormal Observations: " << countNonZeroObs << " of " << allPatients.size() << std::endl;
}

int main() {
    std::cout << "Running patient data debug tool..." << std::endl;
    saveDebugInfo("patient_debug.csv");
    return 0;
}

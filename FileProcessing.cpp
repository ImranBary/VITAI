#include "FileProcessing.h"
#include "MedicalDictionaries.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

// Helper function to check if a file exists
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

std::vector<std::string> listCSVFiles(const std::string& directory) {
    std::vector<std::string> files;
    
    std::cout << "[INFO] Listing CSV files in directory: " << directory << std::endl;
    
    try {
        if (!fs::exists(directory)) {
            std::cout << "[WARNING] Directory does not exist: " << directory << std::endl;
            return files;
        }
        
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.path().extension() == ".csv") {
                files.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Filesystem error: " << e.what() << std::endl;
    }
    
    std::cout << "[INFO] Found " << files.size() << " CSV files." << std::endl;
    return files;
}

void processConditionsInBatches(const std::string &path,
                             std::function<void(const ConditionRow&)> callback) {
    if (!fileExists(path)) {
        std::cerr << "Error: File " << path << " not found.\n";
        return;
    }

    std::ifstream file(path);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        ConditionRow condition;
        
        // Parse CSV line into ConditionRow fields
        std::getline(ss, condition.PATIENT, ',');
        std::getline(ss, condition.CODE, ',');
        std::getline(ss, condition.DESCRIPTION, ',');
        // ... parse other fields
        
        callback(condition);
    }
}

void processMedicationsInBatches(const std::string &path,
                               std::function<void(const MedicationRow&)> callback) {
    if (!fileExists(path)) {
        std::cerr << "Error: File " << path << " not found.\n";
        return;
    }

    std::ifstream file(path);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        MedicationRow medication;
        
        // Parse CSV line into MedicationRow fields
        std::getline(ss, medication.PATIENT, ',');
        // ... parse other fields
        
        callback(medication);
    }
}

void processObservationsInBatches(const std::string &path,
                                std::function<void(const ObservationRow&)> callback) {
    if (!fileExists(path)) {
        std::cerr << "Error: File " << path << " not found.\n";
        return;
    }

    std::ifstream file(path);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        ObservationRow observation;
        
        // Parse CSV line into ObservationRow fields
        std::getline(ss, observation.PATIENT, ',');
        std::getline(ss, observation.DESCRIPTION, ',');
        std::getline(ss, observation.VALUE, ',');
        // ... parse other fields
        
        callback(observation);
    }
}

void processProceduresInBatches(const std::string &path,
                              std::function<void(const ProcedureRow&)> callback) {
    BatchProcessor::processFile<ProcedureRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> ProcedureRow {
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            
            ProcedureRow p;
            p.PATIENT = getValue("PATIENT");
            p.ENCOUNTER = getValue("ENCOUNTER");
            p.CODE = getValue("CODE");
            p.DESCRIPTION = getValue("DESCRIPTION");
            return p;
        },
        callback
    );
}

void processEncountersInBatches(const std::string &path,
                              std::function<void(const EncounterRow&)> callback) {
    if (!fileExists(path)) {
        std::cerr << "Error: File " << path << " not found.\n";
        return;
    }

    std::ifstream file(path);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        EncounterRow encounter;
        
        // Parse CSV line into EncounterRow fields
        std::getline(ss, encounter.PATIENT, ',');
        std::getline(ss, encounter.ENCOUNTERCLASS, ',');
        // ... parse other fields
        
        callback(encounter);
    }
}

void processPatientsInBatches(const std::string &path,
                            std::function<void(const PatientRecord&)> callback) {
    if (!fileExists(path)) {
        std::cerr << "Error: File " << path << " not found.\n";
        return;
    }

    std::ifstream file(path);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        PatientRecord patient;
        
        // Parse CSV line into PatientRecord fields
        std::getline(ss, patient.Id, ',');
        // ... parse other fields
        
        callback(patient);
    }
}

void countHospitalizationsInBatches(std::function<void(const std::string&, uint16_t)> countCallback) {
    auto encFiles = listCSVFiles("encounters");
    for (auto &path : encFiles) {
        processEncountersInBatches(path, [&](const EncounterRow &e){
            if (e.ENCOUNTERCLASS == "inpatient") {
                // Increment hospitalization count for this patient
                countCallback(e.PATIENT, 1);
            }
        });
    }
}

void computeCharlsonIndexBatched(std::function<void(const std::string&, float)> scoreCallback) {
    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            auto it = CHARLSON_CODE_TO_WEIGHT.find(c.CODE);
            if (it != CHARLSON_CODE_TO_WEIGHT.end()) {
                scoreCallback(c.PATIENT, it->second);
            }
        });
    }
}

void computeElixhauserIndexBatched(std::function<void(const std::string&, float)> scoreCallback) {
    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            auto it = ELIXHAUSER_CODE_TO_WEIGHT.find(c.CODE);
            if (it != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
                scoreCallback(c.PATIENT, it->second);
            }
        });
    }
}

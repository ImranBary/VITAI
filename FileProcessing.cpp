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
        std::cerr << "[ERROR] File " << path << " not found.\n";
        return;
    }

    std::cout << "[DEBUG] Processing conditions file: " << path << std::endl;
    std::ifstream file(path);
    std::string line;
    
    // Skip header line and capture column names
    std::getline(file, line);
    std::vector<std::string> headerFields;
    std::istringstream headerStream(line);
    std::string headerField;
    
    // Parse header to find column indices
    while (std::getline(headerStream, headerField, ',')) {
        headerFields.push_back(headerField);
    }
    
    // Find required column indices
    int patientIdx = -1, codeIdx = -1, descIdx = -1, encIdx = -1;
    for (size_t i = 0; i < headerFields.size(); i++) {
        if (headerFields[i] == "PATIENT") patientIdx = i;
        else if (headerFields[i] == "CODE") codeIdx = i;
        else if (headerFields[i] == "DESCRIPTION") descIdx = i;
        else if (headerFields[i] == "ENCOUNTER") encIdx = i;
    }
    
    if (patientIdx == -1 || codeIdx == -1 || descIdx == -1) {
        std::cerr << "[ERROR] Required columns missing in " << path << std::endl;
        return;
    }
    
    int rowCount = 0;
    int successCount = 0;
    
    // Process each data row
    while (std::getline(file, line)) {
        rowCount++;
        
        // Improved CSV parsing with quote handling
        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;
        
        // Reset field and parse character by character
        field.clear();
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(c);
            }
        }
        fields.push_back(field); // Add the last field
        
        // Skip malformed rows
        if (fields.size() <= std::max({patientIdx, codeIdx, descIdx})) {
            std::cerr << "[WARNING] Skipping malformed CSV line: " << line << "\n";
            continue;
        }
        
        ConditionRow condition;
        condition.PATIENT = fields[patientIdx];
        condition.CODE = fields[codeIdx];
        condition.DESCRIPTION = fields[descIdx];
        if (encIdx != -1 && encIdx < static_cast<int>(fields.size())) {
            condition.ENCOUNTER = fields[encIdx];
        }
        
        // Debug every 1000th row
        if (rowCount % 1000 == 1) {
            std::cout << "[DEBUG] Sample condition: " << condition.PATIENT << ", " 
                      << condition.CODE << ", " << condition.DESCRIPTION << std::endl;
        }
        
        callback(condition);
        successCount++;
    }
    
    std::cout << "[INFO] Processed " << successCount << " of " << rowCount 
              << " conditions from " << path << std::endl;
}

void processMedicationsInBatches(const std::string &path,
                               std::function<void(const MedicationRow&)> callback) {
    if (!fileExists(path)) {
        std::cerr << "[ERROR] File " << path << " not found.\n";
        return;
    }

    std::cout << "[DEBUG] Processing medications file: " << path << std::endl;
    std::ifstream file(path);
    std::string line;
    
    // Skip header line and capture column names
    std::getline(file, line);
    std::vector<std::string> headerFields;
    std::istringstream headerStream(line);
    std::string headerField;
    
    while (std::getline(headerStream, headerField, ',')) {
        headerFields.push_back(headerField);
    }
    
    // Find required column indices
    int patientIdx = -1, codeIdx = -1, descIdx = -1;
    for (size_t i = 0; i < headerFields.size(); i++) {
        if (headerFields[i] == "PATIENT") patientIdx = i;
        else if (headerFields[i] == "CODE") codeIdx = i;
        else if (headerFields[i] == "DESCRIPTION") descIdx = i;
    }
    
    if (patientIdx == -1) {
        std::cerr << "[ERROR] Required PATIENT column missing in " << path << std::endl;
        return;
    }
    
    int rowCount = 0;
    
    while (std::getline(file, line)) {
        rowCount++;
        
        // Parse CSV with quote handling
        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;
        
        field.clear();
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(c);
            }
        }
        fields.push_back(field);
        
        if (fields.size() <= patientIdx) {
            continue;
        }
        
        MedicationRow medication;
        medication.PATIENT = fields[patientIdx];
        if (codeIdx != -1 && codeIdx < static_cast<int>(fields.size())) {
            medication.CODE = fields[codeIdx];
        }
        if (descIdx != -1 && descIdx < static_cast<int>(fields.size())) {
            medication.DESCRIPTION = fields[descIdx];
        }
        
        callback(medication);
    }
    
    std::cout << "[INFO] Processed " << rowCount << " medications from " << path << std::endl;
}

void processObservationsInBatches(const std::string &path,
                                std::function<void(const ObservationRow&)> callback) {
    if (!fileExists(path)) {
        std::cerr << "[ERROR] File " << path << " not found.\n";
        return;
    }

    std::cout << "[DEBUG] Processing observations file: " << path << std::endl;
    std::ifstream file(path);
    std::string line;
    
    // Skip header line and capture column names
    std::getline(file, line);
    std::vector<std::string> headerFields;
    std::istringstream headerStream(line);
    std::string headerField;
    
    while (std::getline(headerStream, headerField, ',')) {
        headerFields.push_back(headerField);
    }
    
    // Find required column indices
    int patientIdx = -1, descIdx = -1, valueIdx = -1, unitsIdx = -1, dateIdx = -1;
    for (size_t i = 0; i < headerFields.size(); i++) {
        if (headerFields[i] == "PATIENT") patientIdx = i;
        else if (headerFields[i] == "DESCRIPTION") descIdx = i;
        else if (headerFields[i] == "VALUE") valueIdx = i;
        else if (headerFields[i] == "UNITS") unitsIdx = i;
        else if (headerFields[i] == "DATE") dateIdx = i;
    }
    
    if (patientIdx == -1 || descIdx == -1 || valueIdx == -1) {
        std::cerr << "[ERROR] Required columns missing in " << path << std::endl;
        return;
    }
    
    int rowCount = 0;
    int abnormalCount = 0;
    
    while (std::getline(file, line)) {
        rowCount++;
        
        // Parse CSV with quote handling
        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;
        
        field.clear();
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(c);
            }
        }
        fields.push_back(field);
        
        if (fields.size() <= std::max({patientIdx, descIdx, valueIdx})) {
            continue;
        }
        
        ObservationRow observation;
        observation.PATIENT = fields[patientIdx];
        observation.DESCRIPTION = fields[descIdx];
        observation.VALUE = fields[valueIdx];
        
        if (unitsIdx != -1 && unitsIdx < static_cast<int>(fields.size())) {
            observation.UNITS = fields[unitsIdx];
        }
        if (dateIdx != -1 && dateIdx < static_cast<int>(fields.size())) {
            observation.DATE = fields[dateIdx];
        }
        
        // Debug abnormal observations check
        try {
            double value = std::stod(observation.VALUE);
            bool isAbnormal = isAbnormalObsFast(observation.DESCRIPTION, value);
            if (isAbnormal) {
                abnormalCount++;
                if (abnormalCount % 100 == 1) {
                    std::cout << "[DEBUG] Abnormal observation: " << observation.DESCRIPTION 
                              << " = " << value << std::endl;
                }
            }
        } catch (const std::exception&) {
            // Non-numeric value, handled by the caller
        }
        
        callback(observation);
    }
    
    std::cout << "[INFO] Processed " << rowCount << " observations with " 
              << abnormalCount << " abnormal values from " << path << std::endl;
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
        std::cerr << "[ERROR] File " << path << " not found.\n";
        return;
    }

    std::cout << "[DEBUG] Processing encounters file: " << path << std::endl;
    std::ifstream file(path);
    std::string line;
    
    // Skip header line and capture column names
    std::getline(file, line);
    std::vector<std::string> headerFields;
    std::istringstream headerStream(line);
    std::string headerField;
    
    while (std::getline(headerStream, headerField, ',')) {
        headerFields.push_back(headerField);
    }
    
    // Find required column indices
    int patientIdx = -1, classIdx = -1, idIdx = -1;
    for (size_t i = 0; i < headerFields.size(); i++) {
        if (headerFields[i] == "PATIENT") patientIdx = i;
        else if (headerFields[i] == "ENCOUNTERCLASS") classIdx = i;
        else if (headerFields[i] == "Id") idIdx = i;
    }
    
    if (patientIdx == -1 || classIdx == -1) {
        std::cerr << "[ERROR] Required columns missing in " << path << std::endl;
        return;
    }
    
    int rowCount = 0;
    int inpatientCount = 0;
    
    while (std::getline(file, line)) {
        rowCount++;
        
        // Parse CSV with quote handling
        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;
        
        field.clear();
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(c);
            }
        }
        fields.push_back(field);
        
        if (fields.size() <= std::max(patientIdx, classIdx)) {
            continue;
        }
        
        EncounterRow encounter;
        encounter.PATIENT = fields[patientIdx];
        encounter.ENCOUNTERCLASS = fields[classIdx];
        if (idIdx != -1 && idIdx < static_cast<int>(fields.size())) {
            encounter.ID = fields[idIdx];
        }
        
        if (encounter.ENCOUNTERCLASS == "inpatient") {
            inpatientCount++;
        }
        
        callback(encounter);
    }
    
    std::cout << "[INFO] Processed " << rowCount << " encounters with " 
              << inpatientCount << " inpatient encounters from " << path << std::endl;
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

static PatientRecord parsePatientCSV(const std::vector<std::string>& header, const std::vector<std::string>& row) {
    std::unordered_map<std::string, std::string> rowMap;
    for (size_t i = 0; i < header.size() && i < row.size(); i++) {
        rowMap[header[i]] = row[i];
    }

    PatientRecord record;
    record.Id = rowMap["Id"];
    record.Birthdate = rowMap["BIRTHDATE"];
    record.Deathdate = rowMap.count("DEATHDATE") ? rowMap["DEATHDATE"] : "";
    record.SSN = rowMap.count("SSN") ? rowMap["SSN"] : "";
    record.Drivers = rowMap.count("DRIVERS") ? rowMap["DRIVERS"] : "";
    record.Passport = rowMap.count("PASSPORT") ? rowMap["PASSPORT"] : "";
    record.Prefix = rowMap.count("PREFIX") ? rowMap["PREFIX"] : "";
    record.First = rowMap.count("FIRST") ? rowMap["FIRST"] : "";
    record.Last = rowMap.count("LAST") ? rowMap["LAST"] : "";
    record.Suffix = rowMap.count("SUFFIX") ? rowMap["SUFFIX"] : "";
    record.Maiden = rowMap.count("MAIDEN") ? rowMap["MAIDEN"] : "";
    record.Gender = rowMap.count("GENDER") ? rowMap["GENDER"] : "";
    record.RaceCategory = rowMap.count("RACE") ? rowMap["RACE"] : "";
    record.EthnicityCategory = rowMap.count("ETHNICITY") ? rowMap["ETHNICITY"] : "";
    record.MaritalStatus = rowMap.count("MARITAL") ? rowMap["MARITAL"] : "";
    
    // Parse numerical fields with proper error handling
    if (rowMap.count("HEALTHCARE_EXPENSES")) {
        try {
            record.HealthcareExpenses = std::stof(rowMap["HEALTHCARE_EXPENSES"]);
        } catch (...) {
            std::cerr << "[WARNING] Invalid HEALTHCARE_EXPENSES value: " << rowMap["HEALTHCARE_EXPENSES"] << "\n";
        }
    }
    
    if (rowMap.count("HEALTHCARE_COVERAGE")) {
        try {
            record.HealthcareCoverage = std::stof(rowMap["HEALTHCARE_COVERAGE"]);
        } catch (...) {
            std::cerr << "[WARNING] Invalid HEALTHCARE_COVERAGE value: " << rowMap["HEALTHCARE_COVERAGE"] << "\n";
        }
    }
    
    if (rowMap.count("INCOME")) {
        try {
            record.Income = std::stof(rowMap["INCOME"]);
        } catch (...) {
            std::cerr << "[WARNING] Invalid INCOME value: " << rowMap["INCOME"] << "\n";
        }
    }
    
    // Calculate age from BIRTHDATE
    if (!record.Birthdate.empty()) {
        try {
            // Simplistic age calculation - just the year difference
            int birthYear = std::stoi(record.Birthdate.substr(0, 4));
            int currentYear = 2025; // Default to compilation year if needed
            
            // Try to get current year from system time
            std::time_t t = std::time(nullptr);
            std::tm* now = std::localtime(&t);
            if (now) {
                currentYear = now->tm_year + 1900;
            }
            
            record.Age = currentYear - birthYear;
        } catch (...) {
            std::cerr << "[WARNING] Invalid BIRTHDATE: " << record.Birthdate << " for patient " << record.Id << "\n";
        }
    }
    
    // Determine if patient is deceased
    record.IsDeceased = !record.Deathdate.empty();
    
    return record;
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

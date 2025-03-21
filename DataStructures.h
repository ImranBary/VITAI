#pragma once

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <functional> // Add this to fix std::function error

// Forward declaration of external variables
extern unsigned int THREAD_COUNT;

// Patient record structure
struct PatientRecord {
    std::string Id;
    std::string Birthdate;
    std::string Deathdate;
    std::string SSN;
    std::string Drivers;
    std::string Passport;
    std::string Prefix;
    std::string First;
    std::string Last;
    std::string Suffix;
    std::string Maiden;
    std::string Gender;
    float CharlsonIndex = 0.0f;
    float ElixhauserIndex = 0.0f;
    float Comorbidity_Score = 0.0f;
    int Hospitalizations_Count = 0;
    int Medications_Count = 0;
    int Abnormal_Observations_Count = 0;
    float Health_Index = 0.0f;
    bool IsDeceased = false;
    std::string RaceCategory;
    std::string EthnicityCategory;
    std::string MaritalStatus;
    float HealthcareExpenses = 0.0f;
    float HealthcareCoverage = 0.0f;
    float Income = 0.0f;
    int Age = 0;  // Added Age field with default value
};

// Condition row structure
struct ConditionRow {
    std::string START;
    std::string STOP;
    std::string PATIENT;
    std::string ENCOUNTER; // Add this field
    std::string CODE;
    std::string DESCRIPTION;
};

// Encounter row structure
struct EncounterRow {
    std::string ID;       // Add this field
    std::string START;
    std::string STOP;
    std::string PATIENT;
    std::string ENCOUNTERCLASS;
};

// Medication row structure
struct MedicationRow {
    std::string START;
    std::string STOP;
    std::string PATIENT;
    std::string CODE;
    std::string DESCRIPTION;
};

// Observation row structure
struct ObservationRow {
    std::string DATE;
    std::string PATIENT;
    std::string CODE;
    std::string DESCRIPTION;
    std::string VALUE;
    std::string UNITS;
};

// Procedure row structure
struct ProcedureRow {
    std::string PATIENT;
    std::string ENCOUNTER;
    std::string CODE;
    std::string DESCRIPTION;
};

// Thread-safe counter for tracking patient metrics
class ThreadSafeCounter {
private:
    std::unordered_map<std::string, int> intCounts;
    std::unordered_map<std::string, float> floatCounts;
    std::mutex mutex;

public:
    void increment(const std::string& key, int amount = 1) {
        std::lock_guard<std::mutex> lock(mutex);
        intCounts[key] += amount;
    }

    void addFloat(const std::string& key, float value) {
        std::lock_guard<std::mutex> lock(mutex);
        floatCounts[key] += value;
    }

    int getInt(const std::string& key) const {
        auto it = intCounts.find(key);
        return (it != intCounts.end()) ? it->second : 0;
    }

    float getFloat(const std::string& key) const {
        auto it = floatCounts.find(key);
        return (it != floatCounts.end()) ? it->second : 0.0f;
    }
    
    // Added for debugging - access to internal maps
    const std::unordered_map<std::string, int>& internalMap() const {
        return intCounts;
    }
    
    const std::unordered_map<std::string, float>& internalFloatMap() const {
        return floatCounts;
    }
};

// External variable declarations
extern unsigned int THREAD_COUNT;
extern std::unordered_map<std::string, std::pair<double, double>> OBS_ABNORMAL_DIRECT;

// Mark BatchProcessor as defined
#define BATCH_PROCESSOR_DEFINED
class BatchProcessor {
public:
    template<typename T>
    static void processFile(
        const std::string& filePath,
        std::function<T(const std::vector<std::string>&, const std::vector<std::string>&)> parser,
        std::function<void(const T&)> callback
    ) {
        // Implementation omitted for brevity
    }
};

#endif // DATA_STRUCTURES_H

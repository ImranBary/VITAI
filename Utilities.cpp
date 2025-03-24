#include "DataStructures.h" // Added this include for data type definitions
#include "Utilities.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>
#include <unordered_map>
#include <functional>
#include <set>
#include <sstream>  // Add this include for std::ostringstream

namespace fs = std::filesystem;

// Initialize global variables with EXACT signatures needed by GenerateAndPredict.cpp
unsigned int const DEFAULT_THREAD_COUNT = 4;  // Changed order to match the reference
bool GPU_INFERENCE_FAILED = false;
size_t DEFAULT_CSV_BATCH_SIZE = 1000;
bool PERFORMANCE_MODE = false;
bool EXTREME_PERFORMANCE = false;
const std::string DATA_DIR = "Data";
const std::string SYN_DIR  = "synthea-master";
const std::string SYN_OUT  = "output/csv";
float memoryUtilTarget = 0.7f;
float cpuUtilTarget = 0.8f;

// Medical dictionaries - these are defined as extern in MedicalDictionaries.h
std::unordered_map<std::string, float> CHARLSON_CODE_TO_WEIGHT_INTERNAL = {
    {"diabetes", 1.0f},
    {"heart_failure", 1.0f},
    {"copd", 1.0f},
    // Make sure to include all codes needed in GenerateAndPredict.cpp
    {"E11.9", 1.0f},    // Type 2 diabetes without complications
    {"I10", 0.5f},      // Essential (primary) hypertension
    {"J44.9", 1.0f}     // COPD
    // Add other codes as needed
};

std::unordered_map<std::string, float> ELIXHAUSER_CODE_TO_WEIGHT_INTERNAL = {
    {"diabetes", 0.5f},
    {"heart_failure", 1.5f},
    {"hypertension", 0.7f},
    // Make sure to include all codes needed in GenerateAndPredict.cpp
    {"E11.9", 0.7f},    // Type 2 diabetes without complications
    {"I10", 0.3f},      // Essential (primary) hypertension
    {"J44.9", 0.9f}     // COPD
    // Add other codes as needed
};

void makeDirIfNeeded(const std::string &dir) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
}

std::string getTimestamp() {
    time_t rawtime;
    struct tm timeinfo;
    time(&rawtime);
#ifdef _WIN32
    localtime_s(&timeinfo, &rawtime);
#else
    localtime_r(&timeinfo, &rawtime);
#endif
    char buffer[30];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &timeinfo);
    return std::string(buffer);
}

void runSynthea(int popSize) {
#ifdef _WIN32
    std::string cmd = "cd " + SYN_DIR + " && run_synthea.bat -p " + std::to_string(popSize);
#else
    std::string cmd = "cd " + SYN_DIR + " && ./run_synthea -p " + std::to_string(popSize);
#endif
    std::cout << "[INFO] Running Synthea: " << cmd << "\n";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[ERROR] Synthea generation failed.\n";
        std::exit(1);
    }
    std::cout << "[INFO] Synthea generation complete.\n";
}

bool isDiffFile(const std::string &fname) {
    return (fname.find("_diff_") != std::string::npos);
}

void copySyntheaOutput() {
    fs::path synOutput = fs::path(SYN_DIR) / SYN_OUT;
    if (!fs::exists(synOutput)) {
        std::cerr << "[ERROR] Synthea output dir " << synOutput << " not found.\n";
        std::exit(1);
    }
    makeDirIfNeeded(DATA_DIR);
    std::vector<std::string> needed = {
        "patients.csv", "encounters.csv", "conditions.csv",
        "medications.csv", "observations.csv", "procedures.csv"
    };
    std::string stamp = getTimestamp();
    for (auto &fname : needed) {
        fs::path src = synOutput / fname;
        if (!fs::exists(src)) {
            std::cerr << "[WARN] Missing " << fname << " in Synthea output.\n";
            continue;
        }
        auto dotPos = fname.rfind('.');
        std::string base = (dotPos == std::string::npos) ? fname : fname.substr(0, dotPos);
        std::string ext  = (dotPos == std::string::npos) ? "" : fname.substr(dotPos);
        std::string newName = base + "_diff_" + stamp + ext;
        fs::path dst = fs::path(DATA_DIR) / newName;
        std::cout << "[INFO] Copying " << src << " => " << dst << "\n";
        try {
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        } catch (const fs::filesystem_error &ex) {
            std::cerr << "[ERROR] copy failed: " << ex.what() << "\n";
        }
    }
}

// REMOVE DUPLICATED FUNCTIONS
// These are already defined in other files, so we'll just declare them here
// but not implement them

// Declare FileProcessing.cpp functions
extern std::vector<std::string> listCSVFiles(const std::string &prefix);
extern void processPatientsInBatches(const std::string& filename, std::function<void(const PatientRecord&)> processor);
extern void processConditionsInBatches(const std::string& filename, std::function<void(const ConditionRow&)> processor);
extern void processEncountersInBatches(const std::string& filename, std::function<void(const EncounterRow&)> processor);
extern void processMedicationsInBatches(const std::string& filename, std::function<void(const MedicationRow&)> processor);
extern void processObservationsInBatches(const std::string& filename, std::function<void(const ObservationRow&)> processor);

// Declare PatientSubsets.cpp functions
extern std::set<std::string> findPatientSubset(const std::string& condition, const std::vector<ConditionRow>& conditions);

// Declare FeatureUtils.cpp functions
extern std::vector<std::string> getFeatureCols(const std::string& featureConfig);
extern void writeFeaturesCSV(const std::vector<PatientRecord>& patients, 
                      const std::string& outputFile,
                      const std::vector<std::string>& featureCols);
extern void saveFinalDataCSV(const std::vector<PatientRecord>& patients, 
                     const std::string& outputFile);

// Declare HealthIndex.cpp functions
extern float computeHealthIndex(const PatientRecord& patient);

// Empty stubs for calling the actual MedicalDictionaries.h implementations
void initializeDirectLookupsCaller() {
    // Call the implementation from MedicalDictionaries.h
    initializeDirectLookups();
}

void initializeObsAbnormalDirectCaller() {
    // Call the implementation from MedicalDictionaries.h
    initializeObsAbnormalDirect();
}

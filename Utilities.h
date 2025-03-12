#pragma once

#include <string>
#include <vector>
#include <functional>
#include <set>
#include <unordered_map>
#include <ctime>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// Forward declarations
struct PatientRecord;
struct ConditionRow;
struct EncounterRow;
struct MedicationRow;
struct ObservationRow;

// Global variables
extern const unsigned int DEFAULT_THREAD_COUNT;
extern unsigned int THREAD_COUNT;
extern bool GPU_INFERENCE_FAILED;
extern size_t DEFAULT_CSV_BATCH_SIZE;
extern bool PERFORMANCE_MODE;
extern bool EXTREME_PERFORMANCE;
extern const std::string DATA_DIR;
extern const std::string SYN_DIR;
extern const std::string SYN_OUT;
extern float memoryUtilTarget;
extern float cpuUtilTarget;

// Use MedicalDictionaries.h for dictionary declarations
#include "MedicalDictionaries.h"

// Utility function declarations
void makeDirIfNeeded(const std::string &dir);
std::string getTimestamp();
void runSynthea(int popSize);
bool isDiffFile(const std::string &fname);
void copySyntheaOutput();
std::vector<std::string> listCSVFiles(const std::string &prefix);

void processPatientsInBatches(const std::string& filename, 
                              std::function<void(const PatientRecord&)> processor);
void processConditionsInBatches(const std::string& filename, 
                               std::function<void(const ConditionRow&)> processor);
void processEncountersInBatches(const std::string& filename, 
                               std::function<void(const EncounterRow&)> processor);
void processMedicationsInBatches(const std::string& filename, 
                                std::function<void(const MedicationRow&)> processor);
void processObservationsInBatches(const std::string& filename, 
                                 std::function<void(const ObservationRow&)> processor);

std::set<std::string> findPatientSubset(const std::string& condition, 
                                       const std::vector<ConditionRow>& conditions);
double findGroupWeightFast(const std::string& code);
bool isAbnormalObsFast(const std::string& description, double value);

std::vector<std::string> getFeatureCols(const std::string& featureConfig);
void writeFeaturesCSV(const std::vector<PatientRecord>& patients, 
                      const std::string& outputFile,
                      const std::vector<std::string>& featureCols);
void saveFinalDataCSV(const std::vector<PatientRecord>& patients, 
                     const std::string& outputFile);

float computeHealthIndex(const PatientRecord& patient);

void initializeDirectLookups();
void initializeObsAbnormalDirect();

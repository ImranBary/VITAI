#pragma once

#include <string>
#include <functional>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "DataStructures.h"
#include "BatchProcessor.h"
#include "Utilities.h"

// Batch file processing functions
void processConditionsInBatches(const std::string &path,
                              std::function<void(const ConditionRow&)> callback);

void processMedicationsInBatches(const std::string &path,
                               std::function<void(const MedicationRow&)> callback);

void processObservationsInBatches(const std::string &path,
                                std::function<void(const ObservationRow&)> callback);

void processProceduresInBatches(const std::string &path,
                              std::function<void(const ProcedureRow&)> callback);

void processEncountersInBatches(const std::string &path,
                              std::function<void(const EncounterRow&)> callback);

void processPatientsInBatches(const std::string &path,
                            std::function<void(const PatientRecord&)> callback);

void countHospitalizationsInBatches(std::function<void(const std::string&, uint16_t)> countCallback);

void computeCharlsonIndexBatched(std::function<void(const std::string&, float)> scoreCallback);

void computeElixhauserIndexBatched(std::function<void(const std::string&, float)> scoreCallback);

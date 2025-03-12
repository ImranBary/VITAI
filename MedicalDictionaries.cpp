#include "MedicalDictionaries.h"
#include <iostream>

// Define the global variables declared as extern in the MedicalDictionaries.h file
// These need to be here, not in Utilities.cpp, to match the extern declarations
const std::unordered_map<std::string, float> CHARLSON_CODE_TO_WEIGHT = {
    {"diabetes", 1.0f},
    {"heart_failure", 1.0f},
    {"copd", 1.0f},
    // Make sure to include all codes needed in GenerateAndPredict.cpp
    {"E11.9", 1.0f},    // Type 2 diabetes without complications
    {"I10", 0.5f},      // Essential (primary) hypertension
    {"J44.9", 1.0f}     // COPD
    // Add other codes as needed
};

const std::unordered_map<std::string, float> ELIXHAUSER_CODE_TO_WEIGHT = {
    {"diabetes", 0.5f},
    {"heart_failure", 1.5f},
    {"hypertension", 0.7f},
    // Make sure to include all codes needed in GenerateAndPredict.cpp
    {"E11.9", 0.7f},    // Type 2 diabetes without complications
    {"I10", 0.3f},      // Essential (primary) hypertension
    {"J44.9", 0.9f}     // COPD
    // Add other codes as needed
};

// Remove all function implementations - they're already defined in the header file as inline functions
// This includes:
// initializeDirectLookups()
// initializeObsAbnormalDirect() 
// findGroupWeightFast()
// isAbnormalObsFast()

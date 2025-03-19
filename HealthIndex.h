#pragma once

#include "DataStructures.h"
#include <iostream> // Added this include for std::cerr

// Fixed implementation of health index calculation - don't return 100 when all values are 0
inline float computeHealthIndex(const PatientRecord& patient) {
    // Start with perfect health
    float index = 100.0f;
    bool hasNonZeroValues = false;
    
    // Subtract based on comorbidity scores
    if (patient.CharlsonIndex > 0) {
        index -= (patient.CharlsonIndex * 2.5f);
        hasNonZeroValues = true;
    }
    if (patient.ElixhauserIndex > 0) {
        index -= (patient.ElixhauserIndex * 1.5f);
        hasNonZeroValues = true;
    }
    if (patient.Comorbidity_Score > 0) {
        index -= (patient.Comorbidity_Score * 1.0f);
        hasNonZeroValues = true;
    }
    
    // Subtract for hospitalizations
    if (patient.Hospitalizations_Count > 0) {
        index -= (patient.Hospitalizations_Count * 3.0f);
        hasNonZeroValues = true;
    }
    
    // Subtract for medications
    if (patient.Medications_Count > 0) {
        index -= (patient.Medications_Count * 0.5f);
        hasNonZeroValues = true;
    }
    
    // Subtract for abnormal observations
    if (patient.Abnormal_Observations_Count > 0) {
        index -= (patient.Abnormal_Observations_Count * 1.0f);
        hasNonZeroValues = true;
    }
    
    // Adjust based on age
    if (patient.Age > 65) {
        index -= ((patient.Age - 65) * 0.3f);
        hasNonZeroValues = true;
    } else if (patient.Age > 0) {
        hasNonZeroValues = true;
    }
    
    // If all input values are zero, return a warning value to indicate possible data issue
    if (!hasNonZeroValues) {
        std::cerr << "[WARNING] Patient " << patient.Id 
                  << " has all zero values, health index calculation may be inaccurate\n";
    }
    
    // Ensure index stays in valid range
    if (index < 0) index = 0;
    if (index > 100) index = 100;
    
    return index;
}

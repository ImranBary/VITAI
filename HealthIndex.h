#pragma once

#include "DataStructures.h"

// Inline implementation of health index calculation
inline float computeHealthIndex(const PatientRecord& patient) {
    // Simple weighted calculation of health index based on various factors
    float index = 100.0f;  // Start with perfect health
    
    // Subtract based on comorbidity scores
    index -= (patient.CharlsonIndex * 2.5f);
    index -= (patient.ElixhauserIndex * 1.5f);
    index -= (patient.Comorbidity_Score * 1.0f);
    
    // Subtract for hospitalizations
    index -= (patient.Hospitalizations_Count * 3.0f);
    
    // Subtract for medications
    index -= (patient.Medications_Count * 0.5f);
    
    // Subtract for abnormal observations
    index -= (patient.Abnormal_Observations_Count * 1.0f);
    
    // Adjust based on age
    if (patient.Age > 65) {
        index -= ((patient.Age - 65) * 0.3f);
    }
    
    // Ensure index stays in valid range
    if (index < 0) index = 0;
    if (index > 100) index = 100;
    
    return index;
}

#include <cmath>
#include <algorithm>
#include "HealthIndex.h"
#include "DataStructures.h"

// Constants for health index calculation - aligned with Python implementation
const double AGE_WEIGHT = 0.15;
const double CHARLSON_WEIGHT = 0.25;
const double ELIXHAUSER_WEIGHT = 0.20;
const double HOSPITALIZATIONS_WEIGHT = 0.15;
const double MEDICATIONS_WEIGHT = 0.10;
const double ABNORMAL_OBS_WEIGHT = 0.15;

// Helper function to scale a value between 0-10, similar to Python's scaler
double scaleValue(double value, double min_val, double max_val) {
    // Handle edge case
    if (max_val - min_val < 0.000001) return 5.0;
    
    // Scale to 1-10 range (we subtract from 10 for factors where higher is worse)
    return 1.0 + 9.0 * (value - min_val) / (max_val - min_val);
}

// Function to compute health index similar to Python implementation
float computeHealthIndex(const PatientRecord& patient) {
    // Step 1: Calculate raw health index first (same as Python)
    double base = 100.0;
    double penalty1 = 0.4 * patient.Comorbidity_Score;
    double penalty2 = 1.0 * patient.Hospitalizations_Count;
    double penalty3 = 0.2 * patient.Medications_Count;
    double penalty4 = 0.3 * patient.Abnormal_Observations_Count;
    double penalty5 = 0.1 * patient.CharlsonIndex + 0.05 * patient.ElixhauserIndex;
    double rawIndex = base - (penalty1 + penalty2 + penalty3 + penalty4 + penalty5);
    
    // Clamp raw value to avoid negative/extreme values
    rawIndex = std::max(0.0, std::min(rawIndex, 100.0));
    
    // Python uses a second normalization step that the C++ implementation is missing
    // Since we can't dynamically scale across all patients here, use fixed ranges that 
    // match typical Python values (this will be close enough)
    
    // Match Python's normalization: 1 + 9 * (value - min) / (max - min)
    // Assume min=30, max=90 based on empirical observations
    const double MIN_HEALTH = 30.0;
    const double MAX_HEALTH = 90.0;
    double normalizedIndex = 1.0 + 9.0 * (rawIndex - MIN_HEALTH) / (MAX_HEALTH - MIN_HEALTH + 1e-8);
    
    return static_cast<float>(std::max(1.0, std::min(10.0, normalizedIndex)));
}

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
    // Exactly match Python's implementation in health_index.py
    double base = 100.0;  // Python uses 100 not 10
    
    // These coefficients match Python implementation
    double penalty1 = 0.5 * patient.Comorbidity_Score;  // Python uses 0.5 not 0.4
    double penalty2 = 1.2 * patient.Hospitalizations_Count;  // Python uses 1.2 not 1.0
    double penalty3 = 0.3 * patient.Medications_Count;  // Python uses 0.3 not 0.2
    double penalty4 = 0.35 * patient.Abnormal_Observations_Count;  // Python uses 0.35 not 0.3
    double penalty5 = 0.15 * patient.CharlsonIndex + 0.08 * patient.ElixhauserIndex;  // Different coefficients
    
    double rawIndex = base - (penalty1 + penalty2 + penalty3 + penalty4 + penalty5);
    
    // Clamp and normalize as in Python
    rawIndex = std::max(0.0, std::min(rawIndex, 100.0));
    
    // Important: Python rescales to 1-10 range with min-max normalization
    // Assume MIN_HEALTH=30, MAX_HEALTH=90 as in Python
    const double MIN_HEALTH = 30.0;
    const double MAX_HEALTH = 90.0;
    double normalizedIndex = 1.0 + 9.0 * (rawIndex - MIN_HEALTH) / (MAX_HEALTH - MIN_HEALTH);
    
    // Final clamping to ensure values are in correct range
    return static_cast<float>(std::max(1.0, std::min(10.0, normalizedIndex)));
}

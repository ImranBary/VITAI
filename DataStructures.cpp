#include "DataStructures.h"

// Initialize the static variable
unsigned int THREAD_COUNT = 4;

// Implementation of any DataStructures methods if needed
// Most are likely to be inline or header-only implementations

std::unordered_map<std::string, std::pair<double, double>> OBS_ABNORMAL_DIRECT = {
    {"Systolic Blood Pressure", {90.0, 140.0}},
    {"Diastolic Blood Pressure", {60.0, 90.0}},
    {"Heart Rate", {60.0, 100.0}},
    {"Glucose", {70.0, 126.0}},
    {"A1C", {4.0, 6.5}}
};

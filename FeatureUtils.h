#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <map>
#include <functional>
#include "DataStructures.h"
#include "Utilities.h"

// Feature utility functions
std::vector<std::string> getFeatureCols(const std::string &feature_config);
void writeFeaturesCSV(const std::vector<PatientRecord> &pats,
                    const std::string &outFile,
                    const std::vector<std::string> &cols);
void saveFinalDataCSV(const std::vector<PatientRecord> &pats,
                    const std::string &outfile);
void cleanupFiles(const std::vector<std::string> &files);

// Add function prototype for getPatientFieldByName
std::string getPatientFieldByName(const PatientRecord& patient, const std::string& fieldName);

// Add this function to convert column names to lowercase when writing to CSV
inline std::string toLowercase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// Make sure feature column names are written in lowercase
inline std::vector<std::string> convertFeatureNamesToLowercase(const std::vector<std::string>& featureCols) {
    std::vector<std::string> lowercaseFeatures;
    lowercaseFeatures.reserve(featureCols.size());
    
    for (const auto& feature : featureCols) {
        lowercaseFeatures.push_back(toLowercase(feature));
    }
    
    return lowercaseFeatures;
}

// Define a function to get the exact feature columns expected by the model
std::vector<std::string> getModelExpectedFeatures();

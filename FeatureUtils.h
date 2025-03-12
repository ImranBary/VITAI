#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
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

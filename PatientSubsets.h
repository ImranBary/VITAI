#pragma once

#include <string>
#include <set>  // Not unordered_set
#include <vector>
#include "DataStructures.h"
#include "Utilities.h"

// Patient subset selection functions
std::unordered_set<std::string> findDiabeticPatientsOptimized(const std::vector<ConditionRow> &conds);
std::unordered_set<std::string> findCKDPatientsOptimized(const std::vector<ConditionRow> &conds);
std::set<std::string> findPatientSubset(const std::string& condition, 
                                       const std::vector<ConditionRow>& conditions);

#include "PatientSubsets.h"
#include <algorithm>
#include <mutex>
#include <thread>
#include <iostream>
#include "DataStructures.h"
#include <string>
#include <set>
#include <vector>

std::unordered_set<std::string> findDiabeticPatientsOptimized(const std::vector<ConditionRow> &conds) {
    std::unordered_set<std::string> out;
    out.reserve(conds.size() / 10); // Pre-allocate with estimate of diabetic patients
    
    // Simple implementation for now
    for (const auto& c : conds) {
        std::string lower = c.DESCRIPTION;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        if (lower.find("diabetes") != std::string::npos || 
            lower.find("diabetic") != std::string::npos) {
            out.insert(c.PATIENT);
        }
    }
    
    return out;
}

std::unordered_set<std::string> findCKDPatientsOptimized(const std::vector<ConditionRow> &conds) {
    std::unordered_set<std::string> out;
    out.reserve(conds.size() / 10); // Pre-allocate with estimate of CKD patients
    
    // Simple implementation for now
    for (const auto& c : conds) {
        std::string lower = c.DESCRIPTION;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        if (lower.find("chronic kidney disease") != std::string::npos ||
            lower.find("ckd") != std::string::npos ||
            lower.find("renal failure") != std::string::npos) {
            out.insert(c.PATIENT);
        }
    }
    
    return out;
}

std::unordered_set<std::string> findPatientSubsetUnordered(const std::string& subsetType, 
                                                const std::vector<ConditionRow> &conds) {
    if (subsetType == "none") {
        return std::unordered_set<std::string>();
    }
    
    if (subsetType == "diabetes") {
        return findDiabeticPatientsOptimized(conds);
    }
    
    if (subsetType == "ckd") {
        return findCKDPatientsOptimized(conds);
    }
    
    std::cerr << "[ERROR] Unknown patient subset type: " << subsetType << std::endl;
    return std::unordered_set<std::string>();
}

std::set<std::string> findPatientSubset(const std::string& condition, const std::vector<ConditionRow>& conditions) {
    std::set<std::string> patientSet;
    std::string lowerCondition = condition;
    
    // Convert condition to lowercase for case-insensitive matching
    std::transform(lowerCondition.begin(), lowerCondition.end(), lowerCondition.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    std::cout << "[INFO] Finding patients with condition related to: " << condition << std::endl;
    
    for (const auto& cond : conditions) {
        std::string description = cond.DESCRIPTION;
        std::transform(description.begin(), description.end(), description.begin(),
                      [](unsigned char c) { return std::tolower(c); });
        
        if (description.find(lowerCondition) != std::string::npos) {
            patientSet.insert(cond.PATIENT);
        }
    }
    
    std::cout << "[INFO] Found " << patientSet.size() << " patients with " << condition << "." << std::endl;
    return patientSet;
}

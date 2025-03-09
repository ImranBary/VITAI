#pragma once

#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <unordered_map>
#include <iostream>

// A memory-efficient batch processor for CSV files
class BatchProcessor {
public:
    template<typename T>
    static void processFile(const std::string &path, 
                          std::function<T(const std::vector<std::string>&, const std::vector<std::string>&)> rowToObj,
                          std::function<void(const T&)> callback,
                          size_t batchSize = 500) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Cannot open " << path << std::endl;
            return;
        }
        
        std::string line;
        std::getline(file, line);  // Read header
        
        // Split header into column names
        std::vector<std::string> header = splitCSV(line);
        
        std::vector<T> batch;
        batch.reserve(batchSize);
        
        while (std::getline(file, line)) {
            std::vector<std::string> values = splitCSV(line);
            
            // Convert row to object using callback
            T obj = rowToObj(header, values);
            
            batch.push_back(obj);
            
            if (batch.size() >= batchSize) {
                // Process batch
                for (const auto &item : batch) {
                    callback(item);
                }
                batch.clear();
            }
        }
        
        // Process remaining items
        for (const auto &item : batch) {
            callback(item);
        }
    }
    
    static std::vector<std::string> splitCSV(const std::string &line) {
        std::vector<std::string> result;
        std::string current;
        bool inQuotes = false;
        
        for (char c : line) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                result.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        result.push_back(current);  // Add the last field
        return result;
    }
};

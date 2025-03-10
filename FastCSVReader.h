#pragma once

#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <memory>
#include <algorithm>
#include <iostream>

/**
 * FastCSVReader - Optimized CSV reader with batched processing
 * 
 * Features:
 * - Uses large buffer for improved I/O performance
 * - Processes rows in batches to reduce memory allocations
 * - Direct string manipulation for faster CSV parsing
 * - Memory-friendly string handling
 */
class FastCSVReader {
private:
    std::ifstream file;
    std::vector<char> buffer;
    size_t buffer_size;
    static constexpr size_t DEFAULT_BUFFER_SIZE = 1024 * 1024; // 1MB buffer

public:
    FastCSVReader(const std::string& path, size_t buffer_size = DEFAULT_BUFFER_SIZE) 
        : buffer_size(buffer_size) {
        file.open(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        buffer.resize(buffer_size);
    }
    
    // Fast CSV string splitting without using stringstream
    static std::vector<std::string> splitCSV(const char* line, size_t length) {
        std::vector<std::string> result;
        result.reserve(20); // Typical number of columns in medical data
        
        const char* start = line;
        bool inQuotes = false;
        
        for (size_t i = 0; i < length; i++) {
            char c = line[i];
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                result.push_back(std::string(start, line + i - start));
                start = line + i + 1;
            }
        }
        
        // Add the last field
        result.push_back(std::string(start, line + length - start));
        return result;
    }
    
    // Overload for std::string
    static std::vector<std::string> splitCSV(const std::string& line) {
        return splitCSV(line.c_str(), line.length());
    }
    
    // Read header line
    std::vector<std::string> readHeader() {
        std::string line;
        std::getline(file, line);
        return splitCSV(line);
    }
    
    // Read the file in batches with callback for each batch
    template<typename T>
    void readBatches(
        std::function<T(const std::vector<std::string>&, const std::vector<std::string>&)> rowToObj,
        std::function<void(const std::vector<T>&)> processBatch,
        size_t batch_size = 2000
    ) {
        std::vector<T> batch;
        batch.reserve(batch_size);
        
        std::string line;
        std::vector<std::string> header = readHeader();
        
        while (std::getline(file, line)) {
            try {
                std::vector<std::string> values = splitCSV(line);
                
                // Convert row to object and add to batch
                T obj = rowToObj(header, values);
                batch.push_back(std::move(obj));
                
                if (batch.size() >= batch_size) {
                    processBatch(batch);
                    batch.clear();
                    batch.reserve(batch_size);
                }
            } catch (const std::exception& e) {
                std::cerr << "[WARN] Error processing CSV row: " << e.what() << std::endl;
                // Continue with next row
            }
        }
        
        // Process remaining items
        if (!batch.empty()) {
            processBatch(batch);
        }
    }
    
    bool isOpen() const {
        return file.is_open();
    }
    
    ~FastCSVReader() {
        if (file.is_open()) {
            file.close();
        }
    }
};

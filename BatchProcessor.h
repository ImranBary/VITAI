#pragma once

#ifndef BATCH_PROCESSOR_H
#define BATCH_PROCESSOR_H

#include "DataStructures.h"
#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <filesystem>
#include "SystemResources.h"

// Only define BatchProcessor if it's not already defined in DataStructures.h
#ifndef BATCH_PROCESSOR_DEFINED
#define BATCH_PROCESSOR_DEFINED

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
        
        // Get file size and adapt batch size accordingly
        size_t fileSize = getFileSize(path);
        if (fileSize > 100 * 1024 * 1024) { // If file is larger than 100MB
            batchSize = adaptBatchSizeToFile(path, batchSize);
            std::cout << "[BATCH] Adjusted batch size to " << batchSize << " for large file: " << path << std::endl;
        }
        
        std::string line;
        std::getline(file, line);  // Read header
        
        // Split header into column names
        std::vector<std::string> header = splitCSV(line);
        
        std::vector<T> batch;
        batch.reserve(batchSize);
        
        size_t rowsProcessed = 0;
        while (std::getline(file, line)) {
            std::vector<std::string> values = splitCSV(line);
            
            // Skip empty or malformed rows
            if (values.size() < header.size() / 2) {
                continue;
            }
            
            // Convert row to object using callback
            T obj = rowToObj(header, values);
            
            batch.push_back(obj);
            rowsProcessed++;
            
            if (batch.size() >= batchSize) {
                // Process batch
                for (const auto &item : batch) {
                    callback(item);
                }
                batch.clear();
                batch.reserve(batchSize); // Ensure capacity is maintained
                
                // Every 1M rows, output progress
                if (rowsProcessed % 1000000 == 0) {
                    std::cout << "[PROGRESS] Processed " << rowsProcessed << " rows from " << path << std::endl;
                }
            }
        }
        
        // Process remaining items
        for (const auto &item : batch) {
            callback(item);
        }
    }
    
    static std::vector<std::string> splitCSV(const std::string &line) {
        std::vector<std::string> result;
        result.reserve(20); // Pre-allocate space for typical CSV columns
        std::string current;
        current.reserve(64); // Pre-allocate for efficiency
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
    
    // Process large files with adaptive batch sizing
    template<typename T>
    static void processLargeFile(
        const std::string &path,
        std::function<T(const std::vector<std::string>&, const std::vector<std::string>&)> rowToObj,
        std::function<void(const T&)> callback)
    {
        // Get optimal batch size based on file size and memory
        size_t optimalBatchSize = adaptBatchSizeToFile(path, 1000);
        processFile(path, rowToObj, callback, optimalBatchSize);
    }

private:
    static size_t getFileSize(const std::string& path) {
        std::filesystem::path fs_path(path);
        if (std::filesystem::exists(fs_path)) {
            return static_cast<size_t>(std::filesystem::file_size(fs_path));
        }
        return 0;
    }
    
    static size_t adaptBatchSizeToFile(const std::string& path, size_t defaultBatchSize) {
        size_t fileSize = getFileSize(path);
        if (fileSize == 0) return defaultBatchSize;
        
        // Use the file size to determine an appropriate batch size
        // For very large files, use larger batches for better efficiency
        if (fileSize > 1 * 1024 * 1024 * 1024) { // >1GB
            defaultBatchSize = defaultBatchSize * 10;
        } else if (fileSize > 100 * 1024 * 1024) { // >100MB
            defaultBatchSize = defaultBatchSize * 5;
        } else if (fileSize > 10 * 1024 * 1024) { // >10MB
            defaultBatchSize = defaultBatchSize * 2;
        }
        
        // Consider available memory - use up to 5% of available memory for batch
        size_t availableMemoryMB = SystemResources::getAvailableMemoryMB();
        size_t memoryBasedSize = (availableMemoryMB * 1024 * 1024 * 0.05) / 500; // Assume ~500 bytes per record
        
        // Check if extreme performance mode is active (this requires access to the global flag)
        bool extremeMode = false;
        try {
            // Try to detect if we're running in extreme mode by looking at memory utilization target
            if (SystemResources::getCPUUtilization() > 0.9f) {
                extremeMode = true;
            }
        } catch(...) {}
        
        // In extreme mode, be more aggressive with batch sizing
        if (extremeMode) {
            memoryBasedSize = memoryBasedSize * 2;
            defaultBatchSize = defaultBatchSize * 2;
        }
        
        // Return the capped batch size
        size_t result = std::max(defaultBatchSize, memoryBasedSize);
        
        // Put an upper cap on batch size to avoid memory issues
        return std::min(result, static_cast<size_t>(1000000));
    }
};

#endif // BATCH_PROCESSOR_DEFINED

#endif // BATCH_PROCESSOR_H

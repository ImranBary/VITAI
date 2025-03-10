#pragma once

#include <thread>
#include <string>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <chrono>
#include <cstddef>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#pragma comment(lib, "pdh.lib")
#else
#include <unistd.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#endif

/**
 * SystemResources - Utilities for system resource detection and monitoring
 * 
 * Provides:
 * - Detection of CPU cores and memory
 * - Monitoring of memory usage
 * - Auto-configuration based on available resources
 */
class SystemResources {
public:
    // Get number of available CPU cores
    static unsigned int getCpuCores() {
        return std::max(2u, std::thread::hardware_concurrency());
    }
    
    // Get total system memory in GB
    static double getTotalMemoryGB() {
    #ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return static_cast<double>(memInfo.ullTotalPhys) / (1024 * 1024 * 1024);
    #else
        uint64_t totalPhysMem = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
        return static_cast<double>(totalPhysMem) / (1024 * 1024 * 1024);
    #endif
    }
    
    // Get current process memory usage in MB
    static double getCurrentMemoryUsageMB() {
    #ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return static_cast<double>(pmc.WorkingSetSize) / (1024 * 1024);
        }
        return 0.0;
    #else
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    #endif
    }
    
    // Get free system memory in GB
    static double getFreeMemoryGB() {
    #ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return static_cast<double>(memInfo.ullAvailPhys) / (1024 * 1024 * 1024);
    #else
        uint64_t freePages = sysconf(_SC_AVPHYS_PAGES);
        uint64_t pageSize = sysconf(_SC_PAGE_SIZE);
        return static_cast<double>(freePages * pageSize) / (1024 * 1024 * 1024);
    #endif
    }
    
    // Check if we're on a low-resource system
    static bool isLowResourceSystem() {
        unsigned int cores = getCpuCores();
        double totalMemGB = getTotalMemoryGB();
        
        // Definition of "low resource" - adjust as needed
        bool lowCpu = (cores <= 4);
        bool lowMem = (totalMemGB <= 4.0); // 4GB
        
        return lowCpu || lowMem;
    }
    
    // Print system resource information
    static void printSystemInfo() {
        std::cout << "[SYSTEM] CPU Cores: " << getCpuCores() << std::endl;
        std::cout << "[SYSTEM] Total Memory: " << std::fixed << std::setprecision(2) 
                  << getTotalMemoryGB() << " GB" << std::endl;
        std::cout << "[SYSTEM] Free Memory: " << std::fixed << std::setprecision(2)
                  << getFreeMemoryGB() << " GB" << std::endl;
    }
    
    // Get the recommended batch size based on available memory
    static size_t getRecommendedBatchSize() {
        double freeMemGB = getFreeMemoryGB();
        
        if (freeMemGB >= 16.0) return 5000;
        else if (freeMemGB >= 8.0) return 3000;
        else if (freeMemGB >= 4.0) return 2000;
        else if (freeMemGB >= 2.0) return 1000;
        else return 500;
    }
    
    // Get the recommended thread count based on CPU cores and memory
    static unsigned int getRecommendedThreadCount() {
        unsigned int cpuCores = getCpuCores();
        double totalMemGB = getTotalMemoryGB();
        
        // On memory-constrained systems, use fewer threads
        if (totalMemGB < 4.0) {
            return std::max(2u, cpuCores / 2);
        }
        // On normal systems, leave one core free for OS
        else if (cpuCores > 4) {
            return cpuCores - 1;
        }
        // On small systems, use all cores
        else {
            return cpuCores;
        }
    }

    // Helper function for backward compatibility
    static unsigned int getSystemCPUCores() {
        return getCpuCores();
    }

    static size_t getAvailableMemoryMB() {
    #ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return static_cast<size_t>(memInfo.ullAvailPhys / 1024 / 1024);
    #else
        struct sysinfo memInfo;
        sysinfo(&memInfo);
        return static_cast<size_t>(memInfo.freeram * memInfo.mem_unit / 1024 / 1024);
    #endif
    }

    static size_t getTotalMemoryMB() {
    #ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return static_cast<size_t>(memInfo.ullTotalPhys / 1024 / 1024);
    #else
        struct sysinfo memInfo;
        sysinfo(&memInfo);
        return static_cast<size_t>(memInfo.totalram * memInfo.mem_unit / 1024 / 1024);
    #endif
    }

    static float getCPUUtilization() {
        static float lastCpuUsage = 0.0f;
    #ifdef _WIN32
        static PDH_HQUERY cpuQuery;
        static PDH_HCOUNTER cpuTotal;
        static bool firstCall = true;

        if (firstCall) {
            PdhOpenQuery(NULL, 0, &cpuQuery);
            PdhAddEnglishCounter(cpuQuery, "\\Processor(_Total)\\% Processor Time", 0, &cpuTotal);
            PdhCollectQueryData(cpuQuery);
            firstCall = false;
            return lastCpuUsage; // First call returns 0
        }

        PDH_FMT_COUNTERVALUE counterVal;
        PdhCollectQueryData(cpuQuery);
        PdhGetFormattedCounterValue(cpuTotal, PDH_FMT_DOUBLE, NULL, &counterVal);
        lastCpuUsage = static_cast<float>(counterVal.doubleValue / 100.0);
    #else
        std::ifstream loadFile("/proc/loadavg");
        if (loadFile.is_open()) {
            double load1min, load5min, load15min;
            loadFile >> load1min >> load5min >> load15min;
            loadFile.close();
            
            // Normalize by number of processors
            long numProcessors = sysconf(_SC_NPROCESSORS_ONLN);
            lastCpuUsage = static_cast<float>(std::min(1.0, load1min / numProcessors));
        }
    #endif
        return lastCpuUsage;
    }

    static size_t getProcessMemoryUsageMB() {
    #ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
        return static_cast<size_t>(pmc.WorkingSetSize / 1024 / 1024);
    #else
        // Read from /proc/self/status
        std::ifstream statusFile("/proc/self/status");
        if (statusFile.is_open()) {
            std::string line;
            while (std::getline(statusFile, line)) {
                if (line.substr(0, 6) == "VmRSS:") {
                    size_t pos = line.find_first_of("0123456789");
                    if (pos != std::string::npos) {
                        return std::stoul(line.substr(pos)) / 1024; // Convert KB to MB
                    }
                }
            }
            statusFile.close();
        }
        return 0;
    #endif
    }

    // Helper method to recommend batch size based on file size
    static size_t recommendBatchSize(const std::string& filePath, size_t avgRowSizeBytes, float memoryRatio = 0.2f) {
        // Get file size
        size_t fileSize = getFileSize(filePath);
        if (fileSize == 0) return 1000; // Default if can't determine
        
        // Estimate number of rows
        size_t estimatedRows = fileSize / avgRowSizeBytes;
        
        // Calculate available memory we can use
        size_t availableMemoryBytes = getAvailableMemoryMB() * 1024 * 1024 * memoryRatio;
        
        // Calculate optimal batch size
        size_t optimalBatchSize = availableMemoryBytes / avgRowSizeBytes;
        
        // Cap at estimated row count and ensure minimum reasonable size
        return std::max<size_t>(1000, std::min(estimatedRows, optimalBatchSize));
    }
    
    // Get optimal thread count based on current CPU load
    static unsigned int getOptimalThreadCount(float targetUtilization = 0.95f) {
        unsigned int hwThreads = std::thread::hardware_concurrency();
        float currentUtilization = getCPUUtilization();
        
        if (currentUtilization < 0.3f) {
            // CPU is not heavily used, use more threads
            return std::min(hwThreads * 4, hwThreads + 16); // Cap at reasonable maximum
        } else if (currentUtilization > 0.8f) {
            // CPU is already heavily loaded, use fewer threads
            return std::max(2u, static_cast<unsigned int>(hwThreads * 0.75f));
        }
        
        // Dynamically calculate desired thread count
        float availableCapacity = targetUtilization - currentUtilization;
        if (availableCapacity <= 0) return std::max(2u, hwThreads / 2); // Minimum
        
        // Scale thread count based on available CPU capacity
        return static_cast<unsigned int>(hwThreads * (1.0f + availableCapacity));
    }
    
    // Get optimal batch size based on memory and expected row size
    static size_t getOptimalBatchSize(size_t rowSizeEstimateBytes = 500, float memoryUsageTarget = 0.7f) {
        size_t availableMemMB = getAvailableMemoryMB();
        size_t totalMemMB = getTotalMemoryMB();
        
        // Calculate target memory to use (in bytes)
        size_t targetMemoryBytes = static_cast<size_t>(availableMemMB * 1024 * 1024 * memoryUsageTarget);
        
        // Calculate how many rows would fit in target memory
        size_t optimalBatchSize = targetMemoryBytes / rowSizeEstimateBytes;
        
        // Apply scaling based on system characteristics
        if (totalMemMB > 32000) { // More than 32GB RAM
            optimalBatchSize = std::min(optimalBatchSize, static_cast<size_t>(100000));
        } else if (totalMemMB > 16000) { // More than 16GB RAM
            optimalBatchSize = std::min(optimalBatchSize, static_cast<size_t>(50000));
        } else if (totalMemMB > 8000) { // More than 8GB RAM
            optimalBatchSize = std::min(optimalBatchSize, static_cast<size_t>(20000));
        } else { // Less than 8GB RAM
            optimalBatchSize = std::min(optimalBatchSize, static_cast<size_t>(10000));
        }
        
        // Ensure minimum reasonable batch size
        return std::max<size_t>(500, optimalBatchSize);
    }

private:
    static size_t getFileSize(const std::string& filePath) {
    #ifdef _WIN32
        WIN32_FILE_ATTRIBUTE_DATA fileInfo;
        if (GetFileAttributesExA(filePath.c_str(), GetFileExInfoStandard, &fileInfo)) {
            LARGE_INTEGER size;
            size.LowPart = fileInfo.nFileSizeLow;
            size.HighPart = fileInfo.nFileSizeHigh;
            return static_cast<size_t>(size.QuadPart);
        }
    #else
        struct stat st;
        if (stat(filePath.c_str(), &st) == 0) {
            return static_cast<size_t>(st.st_size);
        }
    #endif
        return 0;
    }
};

// For backward compatibility - now just delegates to the class method
int getSystemCPUCores() {
    return SystemResources::getCpuCores();
}

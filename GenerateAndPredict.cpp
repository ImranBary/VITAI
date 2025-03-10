/*****************************************************
 * GenerateAndPredict.cpp
 *
 * Refactored for batch/stream processing using BatchProcessor
 *
 * Author: Imran Feisal
 * Date: 08/03/2025
 *
 * Usage Examples:
 *   GenerateAndPredict.exe --population=100
 *   GenerateAndPredict.exe --population=100 --enable-xai
 *   GenerateAndPredict.exe --population=100 --threads=8  (specify thread count)
 *   GenerateAndPredict.exe --population=100 --performance-mode
 *   GenerateAndPredict.exe --population=100 --extreme-performance (maximum resource usage)
 *   GenerateAndPredict.exe --population=100 --memory-util=90 --cpu-util=95
 *****************************************************/

#define NOMINMAX  // Move this before any includes to prevent Windows.h macro conflicts

#include <Python.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <functional>
#include <sys/stat.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <bitset>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <condition_variable>
#include <memory_resource>
#include <queue>

#include "BatchProcessor.h"       // Your batch-based CSV helper
#include "ThreadSafeCounter.h"    // For concurrency safety
#include "MedicalDictionaries.h"  // Medical dictionaries helper
#include "FastCSVReader.h"
#include "SystemResources.h"

#ifdef _WIN32
#include <direct.h>   // for _mkdir
#include <windows.h>  // For environment variable handling
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Global constants and directories
const unsigned int DEFAULT_THREAD_COUNT = (2u > std::thread::hardware_concurrency() ? 2u : std::thread::hardware_concurrency());
unsigned int THREAD_COUNT = DEFAULT_THREAD_COUNT;
bool GPU_INFERENCE_FAILED = false;
size_t DEFAULT_CSV_BATCH_SIZE = 1000; // Changed from const to allow modification
bool PERFORMANCE_MODE = false; // New flag for aggressive resource utilization
bool EXTREME_PERFORMANCE = false; // New extreme mode that maximizes resource usage
static const std::string DATA_DIR = "Data";
static const std::string SYN_DIR  = "synthea-master";
static const std::string SYN_OUT  = "output/csv";

// Forward declarations
class ResourceMonitor;
float memoryUtilTarget = 0.7f; // Move global to fix reference in processFilesInParallel
float cpuUtilTarget = 0.8f;    // Move global to fix reference in processFilesInParallel

// Model configuration
struct ModelConfig {
    std::string subset;
    std::string feature_config;
};
static std::map<std::string, ModelConfig> MODEL_CONFIG_MAP = {
    {"combined_diabetes_tabnet", {"diabetes", "combined"}},
    {"combined_all_ckd_tabnet",  {"ckd",      "combined_all"}},
    {"combined_none_tabnet",     {"none",     "combined"}}
};
// Data Structures
struct PatientRecord {
    std::string Id;
    std::string BIRTHDATE;
    std::string DEATHDATE;
    std::string GENDER;
    std::string RACE;
    std::string ETHNICITY;
    float HEALTHCARE_EXPENSES = 0.0f;
    float HEALTHCARE_COVERAGE = 0.0f;
    float INCOME = 0.0f;
    std::string MARITAL;
    bool NewData = false;
    float AGE = 0.0f;
    bool DECEASED = false;
    float CharlsonIndex = 0.0f;
    float ElixhauserIndex = 0.0f;
    float Comorbidity_Score = 0.0f;
    uint16_t Hospitalizations_Count = 0;
    uint16_t Medications_Count = 0;
    uint16_t Abnormal_Observations_Count = 0;
    float Health_Index = 0.0f;
};
struct ConditionRow {
    std::string PATIENT;
    std::string DESCRIPTION;
    std::string CODE;
};
struct EncounterRow {
    std::string Id;
    std::string PATIENT;
    std::string ENCOUNTERCLASS;
};
struct MedicationRow {
    std::string PATIENT;
    std::string ENCOUNTER;
    std::string CODE;
    std::string DESCRIPTION;
};
struct ObservationRow {
    std::string PATIENT;
    std::string ENCOUNTER;
    std::string CODE;
    std::string DESCRIPTION;
    double VALUE = 0.0;
    std::string UNITS;
};
struct ProcedureRow {
    std::string PATIENT;
    std::string ENCOUNTER;
    std::string CODE;
    std::string DESCRIPTION;
};
// Utility functions
static void makeDirIfNeeded(const std::string &dir) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
}
static std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    std::tm timeinfo;
#ifdef _WIN32
    localtime_s(&timeinfo, &t_c);
#else
    localtime_r(&timeinfo, &t_c);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &timeinfo);
    return std::string(buf);
}
static void runSynthea(int popSize) {
#ifdef _WIN32
    std::string cmd = "cd " + SYN_DIR + " && run_synthea.bat -p " + std::to_string(popSize);
#else
    std::string cmd = "cd " + SYN_DIR + " && ./run_synthea -p " + std::to_string(popSize);
#endif
    std::cout << "[INFO] Running Synthea: " << cmd << "\n";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[ERROR] Synthea generation failed.\n";
        std::exit(1);
    }
    std::cout << "[INFO] Synthea generation complete.\n";
}
static bool isDiffFile(const std::string &fname) {
    return (fname.find("_diff_") != std::string::npos);
}
static void copySyntheaOutput() {
    fs::path synOutput = fs::path(SYN_DIR) / SYN_OUT;
    if (!fs::exists(synOutput)) {
        std::cerr << "[ERROR] Synthea output dir " << synOutput << " not found.\n";
        std::exit(1);
    }
    makeDirIfNeeded(DATA_DIR);
    std::vector<std::string> needed = {
        "patients.csv", "encounters.csv", "conditions.csv",
        "medications.csv", "observations.csv", "procedures.csv"
    };
    std::string stamp = getTimestamp();
    for (auto &fname : needed) {
        fs::path src = synOutput / fname;
        if (!fs::exists(src)) {
            std::cerr << "[WARN] Missing " << fname << " in Synthea output.\n";
            continue;
        }
        auto dotPos = fname.rfind('.');
        std::string base = (dotPos == std::string::npos) ? fname : fname.substr(0, dotPos);
        std::string ext  = (dotPos == std::string::npos) ? "" : fname.substr(dotPos);
        std::string newName = base + "_diff_" + stamp + ext;
        fs::path dst = fs::path(DATA_DIR) / newName;
        std::cout << "[INFO] Copying " << src << " => " << dst << "\n";
        try {
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        } catch (const fs::filesystem_error &ex) {
            std::cerr << "[ERROR] copy failed: " << ex.what() << "\n";
        }
    }
}
static std::vector<std::string> listCSVFiles(const std::string &prefix) {
    std::vector<std::string> found;
    fs::path dataDir(DATA_DIR);
    if (!fs::exists(dataDir) || !fs::is_directory(dataDir)) {
        std::cerr << "[ERROR] Data directory not found or not a directory: " << dataDir << std::endl;
        return found;
    }
    for (const auto &entry : fs::directory_iterator(dataDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) == 0 &&
            filename.size() > 4 &&
            filename.compare(filename.size() - 4, 4, ".csv") == 0)
        {
            found.push_back(entry.path().string());
        }
    }
    return found;
}
// Resource Monitor class to optimize system resource utilization
class ResourceMonitor {
public:
    ResourceMonitor(float memoryUtilizationTarget = 0.7f, float cpuUtilizationTarget = 0.8f)
        : m_memoryUtilizationTarget(memoryUtilizationTarget), 
          m_cpuUtilizationTarget(cpuUtilizationTarget),
          m_lastCheck(std::chrono::high_resolution_clock::now()),
          m_updateInterval(std::chrono::milliseconds(500)), // More frequent updates (was 2 seconds)
          m_aggressiveScaling(PERFORMANCE_MODE),
          m_extremeScaling(EXTREME_PERFORMANCE)
    {
        update();
        // Auto-tune for performance if resources are abundant
        autoTuneForPerformance();
    }

    void autoTuneForPerformance() {
        // Auto-increase targets if we have lots of resources
        if (m_totalMemoryMB > 16000) { // More than 16GB RAM
            m_memoryUtilizationTarget = std::min(0.90f, m_memoryUtilizationTarget + 0.15f);
            std::cout << "[RESOURCE] Auto-tuned memory target up to " << (m_memoryUtilizationTarget * 100) << "% due to high RAM\n";
        } else if (m_totalMemoryMB > 8000) { // More than 8GB RAM
            m_memoryUtilizationTarget = std::min(0.85f, m_memoryUtilizationTarget + 0.10f);
            std::cout << "[RESOURCE] Auto-tuned memory target up to " << (m_memoryUtilizationTarget * 100) << "% due to moderate RAM\n";
        }
        
        // Check current CPU utilization - if it's very low, be more aggressive
        if (m_cpuUsage < 0.3f) {
            m_cpuUtilizationTarget = std::min(0.98f, m_cpuUtilizationTarget + 0.15f);
            std::cout << "[RESOURCE] Auto-tuned CPU target up to " << (m_cpuUtilizationTarget * 100) << "% due to low usage\n";
        }

        // In extreme performance mode, push the limits further
        if (m_extremeScaling) {
            m_memoryUtilizationTarget = std::min(0.95f, m_memoryUtilizationTarget + 0.05f);
            m_cpuUtilizationTarget = 0.99f;
            std::cout << "[RESOURCE] Extreme performance mode: Memory target=" << (m_memoryUtilizationTarget * 100) 
                      << "%, CPU target=" << (m_cpuUtilizationTarget * 100) << "%\n";
        }
    }

    void update() {
        auto now = std::chrono::high_resolution_clock::now();
        if (now - m_lastCheck < m_updateInterval) {
            return; // Don't check too frequently
        }
        m_lastCheck = now;

        // Update system resource metrics
        m_availableMemoryMB = SystemResources::getAvailableMemoryMB();
        m_totalMemoryMB = SystemResources::getTotalMemoryMB();
        m_cpuUsage = SystemResources::getCPUUtilization();
        
        // Log resource utilization less frequently to reduce console spam
        static int updateCounter = 0;
        if (++updateCounter % 10 == 0) {
            std::cout << "[RESOURCE] Memory: " << m_availableMemoryMB << "MB/" << m_totalMemoryMB 
                    << "MB (" << (m_availableMemoryMB*100.0/m_totalMemoryMB) << "% free), "
                    << "CPU: " << (m_cpuUsage*100.0) << "% utilized" << std::endl;
        }
                  
        // Dynamically adjust targets if we're underutilizing resources
        if (m_aggressiveScaling || m_extremeScaling) {
            // If CPU usage is still low, gradually increase our target
            if (m_cpuUsage < m_cpuUtilizationTarget * 0.5f && m_cpuUtilizationTarget < 0.99f) {
                float increment = m_extremeScaling ? 0.10f : 0.05f;
                m_cpuUtilizationTarget = std::min(0.99f, m_cpuUtilizationTarget + increment);
                std::cout << "[RESOURCE] Dynamically increasing CPU target to " << (m_cpuUtilizationTarget * 100) << "%\n";
            }
            
            // If memory usage is still low, gradually increase our target
            if ((static_cast<float>(m_availableMemoryMB) / m_totalMemoryMB) > 0.3f && m_memoryUtilizationTarget < 0.95f) {
                float increment = m_extremeScaling ? 0.10f : 0.05f;
                m_memoryUtilizationTarget = std::min(0.95f, m_memoryUtilizationTarget + increment);
                std::cout << "[RESOURCE] Dynamically increasing memory target to " << (m_memoryUtilizationTarget * 100) << "%\n";
            }
        }
    }

    size_t getOptimalBatchSize(size_t fileSize = 0) {
        update();
        // Start with default batch size
        size_t optimalBatchSize = DEFAULT_CSV_BATCH_SIZE;
        
        // File-size aware batch sizing
        if (fileSize > 0) {
            // Adjust based on file size - larger files need larger batches
            float fileSizeGB = fileSize / (1024.0f * 1024.0f * 1024.0f);
            if (fileSizeGB > 1.0f) { // For files > 1GB
                optimalBatchSize = static_cast<size_t>(optimalBatchSize * (1.0f + fileSizeGB * 2.0f));
                std::cout << "[BATCH] File size based adjustment: " << optimalBatchSize << " rows\n";
            }
        }
        
        // Calculate memory-based batch size (aim to use target% of available memory)
        float memoryUtilizationRatio = 1.0f - (static_cast<float>(m_availableMemoryMB) / m_totalMemoryMB);
        
        if (memoryUtilizationRatio < m_memoryUtilizationTarget) {
            // We have memory to spare, increase batch size - more aggressively in performance mode
            float availableRatio = (m_memoryUtilizationTarget - memoryUtilizationRatio) / m_memoryUtilizationTarget;
            // Scale batch size: up to 20x in extreme mode, 10x in performance mode, 5x otherwise
            float scaleFactor = 1.0f;
            if (m_extremeScaling)
                scaleFactor += 19.0f * availableRatio;
            else if (m_aggressiveScaling)
                scaleFactor += 9.0f * availableRatio;
            else
                scaleFactor += 4.0f * availableRatio;
                
            optimalBatchSize = static_cast<size_t>(optimalBatchSize * scaleFactor);
        } else {
            // We're using too much memory, decrease batch size
            float overuseRatio = (memoryUtilizationRatio - m_memoryUtilizationTarget) / (1.0f - m_memoryUtilizationTarget);
            // Scale down to as low as 20% of default
            float scaleFactor = 1.0f - (0.8f * std::min(1.0f, overuseRatio));
            optimalBatchSize = static_cast<size_t>(DEFAULT_CSV_BATCH_SIZE * scaleFactor);
        }
        
        // Ensure a reasonable minimum and maximum
        size_t minBatch = 500;
        size_t maxBatch = m_extremeScaling ? 500000 : 100000;
        return std::clamp<size_t>(minBatch, optimalBatchSize, maxBatch);
    }

    unsigned int getOptimalThreadCount() {
        update();
        // Base thread count on hardware
        unsigned int baseThreadCount = std::thread::hardware_concurrency();
        
        // If CPU usage is low, we can use more threads
        if (m_cpuUsage < m_cpuUtilizationTarget) {
            float availableRatio = (m_cpuUtilizationTarget - m_cpuUsage) / m_cpuUtilizationTarget;
            // Scale up to 5x threads for unused CPU in extreme mode, 3x in performance mode, 2x otherwise
            float scaleFactor = 1.0f;
            if (m_extremeScaling)
                scaleFactor += 4.0f * availableRatio;
            else if (m_aggressiveScaling)
                scaleFactor += 2.0f * availableRatio;
            else
                scaleFactor += 1.0f * availableRatio;
                
            return static_cast<unsigned int>(baseThreadCount * scaleFactor);
        } else {
            // If CPU is overloaded, reduce threads but less aggressively in performance mode
            float overuseRatio = (m_cpuUsage - m_cpuUtilizationTarget) / (1.0f - m_cpuUtilizationTarget);
            // Scale down minimum - extreme: 90%, performance: 70%, normal: 50%
            float minScale = m_extremeScaling ? 0.9f : (m_aggressiveScaling ? 0.7f : 0.5f);
            float scaleFactor = 1.0f - ((1.0f - minScale) * std::min(1.0f, overuseRatio));
            return std::max<unsigned int>(2, static_cast<unsigned int>(baseThreadCount * scaleFactor));
        }
    }

    // Get memory usage targets for operations
    size_t getMaxMemoryUsageMB() {
        return static_cast<size_t>(m_totalMemoryMB * m_memoryUtilizationTarget);
    }

    size_t getAvailableMemoryMB() {
        update();
        return m_availableMemoryMB;
    }

    float getCurrentMemoryUtilization() {
        update();
        return 1.0f - (static_cast<float>(m_availableMemoryMB) / m_totalMemoryMB);
    }
    
    void setAggressiveScaling(bool aggressive) {
        m_aggressiveScaling = aggressive;
    }
    
    void setExtremeScaling(bool extreme) {
        m_extremeScaling = extreme;
        if (extreme) {
            // Make updates more frequent in extreme mode
            m_updateInterval = std::chrono::milliseconds(200);
            m_aggressiveScaling = true; // Extreme implies aggressive
        }
    }

private:
    size_t m_availableMemoryMB;
    size_t m_totalMemoryMB;
    float m_cpuUsage;
    float m_memoryUtilizationTarget;
    float m_cpuUtilizationTarget;
    std::chrono::high_resolution_clock::time_point m_lastCheck;
    std::chrono::duration<double> m_updateInterval;
    bool m_aggressiveScaling;
    bool m_extremeScaling;
};

// Simple thread pool with dynamic scaling for parallel processing
class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false), activeThreads(0), 
                                idealThreadCount(threads),
                                lastScaleTime(std::chrono::high_resolution_clock::now()) {
        scaleThreadPool(threads);
    }

    // Scale thread pool to new size
    void scaleThreadPool(size_t newSize) {
        if (newSize == workers.size()) return;
        
        auto now = std::chrono::high_resolution_clock::now();
        // Only scale at most once every second to prevent thrashing
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastScaleTime).count() < 1)
            return;
            
        lastScaleTime = now;

        // If growing pool, add threads
        if (newSize > workers.size()) {
            size_t threadsToAdd = newSize - workers.size();
            std::cout << "[THREADS] Scaling thread pool: " << workers.size() << " → " << newSize << "\n";
            
            for (size_t i = 0; i < threadsToAdd; ++i) {
                workers.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        {   // Acquire lock and wait for a task
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            condition.wait(lock, [this]{ return stop || !tasks.empty(); });
                            if (stop && tasks.empty())
                                return;
                            task = std::move(tasks.front());
                            tasks.pop();
                        }
                        // Track active threads for monitoring
                        activeThreads++;
                        // Execute task
                        task();
                        activeThreads--;
                    }
                });
            }
        }
        // If shrinking pool, mark for disposal (actual disposal happens in wait_all)
        else if (!stop && newSize < workers.size()) {
            std::cout << "[THREADS] Will scale down thread pool from " << workers.size() 
                      << " to " << newSize << " after current tasks\n";
            idealThreadCount = newSize;
        }
        
        // Store the ideal count for future scaling
        idealThreadCount = newSize;
    }
    
    // Try to dispose of excess threads when pool is idle
    void tryScaleDown() {
        if (workers.size() <= idealThreadCount) return;
        
        // Only attempt scale down if queue is empty and most threads are idle
        if (tasks.empty() && activeThreads < workers.size() / 2) {
            std::cout << "[THREADS] Scaling down pool from " << workers.size() 
                      << " to " << idealThreadCount << " threads\n";
                       
            // Create temporary vector of threads to dispose
            std::vector<std::thread> threadsToDispose;
            size_t numToDispose = workers.size() - idealThreadCount;
            
            // Stop the threads we want to dispose
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                // Signal threads to stop (they'll exit when they next check for work)
                stop = true;
            }
            condition.notify_all();
            
            // Move excess threads to disposal vector
            for (size_t i = 0; i < numToDispose && !workers.empty(); ++i) {
                threadsToDispose.push_back(std::move(workers.back()));
                workers.pop_back();
            }
            
            // Join and destroy the threads
            for (std::thread& thread : threadsToDispose) {
                if (thread.joinable()) thread.join();
            }
            
            // Reset stop flag for remaining threads
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = false;
            }
            
            std::cout << "[THREADS] Pool scaled down to " << workers.size() << " threads\n";
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {   // Enqueue the task
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        
        // Scale up if queue is getting too large compared to thread count
        if (tasks.size() > workers.size() * 2) {
            unsigned int optimalThreads = static_cast<unsigned int>(std::min(
                workers.size() * 1.5, // 50% increase
                static_cast<double>(std::thread::hardware_concurrency() * 4) // Cap at 4x cores
            ));
            scaleThreadPool(optimalThreads);
        }
        
        return res;
    }

    ~ThreadPool() {
        {   std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker: workers)
            worker.join();
    }

    void wait_all() {
        // More efficient wait strategy with periodic resource updates
        ResourceMonitor localMonitor(0.9f, 0.9f); // Temporary monitor for wait period
        localMonitor.setAggressiveScaling(PERFORMANCE_MODE);
        localMonitor.setExtremeScaling(EXTREME_PERFORMANCE);
        
        const auto waitStart = std::chrono::high_resolution_clock::now();
        bool longWait = false;
        
        while (true) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (tasks.empty() && activeThreads == 0) {
                    // If we've been waiting a while, try to scale down
                    auto now = std::chrono::high_resolution_clock::now();
                    if (longWait || std::chrono::duration_cast<std::chrono::seconds>(now - waitStart).count() > 5) {
                        tryScaleDown();
                        longWait = true;
                    }
                    break;
                }
            }
            
            // Update resources while waiting to potentially rescale
            localMonitor.update();
            
            // Adaptively adjust sleep time based on queue size
            std::chrono::milliseconds sleepTime;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (tasks.size() > workers.size() * 10) {
                    sleepTime = std::chrono::milliseconds(1);  // Very short sleep if queue is huge
                } else if (tasks.size() > workers.size()) {
                    sleepTime = std::chrono::milliseconds(5);  // Short sleep for long queue
                } else {
                    sleepTime = std::chrono::milliseconds(20); // Longer sleep when less work
                }
            }
            
            std::this_thread::sleep_for(sleepTime);
            
            // Check if we should scale the thread pool based on workload
            unsigned int optimalThreads = localMonitor.getOptimalThreadCount();
            if (std::abs(static_cast<int>(optimalThreads) - static_cast<int>(workers.size())) > 2) {
                scaleThreadPool(optimalThreads);
            }
        }
    }

    // Adds ability to check queue size and active threads for dynamic scaling
    size_t queueSize() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        return tasks.size();
    }
    
    size_t getActiveThreadCount() {
        return activeThreads;
    }
    
    size_t getTotalThreadCount() {
        return workers.size();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    std::atomic<size_t> activeThreads; // Track actively running threads
    size_t idealThreadCount;  // Desired thread count for scaling
    std::chrono::high_resolution_clock::time_point lastScaleTime; // Prevent too frequent scaling
};

// Parallel file processing helper
template<typename FileProcessor>
static void processFilesInParallel(const std::vector<std::string>& files, FileProcessor processor) {
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;
    ResourceMonitor localMonitor(memoryUtilTarget, cpuUtilTarget);

    // Submit initial batch of tasks
    size_t initialBatch = std::min(files.size(), static_cast<size_t>(THREAD_COUNT * 2));
    for (size_t i = 0; i < initialBatch; i++) {
        results.emplace_back(pool.enqueue(processor, files[i]));
    }
    
    // Dynamically submit remaining files based on resource utilization
    size_t nextFile = initialBatch;
    while (nextFile < files.size()) {
        // Check resource utilization
        localMonitor.update();
        
        // If we have capacity, add more files to process
        if (pool.queueSize() < THREAD_COUNT && 
            localMonitor.getCurrentMemoryUtilization() < memoryUtilTarget) {
            
            size_t batchToAdd = std::min(
                files.size() - nextFile,
                static_cast<size_t>(THREAD_COUNT - pool.queueSize())
            );
            
            if (PERFORMANCE_MODE) {
                // In performance mode, add larger batches
                batchToAdd = std::min(files.size() - nextFile, batchToAdd * 2);
            }
            
            for (size_t i = 0; i < batchToAdd; i++, nextFile++) {
                if (nextFile < files.size()) {
                    results.emplace_back(pool.enqueue(processor, files[nextFile]));
                }
            }
        }
        
        // Small delay to prevent busy wait
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    for (auto &result : results) {
        result.get();
    }
}

// For example, you could create functions such as processConditionsFilesParallel(),
// ─── (Additional parallel processing functions would be similarly refactored) ───
// For example, you could create functions such as processConditionsFilesParallel(),
// processEncountersFilesParallel(), processMedicationsFilesParallel(), etc.
// Each should follow a similar pattern: open the CSV file, split the header/rows,
// and process them (ideally using the BatchProcessor::splitCSV helper).

// Example: Processing condition files in parallel
static void processConditionsFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter &charlsonCounter,
    ThreadSafeCounter &elixhauserCounter,
    ThreadSafeCounter &comorbidityCounter)
{
    // Pre-allocate counters based on expected population size
    size_t estimatedPatientCount = files.size() * 100; // Rough estimate
    charlsonCounter.reserve(estimatedPatientCount);
    elixhauserCounter.reserve(estimatedPatientCount);
    comorbidityCounter.reserve(estimatedPatientCount);

    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;

    ResourceMonitor monitor(memoryUtilTarget, cpuUtilTarget);
    
    for (const auto &file : files) {
        results.emplace_back(pool.enqueue([&](const std::string &path) {
            // Get optimal batch size based on file size
            size_t fileSize = 0;
            try {
                fileSize = std::filesystem::file_size(path);
            } catch(...) {}
            
            size_t batchSize = monitor.getOptimalBatchSize(fileSize);
            
            std::unordered_map<std::string, float> charlsonUpdates;
            std::unordered_map<std::string, float> elixhauserUpdates;
            std::unordered_map<std::string, float> comorbidityUpdates;
            
            // Reserve space for the batch maps
            charlsonUpdates.reserve(batchSize / 10);
            elixhauserUpdates.reserve(batchSize / 10);
            comorbidityUpdates.reserve(batchSize / 10);
            
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            std::string line;
            std::getline(csv, line); // Skip header
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, codeIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "CODE") codeIdx = static_cast<int>(i);
            }
            if (patientIdx == -1 || codeIdx == -1) return;
            
            // Process each line
            size_t lineCount = 0;
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= std::max(patientIdx, codeIdx)) continue;
                
                std::string patientId = values[patientIdx];
                std::string code = values[codeIdx];
                
                // Look up weights in dictionaries
                auto charlson_it = CHARLSON_CODE_TO_WEIGHT.find(code);
                if (charlson_it != CHARLSON_CODE_TO_WEIGHT.end()) {
                    charlsonUpdates[patientId] += charlson_it->second;
                }
                
                auto elixhauser_it = ELIXHAUSER_CODE_TO_WEIGHT.find(code);
                if (elixhauser_it != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
                    elixhauserUpdates[patientId] += elixhauser_it->second;
                }
                
                double comorbidity = findGroupWeightFast(code);
                if (comorbidity > 0) {
                    comorbidityUpdates[patientId] += static_cast<float>(comorbidity);
                }
                
                lineCount++;
                
                // Batch updates for better performance
                if (lineCount % batchSize == 0) {
                    charlsonCounter.bulkAdd(charlsonUpdates);
                    elixhauserCounter.bulkAdd(elixhauserUpdates);
                    comorbidityCounter.bulkAdd(comorbidityUpdates);
                    
                    charlsonUpdates.clear();
                    elixhauserUpdates.clear();
                    comorbidityUpdates.clear();
                }
            }
            
            // Process any remaining items
            if (!charlsonUpdates.empty()) {
                charlsonCounter.bulkAdd(charlsonUpdates);
            }
            if (!elixhauserUpdates.empty()) {
                elixhauserCounter.bulkAdd(elixhauserUpdates);
            }
            if (!comorbidityUpdates.empty()) {
                comorbidityCounter.bulkAdd(comorbidityUpdates);
            }
        }, file));
    }

    for (auto &result : results) {
        result.get();
    }
}

// Process encounter files in parallel
static void processEncountersFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter &hospitalizationCounter)
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;

    for (const auto &file : files) {
        results.emplace_back(pool.enqueue([&](const std::string &path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            std::string line;
            std::getline(csv, line); // Skip header
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, classIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "ENCOUNTERCLASS") classIdx = static_cast<int>(i);
            }
            if (patientIdx == -1 || classIdx == -1) return;
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= std::max(patientIdx, classIdx)) continue;
                std::string patientId = values[patientIdx];
                std::string encounterClass = values[classIdx];
                
                // Count inpatient hospitalizations
                if (encounterClass == "inpatient") {
                    hospitalizationCounter.increment(patientId);
                }
            }
        }, file));
    }

    for (auto &result : results) {
        result.get();
    }
}

// Process medication files in parallel
static void processMedicationsFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter &medicationCounter) 
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;

    for (const auto &file : files) {
        results.emplace_back(pool.enqueue([&](const std::string &path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            std::string line;
            std::getline(csv, line); // Skip header
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, codeIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "CODE") codeIdx = static_cast<int>(i);
            }
            if (patientIdx == -1 || codeIdx == -1) return;
            
            std::unordered_set<std::string> uniqueMedsPerPatient;
            std::string currentPatient = "";
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= patientIdx) continue;
                std::string patientId = values[patientIdx];
                
                // If we've moved to a new patient, update counter for previous patient
                if (!currentPatient.empty() && currentPatient != patientId) {
                    medicationCounter.add(currentPatient, static_cast<uint16_t>(uniqueMedsPerPatient.size()));
                    uniqueMedsPerPatient.clear();
                }
                
                currentPatient = patientId;
                if (codeIdx >= 0 && values.size() > codeIdx) {
                    uniqueMedsPerPatient.insert(values[codeIdx]);
                }
            }
            
            // Don't forget to count medications for the last patient
            if (!currentPatient.empty()) {
                medicationCounter.add(currentPatient, static_cast<uint16_t>(uniqueMedsPerPatient.size()));
            }
        }, file));
    }

    for (auto &result : results) {
        result.get();
    }
}

// Process observation files in parallel
static void processObservationsFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter &abnormalObsCounter) 
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;

    for (const auto &file : files) {
        results.emplace_back(pool.enqueue([&abnormalObsCounter](const std::string &path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            std::string line;
            std::getline(csv, line); // Skip header
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1, codeIdx = -1, valueIdx = -1, descIdx = -1, unitsIdx = -1;
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
                else if (header[i] == "CODE") codeIdx = static_cast<int>(i);
                else if (header[i] == "VALUE") valueIdx = static_cast<int>(i);
                else if (header[i] == "DESCRIPTION") descIdx = static_cast<int>(i);
                else if (header[i] == "UNITS") unitsIdx = static_cast<int>(i);
            }
            if (patientIdx == -1 || codeIdx == -1 || valueIdx == -1 || descIdx == -1) return;
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                // Find the maximum index explicitly instead of using std::max
                int maxIdx = patientIdx;
                if (codeIdx > maxIdx) maxIdx = codeIdx;
                if (valueIdx > maxIdx) maxIdx = valueIdx;
                if (descIdx > maxIdx) maxIdx = descIdx;
                if (unitsIdx > maxIdx) maxIdx = unitsIdx;
                
                if (values.size() <= maxIdx) continue;
                
                std::string patientId = values[patientIdx];
                std::string code = values[codeIdx];
                std::string valueStr = values[valueIdx];
                std::string description = values[descIdx];
                std::string units = unitsIdx >= 0 && values.size() > unitsIdx ? values[unitsIdx] : "";
                
                // Check if observation is abnormal using lookup function from MedicalDictionaries.h
                if (!valueStr.empty()) {
                    try {
                        double value = std::stod(valueStr);
                        // Use one of the available abnormal observation functions
                        if (isAbnormalObsFast(description, value)) {
                            abnormalObsCounter.addInt(patientId, 1); // Changed from increment to addInt
                        }
                    } catch (...) {
                        // Handle conversion errors
                    }
                }
            }
        }, file));
    }

    for (auto &result : results) {
        result.get();
    }
}

// Process patient files in parallel 
static void processPatientsFilesParallel(
    const std::vector<std::string>& files,
    std::vector<PatientRecord> &allPatients)
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<std::vector<PatientRecord>>> results;
    std::mutex patientsMutex;
    
    for (const auto &file : files) {
        results.emplace_back(pool.enqueue([&](const std::string &path) {
            std::vector<PatientRecord> localPatients;
            std::ifstream csv(path);
            if (!csv.is_open()) return localPatients;
            
            std::string line;
            std::getline(csv, line); // Skip header
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            std::unordered_map<std::string, int> colMap;
            for (size_t i = 0; i < header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            bool isDiff = isDiffFile(path);
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                PatientRecord p;
                
                // Helper to get values safely
                auto getValue = [&](const std::string &col) -> std::string {
                    auto it = colMap.find(col);
                    if (it != colMap.end() && it->second < static_cast<int>(values.size())) {
                        return values[it->second];
                    }
                    return "";
                };
                
                p.Id = getValue("Id");
                p.BIRTHDATE = getValue("BIRTHDATE");
                p.DEATHDATE = getValue("DEATHDATE");
                p.GENDER = getValue("GENDER");
                p.RACE = getValue("RACE");
                p.ETHNICITY = getValue("ETHNICITY");
                p.MARITAL = getValue("MARITAL");
                p.NewData = isDiff;
                
                // Try to convert numeric fields
                try {
                    std::string val = getValue("HEALTHCARE_EXPENSES");
                    if (!val.empty()) p.HEALTHCARE_EXPENSES = std::stof(val);
                    
                    val = getValue("HEALTHCARE_COVERAGE");
                    if (!val.empty()) p.HEALTHCARE_COVERAGE = std::stof(val);
                    
                    val = getValue("INCOME");
                    if (!val.empty()) p.INCOME = std::stof(val);
                } catch (...) {
                    // Continue if conversion fails
                }
                
                // Calculate age and deceased status
                if (!p.BIRTHDATE.empty() && p.BIRTHDATE != "NaN") {
                    try {
                        int birthYear = std::stoi(p.BIRTHDATE.substr(0, 4));
                        std::time_t t = std::time(nullptr);
                        std::tm* now = std::localtime(&t);
                        p.AGE = (now->tm_year + 1900) - birthYear;
                    } catch (...) {
                        // Default age if calculation fails
                    }
                }
                
                if (!p.DEATHDATE.empty() && p.DEATHDATE != "NaN") {
                    p.DECEASED = true;
                }
                
                localPatients.push_back(p);
            }
            
            return localPatients;
        }, file));
    }
    
    // Collect all patient records
    for (auto &result : results) {
        auto patientBatch = result.get();
        std::lock_guard<std::mutex> lock(patientsMutex);
        allPatients.insert(allPatients.end(), patientBatch.begin(), patientBatch.end());
    }
}

// Process procedure files in parallel
static void processProceduresFilesParallel(
    const std::vector<std::string>& files,
    ThreadSafeCounter &proceduresCounter)
{
    ThreadPool pool(THREAD_COUNT);
    std::vector<std::future<void>> results;
    
    for (const auto &file : files) {
        results.emplace_back(pool.enqueue([&](const std::string &path) {
            std::ifstream csv(path);
            if (!csv.is_open()) return;
            
            std::string line;
            std::getline(csv, line); // Skip header
            std::vector<std::string> header = BatchProcessor::splitCSV(line);
            int patientIdx = -1;
            
            for (size_t i = 0; i < header.size(); i++) {
                if (header[i] == "PATIENT") patientIdx = static_cast<int>(i);
            }
            
            if (patientIdx == -1) return;
            
            // Process each line
            while (std::getline(csv, line)) {
                std::vector<std::string> values = BatchProcessor::splitCSV(line);
                if (values.size() <= patientIdx) continue;
                std::string patientId = values[patientIdx];
                
                // Increment procedure count for this patient
                proceduresCounter.increment(patientId);
            }
        }, file));
    }
    
    for (auto &result : results) {
        result.get();
    }
}

// Forward declarations for functions used in main()
static std::vector<std::string> getFeatureCols(const std::string &feature_config);
static void writeFeaturesCSV(const std::vector<PatientRecord> &pats,
                           const std::string &outFile,
                           const std::vector<std::string> &cols);
static void saveFinalDataCSV(const std::vector<PatientRecord> &pats,
                           const std::string &outfile);

// ─── Main function ───
int main(int argc, char* argv[]) {
    auto PROGRAM_START_TIME = std::chrono::high_resolution_clock::now();
    int popSize = 100;
    bool enableXAI = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--population=", 0) == 0) {
            popSize = std::stoi(arg.substr(13));
        } else if (arg == "--enable-xai") {
            enableXAI = true;
        } else if (arg.rfind("--threads=", 0) == 0) {
            THREAD_COUNT = std::stoi(arg.substr(10));
        } else if (arg.rfind("--memory-util=", 0) == 0) {
            memoryUtilTarget = std::stof(arg.substr(14)) / 100.0f; // Convert percentage to ratio
        } else if (arg.rfind("--cpu-util=", 0) == 0) {
            cpuUtilTarget = std::stof(arg.substr(11)) / 100.0f; // Convert percentage to ratio
        } else if (arg == "--performance-mode") {
            PERFORMANCE_MODE = true;
            // In performance mode, be much more aggressive with resource utilization
            memoryUtilTarget = std::min(0.9f, memoryUtilTarget + 0.1f); 
            cpuUtilTarget = std::min(0.95f, cpuUtilTarget + 0.1f);
            std::cout << "[INFO] Performance mode enabled: Using more aggressive resource utilization\n";
        } else if (arg == "--extreme-performance") {
            EXTREME_PERFORMANCE = true;
            PERFORMANCE_MODE = true;
            // In extreme mode, push the system to its limits
            memoryUtilTarget = 0.95f;
            cpuUtilTarget = 0.99f;
            std::cout << "[INFO] EXTREME PERFORMANCE MODE: Maximum resource utilization enabled!\n";
        }
    }
    
    // Initialize resource monitor with target utilization levels
    ResourceMonitor resourceMonitor(memoryUtilTarget, cpuUtilTarget);
    if (PERFORMANCE_MODE) {
        resourceMonitor.setAggressiveScaling(true);
    }
    if (EXTREME_PERFORMANCE) {
        resourceMonitor.setExtremeScaling(true);
    }
    
    // Check system memory and scale batch size accordingly
    size_t totalMemory = SystemResources::getTotalMemoryMB();
    if (totalMemory > 32000) { // >32GB RAM
        DEFAULT_CSV_BATCH_SIZE = std::min<size_t>(50000, DEFAULT_CSV_BATCH_SIZE * (totalMemory / 4000));
        std::cout << "[INFO] Increased batch size to " << DEFAULT_CSV_BATCH_SIZE 
                  << " based on high system memory: " << totalMemory << "MB\n";
    } else if (totalMemory > 16000) { // >16GB RAM
        // Scale default batch size based on system memory
        DEFAULT_CSV_BATCH_SIZE = std::min<size_t>(20000, DEFAULT_CSV_BATCH_SIZE * (totalMemory / 8000));
        std::cout << "[INFO] Increased batch size to " << DEFAULT_CSV_BATCH_SIZE 
                  << " based on high system memory: " << totalMemory << "MB\n";
    }
    
    // Dynamically adjust thread count based on system resources
    if (THREAD_COUNT == DEFAULT_THREAD_COUNT) {
        THREAD_COUNT = resourceMonitor.getOptimalThreadCount();
    }
    
    // Adjust batch size based on available memory
    size_t dynamicBatchSize = resourceMonitor.getOptimalBatchSize();
    
    std::cout << "[INFO] popSize=" << popSize
              << ", XAI=" << (enableXAI ? "true" : "false")
              << ", threads=" << THREAD_COUNT 
              << ", memoryTarget=" << (memoryUtilTarget * 100) << "%"
              << ", cpuTarget=" << (cpuUtilTarget * 100) << "%"
              << ", batchSize=" << dynamicBatchSize
              << ", performanceMode=" << (PERFORMANCE_MODE ? "true" : "false") << "\n";
    std::cout << "[SYSTEM] Total Memory: " << SystemResources::getTotalMemoryMB() << "MB, "
              << "Available Memory: " << resourceMonitor.getAvailableMemoryMB() << "MB, "
              << "CPU Cores: " << std::thread::hardware_concurrency() << "\n";
    // 1) Run Synthea and copy output
    std::cout << "[INFO] Starting Synthea generation...\n";
    runSynthea(popSize);
    copySyntheaOutput();
    // 2) Initialize lookup tables (assumed implemented in MedicalDictionaries.h)
    std::cout << "[INFO] Building optimized lookup tables...\n";
    initializeDirectLookups();
    initializeObsAbnormalDirect();
    // 3) Create thread-safe data structures for parallel processing
    ThreadSafeCounter charlsonCounter;
    ThreadSafeCounter elixhauserCounter;
    ThreadSafeCounter comorbidityCounter;
    ThreadSafeCounter hospitalizationCounter;
    ThreadSafeCounter medicationCounter;
    ThreadSafeCounter abnormalObsCounter;
    ThreadSafeCounter proceduresCounter;
    // 4) Process condition files in parallel   
    std::cout << "[INFO] Processing condition files in parallel...\n";
    auto condFiles = listCSVFiles("conditions");
    processConditionsFilesParallel(condFiles, charlsonCounter, elixhauserCounter, comorbidityCounter);
    // Process encounters files
    auto encFiles = listCSVFiles("encounters");
    processEncountersFilesParallel(encFiles, hospitalizationCounter);
    // Process medications files
    auto medFiles = listCSVFiles("medications");
    processMedicationsFilesParallel(medFiles, medicationCounter);
    // Process observations files
    auto obsFiles = listCSVFiles("observations");
    processObservationsFilesParallel(obsFiles, abnormalObsCounter);
    // Process patient files (need to initialize allPatients vector first)
    std::vector<PatientRecord> allPatients;
    auto patFiles = listCSVFiles("patients");
    processPatientsFilesParallel(patFiles, allPatients);
    // Process procedure files
    auto procFiles = listCSVFiles("procedures");
    processProceduresFilesParallel(procFiles, proceduresCounter);
    // (Additional processing for encounters, medications, observations, patients, etc.)
    // You would add similar functions here following the above pattern.
    // 5) Finalize predictions and (optionally) run XAI inference
    // This section would include building feature CSVs, running Python inference,
    // and optionally calling runExplainability().
    // Update patient records with metrics from counters
    for (auto &patient : allPatients) {
        patient.CharlsonIndex = charlsonCounter.getFloat(patient.Id);
        patient.ElixhauserIndex = elixhauserCounter.getFloat(patient.Id);
        patient.Comorbidity_Score = comorbidityCounter.getFloat(patient.Id);
        patient.Hospitalizations_Count = hospitalizationCounter.getInt(patient.Id);
        patient.Medications_Count = medicationCounter.getInt(patient.Id);
        patient.Abnormal_Observations_Count = abnormalObsCounter.getInt(patient.Id);
        // Calculate health index directly here since the function isn't found
        double raw = 10.0 - (
            0.2 * patient.CharlsonIndex +
            0.1 * patient.ElixhauserIndex +
            0.1 * std::min(10.0, (double)patient.Hospitalizations_Count) +
            0.05 * std::min(20.0, (double)patient.Medications_Count) +
            0.05 * std::min(20.0, (double)patient.Abnormal_Observations_Count)
        );
        // Apply penalty for extreme age
        double agePenalty = 0.0;
        if (patient.AGE > 65) {
            agePenalty = (patient.AGE - 65) * 0.05;
        }
        // Apply deceased penalty
        double deceasedPenalty = patient.DECEASED ? 2.0 : 0.0;
        // Adjust score
        raw -= (agePenalty + deceasedPenalty);
        // Clamp to valid range
        if (raw < 1.0) raw = 1.0;
        if (raw > 10.0) raw = 10.0;
        patient.Health_Index = raw;
    }
    // 6) Wrap up and report elapsed time
    auto endTime = std::chrono::high_resolution_clock::now();
    double durationSec = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - PROGRAM_START_TIME).count() / 1000.0;
    std::cout << "[TIME] Total execution time: " << durationSec << " seconds\n";
    // Save the processed patient data
    std::string outputFile = "Data/processed_patients_" + getTimestamp() + ".csv";
    saveFinalDataCSV(allPatients, outputFile);
    // Optionally, write features for model input
    std::string featureFile = "Data/features_" + getTimestamp() + ".csv";
    std::vector<std::string> featureCols = getFeatureCols("combined_all");
    writeFeaturesCSV(allPatients, featureFile, featureCols);
    // Finalize Python (if used)
    if (Py_IsInitialized())
        Py_Finalize();
    return 0;
}

/****************************************************
 * PATIENT SUBSET SELECTION: DIABETES, CKD, ETC.
 ****************************************************/

static std::unordered_set<std::string> findDiabeticPatientsOptimized(const std::vector<ConditionRow> &conds) {
    std::unordered_set<std::string> out;
    out.reserve(conds.size() / 10); // Pre-allocate with estimate of diabetic patients
    
    // Process in parallel if large dataset
    if (conds.size() > 50000 && THREAD_COUNT > 1) {
        std::mutex resultMutex;
        std::vector<std::thread> threads;
        const size_t chunkSize = (conds.size() + THREAD_COUNT - 1) / THREAD_COUNT;
        
        for (unsigned int t = 0; t < THREAD_COUNT; t++) {
            threads.emplace_back([&, t]() {
                std::unordered_set<std::string> localOut;
                localOut.reserve(chunkSize / 10);
                
                size_t start = t * chunkSize;
                size_t end = std::min(start + chunkSize, conds.size());
                
                for (size_t i = start; i < end; i++) {
                    std::string lower = conds[i].DESCRIPTION;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    
                    if (lower.find("diabetes") != std::string::npos || 
                        lower.find("diabetic") != std::string::npos) {
                        localOut.insert(conds[i].PATIENT);
                    }
                }
                
                // Merge results
                std::lock_guard<std::mutex> lock(resultMutex);
                out.insert(localOut.begin(), localOut.end());
            });
        }
        
        // Join threads
        for (auto &t : threads) {
            if (t.joinable()) t.join();
        }
    } 
    else {
        // Original sequential code for smaller datasets
        for (const auto& c : conds) {
            std::string lower = c.DESCRIPTION;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            if (lower.find("diabetes") != std::string::npos || 
                lower.find("diabetic") != std::string::npos) {
                out.insert(c.PATIENT);
            }
        }
    }
    
    return out;
}

static std::unordered_set<std::string> findCKDPatientsOptimized(const std::vector<ConditionRow> &conds) {
    std::unordered_set<std::string> out;
    out.reserve(conds.size() / 10); // Pre-allocate with estimate of CKD patients
    
    // Process in parallel if large dataset
    if (conds.size() > 50000 && THREAD_COUNT > 1) {
        std::mutex resultMutex;
        std::vector<std::thread> threads;
        const size_t chunkSize = (conds.size() + THREAD_COUNT - 1) / THREAD_COUNT;
        
        for (unsigned int t = 0; t < THREAD_COUNT; t++) {
            threads.emplace_back([&, t]() {
                std::unordered_set<std::string> localOut;
                localOut.reserve(chunkSize / 10);
                
                size_t start = t * chunkSize;
                size_t end = std::min(start + chunkSize, conds.size());
                
                for (size_t i = start; i < end; i++) {
                    std::string lower = conds[i].DESCRIPTION;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    
                    if (lower.find("chronic kidney disease") != std::string::npos ||
                        lower.find("ckd") != std::string::npos ||
                        lower.find("renal failure") != std::string::npos) {
                        localOut.insert(conds[i].PATIENT);
                    }
                }
                
                // Merge results
                std::lock_guard<std::mutex> lock(resultMutex);
                out.insert(localOut.begin(), localOut.end());
            });
        }
        
        // Join threads
        for (auto &t : threads) {
            if (t.joinable()) t.join();
        }
    } 
    else {
        // Original sequential code for smaller datasets
        for (const auto& c : conds) {
            std::string lower = c.DESCRIPTION;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            if (lower.find("chronic kidney disease") != std::string::npos ||
                lower.find("ckd") != std::string::npos ||
                lower.find("renal failure") != std::string::npos) {
                out.insert(c.PATIENT);
            }
        }
    }
    
    return out;
}

static std::unordered_set<std::string> findPatientSubset(const std::string& subsetType, 
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

/****************************************************
 * FEATURE UTILS
 ****************************************************/

static std::vector<std::string> getFeatureCols(const std::string &feature_config) {
    // Base features that are always included
    std::vector<std::string> base = {
        "Id","GENDER","RACE","ETHNICITY","MARITAL",
        "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","INCOME",
        "AGE","DECEASED",
        "Hospitalizations_Count","Medications_Count","Abnormal_Observations_Count"
    };
    
    if (feature_config == "composite") {
        base.push_back("Health_Index");
    } else if (feature_config == "cci") {
        base.push_back("CharlsonIndex");
    } else if (feature_config == "eci") {
        base.push_back("ElixhauserIndex");
    } else if (feature_config == "combined") {
        base.push_back("Health_Index");
        base.push_back("CharlsonIndex");
    } else if (feature_config == "combined_eci") {
        base.push_back("Health_Index");
        base.push_back("ElixhauserIndex");
    } else if (feature_config == "combined_all") {
        base.push_back("Health_Index");
        base.push_back("CharlsonIndex");
        base.push_back("ElixhauserIndex");
    } else {
        std::cerr << "[ERROR] invalid feature_config: " << feature_config << "\n";
    }
    return base;
}

static void writeFeaturesCSV(const std::vector<PatientRecord> &pats,
                            const std::string &outFile,
                            const std::vector<std::string> &cols) {
    std::ofstream ofs(outFile);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] cannot open " << outFile << "\n";
        return;
    }
    // Write header
    for (size_t i=0; i<cols.size(); i++) {
        ofs << cols[i];
        if (i+1 < cols.size()) ofs << ",";
    }
    ofs << "\n";
    // Write data rows
    for (auto &p : pats) {
        for (size_t c=0; c<cols.size(); c++) {
            std::string col = cols[c];
            if (c > 0) ofs << ",";
            if (col=="Id")                  { ofs << p.Id; }
            else if (col=="GENDER")         { ofs << p.GENDER; }
            else if (col=="RACE")           { ofs << p.RACE; }
            else if (col=="ETHNICITY")      { ofs << p.ETHNICITY; }
            else if (col=="MARITAL")        { ofs << p.MARITAL; }
            else if (col=="HEALTHCARE_EXPENSES") { ofs << p.HEALTHCARE_EXPENSES; }
            else if (col=="HEALTHCARE_COVERAGE") { ofs << p.HEALTHCARE_COVERAGE; }
            else if (col=="INCOME")         { ofs << p.INCOME; }
            else if (col=="AGE")            { ofs << p.AGE; }
            else if (col=="DECEASED")       { ofs << (p.DECEASED ? "1":"0"); }
            else if (col=="Hospitalizations_Count") { ofs << p.Hospitalizations_Count; }
            else if (col=="Medications_Count") { ofs << p.Medications_Count; }
            else if (col=="Abnormal_Observations_Count") { ofs << p.Abnormal_Observations_Count; }
            else if (col=="CharlsonIndex")  { ofs << p.CharlsonIndex; }
            else if (col=="ElixhauserIndex"){ ofs << p.ElixhauserIndex; }
            else if (col=="Health_Index")   { ofs << p.Health_Index; }
            else if (col=="Comorbidity_Score") { ofs << p.Comorbidity_Score; }
        }
        ofs << "\n";
    }
    std::cout << "[INFO] Wrote " << pats.size() << " patient records to " << outFile << "\n";
}

static void saveFinalDataCSV(const std::vector<PatientRecord> &pats, const std::string &outfile) {
    std::ofstream ofs(outfile);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Cannot open " << outfile << " for writing.\n";
        return;
    }
    // Write comprehensive header with all fields
    ofs << "Id,GENDER,RACE,ETHNICITY,MARITAL,BIRTHDATE,DEATHDATE,"
        << "HEALTHCARE_EXPENSES,HEALTHCARE_COVERAGE,INCOME,"
        << "AGE,DECEASED,CharlsonIndex,ElixhauserIndex,Comorbidity_Score,"
        << "Hospitalizations_Count,Medications_Count,Abnormal_Observations_Count,Health_Index\n";
    // Write data rows
    for (const auto &p : pats) {
        ofs << p.Id << "," 
            << p.GENDER << "," 
            << p.RACE << "," 
            << p.ETHNICITY << "," 
            << p.MARITAL << "," 
            << p.BIRTHDATE << "," 
            << p.DEATHDATE << ","
            << p.HEALTHCARE_EXPENSES << "," 
            << p.HEALTHCARE_COVERAGE << "," 
            << p.INCOME << ","
            << p.AGE << "," 
            << (p.DECEASED ? "1" : "0") << "," 
            << p.CharlsonIndex << "," 
            << p.ElixhauserIndex << "," 
            << p.Comorbidity_Score << ","
            << p.Hospitalizations_Count << "," 
            << p.Medications_Count << "," 
            << p.Abnormal_Observations_Count << "," 
            << p.Health_Index;
        ofs << "\n";
    }
    std::cout << "[INFO] Saved " << pats.size() << " complete patient records to " << outfile << "\n";
}

// Process any remaining items
static void processConditionsInBatches(const std::string &path,
                                       std::function<void(const ConditionRow&)> callback) {
    BatchProcessor::processFile<ConditionRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> ConditionRow {
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            
            ConditionRow c;
            c.PATIENT = getValue("PATIENT");
            c.CODE = getValue("CODE");
            c.DESCRIPTION = getValue("DESCRIPTION");
            return c;
        },
        callback
    );
}

static void processMedicationsInBatches(const std::string &path,
                                        std::function<void(const MedicationRow&)> callback) {
    BatchProcessor::processFile<MedicationRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> MedicationRow {
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            
            MedicationRow m;
            m.PATIENT = getValue("PATIENT");
            m.ENCOUNTER = getValue("ENCOUNTER");
            m.CODE = getValue("CODE");
            m.DESCRIPTION = getValue("DESCRIPTION");
            return m;
        },
        callback
    );
}

static void processObservationsInBatches(const std::string &path,
                                         std::function<void(const ObservationRow&)> callback) {
    BatchProcessor::processFile<ObservationRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> ObservationRow {
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            
            ObservationRow o;
            o.PATIENT = getValue("PATIENT");
            o.ENCOUNTER = getValue("ENCOUNTER");
            o.CODE = getValue("CODE");
            o.DESCRIPTION = getValue("DESCRIPTION");
            try {
                std::string valueStr = getValue("VALUE");
                if (!valueStr.empty()) {
                    o.VALUE = std::stod(valueStr);
                }
            } catch (...) {
                // If conversion fails, keep default value
            }
            o.UNITS = getValue("UNITS");
            return o;
        },
        callback
    );
}

static void processProceduresInBatches(const std::string &path,
                                       std::function<void(const ProcedureRow&)> callback) {
    BatchProcessor::processFile<ProcedureRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> ProcedureRow {
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            
            ProcedureRow p;
            p.PATIENT = getValue("PATIENT");
            p.ENCOUNTER = getValue("ENCOUNTER");
            p.CODE = getValue("CODE");
            p.DESCRIPTION = getValue("DESCRIPTION");
            return p;
        },
        callback
    );
}

static void processEncountersInBatches(const std::string &path,
                                       std::function<void(const EncounterRow&)> callback) {
    BatchProcessor::processFile<EncounterRow>(
        path,
        [](const std::vector<std::string> &header,
           const std::vector<std::string> &values) -> EncounterRow {
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            
            EncounterRow e;
            e.Id = getValue("Id");
            e.PATIENT = getValue("PATIENT");
            e.ENCOUNTERCLASS = getValue("ENCOUNTERCLASS");
            return e;
        },
        callback
    );
}

static void processPatientsInBatches(const std::string &path,
                                     std::function<void(const PatientRecord&)> callback) {
    BatchProcessor::processFile<PatientRecord>(
        path,
        [path](const std::vector<std::string> &header,  // Fix: explicitly capture path
           const std::vector<std::string> &values) -> PatientRecord {
            std::unordered_map<std::string,int> colMap;
            for (size_t i=0; i<header.size(); i++) {
                colMap[header[i]] = static_cast<int>(i);
            }
            
            auto getValue = [&](const std::string &col) -> std::string {
                auto it = colMap.find(col);
                if (it != colMap.end() && it->second < (int)values.size()) {
                    return values[it->second];
                }
                return "";
            };
            
            PatientRecord p;
            p.Id = getValue("Id");
            p.BIRTHDATE = getValue("BIRTHDATE");
            p.DEATHDATE = getValue("DEATHDATE");
            p.GENDER = getValue("GENDER");
            p.RACE = getValue("RACE");
            p.ETHNICITY = getValue("ETHNICITY");
            p.MARITAL = getValue("MARITAL");
            p.NewData = isDiffFile(path);
            
            // Convert numeric fields
            try {
                std::string val;
                val = getValue("HEALTHCARE_EXPENSES");
                if (!val.empty()) p.HEALTHCARE_EXPENSES = std::stof(val);
                
                val = getValue("HEALTHCARE_COVERAGE");
                if (!val.empty()) p.HEALTHCARE_COVERAGE = std::stof(val);
                
                val = getValue("INCOME");
                if (!val.empty()) p.INCOME = std::stof(val);
                
                // Calculate age if birthdate is available
                if (!p.BIRTHDATE.empty() && p.BIRTHDATE != "NaN") {
                    // Simple age calculation (could be enhanced)
                    int birthYear = std::stoi(p.BIRTHDATE.substr(0, 4));
                    int currentYear = std::chrono::system_clock::now().time_since_epoch().count() / 31557600000000000LL + 1970;
                    p.AGE = currentYear - birthYear;
                }
                
                // Set DECEASED flag if deathdate is present
                if (!p.DEATHDATE.empty() && p.DEATHDATE != "NaN") {
                    p.DECEASED = true;
                }
            } catch (...) {
                // Continue with default values if conversion fails
            }
            
            return p;
        },
        callback
    );
}

static void countHospitalizationsInBatches(std::function<void(const std::string&, uint16_t)> countCallback) {
    auto encFiles = listCSVFiles("encounters");
    for (auto &path : encFiles) {
        processEncountersInBatches(path, [&](const EncounterRow &e){
            if (e.ENCOUNTERCLASS == "inpatient") {
                // Increment hospitalization count for this patient
                countCallback(e.PATIENT, 1);
            }
        });
    }
}

static void computeCharlsonIndexBatched(std::function<void(const std::string&, float)> scoreCallback) {
    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            auto it = CHARLSON_CODE_TO_WEIGHT.find(c.CODE);
            if (it != CHARLSON_CODE_TO_WEIGHT.end()) {
                scoreCallback(c.PATIENT, it->second);
            }
        });
    }
}

static void computeElixhauserIndexBatched(std::function<void(const std::string&, float)> scoreCallback) {
    auto condFiles = listCSVFiles("conditions");
    for (auto &path : condFiles) {
        processConditionsInBatches(path, [&](const ConditionRow &c){
            auto it = ELIXHAUSER_CODE_TO_WEIGHT.find(c.CODE);
            if (it != ELIXHAUSER_CODE_TO_WEIGHT.end()) {
                scoreCallback(c.PATIENT, it->second);
            }
        });
    }
}

static double computeHealthIndex(const PatientRecord &p) {
    // Calculate health index using a weighted formula
    double raw = 10.0 - (
        0.2 * p.CharlsonIndex + 
        0.1 * p.ElixhauserIndex +
        0.1 * std::min(10.0, (double)p.Hospitalizations_Count) +
        0.05 * std::min(20.0, (double)p.Medications_Count) +
        0.05 * std::min(20.0, (double)p.Abnormal_Observations_Count)
    );
    
    // Apply penalty for extreme age
    double agePenalty = 0.0;
    if (p.AGE > 65) {
        agePenalty = (p.AGE - 65) * 0.05;
    }
    
    // Apply deceased penalty
    double deceasedPenalty = p.DECEASED ? 2.0 : 0.0;
    
    // Adjust score
    raw -= (agePenalty + deceasedPenalty);
    
    // Clamp to valid range
    if (raw < 1.0) raw = 1.0;
    if (raw > 10.0) raw = 10.0;
    
    return raw;
}

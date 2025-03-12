#pragma once

#include "SystemResources.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <functional>
#include <atomic>
#include <mutex>

/**
 * AggressiveResourceManager - A utility class for maximizing system resource usage
 * 
 * This class provides:
 * - Background monitoring of system resources
 * - Dynamic adjustment of thread count and batch sizes
 * - Automatic scaling based on workload patterns
 * - Forces the system to utilize available resources
 */
class AggressiveResourceManager {
public:
    AggressiveResourceManager(bool extremeMode = false) 
        : m_running(false), 
          m_extremeMode(extremeMode),
          m_memoryTarget(extremeMode ? 0.95f : 0.90f),
          m_cpuTarget(extremeMode ? 0.99f : 0.95f),
          m_currentThreads(std::thread::hardware_concurrency()),
          m_currentBatchSize(5000)
    {
        // Get system capabilities
        m_totalMemoryMB = SystemResources::getTotalMemoryMB();
        m_hardwareThreads = std::thread::hardware_concurrency();
        
        // Auto-tune based on system specs
        if (m_extremeMode) {
            // More aggressive tuning for extreme mode
            try {
                // Try to use the specialized methods if available
                m_memoryTarget = SystemResources::getAggressiveMemoryTarget();
                m_cpuTarget = SystemResources::getAggressiveCpuTarget();
            } catch(...) {
                // Fall back to hardcoded values if methods don't exist
                size_t totalMemoryMB = SystemResources::getTotalMemoryMB();
                unsigned int cores = SystemResources::getCpuCores();
                
                // Set memory target based on system memory
                if (totalMemoryMB > 32000) m_memoryTarget = 0.95f;
                else if (totalMemoryMB > 16000) m_memoryTarget = 0.92f;
                else if (totalMemoryMB > 8000) m_memoryTarget = 0.88f;
                else m_memoryTarget = 0.85f;
                
                // Set CPU target based on core count
                if (cores > 16) m_cpuTarget = 0.98f;
                else if (cores > 8) m_cpuTarget = 0.96f;
                else if (cores > 4) m_cpuTarget = 0.94f;
                else m_cpuTarget = 0.92f;
            }
            
            // Start with higher batch size in extreme mode
            if (m_totalMemoryMB > 32000) { // >32GB RAM
                m_currentBatchSize = 50000;
            } else if (m_totalMemoryMB > 16000) { // >16GB RAM
                m_currentBatchSize = 25000;
            } else if (m_totalMemoryMB > 8000) { // >8GB RAM
                m_currentBatchSize = 15000;
            }
        }
        
        std::cout << "[RESOURCE-MGR] Initialized with memory target: " << (m_memoryTarget * 100)
                  << "%, CPU target: " << (m_cpuTarget * 100) << "%" << std::endl;
    }
    
    ~AggressiveResourceManager() {
        stop();
    }
    
    // Start background monitoring thread
    void start() {
        if (m_running) return;
        
        m_running = true;
        m_monitorThread = std::thread(&AggressiveResourceManager::monitorLoop, this);
        
        std::cout << "[RESOURCE-MGR] Background resource monitoring started" << std::endl;
    }
    
    // Stop background monitoring
    void stop() {
        if (!m_running) return;
        
        m_running = false;
        if (m_monitorThread.joinable()) {
            m_monitorThread.join();
        }
    }
    
    // Register thread pool for auto-scaling
    void registerThreadPool(std::function<void(unsigned int)> scaleCallback) {
        std::lock_guard<std::mutex> lock(m_callbackMutex);
        m_threadScaleCallbacks.push_back(scaleCallback);
    }
    
    // Register batch size handler for auto-scaling
    void registerBatchSizeCallback(std::function<void(size_t)> batchCallback) {
        std::lock_guard<std::mutex> lock(m_callbackMutex);
        m_batchSizeCallbacks.push_back(batchCallback);
    }
    
    // Get optimal thread count for current system state
    unsigned int getOptimalThreadCount() const {
        unsigned int baseThreadCount = m_hardwareThreads;
        float cpuUsage = SystemResources::getCPUUtilization();
        
        // If using aggressive mode, use the aggressive version
        if (m_extremeMode) {
            return SystemResources::getAggressiveThreadCount(m_cpuTarget);
        }
        
        // If we're below target utilization, scale up threads
        if (cpuUsage < m_cpuTarget) {
            float availableRatio = (m_cpuTarget - cpuUsage) / m_cpuTarget;
            
            // Extreme mode: up to 8x hardware threads
            // Normal mode: up to 4x hardware threads
            float maxMultiplier = m_extremeMode ? 8.0f : 4.0f;
            float multiplier = 1.0f + (maxMultiplier - 1.0f) * availableRatio;
            
            return static_cast<unsigned int>(baseThreadCount * multiplier);
        } else {
            // Even when at/above target, keep at least hardware concurrency level
            return baseThreadCount;
        }
    }
    
    // Get optimal batch size for current memory state
    size_t getOptimalBatchSize(size_t rowSizeEstimateBytes = 500) const {
        // If using aggressive mode, use the aggressive version
        if (m_extremeMode) {
            return SystemResources::getAggressiveBatchSize(rowSizeEstimateBytes, m_memoryTarget);
        }
        
        size_t availableMemMB = SystemResources::getAvailableMemoryMB();
        float memoryUtilizationRatio = 1.0f - (static_cast<float>(availableMemMB) / m_totalMemoryMB);
        
        // Base batch size on total system memory
        size_t baseBatchSize;
        if (m_totalMemoryMB > 32000) { // >32GB RAM
            baseBatchSize = 25000;
        } else if (m_totalMemoryMB > 16000) { // >16GB RAM
            baseBatchSize = 15000;
        } else if (m_totalMemoryMB > 8000) { // >8GB RAM
            baseBatchSize = 10000;
        } else { // <8GB RAM
            baseBatchSize = 5000;
        }
        
        // If we're below target memory utilization, increase batch size
        if (memoryUtilizationRatio < m_memoryTarget) {
            float availableRatio = (m_memoryTarget - memoryUtilizationRatio) / m_memoryTarget;
            
            // Aggressive scaling - extreme mode: up to 20x base, normal: up to 10x base
            float maxMultiplier = m_extremeMode ? 20.0f : 10.0f;
            float multiplier = 1.0f + (maxMultiplier - 1.0f) * availableRatio;
            
            return static_cast<size_t>(baseBatchSize * multiplier);
        } else if (memoryUtilizationRatio > m_memoryTarget + 0.05f) {
            // If we're over target + buffer, decrease batch size
            float overuseRatio = (memoryUtilizationRatio - m_memoryTarget) / (1.0f - m_memoryTarget);
            
            // Scale down to minimum 20% of base
            float multiplier = std::max(0.2f, 1.0f - overuseRatio);
            
            return static_cast<size_t>(baseBatchSize * multiplier);
        } else {
            // Within target range, keep current base batch size
            return baseBatchSize;
        }
    }
    
    // Force CPU activity to increase utilization
    void forceCpuUtilization() {
        float currentCpuUsage = SystemResources::getCPUUtilization();
        
        // Only force utilization if we're significantly below target
        if (currentCpuUsage < m_cpuTarget * 0.8f) {
            // Launch specified number of busy threads that burn CPU
            unsigned int threadsToLaunch = static_cast<unsigned int>(
                (m_cpuTarget - currentCpuUsage) * m_hardwareThreads * 2.0f
            );
            
            if (threadsToLaunch > 0) {
                std::vector<std::thread> busyThreads;
                std::atomic<bool> stopBusy{false};
                
                for (unsigned int i = 0; i < threadsToLaunch; i++) {
                    busyThreads.emplace_back([&stopBusy](){
                        // Do meaningless work to consume CPU cycles
                        volatile double result = 0.0;
                        while (!stopBusy) {
                            for (int j = 0; j < 10000000; j++) {
                                result += std::sqrt(j * 1.0);
                            }
                        }
                    });
                }
                
                // Let them run for 100ms
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                // Stop busy threads
                stopBusy = true;
                for (auto& thread : busyThreads) {
                    thread.join();
                }
                
                std::cout << "[RESOURCE-MGR] Forced CPU utilization with " 
                          << threadsToLaunch << " busy threads" << std::endl;
            }
        }
    }

private:
    // Background monitor thread
    void monitorLoop() {
        // Monitor at different rates based on mode
        auto updateInterval = m_extremeMode ? 
            std::chrono::milliseconds(250) : std::chrono::milliseconds(500);
            
        while (m_running) {
            // Get current resource levels
            size_t availableMemMB = SystemResources::getAvailableMemoryMB();
            float memoryUtilizationRatio = 1.0f - (static_cast<float>(availableMemMB) / m_totalMemoryMB);
            float cpuUsage = SystemResources::getCPUUtilization();
            
            // Calculate optimal resources
            unsigned int optimalThreads = getOptimalThreadCount();
            size_t optimalBatchSize = getOptimalBatchSize();
            
            // Only update if significant changes
            bool shouldUpdateThreads = std::abs(static_cast<int>(optimalThreads) - 
                                             static_cast<int>(m_currentThreads)) > 2;
            
            bool shouldUpdateBatchSize = std::abs(static_cast<long>(optimalBatchSize) - 
                                               static_cast<long>(m_currentBatchSize)) > 
                                              (m_currentBatchSize / 4); // 25% change
            
            // Update threads if needed
            if (shouldUpdateThreads) {
                std::lock_guard<std::mutex> lock(m_callbackMutex);
                m_currentThreads = optimalThreads;
                
                for (auto& callback : m_threadScaleCallbacks) {
                    callback(optimalThreads);
                }
                
                std::cout << "[RESOURCE-MGR] Auto-scaled to " << optimalThreads 
                          << " threads (CPU: " << (cpuUsage * 100) << "%)" << std::endl;
            }
            
            // Update batch size if needed
            if (shouldUpdateBatchSize) {
                std::lock_guard<std::mutex> lock(m_callbackMutex);
                m_currentBatchSize = optimalBatchSize;
                
                for (auto& callback : m_batchSizeCallbacks) {
                    callback(optimalBatchSize);
                }
                
                std::cout << "[RESOURCE-MGR] Auto-scaled batch size to " << optimalBatchSize
                          << " (Memory: " << (memoryUtilizationRatio * 100) << "%)" << std::endl;
            }
            
            // In extreme mode, force utilization
            if (m_extremeMode && cpuUsage < m_cpuTarget * 0.85f) {
                forceCpuUtilization();
            }
            
            // Sleep until next update
            std::this_thread::sleep_for(updateInterval);
        }
    }
    
    // Configuration
    bool m_extremeMode;
    float m_memoryTarget;
    float m_cpuTarget;
    
    // System information
    size_t m_totalMemoryMB;
    unsigned int m_hardwareThreads;
    
    // Current settings
    unsigned int m_currentThreads;
    size_t m_currentBatchSize;
    
    // Thread management
    std::atomic<bool> m_running;
    std::thread m_monitorThread;
    
    // Callbacks for resource updates
    std::mutex m_callbackMutex;
    std::vector<std::function<void(unsigned int)>> m_threadScaleCallbacks;
    std::vector<std::function<void(size_t)>> m_batchSizeCallbacks;
};

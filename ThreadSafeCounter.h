#pragma once

#include <unordered_map>
#include <string>
#include <mutex>
#include <cstdint>
#include <shared_mutex> // For read-write lock

/**
 * ThreadSafeCounter
 * 
 * - Enhanced version with read/write lock for better concurrency
 * - Manages float counters (e.g., Charlson scores) and
 *   int counters (e.g., hospitalizations) safely under multiple threads.
 * - Uses shared_mutex for read-heavy workloads to improve performance
 */
class ThreadSafeCounter {
private:
    mutable std::shared_mutex mapMutex; // Read-write lock for better concurrency
    std::unordered_map<std::string, float> floatValues;
    std::unordered_map<std::string, uint16_t> intValues;

    // Optional - to reduce hash map contention with many threads
    static constexpr size_t SHARD_COUNT = 16;
    mutable std::shared_mutex shardMutexes[SHARD_COUNT];
    
    // Simple hash function to determine shard
    size_t getShardIndex(const std::string& key) const {
        return std::hash<std::string>{}(key) % SHARD_COUNT;
    }

public:
    // Increment a float value by 'value' (write operation)
    void addFloat(const std::string &key, float value) {
        size_t shardIdx = getShardIndex(key);
        std::unique_lock<std::shared_mutex> lock(shardMutexes[shardIdx]);
        floatValues[key] += value;
    }

    // Increment an int value by 'value' (write operation)
    void addInt(const std::string &key, uint16_t value) {
        size_t shardIdx = getShardIndex(key);
        std::unique_lock<std::shared_mutex> lock(shardMutexes[shardIdx]);
        intValues[key] += value;
    }

    // Get the float total for a key (read operation - now uses shared lock)
    float getFloat(const std::string &key) const {
        size_t shardIdx = getShardIndex(key);
        std::shared_lock<std::shared_mutex> lock(shardMutexes[shardIdx]);
        auto it = floatValues.find(key);
        return (it == floatValues.end()) ? 0.0f : it->second;
    }

    // Get the int total for a key (read operation - now uses shared lock)
    uint16_t getInt(const std::string &key) const {
        size_t shardIdx = getShardIndex(key);
        std::shared_lock<std::shared_mutex> lock(shardMutexes[shardIdx]);
        auto it = intValues.find(key);
        return (it == intValues.end()) ? 0 : it->second;
    }
    
    // Batch processing - acquire all data at once to reduce lock contention
    std::unordered_map<std::string, float> getAllFloats() const {
        std::shared_lock<std::shared_mutex> lock(mapMutex);
        return floatValues;
    }
    
    std::unordered_map<std::string, uint16_t> getAllInts() const {
        std::shared_lock<std::shared_mutex> lock(mapMutex);
        return intValues;
    }
};

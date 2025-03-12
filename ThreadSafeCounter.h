#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <string>
#include <atomic>
#include <vector>
#include <thread>
#include <memory>

// A thread-safe counter optimized for high-concurrency scenarios
class ThreadSafeCounter {
private:
    class Shard {
    public:
        std::shared_mutex mutex;
        std::unordered_map<std::string, float> counters;
        std::unordered_map<std::string, uint16_t> intCounters;
        
        // Add move constructor to allow unique_ptr to work with this class
        Shard() = default;
        Shard(Shard&&) noexcept = default;
        Shard& operator=(Shard&&) noexcept = default;
        
        // Delete copy operations since shared_mutex is not copyable
        Shard(const Shard&) = delete;
        Shard& operator=(const Shard&) = delete;
    };
    
    // Change vector to store unique_ptr to Shard
    std::vector<std::unique_ptr<Shard>> shards;
    size_t numShards;
    
    // Hash function to distribute keys across shards
    size_t getShard(const std::string& key) const {
        std::hash<std::string> hasher;
        return hasher(key) % numShards;
    }

public:
    // Auto-scale shard count based on hardware
    ThreadSafeCounter(size_t shardCount = 0) {
        // If shardCount is 0, auto-scale based on hardware
        if (shardCount == 0) {
            unsigned int cpuCount = std::thread::hardware_concurrency();
            numShards = cpuCount * 8; // Much more aggressive sharding - 8x CPU count
            numShards = std::max<size_t>(32, numShards); // Minimum 32 shards for better distribution
        } else {
            numShards = shardCount;
        }
        
        // Initialize shards with unique_ptr
        shards.resize(numShards);
        for (size_t i = 0; i < numShards; i++) {
            shards[i] = std::make_unique<Shard>();
        }
    }
    
    // Optimized increment using reader-writer locks more efficiently
    void increment(const std::string& key) {
        size_t shardIdx = getShard(key);
        std::unique_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        shards[shardIdx]->counters[key] += 1.0f;
    }
    
    // Optimized add with hint for initial capacity to reduce rehashing
    void add(const std::string& key, float value) {
        size_t shardIdx = getShard(key);
        std::unique_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        shards[shardIdx]->counters[key] += value;
    }
    
    void add(const std::string& key, uint16_t value) {
        size_t shardIdx = getShard(key);
        std::unique_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        shards[shardIdx]->intCounters[key] += value;
    }

    void addInt(const std::string& key, uint16_t value) {
        size_t shardIdx = getShard(key);
        std::unique_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        shards[shardIdx]->intCounters[key] += value;
    }

    void addFloat(const std::string& key, float value) {
        size_t shardIdx = getShard(key);
        std::unique_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        shards[shardIdx]->counters[key] += value;
    }
    
    float getFloat(const std::string& key) const {
        size_t shardIdx = getShard(key);
        std::shared_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        auto it = shards[shardIdx]->counters.find(key);
        if (it != shards[shardIdx]->counters.end()) {
            return it->second;
        }
        return 0.0f;
    }
    
    uint16_t getInt(const std::string& key) const {
        size_t shardIdx = getShard(key);
        std::shared_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        auto it = shards[shardIdx]->intCounters.find(key);
        if (it != shards[shardIdx]->intCounters.end()) {
            return it->second;
        }
        return 0;
    }
    
    // Optimized bulk add with better locking strategy
    void bulkAdd(const std::unordered_map<std::string, float>& updates) {
        // Pre-sort updates by shard for more efficient processing
        std::vector<std::unordered_map<std::string, float>> shardUpdates(numShards);
        
        // First pass: group by shard without locking
        for (const auto& [key, value] : updates) {
            size_t shardIdx = getShard(key);
            shardUpdates[shardIdx][key] = value;
        }
        
        // Second pass: acquire locks only for shards that have updates
        #pragma omp parallel for if(numShards > 16) // Use OpenMP if many shards
        for (size_t i = 0; i < numShards; i++) {
            if (!shardUpdates[i].empty()) {
                std::unique_lock<std::shared_mutex> lock(shards[i]->mutex);
                for (const auto& [key, value] : shardUpdates[i]) {
                    shards[i]->counters[key] += value;
                }
            }
        }
    }
    
    // More aggressive pre-allocation
    void reserve(size_t expectedEntries) {
        // Calculate per-shard capacity with extra headroom for better distribution
        size_t entriesPerShard = (expectedEntries + numShards - 1) / numShards;
        entriesPerShard = static_cast<size_t>(entriesPerShard * 1.5); // Add 50% extra capacity
        
        #pragma omp parallel for if(numShards > 16) // Parallelize reservation
        for (size_t i = 0; i < numShards; i++) {
            std::unique_lock<std::shared_mutex> lock(shards[i]->mutex);
            shards[i]->counters.reserve(entriesPerShard);
            shards[i]->intCounters.reserve(entriesPerShard);
        }
    }
};

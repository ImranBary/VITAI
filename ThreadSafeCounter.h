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
    ThreadSafeCounter(size_t shardCount = 16) : numShards(shardCount) {
        // Initialize shards with unique_ptr
        shards.resize(numShards);
        for (size_t i = 0; i < numShards; i++) {
            shards[i] = std::make_unique<Shard>();
        }
    }
    
    // Methods that access shards need to be updated to use -> instead of . for pointer access
    void increment(const std::string& key) {
        size_t shardIdx = getShard(key);
        std::unique_lock<std::shared_mutex> lock(shards[shardIdx]->mutex);
        shards[shardIdx]->counters[key] += 1.0f;
    }
    
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
    
    void bulkAdd(const std::unordered_map<std::string, float>& updates) {
        // Group updates by shard to minimize lock contention
        std::vector<std::unordered_map<std::string, float>> shardUpdates(numShards);
        
        for (const auto& [key, value] : updates) {
            size_t shardIdx = getShard(key);
            shardUpdates[shardIdx][key] = value;
        }
        
        // Apply updates to each shard with appropriate locking
        for (size_t i = 0; i < numShards; i++) {
            if (!shardUpdates[i].empty()) {
                std::unique_lock<std::shared_mutex> lock(shards[i]->mutex);
                for (const auto& [key, value] : shardUpdates[i]) {
                    shards[i]->counters[key] += value;
                }
            }
        }
    }
    
    // Pre-allocate space in each shard for expected number of entries
    void reserve(size_t expectedEntries) {
        size_t entriesPerShard = (expectedEntries + numShards - 1) / numShards;
        for (size_t i = 0; i < numShards; i++) {
            std::unique_lock<std::shared_mutex> lock(shards[i]->mutex);
            shards[i]->counters.reserve(entriesPerShard);
            shards[i]->intCounters.reserve(entriesPerShard);
        }
    }
};

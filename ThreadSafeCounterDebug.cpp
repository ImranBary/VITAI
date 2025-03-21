#include <iostream>
#include <string>
#include <map>
#include "DataStructures.h"

// Add method to ThreadSafeCounter class in DataStructures.h:
// Add this to the public section:
// const std::unordered_map<std::string, int>& internalMap() const { return intCounts; }
// const std::unordered_map<std::string, float>& internalFloatMap() const { return floatCounts; }

void testThreadSafeCounter() {
    ThreadSafeCounter counter;
    
    // Test integer counts
    std::cout << "Testing integer counts:\n";
    counter.increment("patient1", 1);
    counter.increment("patient1", 2);
    counter.increment("patient2", 1);
    
    std::cout << "patient1 count: " << counter.getInt("patient1") << " (expected 3)\n";
    std::cout << "patient2 count: " << counter.getInt("patient2") << " (expected 1)\n";
    std::cout << "patient3 count: " << counter.getInt("patient3") << " (expected 0)\n";
    
    // Test float counts
    std::cout << "\nTesting float counts:\n";
    counter.addFloat("patient1", 1.5f);
    counter.addFloat("patient1", 2.5f);
    counter.addFloat("patient2", 0.5f);
    
    std::cout << "patient1 float: " << counter.getFloat("patient1") << " (expected 4.0)\n";
    std::cout << "patient2 float: " << counter.getFloat("patient2") << " (expected 0.5)\n";
    std::cout << "patient3 float: " << counter.getFloat("patient3") << " (expected 0)\n";
    
    // Test map access
    std::cout << "\nDumping full counter contents:\n";
    std::cout << "Integer counts:\n";
    for (const auto& entry : counter.internalMap()) {
        std::cout << "  " << entry.first << ": " << entry.second << "\n";
    }
    
    std::cout << "Float counts:\n";
    for (const auto& entry : counter.internalFloatMap()) {
        std::cout << "  " << entry.first << ": " << entry.second << "\n";
    }
}

int main() {
    std::cout << "=== Testing ThreadSafeCounter Class ===\n";
    testThreadSafeCounter();
    std::cout << "Done!\n";
    return 0;
}

#pragma once

#ifndef SYSTEM_RESOURCES_H
#define SYSTEM_RESOURCES_H

#include <string>
#include <vector>
#include <algorithm>
#include <utility>

// Simple class to avoid the syntax errors
class SystemResources {
public:
    static double getSystemMemoryUsageMB() {
        return 1000.0; // Default implementation
    }
    
    static double getSystemMemorySizeMB() {
        return 8192.0; // Default implementation
    }
    
    static double getSystemCpuUsage() {
        return 25.0; // Default implementation
    }
    
    static int getSystemCoreCount() {
        return 4; // Default implementation
    }
};

#endif // SYSTEM_RESOURCES_H

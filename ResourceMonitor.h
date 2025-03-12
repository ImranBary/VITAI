#pragma once

#ifndef RESOURCE_MONITOR_H
#define RESOURCE_MONITOR_H

#include <string>

class ResourceMonitor {
public:
    static void initialize();
    static double getMemoryUsageMB();
    static double getCpuUsagePercent();
    static void printResourceUsage(const std::string& tag);
};

#endif // RESOURCE_MONITOR_H

#include "ResourceMonitor.h"
#include "SystemResources.h"
#include <iostream>
#include <thread>

void ResourceMonitor::initialize() {
    std::cout << "[INFO] Initializing resource monitor" << std::endl;
}

double ResourceMonitor::getMemoryUsageMB() {
    return SystemResources::getSystemMemoryUsageMB();
}

double ResourceMonitor::getCpuUsagePercent() {
    return SystemResources::getSystemCpuUsage();
}

void ResourceMonitor::printResourceUsage(const std::string& tag) {
    std::cout << "[RESOURCE] " << tag 
              << " Memory: " << getMemoryUsageMB() << " MB, "
              << "CPU: " << getCpuUsagePercent() << "%" << std::endl;
}

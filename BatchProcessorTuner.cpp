#include "BatchProcessorTuner.h"
#include <algorithm>

size_t BatchProcessorTuner::getOptimalBatchSize(size_t recordCount) {
    // Simple implementation - in a real system this would be more sophisticated
    return std::min(recordCount, static_cast<size_t>(1000));
}

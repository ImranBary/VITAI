#pragma once

#ifndef BATCH_PROCESSOR_TUNER_H
#define BATCH_PROCESSOR_TUNER_H

#include <cstddef>

class BatchProcessorTuner {
public:
    static size_t getOptimalBatchSize(size_t recordCount);
};

#endif // BATCH_PROCESSOR_TUNER_H

#pragma once

#include <JuceHeader.h>

#include "Fifo.h"

class Metrics {
public:
    Metrics();

    CircularFifo<int32_t> Clock_offsetError;
};

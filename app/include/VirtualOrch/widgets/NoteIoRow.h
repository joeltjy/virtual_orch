#pragma once

#include <cstdint>

struct NoteIoRow {
    int32_t onset = 0;
    int32_t durationCs = 0;
    int32_t pitch = 0;
    int32_t velocity = 0;
    int32_t localInstrumentId = -1;
    int32_t channel = -1;
};

#include "VirtualOrch/InstrumentConstants.h"

namespace InstrumentConstants {

auto toGmInstrumentId(int32_t localInstrumentId) -> std::optional<int32_t> {
    if (kInstrumentMappings.empty()) {
        if (localInstrumentId < 0 || localInstrumentId >= kGmInstrumentCount)
            return std::nullopt;
        return localInstrumentId;
    }

    for (const auto &[localId, gmId] : kInstrumentMappings) {
        if (localId == localInstrumentId)
            return gmId;
    }
    return std::nullopt;
}

auto toLocalInstrumentId(int32_t gmInstrumentId) -> std::optional<int32_t> {
    if (kInstrumentMappings.empty()) {
        if (gmInstrumentId < 0 || gmInstrumentId >= kGmInstrumentCount)
            return std::nullopt;
        return gmInstrumentId;
    }

    for (const auto &[localId, gmId] : kInstrumentMappings) {
        if (gmId == gmInstrumentId)
            return localId;
    }
    return std::nullopt;
}

auto isValidLocalInstrumentId(int32_t localInstrumentId) -> bool {
    return toGmInstrumentId(localInstrumentId).has_value();
}

} // namespace InstrumentConstants

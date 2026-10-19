#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "VirtualOrch/MusicToken.h"

/**
 * Accompaniment mode's cap on how many generated notes may sound at once.
 *
 * The next generated note may not start before the cap leaves room for it: when at least `cap`
 * generated notes are sounding at the generation head, its earliest onset is the
 * (count - cap + 1)-th earliest offset among them (with count == cap: the next offset).
 */
namespace PolyphonyCap {

inline constexpr int32_t kMinCap = 1;
inline constexpr int32_t kMaxCap = 4;

[[nodiscard]] inline auto isSoundingNote(const Token &token) -> bool {
    return token.note >= static_cast<int32_t>(Vocab::NoteOffset)
           && token.note < static_cast<int32_t>(Vocab::Rest);
}

/**
 * @param generated  notes the transformer has generated (other tokens are ignored)
 * @param headCs     the generation head: the earliest onset the next note would have
 * @param cap        0 = no cap
 * @return the earliest allowed onset, never below headCs
 */
[[nodiscard]] inline auto earliestOnsetCs(const std::vector<Token> &generated,
                                          int32_t headCs,
                                          int32_t cap) -> int32_t {
    if (cap <= 0)
        return headCs;
    std::vector<int32_t> offsets;
    for (const auto &token: generated) {
        if (! isSoundingNote(token))
            continue;
        const int32_t end = token.time + token.getRealDuration();
        if (token.time <= headCs && headCs < end)
            offsets.push_back(end);
    }
    if (static_cast<int32_t>(offsets.size()) < cap)
        return headCs;
    std::sort(offsets.begin(), offsets.end());
    const auto count = static_cast<int32_t>(offsets.size());
    return std::max(headCs, offsets[static_cast<size_t>(count - cap)]);
}

} // namespace PolyphonyCap

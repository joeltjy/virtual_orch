#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/**
 * Model-neutral vocsep ensemble: smooths each note's choice (a sorted set of integer keys)
 * over the notes of the same voice.
 *
 * For note idx the effective choice B is built from the previous same-voice B (or the raw
 * choice A when the note is first in its voice or has no voice). Then, over the last
 * kHistoryNotes same-voice notes (including this one), every key whose share of A choices
 * is above addP is added and every key whose share is below removeP is removed.
 *
 * "Same voice" means the same stream (user / reduction) AND the same voiceId >= 0; a note
 * without a voice (voiceId < 0) is its own voice.
 *
 * Model-specific rules (for example IOD's octave filters) are applied by the caller after
 * this, and the caller falls back to A when the result is empty.
 */
namespace VocsepEnsemble {

inline constexpr size_t kHistoryNotes = 10;

using Keys = std::vector<int32_t>;

[[nodiscard]] inline auto sortedUnique(Keys keys) -> Keys {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

[[nodiscard]] inline auto contains(const Keys &sorted, int32_t key) -> bool {
    return std::binary_search(sorted.begin(), sorted.end(), key);
}

/** B keys for notes[idx]. notes[idx].keysA must be set; earlier notes keep their keysB. */
[[nodiscard]] inline auto computeKeysB(const std::vector<OrchestrationModel::InputNote> &notes,
                                       size_t idx, float addP, float removeP) -> Keys {
    if (idx >= notes.size())
        return {};
    const auto &cur = notes[idx];
    const int32_t voiceId = cur.token.voiceId;
    const auto sameVoiceAs = [&](const OrchestrationModel::InputNote &other) {
        return other.token.voiceId == voiceId && other.stream == cur.stream;
    };

    // B0: the previous same-voice B, or A if first in its voice / no voice.
    Keys b0 = cur.keysA;
    if (voiceId >= 0) {
        for (size_t i = idx; i-- > 0;) {
            if (! sameVoiceAs(notes[i]) || ! notes[i].hasResult)
                continue;
            b0 = notes[i].keysB;
            break;
        }
    }

    // The last kHistoryNotes same-voice notes ending at idx (inclusive).
    std::vector<size_t> sameVoice;
    if (voiceId < 0) {
        sameVoice.push_back(idx);
    } else {
        for (size_t i = 0; i <= idx; ++i) {
            if (! sameVoiceAs(notes[i]))
                continue;
            if (i < idx && ! notes[i].hasResult)
                continue;
            sameVoice.push_back(i);
        }
    }
    if (sameVoice.size() > kHistoryNotes)
        sameVoice.erase(sameVoice.begin(),
                        sameVoice.end() - static_cast<std::ptrdiff_t>(kHistoryNotes));

    const float n = static_cast<float>(sameVoice.size());
    if (n <= 0.0f)
        return cur.keysA;

    Keys candidates = b0;
    for (const size_t i: sameVoice)
        candidates.insert(candidates.end(), notes[i].keysA.begin(), notes[i].keysA.end());
    candidates = sortedUnique(std::move(candidates));

    Keys b = sortedUnique(b0);
    for (const int32_t key: candidates) {
        int32_t count = 0;
        for (const size_t i: sameVoice) {
            if (contains(notes[i].keysA, key))
                ++count;
        }
        const float prop = static_cast<float>(count) / n;
        const bool inB = contains(b, key);
        if (prop > addP && ! inB) {
            b.push_back(key);
            std::sort(b.begin(), b.end());
        } else if (prop < removeP && inB) {
            b.erase(std::remove(b.begin(), b.end(), key), b.end());
        }
    }
    return b;
}

} // namespace VocsepEnsemble

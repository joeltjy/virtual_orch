#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "VirtualOrch/MusicToken.h"

/**
 * Reduction notes waiting to be orchestrated.
 *
 * The reduction transformer runs several seconds ahead of the clock. If the orchestration
 * transformer orchestrated every note on arrival, an instrument toggle would only affect notes
 * that arrive later, and several seconds of already orchestrated notes would sit in the playback
 * queue with the old instruments. Instead a reduction note waits here, un-orchestrated, until
 * its onset is within `windowCs` of the clock; only then does the model see it, with the
 * instrument list that is current at that moment.
 *
 * Order matters: a ClearQueue token (sustain pedal, or a regenerate clear) erases the held
 * notes it covers, so notes that arrived before it never come out after it.
 */
class HoldBackQueue {
public:
    /** A note is released once its onset is at most this far ahead of the clock (1 s). */
    static constexpr int32_t kDefaultWindowCs = 100;
    /** Safety bound; the reduction is throttled to a few seconds ahead, so this is never hit. */
    static constexpr size_t kMaxHeld = 256;

    /**
     * Take a batch in arrival order. Notes are held (sorted by onset); a ClearQueue token erases
     * the held notes with onset >= its time. Returns how many held notes the clears erased.
     */
    auto accept(const std::vector<Token> &batch) -> size_t {
        size_t erased = 0;
        for (const auto &token: batch) {
            if (token.note == static_cast<int32_t>(Vocab::ClearQueue)) {
                const auto before = held.size();
                held.erase(std::remove_if(held.begin(), held.end(),
                                          [&](const Token &t) { return t.time >= token.time; }),
                           held.end());
                erased += before - held.size();
                continue;
            }
            // Only sounding notes are held; rests and separators pass through the caller as before.
            if (token.note < static_cast<int32_t>(Vocab::NoteOffset)
                || token.note >= static_cast<int32_t>(Vocab::Rest))
                continue;
            const auto at = std::upper_bound(
                held.begin(), held.end(), token,
                [](const Token &a, const Token &b) { return a.time < b.time; });
            held.insert(at, token);
        }
        if (held.size() > kMaxHeld)
            held.erase(held.begin(), held.begin() + static_cast<std::ptrdiff_t>(held.size() - kMaxHeld));
        return erased;
    }

    /**
     * The held notes whose onset is within windowCs of nowCs, in onset order, removed from the
     * queue. With nowCs < 0 (no clock) everything is released.
     */
    auto release(int32_t nowCs, int32_t windowCs = kDefaultWindowCs) -> std::vector<Token> {
        std::vector<Token> out;
        const auto split = nowCs < 0 ? held.end()
                                     : std::find_if(held.begin(), held.end(), [&](const Token &t) {
                                           return t.time > nowCs + windowCs;
                                       });
        out.assign(held.begin(), split);
        held.erase(held.begin(), split);
        return out;
    }

    /** While the orchestration is paused nothing is released, and notes that are past are dropped. */
    auto dropPast(int32_t nowCs) -> size_t {
        const auto before = held.size();
        held.erase(std::remove_if(held.begin(), held.end(),
                                  [&](const Token &t) { return t.time < nowCs; }),
                   held.end());
        return before - held.size();
    }

    /** Vocsep re-stamped a note that may still be waiting here. */
    auto applyRestamp(const VoiceRestamp &restamp) -> int {
        return applyVoiceRestampToTokens(held, restamp);
    }

    auto clear() -> void { held.clear(); }
    [[nodiscard]] auto size() const -> size_t { return held.size(); }
    [[nodiscard]] auto notes() const -> const std::vector<Token> & { return held; }

private:
    std::vector<Token> held; // sorted by onset
};

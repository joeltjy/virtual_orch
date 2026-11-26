#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

/**
 * Register control (Launchpad row 6): limits the pitch of every generated reduction note.
 * Every register, AUTO (the default) included, stays within 2.5 octaves of C4 (F#1 to F#6, 30..90; Nov 25 2026:
 * the reduction went too high). Pitches are MIDI numbers with C4 = 60, so
 *   AUTO   F#1 to F#6 30..90
 *   LOW    F#1 to E3  30..52
 *   MEDIUM G3 to D5   55..74
 *   HIGH   F5 to F#6  77..90
 * The ranges do not overlap (a minor third between them), so each sounds clearly apart. Like TemporalDensity, only LaunchpadGrid writes it.
 */
class RegisterRange {
public:
    static constexpr int32_t kAuto = 0;
    static constexpr int32_t kLow = 1;
    static constexpr int32_t kMedium = 2;
    static constexpr int32_t kHigh = 3;
    static constexpr int32_t kPianoLow = 21;   // A0
    static constexpr int32_t kPianoHigh = 108; // C8
    /** The reduction's limits in every register: C4 -/+ 2.5 octaves. */
    static constexpr int32_t kRtLow = 30;  // F#1
    static constexpr int32_t kRtHigh = 90; // F#6

    [[nodiscard]] static auto get() -> int32_t { return value.load(std::memory_order_relaxed); }
    [[nodiscard]] static auto isAuto() -> bool { return get() == kAuto; }
    [[nodiscard]] static auto isValid(int32_t v) -> bool { return v >= kAuto && v <= kHigh; }

    /** Lowest / highest allowed pitch (inclusive) of a register value; AUTO = kRtLow..kRtHigh. */
    [[nodiscard]] static auto lowPitch(int32_t v) -> int32_t {
        switch (v) {
            case kMedium: return 55; // G3
            case kHigh: return 77;   // F5
            default: return kRtLow;  // AUTO, LOW
        }
    }
    [[nodiscard]] static auto highPitch(int32_t v) -> int32_t {
        switch (v) {
            case kLow: return 52;    // E3
            case kMedium: return 74; // D5
            default: return kRtHigh; // AUTO, HIGH
        }
    }
    [[nodiscard]] static auto lowPitch() -> int32_t { return lowPitch(get()); }
    [[nodiscard]] static auto highPitch() -> int32_t { return highPitch(get()); }

    /**
     * Mask the note ids of a logit block (`instruments` bands of `pitchesPerInstrument` ids from
     * `noteOffset`) whose pitch lies outside the current register (in AUTO: outside kRtLow..kRtHigh).
     * Does nothing when it would leave no finite logit in the block (the register then conflicts with
     * what the model may play, and generating at all is better than not).
     */
    static auto maskNoteBlock(std::vector<float> &logits, size_t noteOffset, int32_t instruments,
                              int32_t pitchesPerInstrument = 128) -> void {
        const int32_t low = lowPitch();
        const int32_t high = highPitch();
        const auto id = [&](int32_t instr, int32_t pitch) {
            return noteOffset + static_cast<size_t>(instr) * static_cast<size_t>(pitchesPerInstrument)
                   + static_cast<size_t>(pitch);
        };
        bool anyLeft = false;
        for (int32_t instr = 0; instr < instruments && ! anyLeft; ++instr)
            for (int32_t pitch = low; pitch <= high && pitch < pitchesPerInstrument; ++pitch)
                if (id(instr, pitch) < logits.size() && std::isfinite(logits[id(instr, pitch)])) {
                    anyLeft = true;
                    break;
                }
        if (! anyLeft)
            return;
        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        for (int32_t instr = 0; instr < instruments; ++instr)
            for (int32_t pitch = 0; pitch < pitchesPerInstrument; ++pitch)
                if ((pitch < low || pitch > high) && id(instr, pitch) < logits.size())
                    logits[id(instr, pitch)] = kNegInf;
    }

private:
    friend class LaunchpadGrid;
    friend struct RegisterTestAccess;

    static auto set(int32_t newValue) -> void {
        if (isValid(newValue))
            value.store(newValue, std::memory_order_relaxed);
    }

    static inline std::atomic<int32_t> value{kAuto};
};

#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <vector>

#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/LaunchpadLighting.h"
#include "VirtualOrch/TemporalDensity.h"

/**
 * Logic behind the performance-view widgets: which seating-plan sectors are active, the share
 * of strings / woodwinds / brass in the last seconds of playback, the TD bar, and the
 * Launchpad action console. No UI here (the widgets only draw what this computes).
 */
namespace PerformanceActivity {

// ---- Seating plan -------------------------------------------------------------------------

/** Order = pixel value - 1 in orch_seating_plan_sectors.png. */
enum class Sector : int {
    Horns,
    FirstViolins,
    Flutes,
    SecondViolins,
    Oboes,
    Violas,
    Clarinets,
    Cellos,
    Bassoons,
    DoubleBasses,
    Trumpets,
    Trombones,
    Tuba,
    Piano,
};
inline constexpr int kSectorCount = 14;

/**
 * The sectors an instrument lights. Violin lights both violin sections. Tremolo, pizzicato and
 * string ensemble go to the first solo string (violin, viola, cello, bass) whose GM range holds
 * the pitch. Timpani, snare, guitar and choir have no sector.
 */
[[nodiscard]] inline auto sectorsFor(int32_t localId, int32_t pitch) -> std::vector<Sector> {
    const auto stringSectors = [](int32_t solo) -> std::vector<Sector> {
        switch (solo) {
            case 0: return {Sector::FirstViolins, Sector::SecondViolins};
            case 1: return {Sector::Violas};
            case 2: return {Sector::Cellos};
            default: return {Sector::DoubleBasses};
        }
    };
    switch (localId) {
        case 0: case 1: case 2: case 3:
            return stringSectors(localId);
        case 4: case 5: case 6:
            for (const int32_t solo: InstrumentConstants::kSoloStringLocalIds)
                if (InstrumentConstants::localInstrumentAllowsPitch(solo, pitch))
                    return stringSectors(solo);
            return {};
        case 7: return {Sector::Flutes};
        case 8: return {Sector::Oboes};
        case 9: return {Sector::Clarinets};
        case 10: return {Sector::Bassoons};
        case 11: return {Sector::Trumpets};
        case 12: return {Sector::Horns};
        case 13: return {Sector::Trombones};
        case 14: return {Sector::Tuba};
        case 15: return {Sector::Piano};
        default: return {};
    }
}

// ---- Activity log -------------------------------------------------------------------------

/** What the seating plan and the group bars read. Thread-safe; times are wall-clock ms. */
class ActivityLog {
public:
    /** A sector turns off only after this long without a sounding note. */
    static constexpr int64_t kHoldMs = 2000;
    /** Window of the group proportions. */
    static constexpr int64_t kGroupWindowMs = 5000;

    /** A note that starts sounding at nowMs and lasts durationMs. */
    auto noteOn(int64_t nowMs, int32_t localId, int32_t pitch, int64_t durationMs) -> void {
        const juce::ScopedLock lock(mutex);
        const int64_t end = nowMs + std::max<int64_t>(durationMs, 0);
        for (const auto sector: sectorsFor(localId, pitch)) {
            auto &until = soundingUntil[static_cast<size_t>(sector)];
            until = std::max(until, end);
        }
        if (const auto family = LaunchpadLighting::familyForLocalInstrument(localId);
            family.has_value() && *family != LaunchpadLighting::Family::Other
            && ! sectorsFor(localId, pitch).empty()) {
            intervals.push_back({nowMs, end, static_cast<int>(*family)});
        }
        prune(nowMs);
    }

    /** MIDI keyboard: the piano sector is lit while a key is down. */
    auto keyboardNoteOn(int64_t nowMs, int32_t pitch) -> void {
        const juce::ScopedLock lock(mutex);
        keysDown.insert(pitch);
        pianoReleasedMs = nowMs;
    }

    auto keyboardNoteOff(int64_t nowMs, int32_t pitch) -> void {
        const juce::ScopedLock lock(mutex);
        keysDown.erase(pitch);
        pianoReleasedMs = nowMs;
    }

    /** 0 while a piano key is down, else ms since the last release (INT64_MAX if none yet). */
    [[nodiscard]] auto pianoSilentForMs(int64_t nowMs) const -> int64_t {
        const juce::ScopedLock lock(mutex);
        if (! keysDown.empty())
            return 0;
        if (pianoReleasedMs == kNever)
            return INT64_MAX;
        return std::max<int64_t>(0, nowMs - pianoReleasedMs);
    }

    [[nodiscard]] auto activeSectors(int64_t nowMs) const -> std::array<bool, kSectorCount> {
        const juce::ScopedLock lock(mutex);
        std::array<bool, kSectorCount> active{};
        for (int i = 0; i < kSectorCount; ++i)
            active[static_cast<size_t>(i)] =
                soundingUntil[static_cast<size_t>(i)] != kNever
                && nowMs < soundingUntil[static_cast<size_t>(i)] + kHoldMs;
        if (! keysDown.empty() || (pianoReleasedMs != kNever && nowMs < pianoReleasedMs + kHoldMs))
            active[static_cast<size_t>(Sector::Piano)] = true;
        return active;
    }

    /** Sounding time per family (strings, woodwinds, brass) over the last kGroupWindowMs. */
    [[nodiscard]] auto groupSoundingMs(int64_t nowMs) const -> std::array<double, 3> {
        const juce::ScopedLock lock(mutex);
        std::array<double, 3> total{};
        const int64_t from = nowMs - kGroupWindowMs;
        for (const auto &iv: intervals) {
            const int64_t lo = std::max(iv.start, from);
            const int64_t hi = std::min(iv.end, nowMs);
            if (hi > lo && iv.family >= 0 && iv.family < 3)
                total[static_cast<size_t>(iv.family)] += static_cast<double>(hi - lo);
        }
        return total;
    }

    /** The three groups normalised among themselves (all zero when nothing sounded). */
    [[nodiscard]] auto groupProportions(int64_t nowMs) const -> std::array<float, 3> {
        const auto ms = groupSoundingMs(nowMs);
        const double sum = ms[0] + ms[1] + ms[2];
        if (sum <= 0.0)
            return {0.0f, 0.0f, 0.0f};
        return {static_cast<float>(ms[0] / sum), static_cast<float>(ms[1] / sum),
                static_cast<float>(ms[2] / sum)};
    }

    auto clear() -> void {
        const juce::ScopedLock lock(mutex);
        soundingUntil.fill(kNever);
        intervals.clear();
        keysDown.clear();
        pianoReleasedMs = kNever;
    }

private:
    struct Interval {
        int64_t start;
        int64_t end;
        int family;
    };
    static constexpr int64_t kNever = INT64_MIN;

    auto prune(int64_t nowMs) -> void {
        const int64_t cutoff = nowMs - kGroupWindowMs;
        while (! intervals.empty() && intervals.front().end < cutoff)
            intervals.pop_front();
        // Intervals arrive in start order but end at different times: drop stale ones anywhere.
        if (intervals.size() > 4096)
            intervals.erase(std::remove_if(intervals.begin(), intervals.end(),
                                           [&](const Interval &iv) { return iv.end < cutoff; }),
                            intervals.end());
    }

    mutable juce::CriticalSection mutex;
    std::array<int64_t, kSectorCount> soundingUntil = [] {
        std::array<int64_t, kSectorCount> a{};
        a.fill(kNever);
        return a;
    }();
    std::deque<Interval> intervals;
    std::set<int32_t> keysDown;
    int64_t pianoReleasedMs = kNever;
};

// ---- Pedal freeze ---------------------------------------------------------------------------

/** The orchestra is shown as frozen once the pedal has been down this long. */
inline constexpr int64_t kPedalFreezeShowAfterMs = 1000;

/** pedalDownForMs: -1 while the pedal is up (MidiInputProcess::getPedalDownForMs). */
[[nodiscard]] inline auto pedalShowsFrozen(int64_t pedalDownForMs) -> bool {
    return pedalDownForMs >= kPedalFreezeShowAfterMs;
}

// ---- TD bar -------------------------------------------------------------------------------

/** Position (0..1) of the manual TD value on the log-spaced bar of 7 steps; nullopt for AUTO. */
[[nodiscard]] inline auto tdBarFraction(int32_t tdValue) -> std::optional<float> {
    if (tdValue == TemporalDensity::kAuto)
        return std::nullopt;
    const auto &levels = TemporalDensity::kLevels;
    const auto it = std::find(levels.begin(), levels.end(), tdValue);
    if (it == levels.end())
        return std::nullopt;
    const auto index = static_cast<float>(it - levels.begin());
    return (index + 1.0f) / static_cast<float>(levels.size());
}

// ---- Launchpad action console -------------------------------------------------------------

/** One thing the user did on the Launchpad. */
struct LaunchpadAction {
    enum class Kind {
        InstrumentPad, // local = instrument, on = new state, model = Model row (else Input)
        Flush,
        TdLevel,       // value = level
        TdAuto,        // on = AUTO now on
        Mode,          // value = 0 Edit, 1 Jam, 2 Accompaniment
        Polyphony,     // value = max polyphony, 0 = off
        Pause,         // value = 1 RT, 2 OT; on = now paused
        ForceRestart,
        Register,      // value = 1 LOW, 2 MEDIUM, 3 HIGH
        RegisterAuto,  // on = AUTO now on
    };
    Kind kind = Kind::Flush;
    int32_t local = 0;
    int32_t value = 0;
    bool on = false;
    bool model = false;
};

/** The action in words. `nameFor` gives an instrument's name. */
template <typename NameFn>
[[nodiscard]] inline auto describe(const LaunchpadAction &a, NameFn &&nameFor) -> juce::String {
    using Kind = LaunchpadAction::Kind;
    switch (a.kind) {
        case Kind::InstrumentPad:
            return juce::String(a.model ? "Model: " : "Input: ") + nameFor(a.local)
                   + (a.on ? " on" : " off");
        case Kind::Flush:
            return "Flush: input instruments moved to the model";
        case Kind::TdLevel:
            return "Temporal density set to " + juce::String(a.value);
        case Kind::TdAuto:
            return a.on ? "Temporal density AUTO on" : "Temporal density AUTO off";
        case Kind::Mode:
            return a.value == 2 ? "Mode: Accompaniment" : (a.value == 1 ? "Mode: Jam" : "Mode: Edit");
        case Kind::Polyphony:
            return a.value > 0 ? "Max polyphony " + juce::String(a.value)
                               : juce::String("Max polyphony off");
        case Kind::ForceRestart:
            return "Force restart";
        case Kind::Register:
            return a.value == 1   ? juce::String("Register LOW (below C4)")
                   : a.value == 2 ? juce::String("Register MEDIUM (C3 to C6)")
                                  : juce::String("Register HIGH (above C5)");
        case Kind::RegisterAuto:
            return a.on ? "Register AUTO on (no limit)" : "Register AUTO off";
        case Kind::Pause: {
            const juce::String who = a.value == 1 ? "Reduction" : "Orchestration";
            return who + (a.on ? " paused" : " resumed");
        }
    }
    return {};
}

/** Newest lines last; thread-safe. */
class ConsoleLog {
public:
    static constexpr size_t kKeep = 50;

    auto add(const juce::String &line) -> void {
        const juce::ScopedLock lock(mutex);
        lines.push_back(line);
        while (lines.size() > kKeep)
            lines.pop_front();
    }

    /** The newest `count` lines, oldest first. */
    [[nodiscard]] auto latest(size_t count) const -> std::vector<juce::String> {
        const juce::ScopedLock lock(mutex);
        const size_t n = std::min(count, lines.size());
        return {lines.end() - static_cast<std::ptrdiff_t>(n), lines.end()};
    }

private:
    mutable juce::CriticalSection mutex;
    std::deque<juce::String> lines;
};

} // namespace PerformanceActivity

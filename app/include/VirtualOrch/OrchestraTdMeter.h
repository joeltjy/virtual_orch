#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <set>
#include <vector>

#include "VirtualOrch/TemporalDensity.h"

/**
 * Temporal density of the ORCHESTRA as it will sound: measured on the orchestration transformer's output notes
 * (after instruments are chosen), not on the reduction (Nov 25 2026, Joel). In TD units, like TdController:
 * events in the 3 s ending at the newest orchestrated onset, x 200 / 300 (events per 2 s).
 *
 *  - every distinct onset of a non-tremolo note is one event (a chord counts once, as before);
 *  - every tremolo note held for x seconds (of the window) counts as kTremoloPerSecond x x events, each note on its
 *    own (a 3-note tremolo chord held 1 s = 24), so held tremolo keeps the density up (and can reach TD 64) without
 *    the reduction having to draw more notes. Its onset is not also counted as a plain onset.
 *
 * Written by the OT thread (add / clearFrom / clear), read by the RT's TD tick (read): a mutex.
 */
class OrchestraTdMeter {
public:
    static constexpr double kTremoloPerSecond = 8.0;
    static constexpr int32_t kWindowCs = 300; // as TdController::kWindowCs
    /** InstrumentConstants: local id 4 = Tremolo Strings (GM 44). */
    static constexpr int32_t kTremoloLocalId = 4;
    /** Notes older than this before the newest onset are dropped. */
    static constexpr int32_t kKeepCs = 1500;

    struct Note {
        int32_t onsetCs = 0;
        int32_t durationCs = 0;
        bool tremolo = false;
        int32_t sourceNoteId = -1; // the reduction note it orchestrates (-1: unknown / the pianist's)
    };

    struct Reading {
        double td = 0.0;            // events per 2 s
        double events = 0.0;        // in the window
        double tremoloEvents = 0.0; // of which from tremolo notes (8 per second held, per note)
        int32_t endCs = 0;          // the window's end: the newest onset, or now if later
    };

    /**
     * The reading of `notes` (any order) over the 3 s ending at the newest onset, or at `nowCs` if that is later (a
     * held tremolo with nothing newer still counts while it sounds); empty when there are no notes.
     */
    [[nodiscard]] static auto measureNotes(const std::vector<Note> &notes, std::optional<int32_t> nowCs = std::nullopt)
        -> std::optional<Reading> {
        if (notes.empty())
            return std::nullopt;
        int32_t end = INT32_MIN;
        for (const auto &n: notes)
            end = std::max(end, n.onsetCs);
        if (nowCs.has_value())
            end = std::max(end, *nowCs);
        const int32_t start = end - kWindowCs; // (start, end]
        std::set<int32_t> plain;
        double tremolo = 0.0;
        for (const auto &n: notes) {
            if (! n.tremolo) {
                if (n.onsetCs > start && n.onsetCs <= end)
                    plain.insert(n.onsetCs);
                continue;
            }
            // the part of the note inside the window, in seconds, x 8
            const int32_t from = std::max(n.onsetCs, start);
            const int32_t to = std::min(n.onsetCs + std::max(1, n.durationCs), end);
            if (to > from)
                tremolo += kTremoloPerSecond * static_cast<double>(to - from) / 100.0;
        }
        Reading r;
        r.tremoloEvents = tremolo;
        r.events = static_cast<double>(plain.size()) + tremolo;
        r.td = r.events * 200.0 / static_cast<double>(kWindowCs);
        r.endCs = end;
        return r;
    }

    auto add(int32_t onsetCs, int32_t durationCs, bool tremolo, int32_t sourceNoteId = -1) -> void {
        const std::lock_guard<std::mutex> lock(mutex);
        notes.push_back({onsetCs, durationCs, tremolo, sourceNoteId});
        newest = std::max(newest, onsetCs);
        while (! notes.empty() && notes.front().onsetCs < newest - kKeepCs)
            notes.pop_front();
    }

    /**
     * The real length of the notes made from reduction note `sourceNoteId` (Hybrid v1.3: a note goes out with a
     * provisional length until the model's OFF); -1 changes nothing.
     */
    auto setDuration(int32_t sourceNoteId, int32_t durationCs) -> void {
        if (sourceNoteId < 0)
            return;
        const std::lock_guard<std::mutex> lock(mutex);
        for (auto &n: notes)
            if (n.sourceNoteId == sourceNoteId)
                n.durationCs = durationCs;
    }

    /** A ClearQueue at `fromCs`: the notes from it on will not sound. */
    auto clearFrom(int32_t fromCs) -> void {
        const std::lock_guard<std::mutex> lock(mutex);
        notes.erase(std::remove_if(notes.begin(), notes.end(), [&](const Note &n) { return n.onsetCs >= fromCs; }),
                    notes.end());
        newest = INT32_MIN;
        for (const auto &n: notes)
            newest = std::max(newest, n.onsetCs);
    }

    auto clear() -> void {
        const std::lock_guard<std::mutex> lock(mutex);
        notes.clear();
        newest = INT32_MIN;
    }

    [[nodiscard]] auto read(std::optional<int32_t> nowCs = std::nullopt) const -> std::optional<Reading> {
        std::vector<Note> copy;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            copy.assign(notes.begin(), notes.end());
        }
        return measureNotes(copy, nowCs);
    }

private:
    mutable std::mutex mutex;
    std::deque<Note> notes;
    int32_t newest = INT32_MIN;
};

/**
 * The OT's second lever on TD: a logit bonus for tremolo strings (Nov 25 2026). Every tick (the OT's 2 s): below the
 * TD target the bonus grows, at or above it (or at TD AUTO / TD <= 4) it decays; it is never negative (it only ever
 * adds density, it never takes tremolo away from what the model would pick).
 *   bonus += kGain x clamp(log2(target / measured), -2, 2)    (target > 4, a reading)
 *   bonus *= kDecay                                           (otherwise, or when measured >= target)
 * The OT adds the bonus to the GROUPS tokens that include strings and to the strings tokens that include tremolo.
 */
class TremoloBiasController {
public:
    static constexpr double kTickMs = 2000.0;
    static constexpr double kGain = 0.5;
    static constexpr double kDecay = 0.5;
    static constexpr double kMax = 4.0;

    auto tick(int32_t target, std::optional<double> measured) -> double {
        if (target == TemporalDensity::kAuto || target <= 4 || ! measured.has_value()) {
            bonusValue *= kDecay;
        } else {
            const double e = std::clamp(std::log2(static_cast<double>(target) / std::max(*measured, 0.1)), -2.0, 2.0);
            if (e > 0.0)
                bonusValue = std::min(kMax, bonusValue + kGain * e);
            else
                bonusValue *= kDecay;
        }
        if (bonusValue < 1e-3)
            bonusValue = 0.0;
        return bonusValue;
    }

    [[nodiscard]] auto bonus() const -> double { return bonusValue; }
    auto reset() -> void { bonusValue = 0.0; }

private:
    double bonusValue = 0.0;
};

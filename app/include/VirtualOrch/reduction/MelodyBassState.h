#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/MelodyBassSkyline.h"

/**
 * The user's notes as MelodyFirstReductionTransformer keeps them:
 *   L = every incoming note (q5v8, on instrument 0), ordered by onset;
 *   R = the running melody + bass of those notes.
 * R is refreshed by running algorithm A on the notes that sound in the window [t - 2 s, t]:
 * every note is truncated at t (a held key counts as lasting until t), a note that began before
 * the window but still sounds is kept at its real onset. A's output REPLACES whatever R held for
 * exactly the notes A was given, so a note A no longer returns disappears from R.
 */
class MelodyBassState {
public:
    static constexpr int32_t kWindowCs = 200;
    /** R entries ending longer ago than this are dropped. */
    static constexpr int32_t kKeepCs = 6000;
    static constexpr int kMelodyLine = 0;
    static constexpr int kBassLine = 1;

    struct UserNote {
        int64_t key = 0;
        int32_t onset = 0;
        int32_t durationCs = 0; // final once not held
        int pitch = 60;
        int velocity = 64;
        bool held = false;
    };

    struct Piece {
        int64_t key = 0; // the L note this piece comes from
        int32_t onset = 0;
        int32_t durationCs = 0;
        int pitch = 60;
        int velocity = 64;
        int line = kMelodyLine;
        bool provisional = false; // its source still sounds (no note-off yet)

        auto operator==(const Piece &o) const -> bool {
            return key == o.key && onset == o.onset && durationCs == o.durationCs && pitch == o.pitch
                   && velocity == o.velocity && line == o.line && provisional == o.provisional;
        }
    };

    auto reset() -> void {
        notes.clear();
        pieces.clear();
        nextKey = 0;
    }

    /** A note arrives (q5v8 snapped here). Returns its key. */
    auto addNote(int32_t onset, int pitch, int velocity, int32_t durationCs, bool held) -> int64_t {
        UserNote n;
        n.key = nextKey++;
        n.onset = onset;
        n.pitch = DenseQuantize::snapPianoPitch(pitch);
        n.velocity = std::max(DenseQuantize::VelocityStep, DenseQuantize::snapVelocity(velocity));
        n.durationCs = held ? 0 : DenseQuantize::snapDurationCs(durationCs);
        n.held = held;
        notes.push_back(n);
        std::stable_sort(notes.begin(), notes.end(), [](const UserNote &a, const UserNote &b) {
            return a.onset != b.onset ? a.onset < b.onset : a.pitch < b.pitch;
        });
        return n.key;
    }

    /** Note-off for a held note. Returns false when no such held note exists. */
    auto noteOff(int32_t onset, int pitch, int32_t finalDurationCs) -> bool {
        const int snapped = DenseQuantize::snapPianoPitch(pitch);
        for (auto &n: notes)
            if (n.held && n.onset == onset && n.pitch == snapped) {
                n.held = false;
                n.durationCs = DenseQuantize::snapDurationCs(finalDurationCs);
                return true;
            }
        return false;
    }

    /** Run A on the window ending at nowCs. Returns true when R changed. */
    auto runWindow(int32_t nowCs) -> bool {
        const int32_t windowStart = nowCs - kWindowCs;
        std::vector<MelodyBass::Note> input;
        std::vector<int64_t> keyOfIndex;
        std::map<int64_t, bool> heldOf;
        for (const auto &n: notes) {
            if (n.onset > nowCs)
                continue;
            const int32_t end = n.held ? nowCs : std::min(n.onset + n.durationCs, nowCs);
            const int32_t fullEnd = n.held ? nowCs : n.onset + n.durationCs;
            if (fullEnd <= windowStart || end - n.onset < 1)
                continue;
            input.push_back({static_cast<int>(input.size()), n.onset, end, n.pitch, n.velocity});
            keyOfIndex.push_back(n.key);
            heldOf[n.key] = n.held;
        }

        const auto before = pieces;
        std::set<int64_t> given(keyOfIndex.begin(), keyOfIndex.end());
        pieces.erase(std::remove_if(pieces.begin(), pieces.end(),
                                    [&](const Piece &p) { return given.count(p.key) > 0; }),
                     pieces.end());

        const auto result = MelodyBass::skylineAndBass(input);
        const auto add = [&](const std::vector<MelodyBass::Segment> &segments, int line) {
            for (const auto &s: segments) {
                Piece p;
                p.key = keyOfIndex[static_cast<size_t>(s.sourceIndex)];
                p.onset = s.start;
                p.durationCs = s.end - s.start;
                p.pitch = s.pitch;
                p.velocity = s.velocity;
                p.line = line;
                p.provisional = heldOf[p.key];
                pieces.push_back(p);
            }
        };
        add(result.melody, kMelodyLine);
        add(result.bass, kBassLine);

        std::stable_sort(pieces.begin(), pieces.end(), [](const Piece &a, const Piece &b) {
            if (a.onset != b.onset)
                return a.onset < b.onset;
            if (a.pitch != b.pitch)
                return a.pitch < b.pitch;
            return a.line < b.line;
        });
        prune(nowCs);
        return pieces != before;
    }

    [[nodiscard]] auto getPieces() const -> const std::vector<Piece> & { return pieces; }
    [[nodiscard]] auto getNotes() const -> const std::vector<UserNote> & { return notes; }
    [[nodiscard]] auto hasHeldNotes() const -> bool {
        return std::any_of(notes.begin(), notes.end(), [](const UserNote &n) { return n.held; });
    }

private:
    auto prune(int32_t nowCs) -> void {
        const int32_t cutoff = nowCs - kKeepCs;
        notes.erase(std::remove_if(notes.begin(), notes.end(),
                                   [&](const UserNote &n) {
                                       return ! n.held && n.onset + n.durationCs < cutoff;
                                   }),
                    notes.end());
        pieces.erase(std::remove_if(pieces.begin(), pieces.end(),
                                    [&](const Piece &p) { return p.onset + p.durationCs < cutoff; }),
                     pieces.end());
    }

    std::vector<UserNote> notes;
    std::vector<Piece> pieces;
    int64_t nextKey = 0;
};

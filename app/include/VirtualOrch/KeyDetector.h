#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "VirtualOrch/Spelling.h"

/**
 * The key of what is playing (the orchestra's notes and the piano's), for the score: which
 * major or minor scale fits the last notes best.
 *
 *  - Notes: the last kMaxNotes notes that started by now and ended at most kMaxAgeCs ago.
 *  - Weight of a note: recency 2^(-age / kHalfLifeCs), age counted from the note's end (0 while
 *    it sounds), x (0.5 + 0.5 min(duration / 1 s, 1)), x 1.5 for a bass note.
 *  - Candidates: 12 major and 12 minor scales (minor: natural minor + raised 6th and 7th).
 *    fit = weight inside the scale / all weight; score = fit + 0.1 x tonic evidence, where the
 *    tonic evidence is the weight on the tonic triad (bass notes doubled) / all weight, + 0.25
 *    for a minor key whose raised 7th is heard.
 *  - Hysteresis: the shown key changes when it has been beaten by at least kSwitchMargin (by any
 *    key) for kSwitchNotes new notes in a row; it becomes the best key at that moment. Counting
 *    one challenger only got stuck: F major music ranks F major and D minor (the same notes) in
 *    turn, every change of leader reset the count, and the first key heard stayed (Nov 2026). With fewer than kMinNotes notes, or a fit below kSureFit,
 *    the last key stays and is marked uncertain.
 *  - Naming: the spelling with fewer accidentals (Db major, G# minor); F# / Gb major and
 *    D# / Eb minor take the one nearer the previous key on the line of fifths (default F# major,
 *    Eb minor).
 */
class KeyDetector {
public:
    static constexpr int kMaxNotes = 40;
    static constexpr int32_t kMaxAgeCs = 500;
    static constexpr double kHalfLifeCs = 200.0;
    static constexpr double kSwitchMargin = 0.05;
    static constexpr int kSwitchNotes = 8;
    static constexpr int kMinNotes = 8;
    static constexpr double kSureFit = 0.75;
    static constexpr double kRunnerUpWithin = 0.03;

    struct Note {
        int32_t onsetCs = 0;
        int32_t durationCs = 0;
        int pitch = 60;
        bool bass = false;
    };

    struct Candidate {
        int tonicPc = 0;
        bool minor = false;
        double fit = 0.0;
        double score = 0.0;
    };

    struct Result {
        std::optional<Spelling::Key> key;      // none until enough notes were heard
        std::optional<Spelling::Key> runnerUp; // a close second, if any
        bool uncertain = true;                 // too few notes, or a poor fit: shown greyed
    };

    /** The window of notes used at `nowCs` (see class comment), oldest first. */
    [[nodiscard]] static auto window(std::vector<Note> notes, int32_t nowCs) -> std::vector<Note> {
        notes.erase(std::remove_if(notes.begin(), notes.end(),
                                   [nowCs](const Note &n) {
                                       return n.onsetCs > nowCs || n.onsetCs + n.durationCs < nowCs - kMaxAgeCs;
                                   }),
                    notes.end());
        std::stable_sort(notes.begin(), notes.end(), [](const Note &a, const Note &b) { return a.onsetCs < b.onsetCs; });
        if (notes.size() > static_cast<size_t>(kMaxNotes))
            notes.erase(notes.begin(), notes.end() - kMaxNotes);
        return notes;
    }

    /** Every candidate key for these notes (already a window), best first. */
    [[nodiscard]] static auto rank(const std::vector<Note> &notes, int32_t nowCs) -> std::vector<Candidate> {
        std::array<double, 12> weight{};
        std::array<double, 12> triadWeight{}; // bass doubled
        double total = 0.0;
        for (const auto &n: notes) {
            const double age = std::max(0, nowCs - (n.onsetCs + n.durationCs));
            const double w = std::pow(2.0, -age / kHalfLifeCs)
                             * (0.5 + 0.5 * std::min(n.durationCs / 100.0, 1.0)) * (n.bass ? 1.5 : 1.0);
            const auto pc = static_cast<size_t>(Spelling::mod(n.pitch, 12));
            weight[pc] += w;
            triadWeight[pc] += n.bass ? 2.0 * w : w;
            total += w;
        }
        std::vector<Candidate> out;
        if (total <= 0.0)
            return out;
        for (const bool minor: {false, true})
            for (int tonic = 0; tonic < 12; ++tonic) {
                const Spelling::Key key{lofForTonic(tonic, minor, std::nullopt), minor};
                double in = 0.0;
                for (int pc = 0; pc < 12; ++pc)
                    if (Spelling::inScale(pc, key))
                        in += weight[static_cast<size_t>(pc)];
                const int third = minor ? 3 : 4;
                double triad = 0.0, triadTotal = 0.0;
                for (int pc = 0; pc < 12; ++pc)
                    triadTotal += triadWeight[static_cast<size_t>(pc)];
                for (const int iv: {0, third, 7})
                    triad += triadWeight[static_cast<size_t>(Spelling::mod(tonic + iv, 12))];
                double tonicEvidence = triadTotal > 0.0 ? triad / triadTotal : 0.0;
                if (minor && weight[static_cast<size_t>(Spelling::mod(tonic + 11, 12))] > 0.0)
                    tonicEvidence += 0.25;
                const double fit = in / total;
                out.push_back({tonic, minor, fit, fit + 0.1 * tonicEvidence});
            }
        std::stable_sort(out.begin(), out.end(), [](const Candidate &a, const Candidate &b) { return a.score > b.score; });
        return out;
    }

    /**
     * Line-of-fifths position for a tonic: the spelling with fewer accidentals; the tied keys
     * (F# / Gb major, D# / Eb minor) take the one nearer `previous` (default F# major, Eb minor).
     */
    [[nodiscard]] static auto lofForTonic(int pc, bool minor, std::optional<Spelling::Key> previous) -> int {
        // major: C Db D Eb E F F# G Ab A Bb B; minor: C C# D Eb E F F# G G# A Bb B
        static constexpr std::array<int, 12> kMajor{0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5};
        static constexpr std::array<int, 12> kMinor{0, 7, 2, -3, 4, -1, 6, 1, 8, 3, -2, 5};
        const auto p = static_cast<size_t>(Spelling::mod(pc, 12));
        const int a = minor ? kMinor[p] : kMajor[p];
        const bool tied = (! minor && p == 6) || (minor && p == 3);
        if (! tied || ! previous.has_value())
            return a;
        const int b = a > 0 ? a - 12 : a + 12; // the other spelling
        return std::abs(b - previous->tonicLof) < std::abs(a - previous->tonicLof) ? b : a;
    }

    /** "C major", "G# minor", "Eb major". */
    [[nodiscard]] static auto name(const Spelling::Key &key) -> std::string {
        return Spelling::nameOf(key.tonicLof) + (key.minor ? " minor" : " major");
    }

    /** Feed every note heard so far that may be in the window (past notes only), at `nowCs`. */
    auto update(const std::vector<Note> &notes, int32_t nowCs) -> const Result & {
        const auto w = window(notes, nowCs);
        int fresh = 0;
        int32_t newest = lastOnset;
        for (const auto &n: w)
            if (n.onsetCs > lastOnset) {
                ++fresh;
                newest = std::max(newest, n.onsetCs);
            }
        lastOnset = newest;

        if (static_cast<int>(w.size()) < kMinNotes) {
            result.uncertain = true;
            result.runnerUp.reset();
            return result;
        }
        const auto ranked = rank(w, nowCs);
        if (ranked.empty())
            return result;
        const auto &best = ranked.front();
        const auto scoreOf = [&](int pc, bool minor) -> const Candidate * {
            for (const auto &c: ranked)
                if (c.tonicPc == pc && c.minor == minor)
                    return &c;
            return nullptr;
        };
        if (! result.key.has_value()) {
            result.key = Spelling::Key{lofForTonic(best.tonicPc, best.minor, std::nullopt), best.minor};
            beatenNotes = 0;
        } else {
            const int curPc = Spelling::tonicPc(*result.key);
            const auto *cur = scoreOf(curPc, result.key->minor);
            const bool other = best.tonicPc != curPc || best.minor != result.key->minor;
            if (other && (cur == nullptr || best.score >= cur->score + kSwitchMargin)) {
                beatenNotes += fresh; // beaten, by whichever key leads now
                if (beatenNotes >= kSwitchNotes) {
                    result.key = Spelling::Key{lofForTonic(best.tonicPc, best.minor, result.key), best.minor};
                    beatenNotes = 0;
                }
            } else {
                beatenNotes = 0;
            }
        }
        const int curPc = Spelling::tonicPc(*result.key);
        const auto *cur = scoreOf(curPc, result.key->minor);
        result.uncertain = cur == nullptr || cur->fit < kSureFit;
        result.runnerUp.reset();
        for (const auto &c: ranked) {
            if (c.tonicPc == curPc && c.minor == result.key->minor)
                continue;
            if (cur != nullptr && std::abs(c.score - cur->score) <= kRunnerUpWithin)
                result.runnerUp = Spelling::Key{lofForTonic(c.tonicPc, c.minor, result.key), c.minor};
            break;
        }
        return result;
    }

    [[nodiscard]] auto current() const -> const Result & { return result; }

private:
    Result result;
    int beatenNotes = 0; // new notes in a row with the shown key beaten by kSwitchMargin
    int32_t lastOnset = INT32_MIN;
};

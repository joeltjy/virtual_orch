#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

/**
 * Chord-token model 2 ("m2c", B3f 7c): turns a CHORD block [CHORD (root, type, n)][SDUR][BASS][SVEL] into the notes
 * the orchestration transformer receives. Only the OT gets these notes; model 2's context keeps the four tokens.
 *
 * Bank (orchestration/overnight/build_voicing_bank.py -> results/voicing_bank_m2c.json):
 *   {"bank": {"type|n|bass_iv": [[semitone offsets from the bass, first 0], ... up to 5, most common first]}}
 *   bass_iv = (bass pitch class - root) mod 12, root = the CHORD token's root = the TRUE root pitch class.
 * Decoding: one voicing at random, transposed so its 0 sits on BASS. A missing key falls back to the same type and
 * bass_iv with the nearest n, else to the chord tones stacked upward from the bass.
 *
 * Roots are true pitch classes (a root-position C major is "maj|3|0"). The first m2c tokenizer named roots from
 * the token id (11128 + p, i.e. + 4 semitones); that was fixed in wait3.analyse_stack on 2026-11-24, before the
 * run this engine targets, and the bank was rebuilt to match.
 */
class VoicingBank {
public:
    static constexpr int kTypes = 18;
    /** wait3.TYPES, in token order. */
    static constexpr std::array<const char *, kTypes> kTypeNames{
        "unison", "ic1", "ic2", "ic3", "ic4", "ic5", "ic6", "maj", "min", "dim", "aug",
        "dom7", "min7", "maj7", "minmaj7", "hdim7", "dim7", "augmaj7"};

    [[nodiscard]] static auto typeIndex(const std::string &name) -> int;
    /** Pitch classes of a type: unison 1, ic1..ic6 2, triads 3, sevenths 4 (the k of [K k, +1]). */
    [[nodiscard]] static auto pitchClassCount(int type) -> int;
    /** The type's intervals above its root (unison {0}, icN {0, N}, maj {0, 4, 7}, ...). */
    [[nodiscard]] static auto intervals(int type) -> std::vector<int>;
    [[nodiscard]] static auto key(int type, int n, int bassIv) -> std::string;

    /** Parse the bank JSON text (juce::JSON); false (and `error`) when it has no usable "bank" object. */
    auto loadJson(const std::string &text, std::string &error) -> bool;
    /** Tests / callers: add one entry. */
    auto add(int type, int n, int bassIv, std::vector<std::vector<int>> voicings) -> void;
    [[nodiscard]] auto size() const -> size_t { return bank.size(); }
    [[nodiscard]] auto empty() const -> bool { return bank.empty(); }

    struct Decoded {
        std::vector<int32_t> pitches; // ascending, starting on the bass
        enum class Source { Bank, NearestN, Stacked } source = Source::Bank;
    };
    /**
     * The chord's pitches with the bass on `bassPitch`: type 0..17, root 0..11 (pitch class), n 1..10.
     * Pitches outside 21..108 are dropped (the bass is always kept).
     */
    [[nodiscard]] auto decode(int type, int root, int n, int32_t bassPitch, std::mt19937_64 &rng) const -> Decoded;

private:
    std::map<std::string, std::vector<std::vector<int>>> bank;
};

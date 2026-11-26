#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct Token;

/**
 * Hybrid v1.4 group tokens (orchestration docs/model_specs/v14_group_tokens.txt; Python: overnight/pkg/groups4.py).
 *
 * A group of orchestra notes on one onset and one duration class is ONE item: its lowest pitch (the bass) and a
 * SHAPE x N token = the pitch classes relative to the bass (43 shapes: unison/octave, 11 dyads, 10 triad inversions,
 * 21 seventh inversions) and N = the notes it stands for. 328 tokens, at shape0 + offset (m1_v4: 11768, m2_v4: 11911).
 *
 * The RT sends a group to the OT / playback as one Token whose pitch is the bass and whose chordShape is the
 * offset (-1 = a plain note): the chord is not decoded in the RT. The consumers expand it with expand() (the v4
 * voicing bank, seeded by the token's noteId, so the OT and reduction playback voice it alike).
 */
namespace GroupChord {

constexpr int kShapes = 328;

enum class Kind { Monad, Dyad, Triad, Seventh };

struct Shape {
    std::vector<int32_t> intervals; // pitch classes above the bass (0 first)
    Kind kind = Kind::Monad;
    int32_t n = 2;                  // notes the group stands for
    std::string name;               // "maj 1st", "+7", ...
};

/** The shape of a SHAPE x N offset (0..327). */
[[nodiscard]] auto shape(int32_t offset) -> const Shape &;
/** The offset for these intervals above the bass and N, or -1 (not a group shape, or N out of range). */
[[nodiscard]] auto offsetOf(const std::vector<int32_t> &intervals, int32_t n) -> int32_t;
/** Offsets whose shape has exactly k pitch classes (k 1..4). */
[[nodiscard]] auto offsetsWithPitchClasses(int32_t k) -> const std::vector<int32_t> &;
/** Pitch-class mask (bit pc) of a group with this bass pitch and shape offset. */
[[nodiscard]] auto pitchClassMask(int32_t bassPitch, int32_t offset) -> uint16_t;
/** "C3 maj 1st x4" */
[[nodiscard]] auto label(int32_t bassPitch, int32_t offset) -> std::string;

/** The v4 voicing bank: per offset, the interval sets above the bass seen in the corpus, with counts. */
class VoicingBankV4 {
public:
    /** {"bank": {"<offset>": [[[intervals...], count], ...]}} (overnight/build_voicing_bank_v4.py). */
    auto loadJson(const std::string &text, std::string &error) -> bool;
    auto loadFile(const std::string &path, std::string &error) -> bool;
    [[nodiscard]] auto size() const -> size_t { return bank.size(); }
    [[nodiscard]] auto empty() const -> bool { return bank.empty(); }
    /**
     * The pitches of a group: an interval set drawn by count with a generator seeded by `seed` (the same seed
     * gives the same voicing), added to the bass; no example: the shape stacked upward from the bass in close
     * position until N notes. Pitches above `top` are dropped (the bass always stays).
     */
    [[nodiscard]] auto decode(int32_t bassPitch, int32_t offset, uint64_t seed, int32_t top = 127) const
        -> std::vector<int32_t>;

private:
    std::map<int32_t, std::vector<std::pair<std::vector<int32_t>, int64_t>>> bank;
};

/** The bank the consumers (OT, reduction playback) decode with; set when Hybrid v1.4 loads. */
auto setSharedBank(std::shared_ptr<const VoicingBankV4> bank) -> void;
[[nodiscard]] auto sharedBank() -> std::shared_ptr<const VoicingBankV4>;

/** A chord token (chordShape >= 0) as its notes (same onset, duration, velocity, ids); any other token as is. */
[[nodiscard]] auto expand(const Token &token) -> std::vector<Token>;
/** expand() over a batch, in order. */
[[nodiscard]] auto expandAll(const std::vector<Token> &tokens) -> std::vector<Token>;

} // namespace GroupChord

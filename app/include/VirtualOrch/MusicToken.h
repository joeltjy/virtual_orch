#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace Config {
    constexpr int32_t MaxTimeInSeconds = 100;
    constexpr int32_t MaxDurationInSeconds = 10;
    constexpr int32_t TimeResolution = 100;

    constexpr int32_t MaxPitch = 128;
    constexpr int32_t MaxInstr = 129;
    constexpr int32_t MaxNote = MaxPitch * MaxInstr;

    constexpr int32_t MaxTime = TimeResolution * MaxTimeInSeconds;
    constexpr int32_t MaxDur = TimeResolution * MaxDurationInSeconds;

    constexpr int32_t AnticipationDelta = 0 * TimeResolution;
} // namespace Config

namespace Vocab {
    // Event Block
    constexpr size_t EventOffset = 0;
    constexpr size_t TimeOffset = EventOffset + 0;
    constexpr size_t DurOffset = TimeOffset + Config::MaxTime;
    constexpr size_t NoteOffset = DurOffset + Config::MaxDur;

    // SPECIAL ALIASES
    constexpr size_t BarClick = NoteOffset + Config::MaxPitch * 128 + 36;
    constexpr size_t BeatClick = NoteOffset + Config::MaxPitch * 128 + 42;

    constexpr size_t Rest = NoteOffset + Config::MaxNote;

    // Control Block
    constexpr size_t ControlOffset = NoteOffset + Config::MaxNote + 1;
    constexpr size_t AtimeOffset = ControlOffset + 0;
    constexpr size_t AdurOffset = AtimeOffset + Config::MaxTime;
    constexpr size_t AnoteOffset = AdurOffset + Config::MaxDur;

    // Special Block
    constexpr size_t SpecialOffset = AnoteOffset + Config::MaxNote;
    constexpr size_t Separator = SpecialOffset + 0;
    constexpr size_t AutoRegress = SpecialOffset + 1;
    constexpr size_t Anticipate = SpecialOffset + 2;
    constexpr size_t VocabSize = Anticipate + 1;

    // Added events
    constexpr size_t BarSeparator = VocabSize + 1;
    constexpr size_t ClearQueue = VocabSize + 2;
} // namespace Vocab

template<typename T>
static void softmax(T &input, const float temperature) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0F;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp((input[i] - rowmax) / temperature);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

/** Non-owning logit row for in-place sampleTopP (avoids copying full vocab). */
struct LogitSpan {
    float *ptr = nullptr;
    size_t len = 0;

    [[nodiscard]] auto begin() -> float * { return ptr; }
    [[nodiscard]] auto end() -> float * { return ptr + len; }
    [[nodiscard]] auto begin() const -> const float * { return ptr; }
    [[nodiscard]] auto end() const -> const float * { return ptr + len; }
    [[nodiscard]] auto size() const -> size_t { return len; }
    auto operator[](size_t i) -> float & { return ptr[i]; }
    auto operator[](size_t i) const -> float { return ptr[i]; }
    [[nodiscard]] auto data() -> float * { return ptr; }
    [[nodiscard]] auto data() const -> const float * { return ptr; }
};

/**
 * Nucleus then temperature (amt_causal / offline generate_sample order):
 * 1. Softmax at T=1 over finite logits → build nucleus (smallest prefix of
 *    descending mass with cumulative prob >= p; if p >= 1 keep all).
 * 2. Softmax(/temperature) on that truncated set only.
 * 3. Multinomial.
 * Ranking by raw logit matches T=1 softmax order. temperature <= 0 → argmax
 * on the nucleus (or all finite if p >= 1).
 */
template<typename T>
static int32_t sampleTopP(T &scores, const float p, const float temperature) {
    thread_local std::mt19937 gen{std::random_device{}()};

    std::vector<int32_t> finite;
    finite.reserve(256);
    for (size_t i = 0; i < scores.size(); ++i) {
        if (std::isfinite(scores[i]))
            finite.push_back(static_cast<int32_t>(i));
    }
    if (finite.empty())
        return 0;

    std::sort(finite.begin(), finite.end(), [&](const int32_t a, const int32_t b) {
        return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    });

    const float rowmax = scores[static_cast<size_t>(finite.front())];
    std::vector<float> untemp(finite.size());
    float sumU = 0.0f;
    for (size_t k = 0; k < finite.size(); ++k) {
        untemp[k] = std::exp(scores[static_cast<size_t>(finite[k])] - rowmax);
        sumU += untemp[k];
    }
    if (!(sumU > 0.0f) || ! std::isfinite(sumU))
        return finite.front();
    for (float &u: untemp)
        u /= sumU;

    size_t keep = finite.size();
    if (p < 1.0f) {
        float cum = 0.0f;
        keep = 0;
        for (; keep < finite.size(); ++keep) {
            cum += untemp[keep];
            if (cum >= p) {
                ++keep;
                break;
            }
        }
        if (keep == 0)
            keep = 1;
    }

    if (temperature <= 0.0f)
        return finite.front();

    std::vector<float> probs(keep);
    float sum = 0.0f;
    for (size_t k = 0; k < keep; ++k) {
        probs[k] = std::exp((scores[static_cast<size_t>(finite[k])] - rowmax) / temperature);
        sum += probs[k];
    }
    if (!(sum > 0.0f) || ! std::isfinite(sum))
        return finite.front();
    for (float &prob: probs)
        prob /= sum;

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float threshold = dis(gen);
    for (size_t k = 0; k < keep; ++k) {
        threshold -= probs[k];
        if (threshold <= 0.0f)
            return finite[k];
    }
    return finite[keep - 1];
}

static int32_t minTime(std::vector<int32_t> &tokens) {
    int32_t minTime = INT_MAX;
    for (size_t i = 0; i < tokens.size(); i += 3) {
        int32_t time = tokens[i];
        int32_t note = tokens[i + 2];

        // Stop calculating at sequence separator
        if (note == Vocab::Separator) {
            break;
        }

        if (note < Vocab::ControlOffset) {
            time -= Vocab::TimeOffset;
        } else {
            time -= Vocab::AtimeOffset;
        }

        minTime = std::min(minTime, time);
    }

    return minTime;
}

struct Token {
    int32_t time;
    int32_t duration;
    int32_t note;
    /** MIDI velocity 0–127. Default 100 matches AMT / orchestration when unset. */
    int32_t velocity = 100;
    /**
     * Stable vocsep voice label for OT / VocsepOutput.
     * -1 = unset / vocsep off / non-note; >= 0 inherits from parent or is newly allocated.
     */
    int32_t voiceId = -1;

    auto getRealDuration() const -> int32_t {
        return duration - Vocab::DurOffset;
    }

    auto getInstrument() const -> int32_t {
        return (note - Vocab::NoteOffset) / Config::MaxPitch;
    }

    auto getPitch() const -> int32_t {
        return (note - Vocab::NoteOffset) % Config::MaxPitch;
    }

    /** Re-encode pitch with a local instrument id (preserves time/duration/velocity). */
    auto withInstrument(int32_t localInstrumentId) const -> Token {
        Token out = *this;
        if (note >= static_cast<int32_t>(Vocab::NoteOffset)
            && note < static_cast<int32_t>(Vocab::Rest)) {
            out.note = static_cast<int32_t>(Vocab::NoteOffset
                                           + Config::MaxPitch * localInstrumentId + getPitch());
        }
        return out;
    }

    std::string toString() const {
        return "(" + std::to_string(time) + ", " + std::to_string(duration) + ", "
               + std::to_string(note) + ", " + std::to_string(velocity) + ")";
    }

    std::string toUnderstandableString() const {
        // Check for special tokens
        if (note == Vocab::BarSeparator) {
            return "(" + std::to_string(time) + ", Bar Separator)";
        }
        if (note == Vocab::ClearQueue) {
            return "(" + std::to_string(time) + ", Clear Queue)";
        }
        return "(" + std::to_string(time) + ", "
               + std::to_string(duration - Vocab::DurOffset) + ", "
               + std::to_string(getInstrument()) + " - "
               + std::to_string(getPitch()) + ", vel "
               + std::to_string(velocity) + ")";
    }

    juce::MemoryBlock toMemoryBlock() const {
        juce::MemoryOutputStream stream;

        // Error Token
        if ((time > std::numeric_limits<uint32>::max())
            || (getRealDuration() > std::numeric_limits<uint16>::max())
            || (getInstrument() > std::numeric_limits<uint8>::max()
                && note != Vocab::BarSeparator && note != Vocab::ClearQueue)
            || (note >= Vocab::Rest
                && note != Vocab::BarSeparator && note != Vocab::ClearQueue)) {
            stream.writeInt(0);
            stream.writeShort(0);
            stream.writeByte(0);
            stream.writeByte(0);
        } else {
            stream.writeInt(time);
            stream.writeShort(getRealDuration());
            if (note == Vocab::BarSeparator) {
                stream.writeByte(0);
                stream.writeByte(129);
            } else if (note == Vocab::ClearQueue) {
                stream.writeByte(0);
                stream.writeByte(130);
            } else {
                stream.writeByte(getInstrument());
                stream.writeByte(getPitch());
            }
        }
        return stream.getMemoryBlock();
    }
};

inline bool operator<(const Token &lhs, const Token &rhs) {
    return lhs.time < rhs.time;
}

inline auto tokenEquals(const Token &lhs, const Token &rhs) -> bool {
    return lhs.time == rhs.time && lhs.duration == rhs.duration && lhs.note == rhs.note
           && lhs.velocity == rhs.velocity;
}

/** Match onset/dur/vel/pitch, ignoring local-instrument embedding in `note`. */
inline auto tokenEqualsByOnsetPitch(const Token &lhs, const Token &rhs) -> bool {
    if (lhs.time != rhs.time || lhs.duration != rhs.duration || lhs.velocity != rhs.velocity)
        return false;
    const bool lhsNote = lhs.note >= static_cast<int32_t>(Vocab::NoteOffset)
                         && lhs.note < static_cast<int32_t>(Vocab::Rest);
    const bool rhsNote = rhs.note >= static_cast<int32_t>(Vocab::NoteOffset)
                         && rhs.note < static_cast<int32_t>(Vocab::Rest);
    if (! lhsNote || ! rhsNote)
        return lhs.note == rhs.note;
    return lhs.getPitch() == rhs.getPitch();
}

inline auto tokenOrderLess(const Token &lhs, const Token &rhs) -> bool {
    if (lhs.time != rhs.time)
        return lhs.time < rhs.time;
    if (lhs.duration != rhs.duration)
        return lhs.duration < rhs.duration;
    if (lhs.note != rhs.note)
        return lhs.note < rhs.note;
    return lhs.velocity < rhs.velocity;
}

/** Replace a note. For example, when noteOffs come in. */
struct TokenUpdate {
    Token oldNote;
    Token newNote;
};

/** Find oldNote scanning backwards; replace with newNote; reinsert if order key changed. */
inline auto applyTokenUpdateToHistory(std::vector<Token> &history, const TokenUpdate &update) -> bool {
    for (auto rit = history.rbegin(); rit != history.rend(); ++rit) {
        if (! tokenEquals(*rit, update.oldNote))
            continue;

        auto pos = std::prev(rit.base());

        const bool orderKeyChanged = update.newNote.time != update.oldNote.time
                                     || update.newNote.duration != update.oldNote.duration
                                     || update.newNote.note != update.oldNote.note
                                     || update.newNote.velocity != update.oldNote.velocity;

        if (! orderKeyChanged) {
            *pos = update.newNote;
            return true;
        }

        history.erase(pos);
        const auto insertAt = std::lower_bound(history.begin(), history.end(), update.newNote, tokenOrderLess);
        history.insert(insertAt, update.newNote);
        return true;
    }

    return false;
}

enum ModelType : uint8_t {
    Small,
    Medium
};

constexpr int SMALL_HIDDEN_SIZE = 12;
constexpr int MEDIUM_HIDDEN_SIZE = 16;

constexpr int SMALL_N_HEADS = 12;
constexpr int MEDIUM_N_HEADS = 24;


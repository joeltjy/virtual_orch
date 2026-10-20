#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "VirtualOrch/TdTokens.h"
#include "VirtualOrch/reduction/AnticipatedStream.h"
#include "VirtualOrch/reduction/AnticipationSampler.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * One generating model of the anticipation family (T3'', T8'', T7''): its token stream, its
 * committed + scratch KV cache, its sampler and its TD ids, plus the loop that draws the next
 * instrument-1 note:
 *
 *   1. the stream's given notes whose actual time the event clock has reached are written first;
 *   2. the onset is drawn; if it lies at or after the next GIVEN note, that note is written first
 *      and the onset is drawn again (the AMT-native vocab also writes a visible 1 s REST quad for
 *      every step before the drawn onset, and releases the melody controls as the clock moves);
 *   3. duration, note and velocity are drawn, the event is written, and the clock moves to it.
 *
 * Dense stages (T3'', T8''): no rests, the clock is max(last event, app clock). AMT stage (T7''):
 * visible rests, the clock follows the rest chain and catches up with the app clock one rest at a
 * time. A note at a tie goes given-first unless tieFirstGenerated is set (the T3'' training order).
 */
class AnticipationStage {
public:
    struct Params {
        int32_t now = 0;              // app clock (cs)
        int32_t lowerAbs = 0;         // earliest onset: polyphony cap, never in the past
        std::array<float, 4> temperatures{0.5f, 0.5f, 0.5f, 0.5f};
        bool pitchBias = true;
        bool biases = true;           // BiasSettings gate: false turns all three biases off
        int32_t pitchLow = DenseQuantize::PianoKeyLow;   // register limits (inclusive)
        int32_t pitchHigh = DenseQuantize::PianoKeyHigh;
    };

    AnticipationStage(AnticipatedStream::Vocab vocabIn, bool visibleRestsIn)
        : stream(vocabIn),
          sampler(vocabIn == AnticipatedStream::Vocab::Amt ? AnticipationSampler::kAmtVelocityBase
                                                           : AnticipationSampler::kDenseVelocityBase),
          visibleRests(visibleRestsIn),
          tdResolver(vocabIn == AnticipatedStream::Vocab::Amt ? TdTokens::kAmtFirst
                                                              : TdTokens::kDenseFirst) {
        stream.tdTokenProvider = [this] {
            return tdResolver.resolve(tdValue ? tdValue() : TemporalDensity::kAuto,
                                      static_cast<int32_t>(vocabSize));
        };
    }

    auto setBackend(std::unique_ptr<StreamingKv::Backend> newBackend, size_t vocab) -> void {
        backend = std::move(newBackend);
        vocabSize = vocab;
        cache = std::make_unique<StreamingKv::CommittedScratchCache>(*backend);
    }
    [[nodiscard]] auto isLoaded() const -> bool { return backend != nullptr; }

    /** Start (or restart) the stream at the app clock. */
    auto reset(int32_t nowCs) -> void {
        stream.reset(nowCs);
        if (cache != nullptr)
            cache->reset();
        sampler.reset();
        tdResolver = TdTokens::LevelTdResolver(stream.vocabKind() == AnticipatedStream::Vocab::Amt
                                                   ? TdTokens::kAmtFirst
                                                   : TdTokens::kDenseFirst);
    }

    /** Draw the next note, or nothing (no context yet, or an error: see lastError). */
    auto generate(const Params &params) -> std::optional<AnticipatedStream::Note>;

    AnticipatedStream stream;
    AnticipationSampler sampler;
    std::unique_ptr<StreamingKv::CommittedScratchCache> cache;
    std::function<int32_t()> tdValue; // Launchpad value (TemporalDensity::kAuto or a level)
    bool tieFirstGenerated = false;
    std::string lastError;

private:
    static constexpr int kMaxRedraws = 256;
    static constexpr int32_t kRestStepCs = 100;

    std::unique_ptr<StreamingKv::Backend> backend;
    size_t vocabSize = 0;
    bool visibleRests;
    TdTokens::LevelTdResolver tdResolver;
};

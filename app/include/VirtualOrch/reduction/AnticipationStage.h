#pragma once

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
 *
 * W2/W3 (waitToken >= 0): at an onset decision WAIT is allowed unless a 1 s wait at this point
 * already timed out. Threshold rule: WAIT when P(play) < waitMinPlay (P from the masked onset +
 * WAIT distribution at T 1, before top-p). WAIT with a given group available writes that group
 * (at the clock or later) and decides again; without one the stage WAITS (generate returns nothing,
 * lastWaited()) until a group arrives, or until max(clock, now at the WAIT) + 1 s, then plays on
 * with WAIT masked. A generated note at the onset of the previous one may not go below its pitch.
 * As in training (the clock is the last written event, given or generated), the onset step right
 * after the soloist's group may start AT that group's onset, though it is already a little in
 * the past live (input latency), if it is at most kAttackLookbackCs old; playback plays such a
 * note at once and the context keeps its true onset.
 *
 * Late input (Params::reorderLate, Accompaniment): given notes behind the clock are held back and
 * written in onset order by AnticipatedStream::reorderLate at most once every kReorderEveryCs.
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
        // Per-model switches (T3'' and W2/W3; the BiasSettings gate `biases` applies on top).
        bool durationBias = true;
        bool onsetGapBias = true;
        float pitchTau = DensePitchBias::TauBase;
        float pitchTauPrime = DensePitchBias::TauBase;
        float durationTau = DenseDurationBias::TauBase;
        float durationTauPrime = DenseDurationBias::TauPrimeBase;
        float onsetGapTau = DenseOnsetGapBias::TauBase;
        TdOnsetBias::Options onsetOptions; // short-gap penalty, LogitTD
        /** Accompaniment: hold late given notes and reorder them in at most once a second. */
        bool reorderLate = false;
        /** How often late notes are reordered in (Edit: at once, 0). */
        int32_t reorderEveryCs = 100;
        /** W2/W3 threshold: WAIT when P(play) is below this. */
        float waitMinPlay = 0.30f;
    };

    static constexpr int32_t kReorderEveryCs = 100;
    static constexpr int32_t kWaitTimeoutCs = 100;
    static constexpr int32_t kAttackLookbackCs = 30;
    /** W2/W3: a generated onset is at most this far past the clock (3 s). */
    static constexpr int32_t kMaxAheadCs = 300;
    int32_t maxAheadCs = kMaxAheadCs; // (tests widen it)
    /** W2/W3: the WAIT id; -1 = no WAIT (T3'', T8'', T7''). Also set on the stream. */
    int32_t waitToken = -1;
    /** W2/W3: a generated note at the onset of the previous one is not lower than it. */
    bool ascendingTies = false;

    /** The last generate() returned nothing because the model is waiting for the soloist. */
    [[nodiscard]] auto lastWaited() const -> bool { return waiting; }
    /** WAIT events for the log: "wait", "resume", "timeout", "reorder"; value = clock / count. */
    std::function<void(const char *, int32_t)> onWaitEvent;

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
        clearWaitState();
        mirrored.clear();
        lastReorderCs = INT32_MIN;
        tdResolver = TdTokens::LevelTdResolver(stream.vocabKind() == AnticipatedStream::Vocab::Amt
                                                   ? TdTokens::kAmtFirst
                                                   : TdTokens::kDenseFirst);
    }

    /**
     * Leaving Accompaniment: drop the notes generated from cutCs on (stream and bias history) and
     * the given notes `drop` selects; the cache is rebuilt from what is kept. True if it changed.
     */
    auto rewindTo(int32_t cutCs,
                  const std::function<bool(const AnticipatedStream::ControlSpec &,
                                           const AnticipatedStream::Note &)> &drop = {}) -> bool {
        const bool changed = stream.rewindTo(cutCs, drop);
        sampler.forgetGeneratedFrom(cutCs);
        clearWaitState();
        std::erase_if(mirrored, [cutCs](const AnticipatedStream::Note &n) { return n.onsetCs >= cutCs; });
        if (changed && cache != nullptr)
            cache->reset();
        return changed;
    }

    /**
     * Hybrid: the OTHER model generated `note`. It is queued and written into this stream (given
     * notes due by then first) when this model next generates, so both keep the same context; a
     * cut before then drops it (a far-future note never moves this stream's origin for nothing).
     * Cancels a pending WAIT.
     */
    auto mirrorEvent(const AnticipatedStream::Note &note, int32_t /*now*/, bool /*reorderLate*/) -> void {
        if (! stream.isStarted())
            return;
        mirrored.push_back(note);
        clearWaitState();
    }
    [[nodiscard]] auto mirroredCount() const -> size_t { return mirrored.size(); }

    /** Draw the next note, or nothing (no context yet, or an error: see lastError). */
    auto generate(const Params &params) -> std::optional<AnticipatedStream::Note>;

    AnticipatedStream stream;
    AnticipationSampler sampler;
    std::unique_ptr<StreamingKv::CommittedScratchCache> cache;
    std::function<int32_t()> tdValue; // Launchpad value (TemporalDensity::kAuto or a level)
    bool tieFirstGenerated = false;
    std::string lastError;

private:
    auto clearWaitState() -> void {
        waiting = false;
        waitDeadlineCs = INT32_MIN;
        timedOutAtItems = SIZE_MAX;
    }
    auto event(const char *what, int32_t value) -> void {
        if (onWaitEvent)
            onWaitEvent(what, value);
    }

    /** Write the queued notes of the other model (hybrid) in onset order. */
    auto applyMirrored(int32_t now, bool reorderLate) -> void {
        for (const auto &note: mirrored) {
            while (stream.needsRebase(note.onsetCs))
                stream.rebase();
            if (reorderLate)
                stream.holdLate();
            stream.insertDue(note.onsetCs, now, ! tieFirstGenerated);
            stream.appendEvent(note);
            sampler.recordGenerated(note.onsetCs, note.durationCs, note.pitch, note.velocity);
        }
        mirrored.clear();
    }

    std::vector<AnticipatedStream::Note> mirrored;
    bool waiting = false;
    int32_t waitDeadlineCs = INT32_MIN;
    size_t timedOutAtItems = SIZE_MAX; // WAIT masked while nothing new has been written
    int32_t lastReorderCs = INT32_MIN;

    static constexpr int kMaxRedraws = 256;
    static constexpr int32_t kRestStepCs = 100;

    std::unique_ptr<StreamingKv::Backend> backend;
    size_t vocabSize = 0;
    bool visibleRests;
    TdTokens::LevelTdResolver tdResolver;
};

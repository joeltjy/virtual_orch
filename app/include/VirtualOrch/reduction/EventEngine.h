#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "VirtualOrch/OpLogDraw.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"
#include "VirtualOrch/reduction/VoicingBank.h"

/**
 * Hybrid v1.3: the event-tokenized models (orchestration/docs/model_specs/event_pair_proposal.txt; tokenizer
 * orchestration/overnight/pkg/events.py). A note is two events instead of one quad:
 *   ON  = time, note, vel     (time 0..9999 cs from the context's base; note 11000 + 128 instr + pitch; vel 11640 + v)
 *   OFF = time, OFF0 + 128 instr + pitch
 * in time order; at equal times every OFF before every ON, then by note id (instrument, pitch). Piano key-down and
 * key-up are written as events at once: no duration estimate and no rewriting of the context.
 *
 * This class is model 1 event (m1e, continuation, vocab 12024, OFF0 11768). Its sampling rule (spec section 8):
 * after a drawn time the model may put its mass on a PIANO event (it expects the pianist); such a step is a no-op
 * ("nothing happens"), and the next drawn time must be later. When the orchestra has been silent for
 * forceAfterCs, the piano tokens are masked instead, so an orchestra event is forced after a long hold.
 * An orchestra note the model never releases is released after maxHoldCs.
 *
 * Each drawn event waits as the scratch tail of the KV cache until the clock reaches its time; a piano event that
 * comes first throws it away (it is drawn again after the piano event), so the context stays in time order.
 *
 * Settings::chordTokens: model 2 event (m2e, vocab 14328, OFF0 14071, STACK_OFF 14327, plus wait3's WAIT / K /
 * EXPIRED / LATE / CHORD ids; events2.py, spec section 4):
 *   - where a time would be drawn, the model may draw [WAIT d][K k,+1] instead (d >= 6 cs, 12 with +1), measured
 *     from the event before it; while it is pending the orchestra may release notes (OFFs) but not start any;
 *   - a piano ON from 6 cs before the target to the window close (ref + d + 6) fires: the piano ONs of that onset
 *     (within 5 cs, written at its time), then the block [CHORD (k pitch classes)][BASS (a chord tone)][SVEL], decoded
 *     into band notes with the voicing bank, then the leftover ON at the onset when +1; the model later releases
 *     the band notes with one (time, STACK_OFF) event, the leftover with its own OFF;
 *   - at a window close the model draws [LATE d] (at most maxLates; a piano ON already past the close sets d) or
 *     [EXPIRED]; after [EXPIRED] it draws a WAIT (re-wait, the same K) or a CHORD (give up: the block plays alone at
 *     the close).
 */
class EventEngine {
public:
    static constexpr int32_t kMaxTime = 9999;
    static constexpr int32_t kNote0 = 11000;
    static constexpr int32_t kOrch0 = 11128; // note token of instrument 1 (the orchestra / reduction)
    static constexpr int32_t kVel0 = 11640;
    static constexpr int32_t kM1eOff0 = 11768;
    static constexpr int32_t kM1eVocab = 12024;
    static constexpr int32_t kPianoLow = 21;
    static constexpr int32_t kPianoHigh = 108;

    struct Settings {
        int32_t off0 = kM1eOff0;
        int32_t vocab = kM1eVocab;
        uint64_t seed = 1;
        float onsetTemperature = 1.0f;
        float otherTemperature = 1.0f;
        float minP = 0.0f;
        /** Orchestra pitches drawn in [pitchLow, pitchHigh] (the Launchpad register). */
        int32_t pitchLow = kPianoLow;
        int32_t pitchHigh = kPianoHigh;
        /** Earliest time of a new orchestra ON (the polyphony cap's lower bound, absolute cs). */
        int32_t minOnsetCs = INT32_MIN;
        /** At most this many orchestra notes sounding (0 = no cap): an ON beyond it is masked. */
        int32_t maxSounding = 0;
        /** A drawn time is at most this far after the floor (cs). */
        int32_t maxAheadCs = 300;
        /** Orchestra silent (no orchestra event) this long: piano tokens are masked (spec section 8). */
        int32_t forceAfterCs = 200;
        /** An orchestra note sounding this long is released by the engine (Q7: 8 s; the RT plays at most 5 s). */
        int32_t maxHoldCs = 500;
        /** Model 2 event: the WAIT family, chord blocks and STACK_OFF (see the class comment). */
        bool chordTokens = false;
        int32_t stackOff = -1;
        std::shared_ptr<const VoicingBank> voicings;
        int32_t maxLates = 2;
        /** A window closes this long after its time (live input arrives ~12 cs late). */
        int32_t expiryGraceCs = 20;
        /** The context is rebuilt once its time span passes this (time tokens stop at 9999). */
        int32_t maxSpanCs = 9000;
        /** ... keeping about this many of its newest tokens. */
        size_t keepTokens = 384;
        /**
         * Joel's logit adjustments (as v1.1 / v1.2, Nov 25 2026; set by the transformer from RtLogitAdjustments), on
         * the orchestra's ON history (own and mirrored notes):
         *   onsetGapBias: -tau log s_gap(t - last orchestra ON) on the drawn time tokens (only the time range, not the
         *     WAIT family), as DenseOnsetGapBias on the dense models;
         *   pitchBias: -tau log pi(p) - tau' log s(p - last pitch) on the orchestra ON pitches of the "what happens"
         *     pick, then a constant so their total probability is unchanged: the bias picks which pitch, not whether
         *     an ON, an OFF or a piano no-op happens. The chord block's BASS and leftover picks are not biased.
         * The event models have no duration token: no duration bias.
         */
        bool pitchBias = false;
        bool onsetGapBias = false;
        float pitchTau = DensePitchBias::TauBase;
        float pitchTauPrime = DensePitchBias::TauBase;
        float onsetGapTau = 0.5f; // DenseOnsetGapBias::TauBase
    };

    enum class Kind : uint8_t { PianoOn, PianoOff, OrchOn, OrchOff, StackOff, Special };

    // wait3 ids (model 2 event keeps every m2c id)
    static constexpr int32_t kWait0 = 11768;
    static constexpr int32_t kK0 = 11868;
    static constexpr int32_t kExpired = 11876;
    static constexpr int32_t kLate0 = 11877;
    static constexpr int32_t kChord0 = 11911;
    static constexpr int32_t kChordEnd = 14071;
    static constexpr int32_t kGrid = 3;
    static constexpr int32_t kMaxWait = 300;
    static constexpr int32_t kLateMax = 99;
    static constexpr int32_t kEps = 6;
    static constexpr int32_t kWinEarly = 6;
    static constexpr int32_t kMergeCs = 5;
    static constexpr int32_t kFloor = 6;
    static constexpr int32_t kFloorPlus = 12;

    struct Event {
        Kind kind = Kind::OrchOn;
        int32_t timeCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
        bool generated = false; // drawn by this engine (not the piano, not mirrored)
        std::vector<int32_t> toks; // Special: the tokens (WAIT + K, LATE, EXPIRED, the chord block)
    };

    /** What step() produced for the transformer: an orchestra ON or OFF, at its time. */
    struct Out {
        bool on = true;
        int32_t timeCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
        bool forcedRelease = false; // released by the engine (maxHoldCs), not by the model
        bool chord = false;         // a chord block's band note (or its release), or the block's leftover
        /** The draws that made this note (onset, event, vel), while OpLog::on(): the operation log's RT_DRAW. */
        std::vector<OpLogDraw::Draw> draws;
    };

    struct Stats {
        int ons = 0;
        int offs = 0;
        int pianoMeant = 0;    // a piano token after a drawn time: nothing happened
        int forced = 0;        // ... with the piano masked (forceAfterCs)
        int forcedReleases = 0;
        int discarded = 0;     // a drawn event thrown away because a piano event came first
        int rebuilds = 0;
        // model 2 event
        int waits = 0;
        int fires = 0;
        int lates = 0;
        int latesForced = 0;
        int expired = 0;
        int reWaits = 0;
        int giveUps = 0;
        int stackOffs = 0;
        int voicingFallbacks = 0;
    };

    EventEngine();

    auto setBackend(StreamingKv::Backend *backendIn) -> void;
    auto setSettings(const Settings &s) -> void { cfg = s; }
    [[nodiscard]] auto settings() const -> const Settings & { return cfg; }
    auto reset(int32_t nowCs) -> void;

    /** Piano key-down / key-up (absolute cs). */
    auto pianoOn(int32_t timeCs, int32_t pitch, int32_t velocity) -> void;
    auto pianoOff(int32_t timeCs, int32_t pitch) -> void;
    /** An orchestra note of the other model (hybrid): written into this context, not played again. */
    auto mirrorOn(int32_t timeCs, int32_t pitch, int32_t velocity) -> void;
    auto mirrorOff(int32_t timeCs, int32_t pitch) -> void;
    /** Drop the pending draw (a model switch, a cut). */
    auto interrupt() -> void {
        pending.reset();
        wait.reset();
        fire.reset();
    }
    /** Generated events from `cutCs` on leave the context (their notes were cut downstream). */
    auto dropGeneratedFrom(int32_t cutCs) -> void;

    /** Draw / commit at clock `nowCs`: the orchestra events that start now. */
    auto step(int32_t nowCs) -> std::vector<Out>;

    /** The context as model tokens (times relative to the base). */
    [[nodiscard]] auto tokens() const -> std::vector<int32_t>;
    [[nodiscard]] auto events() const -> const std::vector<Event> & { return items; }
    [[nodiscard]] auto stats() const -> const Stats & { return st; }
    /** The time the onset tokens count from (cs). */
    [[nodiscard]] auto base() const -> int32_t { return baseCs; }
    [[nodiscard]] auto hasPending() const -> bool { return pending.has_value(); }
    [[nodiscard]] auto hasPendingWait() const -> bool { return wait.has_value(); }
    /** Orchestra notes sounding now (ON without OFF), as (onset, pitch). */
    [[nodiscard]] auto sounding() const -> std::vector<std::pair<int32_t, int32_t>>;

    std::string lastError;

private:
    struct Pending {
        int32_t timeCs = 0;
        bool on = true;
        int32_t pitch = 60;
        int32_t velocity = 64;
        std::vector<int32_t> toks; // the drawn tokens (time, note / OFF [, vel])
        std::vector<OpLogDraw::Draw> draws;
    };

    [[nodiscard]] auto timeToken(int32_t timeCs) const -> int32_t { return timeCs - baseCs; }
    [[nodiscard]] auto noteToken(Kind kind, int32_t pitch) const -> int32_t;
    auto append(const Event &e) -> void;
    auto insertInput(Event e) -> void;
    auto lastTime() const -> int32_t { return items.empty() ? baseCs : items.back().timeCs; }
    auto maybeRebuild(int32_t floorCs) -> void;
    auto logitsFor(const std::vector<int32_t> &desired) -> const std::vector<float> *;
    /**
     * Sample a token from `ranges` of `logits` (min-p, temperature). With `field` set and OpLog::on(), the draw is
     * recorded in drawLog (p = the distribution drawn from, p_model = softmax of `raw` (default: `logits`) at
     * temperature 1, the 10 most probable tokens by p).
     */
    auto pick(const std::vector<float> &logits, const std::vector<std::pair<int32_t, int32_t>> &ranges, float temperature,
              const char *field = nullptr, const std::vector<float> *raw = nullptr) -> int32_t;
    std::vector<OpLogDraw::Draw> drawLog; // draws of the event being drawn (draw())
    auto draw(int32_t nowCs) -> void;
    auto commitPending() -> std::vector<Out>;
    auto releaseLongHolds(int32_t nowCs) -> std::vector<Out>;
    // model 2 event
    struct Wait {
        int32_t refCs = 0;
        int32_t d = 0;
        int32_t kTok = 0;
        int32_t k = 1;
        bool plus = false;
        int32_t closeCs = 0;
        int lates = 0;
    };
    struct Fire {
        int32_t onsetCs = 0;
        int32_t readyCs = 0;
        bool noPiano = false;
        int32_t firstChord = -1; // a give-up's CHORD, drawn after [EXPIRED]
        int32_t k = 1;
        bool plus = false;
    };
    struct Stack {
        int32_t onsetCs = 0;
        std::vector<int32_t> pitches;
    };
    [[nodiscard]] auto chordMode() const -> bool { return cfg.chordTokens; }
    auto special(int32_t timeCs, std::vector<int32_t> toks) -> void;
    auto drawWaitK(std::vector<int32_t> desired, int32_t waitTok, int32_t refCs, int32_t sameKTok) -> bool;
    auto onPianoOnChord(Event &e) -> void;
    auto closeWindow(int32_t nowCs) -> void;
    auto writeBlock(int32_t nowCs) -> std::vector<Out>;
    [[nodiscard]] auto chordRanges(int32_t k) const -> std::vector<std::pair<int32_t, int32_t>>;
    [[nodiscard]] auto waitRanges(int32_t refCs, int32_t nowCs) const -> std::vector<std::pair<int32_t, int32_t>>;
    [[nodiscard]] auto soundingAll() const -> std::vector<int32_t>;
    /** The orchestra ONs (own and mirrored) as dense quads (onset, dur, note, vel) for the logit adjustments. */
    [[nodiscard]] auto orchestraQuads() const -> std::vector<int32_t>;
    /** cfg.pitchBias on the orchestra ON ids of `logits` among `ranges`, their total mass kept (see Settings). */
    auto applyPitchBias(std::vector<float> &logits, const std::vector<std::pair<int32_t, int32_t>> &ranges,
                        int32_t timeCs) -> void;

    Settings cfg;
    StreamingKv::Backend *backend = nullptr;
    std::unique_ptr<StreamingKv::CommittedScratchCache> cache;
    std::mt19937_64 rng;
    std::vector<Event> items;
    int32_t baseCs = 0;
    int32_t nowCs = 0;
    int32_t floorCs = INT32_MIN;       // the next drawn time must be >= this (after a piano-meant step: later)
    int32_t lastOrchCs = INT32_MIN;    // the newest orchestra event (forceAfterCs)
    std::optional<Pending> pending;
    std::optional<Wait> wait;
    std::optional<Fire> fire;
    std::vector<Stack> stacks;
    Stats st;
    DensePitchBias::PitchTauScheduler pitchScheduler;
};

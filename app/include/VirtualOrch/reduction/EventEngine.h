#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "VirtualOrch/reduction/StreamingKvCache.h"

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
        /** The context is rebuilt once its time span passes this (time tokens stop at 9999). */
        int32_t maxSpanCs = 9000;
        /** ... keeping about this many of its newest tokens. */
        size_t keepTokens = 384;
    };

    enum class Kind : uint8_t { PianoOn, PianoOff, OrchOn, OrchOff };

    struct Event {
        Kind kind = Kind::OrchOn;
        int32_t timeCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
        bool generated = false; // drawn by this engine (not the piano, not mirrored)
    };

    /** What step() produced for the transformer: an orchestra ON or OFF, at its time. */
    struct Out {
        bool on = true;
        int32_t timeCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
        bool forcedRelease = false; // released by the engine (maxHoldCs), not by the model
    };

    struct Stats {
        int ons = 0;
        int offs = 0;
        int pianoMeant = 0;    // a piano token after a drawn time: nothing happened
        int forced = 0;        // ... with the piano masked (forceAfterCs)
        int forcedReleases = 0;
        int discarded = 0;     // a drawn event thrown away because a piano event came first
        int rebuilds = 0;
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
    auto interrupt() -> void { pending.reset(); }
    /** Generated events from `cutCs` on leave the context (their notes were cut downstream). */
    auto dropGeneratedFrom(int32_t cutCs) -> void;

    /** Draw / commit at clock `nowCs`: the orchestra events that start now. */
    auto step(int32_t nowCs) -> std::vector<Out>;

    /** The context as model tokens (times relative to the base). */
    [[nodiscard]] auto tokens() const -> std::vector<int32_t>;
    [[nodiscard]] auto events() const -> const std::vector<Event> & { return items; }
    [[nodiscard]] auto stats() const -> const Stats & { return st; }
    [[nodiscard]] auto hasPending() const -> bool { return pending.has_value(); }
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
    };

    [[nodiscard]] auto timeToken(int32_t timeCs) const -> int32_t { return timeCs - baseCs; }
    [[nodiscard]] auto noteToken(Kind kind, int32_t pitch) const -> int32_t;
    auto append(const Event &e) -> void;
    auto insertInput(Event e) -> void;
    auto lastTime() const -> int32_t { return items.empty() ? baseCs : items.back().timeCs; }
    auto maybeRebuild(int32_t floorCs) -> void;
    auto logitsFor(const std::vector<int32_t> &desired) -> const std::vector<float> *;
    auto pick(const std::vector<float> &logits, const std::vector<std::pair<int32_t, int32_t>> &ranges, float temperature)
        -> int32_t;
    auto draw(int32_t nowCs) -> void;
    auto commitPending() -> Out;
    auto releaseLongHolds(int32_t nowCs) -> std::vector<Out>;

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
    Stats st;
};

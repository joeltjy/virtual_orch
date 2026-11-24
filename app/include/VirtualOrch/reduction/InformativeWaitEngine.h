#pragma once

#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/DenseOnsetGapBias.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * Informative WAIT (overnight model 2, 2026-11-23; vocab 11879). Token ids:
 *   onset 0..9999 (cs from the line origin), duration 10000 + d (q5), piano note 11000 + p,
 *   orchestra note 11128 + p, velocity 11640 + v (q8),
 *   [WAIT d] = 11768 + d / 3 - 1 (d = 3..300 cs, 3 cs grid), [COUNT k] = 11868 + k - 1 (k = 1..10),
 *   [EXPIRED] = 11878 (sampler-written, never sampled).
 * [WAIT d][COUNT k] = "the pianist's next onset is d cs after the event right before this WAIT; put k
 * orchestra notes on it". The window is [ref + d - 6, ref + d + 6]: a piano onset in it FIRES the WAIT
 * (first piano note, the k orchestra notes on the piano onset, the rest of the chord); an earlier
 * piano onset is ordinary input (the WAIT stays); none by ref + d + 6 writes [EXPIRED] there.
 *
 * This is the live port of overnight/pkg/sample_overnight.py (orchestration repo), three modes:
 *   Unconstrained   the model's own sampling (orchestra chords, WAITs with any COUNT);
 *   LineLegato (A)  one orchestra line; after each note [t_s, t_e] a PROVISIONAL [WAIT x][COUNT 1] is
 *                   written (x = its duration on the WAIT grid); a piano onset in [t_e - 6, t_e] fires
 *                   it, otherwise it is removed at t_e and the next note starts exactly at t_e;
 *   LineRests (B)   as A, but after a removal the model draws any onset >= t_e (rests) or its own
 *                   [WAIT d] with COUNT forced to 1 (fires / expires as in Unconstrained).
 * Sampling as the Python sampler: masks -> min-p 0.25 (duration / pitch / velocity) -> temperature
 * (onset / WAIT / COUNT 1.0, the rest 0.5) -> top-p 0.98. Piano notes, [EXPIRED] and the provisional
 * WAITs are never sampled. Piano notes within 5 cs of a group's first onset join the group (training
 * merge); a drawn orchestra onset is > 5 cs after the last piano onset, and a piano onset arriving
 * at or before a not-yet-played orchestra onset + 5 cuts that note (the piano goes first, as the
 * sampler's redraw rule).
 *
 * Anticipation (batched precompute): when a WAIT becomes pending, the engine feeds the target onset
 * T = ref + d, takes the most likely duration, and keeps the top-5 piano pitches (batch 1, scratch).
 * It then tiles that KV state to 5 branches and, one batched call per step(), feeds each branch's
 * pitch, its most likely velocity, and draws the k orchestra notes forced onto T. If the pianist's
 * first note at the fire is one of the five and its branch is complete, those notes are emitted at
 * once (no decode on the critical path); otherwise they are decoded then (k x 3 calls).
 *
 * Times are absolute app centiseconds. No JUCE: the transformer wraps it (and tests drive it).
 */
class InformativeWaitEngine {
public:
    static constexpr int32_t kOnsetMax = 9999;
    static constexpr int32_t kDur0 = 10000;
    static constexpr int32_t kPiano0 = 11000;
    static constexpr int32_t kOrch0 = 11128;
    static constexpr int32_t kVel0 = 11640;
    static constexpr int32_t kWait0 = 11768;
    static constexpr int32_t kCount0 = 11868;
    static constexpr int32_t kExpired = 11878;
    static constexpr int32_t kVocab = 11879;
    static constexpr int32_t kGrid = 3;
    static constexpr int32_t kWinLo = 6;
    static constexpr int32_t kEps = 6;
    static constexpr int32_t kMaxWait = 300;
    static constexpr int32_t kMaxCount = 10;
    static constexpr int32_t kMergeCs = 5;
    static constexpr int kBranches = 5;

    static constexpr auto waitToken(int32_t d) -> int32_t { return kWait0 + quantizeWait(d) / kGrid - 1; }
    static constexpr auto waitValue(int32_t token) -> int32_t { return (token - kWait0 + 1) * kGrid; }
    static constexpr auto countToken(int32_t k) -> int32_t {
        return kCount0 + (k < 1 ? 1 : (k > kMaxCount ? kMaxCount : k)) - 1;
    }
    static constexpr auto quantizeWait(int32_t d) -> int32_t {
        const int32_t q = ((d + kGrid / 2) / kGrid) * kGrid;
        return q < kGrid ? kGrid : (q > kMaxWait ? kMaxWait : q);
    }
    static auto q5(int32_t d) -> int32_t;
    static auto q8(int32_t v) -> int32_t;

    enum class Mode { Unconstrained, LineLegato, LineRests };

    struct Settings {
        Mode mode = Mode::Unconstrained;
        int32_t bandLow = 55;   // line modes: orchestra pitch band (inclusive)
        int32_t bandHigh = 84;
        float onsetTemperature = 1.0f; // onset / WAIT / COUNT
        float otherTemperature = 0.5f; // duration / pitch / velocity
        float minP = 0.25f;            // duration / pitch / velocity (0 = off)
        float topP = 0.98f;
        bool precompute = true;
        /** A drawn orchestra onset is at most this far after now. */
        int32_t maxAheadCs = 300;
        /** Nothing new is drawn while the last written event is this far ahead of now. */
        int32_t drawAheadCs = 50;
        /** A further chord note may join an orchestra onset at most this far in the past. */
        int32_t chordLateCs = 0;
        /**
         * What a piano onset does to the generated notes not yet started (the legacy Hybrid's modes):
         *   Jam: those at or after (onset - 5) go (the piano goes first);
         *   Edit: those editKeepCs or more after the onset go, the rest stay;
         *   Accompaniment: all stay.
         * Kept notes after the piano's onset stay in the context with the piano notes written among
         * them in onset order.
         */
        enum class InputCut { Jam, Edit, Accompaniment };
        InputCut inputCut = InputCut::Jam;
        int32_t editKeepCs = 200;
        /** Earliest onset of a new orchestra note (the polyphony cap's lower bound, absolute cs). */
        int32_t minOnsetCs = INT32_MIN;
        /** Orchestra durations drawn (cs, on the 5 cs grid). */
        int32_t minDurationCs = 10;
        int32_t maxDurationCs = 500;
        /** A chord's next note: at least the previous pitch (true), or any (false). At most maxChordNotes. */
        bool ascendingChords = true;
        int32_t maxChordNotes = 10;
        /** Line A: a legato fill whose start is more than this in the past is skipped (the engine is behind). */
        int32_t lagGuardCs = 10;
        /** Context window: past `maxTokens` the oldest events go (re-based to onset 0), down to `keepTokens`. */
        size_t maxTokens = 500;
        size_t keepTokens = 300;
        /**
         * Held keys: false (default) = written with an estimated duration (the median of the pianist's last 16
         * released notes, at least the time it has been down) and never rewritten, so the KV cache never re-feeds;
         * true = W3's way: the time it has been down, rewritten at its note-off (scratch tail), committed after
         * holdCommitCs.
         */
        bool rewriteHeldDurations = false;
        int32_t holdCommitCs = 100;
        uint64_t seed = 0;
        /**
         * Overnight model 1 (continuation, vocab 11768): no WAIT / COUNT / EXPIRED; the piano goes first when a
         * drawn orchestra onset is AT or after a piano onset (gap 0, not 5); an orchestra note may share the piano's
         * onset (the T3'' / v1 rules of sample_overnight.generate for m1). Unconstrained mode only.
         */
        bool continuation = false;
        /**
         * Dynamic LogitTD (TdController, set by the transformer): the Launchpad TD level (0 = AUTO) and the
         * strength gamma. At an onset draw the onsets lo..hi (c = t - (lo - 1) cs, c >= 1) get
         * gamma x TdOnsetBias::kernelFor(gamma, tdLevel, c), and then a constant so their total probability
         * at the onset temperature is unchanged: gamma never changes P(chord note) or P(WAIT). Empty or AUTO:
         * the model's own onsets.
         */
        int32_t tdLevel = 0;
        std::optional<float> tdGamma;
        /**
         * The legacy Hybrid's logit adjustments (AnticipationSampler / TdOnsetBias), on the same dense
         * token layout. History: every orchestra note in the context (own and mirrored).
         *   pitch / duration / onsetGap: the dense biases with their tau (the scheduler ramps them);
         *     the transformer folds in the BiasSettings switch and silence gate;
         *   shortGap: p(c) = -ln(1 + (6 / c)^4) on new onsets, c = t - (lo - 1);
         *   logitTd: the chord shift h(k, n) on the chord slot (TD > 1) and at most 10 notes a chord.
         * WAIT / COUNT logits are never touched.
         */
        bool pitchBias = false;
        bool durationBias = false;
        bool onsetGapBias = false;
        bool shortGap = false;
        bool logitTd = false;
        float pitchTau = DensePitchBias::TauBase;
        float pitchTauPrime = DensePitchBias::TauBase;
        float durationTau = DenseDurationBias::TauBase;
        float durationTauPrime = DenseDurationBias::TauPrimeBase;
        float onsetGapTau = DenseOnsetGapBias::TauBase;
        /** The Launchpad register range: orchestra pitches drawn in [pitchLow, pitchHigh]. */
        int32_t pitchLow = 21;
        int32_t pitchHigh = 108;
    };

    struct OutNote {
        int32_t onsetCs = 0;
        int32_t durationCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
        bool precomputed = false; // a fire served from the anticipation branches
    };

    struct Stats {
        int orchestraNotes = 0;
        int pianoGroups = 0;
        int waitsDrawn = 0;       // model-drawn WAITs
        int waitsProvisional = 0; // line modes
        int waitsGe10 = 0;        // model-drawn WAITs with d >= 10 cs
        int fires = 0;
        int expired = 0;          // EXPIRED written (model-drawn WAITs)
        int removed = 0;          // provisional WAITs removed (line modes)
        int earlyPiano = 0;
        int cutNotes = 0;         // generated notes cut by an earlier piano onset
        int anticipated = 0;      // fires scored against an anticipation
        int antTop1 = 0;
        int antTop5 = 0;
        int precomputeStarted = 0;
        int precomputeCompleted = 0;
        int precomputeHits = 0;   // fires served from a complete branch
        int precomputePartial = 0; // fires whose branch had some of the k notes (the rest decoded)
        int fireDecoded = 0;      // fires decoded on the spot
        double precomputeMsTotal = 0;
        double fireDecodeMsTotal = 0;
        int contextRebuilds = 0;
        int lagSkips = 0;         // line A fills skipped because the engine was behind
    };

    /**
     * One record per decision (the transformer logs them; tests read them). "placed" (at the fire's
     * onset): a = the fire's orchestra notes placed, b = microseconds from the firing key landing
     * (noteOn's landedUs; the fire itself when unknown) to the last of them placed; text
     * "precomputed" / "decoded", then "|key" / "|fire" for where the time is counted from.
     */
    struct Event {
        std::string what; // piano, orch, wait, count, fire, expired, removed, provisional, early, cut, anticipate, hit, miss
        int32_t atCs = 0;
        int32_t a = 0;
        int32_t b = 0;
        std::string text;
    };
    std::function<void(const Event &)> onEvent;

    InformativeWaitEngine();
    ~InformativeWaitEngine();

    /** The engine borrows the backend (owned by the caller) and builds its own cache on it. */
    auto setBackend(StreamingKv::Backend *backend) -> void;
    [[nodiscard]] auto isLoaded() const -> bool { return backend != nullptr; }

    auto setSettings(const Settings &s) -> void;
    [[nodiscard]] auto settings() const -> const Settings & { return cfg; }

    /** Start again at `nowCs` (nothing written; the first piano group starts the context). */
    auto reset(int32_t startCs) -> void;

    /**
     * A key went down (velocity 1..127). `landedUs`: when the key landed (KeyLanding, steady-clock
     * microseconds; -1 = unknown): a WAIT it fires is timed from it to the last orchestra note.
     */
    auto noteOn(int32_t onsetCs, int32_t pitch, int32_t velocity, int64_t landedUs = -1) -> void;
    /** That key went up: its final duration. */
    auto noteOff(int32_t onsetCs, int32_t pitch, int32_t durationCs) -> void;

    /**
     * Apply the input, then do at most one model action (an orchestra event, a WAIT, an expiry, or
     * one anticipation call). Returns the orchestra notes to play (several at a fire).
     */
    auto step(int32_t stepNowCs) -> std::vector<OutNote>;
    /** Hybrid, the model that is not drawing: apply input (piano groups, note-offs, mirrored notes) and nothing else. */
    auto absorb(int32_t stepNowCs) -> void;
    /** Hybrid: a note the OTHER model generated joins this context (applied in onset order at the next step). */
    auto mirrorNote(const OutNote &note) -> void;
    /** Hybrid switch: forget a pending WAIT, its anticipation and any stack still being decoded. */
    auto interrupt() -> void;
    /** Hybrid: generated notes from cutCs on leave the context (they were cut from playback). */
    auto dropGeneratedFrom(int32_t cutCs) -> void;

    /** Generated notes from this onset on were cut (the piano came first): INT32_MAX = none. */
    auto takeCancelFrom() -> int32_t;

    [[nodiscard]] auto stats() const -> const Stats & { return st; }
    /**
     * Dynamic LogitTD: the mean sensitivity (TdOnsetBias::sensitivity, d log2 density / d gamma) of the onset
     * draws made with a gamma since the last call, and how many there were; resets.
     */
    auto takeSensitivity() -> std::pair<std::optional<double>, int>;
    [[nodiscard]] auto tokens() const -> std::vector<int32_t>;
    [[nodiscard]] auto base() const -> int32_t { return baseCs; }
    [[nodiscard]] auto hasPendingWait() const -> bool { return pending.has_value(); }
    /** The anticipated top-5 piano pitches of the pending WAIT (empty when none yet). */
    [[nodiscard]] auto anticipatedPitches() const -> std::vector<int32_t>;
    [[nodiscard]] auto precomputeComplete() const -> bool;
    std::string lastError;

private:
    enum class Kind { Piano, Orch, Wait, Count, Expired };
    struct Item {
        Kind kind = Kind::Piano;
        int32_t onsetCs = 0; // absolute (WAIT / COUNT / EXPIRED: the time they refer to)
        int32_t durationCs = 0;
        int32_t pitch = 0;
        int32_t velocity = 0;
        int32_t token = 0;   // WAIT / COUNT id
        bool held = false;   // piano: key still down (duration provisional)
        bool provisional = false; // line-mode WAIT / COUNT that is removed at expiry
    };
    /** The sampler's scalar state; one snapshot per item so a cut can restore it. */
    struct Cursor {
        int32_t clockCs = INT32_MIN; // onset of the last written event (piano, orchestra, expiry)
        int32_t floorCs = INT32_MIN; // orchestra onsets >= this ...
        int32_t exclHiCs = INT32_MIN; // ... and > this, unless == chordOnCs
        int32_t chordOnCs = INT32_MIN;
        int32_t chordCount = 0;
        int32_t lastOrchPitch = -1;
        int32_t lastPianoCs = INT32_MIN;
        int32_t freeFloorCs = INT32_MIN; // line B: next free onset >= this
    };
    struct Pending {
        int32_t refCs = 0;
        int32_t d = 0;
        int32_t k = 1;
        bool provisional = false; // line mode: removed (not EXPIRED) at expiry
        size_t waitItem = 0;      // index of the WAIT item
        [[nodiscard]] auto lo() const -> int32_t { return refCs + d - kWinLo; }
        [[nodiscard]] auto hi(bool line) const -> int32_t { return refCs + d + (line ? 0 : kEps); }
    };
    struct PianoNote {
        int32_t onsetCs = 0; // arrival onset
        int32_t pitch = 0;
        int32_t velocity = 0;
        int32_t durationCs = -1; // -1 = held
        int64_t landedUs = -1;   // when the key landed (steady-clock us), -1 = unknown
    };
    struct OpenGroup {
        int32_t headCs = 0;
        std::vector<PianoNote> notes; // arrival order
        bool fired = false;           // the head fired a WAIT (rest of the chord goes after the stack)
    };
    struct Branch {
        int32_t pitch = 0;
        int32_t velocity = 0;
        std::vector<std::array<int32_t, 3>> notes; // (duration, pitch, velocity) of the orchestra notes
    };
    struct Precompute {
        int32_t targetCs = 0;
        int32_t k = 1;
        int phase = 0; // 0 top-5 (batch 1), 1 tile + [pitch, velocity, T], then 3 per orchestra note
        int32_t durationCs = 0;
        std::vector<float> pianoProbs; // 88 keys, p(A0..C8)
        std::vector<Branch> branches;
        StreamingKv::State state; // batch 1: context + T + D (phase 0)
        StreamingKv::BatchState kv;
        std::vector<float> logits; // batch x vocab of the last batched call
        bool complete = false;
        bool failed = false;
        double ms = 0;
    };

    auto rel(int32_t absCs) const -> int32_t;
    auto isLine() const -> bool { return cfg.mode != Mode::Unconstrained; }
    auto event(const char *what, int32_t at, int32_t a = 0, int32_t b = 0, std::string text = {}) -> void;

    auto append(const Item &item) -> void;
    auto committedLen() const -> size_t;
    auto logitsFor(const std::vector<int32_t> &desired) -> const std::vector<float> *;
    auto pick(const std::vector<float> &logits, const std::vector<std::pair<int32_t, int32_t>> &ranges,
              float temperature, float minP, int32_t step = 1) -> int32_t;
    auto pickFrom(std::vector<float> scores, float temperature) -> int32_t;
    /** Dynamic LogitTD on the onsets loRel..hiRel of `logits` (cfg.tdGamma), and their sensitivity. */
    auto applyTdGamma(std::vector<float> &logits, int32_t loRel, int32_t hiRel) -> void;
    /** The orchestra notes in the context as dense quads (absolute onsets), for the biases. */
    [[nodiscard]] auto orchestraHistory() const -> std::vector<int32_t>;
    /** The register range applied to [lo, hi]; false if no pitch is left (a chord stacked past its top). */
    auto clampToRegister(int32_t &lo, int32_t &hi) const -> bool;
    DensePitchBias::PitchTauScheduler pitchScheduler;
    DenseDurationBias::DurationTauScheduler durationScheduler;
    double sensitivitySum = 0.0;
    int sensitivityDraws = 0;

    auto applyInput() -> std::vector<OutNote>;
    auto onGroupHead(const PianoNote &head) -> std::vector<OutNote>;
    auto closeGroup() -> std::vector<OutNote>;
    auto writePianoNotes(std::vector<PianoNote> notes) -> void;
    auto cutFuture(int32_t pianoOnsetCs) -> void;
    auto truncateTo(size_t index) -> void;
    /**
     * Write piano / orchestra items back after a truncation, in onset order (piano first at a shared onset),
     * moving the cursor as when they were first written. WAIT / COUNT / EXPIRED items are not rewritten.
     */
    auto reappend(std::vector<Item> tail) -> void;
    /** Truncate at the first piano / orchestra item after `onsetCs`; returns what was removed. */
    auto takeAfter(int32_t onsetCs) -> std::vector<Item>;
    auto drawEvent() -> std::vector<OutNote>;
    auto drawBody(int32_t onsetCs, int32_t minPitch) -> std::optional<OutNote>;
    auto fire(const PianoNote &head) -> std::vector<OutNote>;
    auto expire() -> std::vector<OutNote>;
    auto noteAt(int32_t onsetCs) -> std::vector<OutNote>;
    auto addProvisionalWait(const OutNote &note) -> void;
    auto removeProvisionalWait(size_t index) -> void;
    auto startPrecompute() -> void;
    auto precomputeStep() -> void;
    auto scoreAnticipation(int32_t pitch) -> void;
    auto maybeShrinkContext() -> void;
    auto commitLongHolds() -> void;

    Settings cfg;
    StreamingKv::Backend *backend = nullptr;
    std::unique_ptr<StreamingKv::CommittedScratchCache> cache;
    std::mt19937_64 rng;

    std::vector<Item> items;
    std::vector<Cursor> before; // cursor before items[i]
    Cursor cur;
    int32_t baseCs = 0;
    int32_t nowCs = 0;
    bool started = false;
    std::optional<Pending> pending;
    std::optional<OpenGroup> group;
    std::vector<PianoNote> inbox;                       // arrived, not yet applied
    std::vector<std::array<int32_t, 3>> offs;           // (onset, pitch, duration)
    std::vector<int32_t> releasedDurations;             // the pianist's last released notes (cs)
    int32_t lastPianoVelocity = 64;
    auto estimatedDuration(int32_t heldSoFar) const -> int32_t;
    std::optional<Precompute> pre;
    std::vector<OutNote> mirrored;
    auto gap() const -> int32_t { return cfg.continuation ? 0 : kMergeCs; }
    auto vocab() const -> int32_t { return cfg.continuation ? kWait0 : kVocab; }
    auto applyMirrored() -> void;
    struct FireLeft {
        int32_t onsetCs = 0;
        int32_t left = 0;
        int32_t minPitch = 21;
    };
    std::optional<FireLeft> fireLeft; // stacked notes of a fire still to decode (one per step)
    /** When the firing key landed (else when the fire began), and how many notes are placed ("placed"). */
    int64_t fireFromUs = 0;
    bool fireFromKey = false;
    int32_t firePlaced = 0;
    auto firePlacedEvent(int32_t onsetCs, bool precomputed) -> void;
    int32_t cancelFrom = INT32_MAX;
    std::vector<int32_t> emittedFuture;                 // onsets of generated notes not yet started (for cuts)
    Stats st;
};

#pragma once

#include <JuceHeader.h>

#include <functional>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/EventEngine.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"

/**
 * Hybrid v1.3 (todo item 6; orchestration/docs/model_specs/event_pair_proposal.txt section 6): the base of the
 * transformers whose models read and write NOTE ON / NOTE OFF events instead of (onset, duration, pitch, velocity).
 * It works like the other RTs, except that durations no longer need updating in the models' contexts:
 *   - a piano key-down is an ON event at once; its key-up (the filter's TokenUpdate) an OFF event at its time;
 *   - an orchestra ON drawn by a model goes out as a note with the provisional length (kMaxNoteDurationCs, the
 *     RT's cap); the model's OFF for it sends the real length downstream as a TokenUpdate (as a key-up does).
 * The input / output plumbing lives here; the subclass owns the engines and decides who draws.
 */
class EventReductionTransformer : public ReductionTransformer {
public:
    EventReductionTransformer(const juce::String &threadName, ModelConfig &modelConfig);

    [[nodiscard]] auto inputEventWidth() const -> size_t override { return 4; }

    /** Clock (tests replace it). */
    std::function<int32_t()> nowCsProvider;

    struct PianoEvent {
        bool on = true;
        int32_t timeCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
    };

protected:
    [[nodiscard]] auto nowCs() const -> int32_t;
    /**
     * The pianist's events since the last call, in time order: key-downs at once; key-ups when the clock reaches
     * them (a complete note from the queue schedules its OFF at onset + length). Accompaniment ignores key-downs.
     */
    auto drainPianoEvents(int32_t now) -> std::vector<PianoEvent>;
    /** Send a model's orchestra ON (provisional length) or OFF (the real length, as a TokenUpdate). */
    auto emitOrchestra(const EventEngine::Out &o, int32_t velocityBoost, int32_t noteId) -> void;
    /** Forget the notes still open (a cut dropped them downstream). */
    auto forgetOpenFrom(int32_t cutCs) -> void;
    auto clearIo() -> void;
    /** Notes sounding now that a model sent (ON without OFF). */
    [[nodiscard]] auto openOrchestraNotes() const -> size_t { return open.size(); }

private:
    std::vector<PianoEvent> scheduledOffs;
    struct Open {
        Token token; // as sent (provisional length)
    };
    std::vector<Open> open;
};

#include "VirtualOrch/reduction/MelodyFirstReductionTransformer.h"

#include "VirtualOrch/PolyphonyCap.h"
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include <algorithm>

namespace {

constexpr int kEventWidth = 4;
constexpr size_t kMelodyEventsKept = 64;
/** Dense T8'' vocab (11776, or 11775 without level 64); AMT-native T7'' vocab. */
constexpr size_t kMinDenseVocab = static_cast<size_t>(TdTokens::kDenseFirst) + 7;
constexpr size_t kMinAmtVocab = 55293;

auto dense(int32_t time, int32_t durCs, int32_t instrument, int32_t pitch, int32_t velocity)
    -> std::array<int32_t, 4> {
    return {time,
            static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * instrument + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity};
}

} // namespace

MelodyFirstReductionTransformer::MelodyFirstReductionTransformer(ModelConfig &modelConfigIn)
    : ReductionTransformer("Reduction Transformer Melody First", modelConfigIn) {
    tdValueProvider = [] { return TemporalDensity::get(); };
    melodyStage.tdValue = [this] { return tdValueProvider(); };
    reductionStage.tdValue = [this] { return tdValueProvider(); };
}

auto MelodyFirstReductionTransformer::init(const char *reductionModelPath,
                                           const char *melodyModelPath) -> bool {
    auto t7 = std::make_unique<StreamingKv::OnnxBackend>();
    auto t8 = std::make_unique<StreamingKv::OnnxBackend>();
    std::string error;
    if (! t7->init(reductionModelPath, error)) {
        reportSamplingError("Melody-first reduction (T7'') init failed: " + juce::String(error));
        return false;
    }
    if (! t8->init(melodyModelPath, error)) {
        reportSamplingError("Melody-first reduction (T8'') init failed: " + juce::String(error));
        return false;
    }
    if (t8->usesPositionIds()) {
        reportSamplingError("Melody-first reduction: this mb export takes position_ids (the old "
                            "learned-position T8); T8'' is a dense RoPE model");
        return false;
    }
    if (t7->vocabSize() < kMinAmtVocab || t8->vocabSize() < kMinDenseVocab) {
        reportSamplingError("Melody-first reduction: unexpected vocab sizes (T7'' "
                            + juce::String(static_cast<int>(t7->vocabSize())) + ", T8'' "
                            + juce::String(static_cast<int>(t8->vocabSize())) + ")");
        return false;
    }
    const size_t t7Vocab = t7->vocabSize();
    const size_t t8Vocab = t8->vocabSize();
    initWithBackends(std::move(t8), t8Vocab, std::move(t7), t7Vocab);
    return true;
}

auto MelodyFirstReductionTransformer::initWithBackends(
    std::unique_ptr<StreamingKv::Backend> melodyBackendIn, size_t melodyVocab,
    std::unique_ptr<StreamingKv::Backend> reductionBackendIn, size_t reductionVocab) -> void {
    melodyStage.setBackend(std::move(melodyBackendIn), melodyVocab);
    reductionStage.setBackend(std::move(reductionBackendIn), reductionVocab);
}

auto MelodyFirstReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto MelodyFirstReductionTransformer::samplingForStep(int stepIdx) const -> FieldSampling {
    switch (stepIdx % kEventWidth) {
        case 1: return {AnticipationSampler::kTopP, getDurationTemperature()};
        case 2: return {AnticipationSampler::kTopP, getNoteTemperature()};
        case 3: return {AnticipationSampler::kTopP, getVelocityTemperature()};
        default: return {AnticipationSampler::kTopP, getOnsetTemperature()};
    }
}

auto MelodyFirstReductionTransformer::temperatures() const -> std::array<float, 4> {
    return {getOnsetTemperature(), getDurationTemperature(), getNoteTemperature(),
            getVelocityTemperature()};
}

auto MelodyFirstReductionTransformer::resetAll(int32_t now) -> void {
    state.reset();
    melodyStage.reset(now);
    reductionStage.reset(now);
    fedPieces.clear();
    heldPieces.clear();
    aDirty = false;
    userHasPlayed = false;
    melodyEvents.clear();
    latestMelodyOnset = melodyStage.stream.base();
    latestReductionOnset = reductionStage.stream.base();
    currentTime = reductionStage.stream.clock();
}

auto MelodyFirstReductionTransformer::melodyIsTooFarAhead() const -> bool {
    return latestMelodyOnset - latestReductionOnset > kMaxMelodyAheadCs;
}

auto MelodyFirstReductionTransformer::rewindGeneratedFrom(int32_t cutCs) -> void {
    // T8'': the planned melody + bass from the cut on; your melody / bass pieces go back to pending.
    melodyStage.rewindTo(cutCs);
    // T7'': its notes from the cut on, and the melody controls of the rewound melody notes.
    reductionStage.rewindTo(cutCs, [cutCs](const AnticipatedStream::ControlSpec &spec,
                                           const AnticipatedStream::Note &note) {
        return spec.controlToken && note.onsetCs >= cutCs;
    });
    std::erase_if(melodyEvents, [cutCs](const Token &t) { return t.time >= cutCs; });
    rewindInputDataFrom(cutCs);

    latestMelodyOnset = melodyStage.stream.base();
    for (const auto &t: melodyEvents)
        latestMelodyOnset = std::max(latestMelodyOnset, t.time);
    latestReductionOnset = reductionStage.stream.clock();
    currentTime = reductionStage.stream.clock();
}

// ---- input ----

auto MelodyFirstReductionTransformer::drainInput(int32_t now) -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
    }

    bool applied = false;
    Token in{-1, -1, -1};
    while (inputTokenQueue.pull(in)) {
        const int pitch = DenseQuantize::snapPianoPitch(in.getPitch());
        const int velocity = std::max(
            DenseQuantize::VelocityStep,
            DenseQuantize::snapVelocity(juce::jlimit(
                0, DenseConfig::MaxVelocity - 1,
                in.velocity > 0 ? in.velocity : DenseConfig::DefaultVelocity)));
        const bool held = DenseSampling::isProvisionalDurationToken(in.duration,
                                                                    modelConfig.inputDuration);
        const int32_t durCs = DenseQuantize::snapDurationCs(in.getRealDuration());
        state.addNote(in.time, pitch, velocity, durCs, held);

        // T7'': the piano is given, as instrument-0 events (not controls)
        AnticipatedStream::ControlSpec piano;
        piano.instrument = 0;
        piano.priority = 1; // after a melody control at the same actual time
        reductionStage.stream.addControl({in.time, held ? 0 : durCs, pitch, velocity}, held, piano);

        const auto quad = dense(in.time, durCs, DenseConfig::PianoReductionInputInstrument, pitch,
                                velocity);
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        applied = true;
        userHasPlayed = true;
        aDirty = true;
    }

    TokenUpdate update{};
    bool updated = false;
    while (updatesFromFilter.pull(update)) {
        if (orchestrationUpdatesIncoming != nullptr)
            orchestrationUpdatesIncoming->push(update);
        const int pitch = DenseQuantize::snapPianoPitch(update.oldNote.getPitch());
        const int32_t finalDur = DenseQuantize::snapDurationCs(update.newNote.getRealDuration());
        if (state.noteOff(update.oldNote.time, pitch, finalDur))
            aDirty = true;
        reductionStage.stream.confirmControl(update.oldNote.time, pitch, finalDur, 0);
        for (size_t i = 0; i + 3 < inputData.size(); i += kEventWidth) {
            if (inputData[i] == update.oldNote.time
                && DenseQuantize::instrumentOfNoteToken(inputData[i + 2])
                       == DenseConfig::PianoReductionInputInstrument
                && inputData[i + 2] - static_cast<int32_t>(DenseVocab::NoteOffset) == pitch) {
                inputData[i + 1] = static_cast<int32_t>(DenseVocab::DurOffset) + finalDur;
                updated = true;
            }
        }
    }

    if (applied || updated)
        notifyInputDataChanged();
    reductionStage.stream.commitLongHolds(now);
    melodyStage.stream.commitLongHolds(now);
    return applied;
}

auto MelodyFirstReductionTransformer::syncMelodyGiven() -> void {
    const int32_t base = melodyStage.stream.base();
    for (const auto &p: state.getPieces()) {
        const PieceId id{p.key, p.onset, p.pitch, p.line};
        if (fedPieces.count(id) == 0) {
            if (p.onset < base)
                continue; // from before this stream began
            fedPieces.insert(id);
            if (p.provisional)
                heldPieces.insert(id);
            AnticipatedStream::ControlSpec given;
            given.instrument = 0;
            given.line = p.line; // melody and bass stay strictly monophonic
            melodyStage.stream.addControl(
                {p.onset, p.provisional ? 0 : p.durationCs, p.pitch, p.velocity}, p.provisional,
                given);
        } else if (! p.provisional && heldPieces.erase(id) > 0) {
            // the held note ended: its given event gets the final duration
            melodyStage.stream.confirmControl(p.onset, p.pitch, p.durationCs, 0);
        }
    }
    // forget old identities
    for (auto it = fedPieces.begin(); it != fedPieces.end();)
        it = std::get<1>(*it) < base - 1000 ? fedPieces.erase(it) : std::next(it);
}

// ---- melody (T8'') ----

auto MelodyFirstReductionTransformer::generateMelodyNote(int32_t now)
    -> std::optional<AnticipatedStream::Note> {
    AnticipationStage::Params params;
    params.now = now;
    params.temperatures = temperatures();
    params.pitchBias = isPitchBiasEnabled();
    params.biases = biasesActiveNow();
    params.pitchLow = RegisterRange::lowPitch();
    params.pitchHigh = RegisterRange::highPitch();
    // Two voices: at most two notes sounding (fewer with the accompaniment cap); the next onset
    // waits for the earliest end when both are busy.
    const int userCap = getMaxPolyphony();
    const int cap = userCap > 0 ? std::min(userCap, kMelodyVoices) : kMelodyVoices;
    params.lowerAbs = std::max(
        now, PolyphonyCap::earliestOnsetCs(melodyEvents, std::max(melodyStage.stream.clock(), now), cap));
    const auto note = melodyStage.generate(params);
    if (! note.has_value()) {
        if (! melodyStage.lastError.empty())
            reportSamplingError("Melody-first (T8''): " + juce::String(melodyStage.lastError));
        return std::nullopt;
    }

    Token event{note->onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + note->durationCs,
                static_cast<int32_t>(Vocab::NoteOffset) + note->pitch, note->velocity};
    melodyEvents.push_back(event);
    if (melodyEvents.size() > kMelodyEventsKept)
        melodyEvents.erase(melodyEvents.begin());
    latestMelodyOnset = std::max(latestMelodyOnset, note->onsetCs);

    // T8'''s notes are the melody + bass CONTROLS of T7'': anticipated by 3 s, AMT control ids.
    AnticipatedStream::ControlSpec control;
    control.instrument = 0;
    control.deltaCs = kMelodyDeltaCs;
    control.priority = 0;
    control.controlToken = true;
    reductionStage.stream.addControl(*note, false, control);
    return note;
}

// ---- reduction (T7'') ----

auto MelodyFirstReductionTransformer::generateReductionNote(int32_t now) -> std::optional<Token> {
    AnticipationStage::Params params;
    params.now = now;
    params.lowerAbs = std::max(onsetLowerBoundCs(), now);
    params.temperatures = temperatures();
    params.pitchBias = isPitchBiasEnabled();
    params.biases = biasesActiveNow();
    params.pitchLow = RegisterRange::lowPitch();
    params.pitchHigh = RegisterRange::highPitch();
    const auto note = reductionStage.generate(params);
    if (! note.has_value()) {
        if (! reductionStage.lastError.empty())
            reportSamplingError("Melody-first (T7''): " + juce::String(reductionStage.lastError));
        return std::nullopt;
    }

    const auto quad = dense(note->onsetCs, note->durationCs, AnticipatedStream::kEventInstrument,
                            note->pitch, note->velocity);
    inputData.insert(inputData.end(), quad.begin(), quad.end());
    notifyInputDataChanged();

    currentTime = note->onsetCs;
    latestReductionOnset = std::max(latestReductionOnset, note->onsetCs);
    logGeneratedTokenIfResumed(note->onsetCs);

    Token outbound{note->onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + note->durationCs,
                   static_cast<int32_t>(DenseVocab::NoteOffset) + note->pitch, note->velocity};
    pushOutputToken(outbound);
    if (isGeneratedTooFarAhead(note->onsetCs))
        startAheadThrottle();
    return outbound;
}

// ---- loop ----

auto MelodyFirstReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    if (! melodyStage.stream.isStarted())
        resetAll(now);
    if (const auto cut = consumeModeCut())
        rewindGeneratedFrom(*cut);
    drainInput(now);
    if (aDirty) {
        state.runWindow(now);
        syncMelodyGiven();
        aDirty = false;
    }
    if (isGenerationStopped() || ! userHasPlayed || ! isModelLoaded())
        return false;

    bool any = false;
    for (int i = 0; i < kMelodyNotesPerLoop; ++i) {
        if (melodyIsTooFarAhead())
            break;
        if (! generateMelodyNote(now).has_value())
            break;
        any = true;
    }
    if (generateReductionNote(now).has_value())
        any = true;
    return any;
}

void MelodyFirstReductionTransformer::threadRun() {
    if (! isModelLoaded())
        return;
    if (voiceSeparation != nullptr)
        voiceSeparation->reset();

    inputData.clear();
    clearInputTokenQueue();
    clearInputConditioningQueue();
    clearOutputTokenQueue();
    clearUpdatesFromFilter();
    clearOutputHistory();
    resetAll(nowCs());
    notifyInputDataChanged();
    resetAheadThrottle();
    resetLoopTiming();

    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        finishAheadThrottleIfDue();

        // Leaving Accompaniment: drop the generated reduction and melody from the cut on.
        if (const auto cut = consumeModeCut())
            rewindGeneratedFrom(*cut);

        const int32_t now = nowCs();
        drainInput(now);
        const bool inputApplied = aDirty;
        if (aDirty) {
            state.runWindow(now);
            syncMelodyGiven();
            aDirty = false;
        }

        if (isGenerationStopped()) {
            maybeEmitGenerationPauseSoftStopClear();
            if (inputApplied && ! generationPause.get() && clock != nullptr
                && ! keepsQueueOnInput()) {
                const Token clearToken{now + Config::TimeResolution,
                                       static_cast<int32_t>(DenseVocab::DurOffset),
                                       static_cast<int32_t>(Vocab::ClearQueue)};
                pushOutputToken(clearToken);
            } else if (! inputApplied) {
                wait(5);
            }
            recordThreadLoopMs(loopStartMs);
            continue;
        }

        wasGenerationPaused = false;
        bool any = false;
        if (userHasPlayed) {
            for (int i = 0; i < kMelodyNotesPerLoop; ++i) {
                if (melodyIsTooFarAhead())
                    break;
                if (! generateMelodyNote(now).has_value())
                    break;
                any = true;
            }
            if (generateReductionNote(now).has_value())
                any = true;
        }
        if (! any)
            wait(5);
        recordThreadLoopMs(loopStartMs);
    }
}

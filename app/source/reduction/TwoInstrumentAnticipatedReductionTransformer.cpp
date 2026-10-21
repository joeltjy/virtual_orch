#include "VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h"

#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"
#include "VirtualOrch/TemporalDensity.h"

#include <algorithm>
#include <limits>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int kDenseEventWidth = 4;
constexpr int kMaxRedraws = 256;

auto dense(int32_t time, int32_t durCs, int32_t instrument, int32_t pitch, int32_t velocity)
    -> std::array<int32_t, 4> {
    return {time,
            static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * instrument + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity};
}

} // namespace

TwoInstrumentAnticipatedReductionTransformer::TwoInstrumentAnticipatedReductionTransformer(
    ModelConfig &modelConfigIn)
    : ReductionTransformer("Reduction Transformer Anticipated", modelConfigIn) {
    tdValueProvider = [] { return TemporalDensity::get(); };
    stage.tdValue = [this] { return tdValueProvider(); };
}

auto TwoInstrumentAnticipatedReductionTransformer::init(const char *modelPath) -> bool {
    auto onnx = std::make_unique<StreamingKv::OnnxBackend>();
    std::string error;
    if (! onnx->init(modelPath, error)) {
        reportSamplingError("Anticipated reduction init failed: " + juce::String(error));
        return false;
    }
    const size_t vocab = onnx->vocabSize();
    if (vocab < static_cast<size_t>(TdTokens::kDenseFirst + 7)) {
        reportSamplingError("Anticipated reduction: unexpected vocab size "
                            + juce::String(static_cast<int>(vocab)));
        return false;
    }
    initWithBackend(std::move(onnx), vocab);
    return true;
}

auto TwoInstrumentAnticipatedReductionTransformer::initWithBackend(
    std::unique_ptr<StreamingKv::Backend> newBackend, size_t vocab) -> void {
    stage.setBackend(std::move(newBackend), vocab);
}

auto TwoInstrumentAnticipatedReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto TwoInstrumentAnticipatedReductionTransformer::samplingForStep(int stepIdx) const
    -> FieldSampling {
    switch (stepIdx % kDenseEventWidth) {
        case 1: return {AnticipationSampler::kTopP, getDurationTemperature()};
        case 2: return {AnticipationSampler::kTopP, getNoteTemperature()};
        case 3: return {AnticipationSampler::kTopP, getVelocityTemperature()};
        default: return {AnticipationSampler::kTopP, getOnsetTemperature()};
    }
}

auto TwoInstrumentAnticipatedReductionTransformer::resetStream(int32_t now) -> void {
    stage.reset(now);
    currentTime = stage.stream.clock();
}

auto TwoInstrumentAnticipatedReductionTransformer::rewindGeneratedFrom(int32_t cutCs) -> void {
    stage.rewindTo(cutCs);
    rewindInputDataFrom(cutCs);
    currentTime = stage.stream.clock();
}

// ---- input ----

auto TwoInstrumentAnticipatedReductionTransformer::drainInput(int32_t now) -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
    }

    bool applied = false;
    Token in{-1, -1, -1};
    while (inputTokenQueue.pull(in)) {
        const int32_t pitch = DenseQuantize::snapPianoPitch(in.getPitch());
        const int32_t velocity = DenseQuantize::snapVelocity(
            juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                         in.velocity > 0 ? in.velocity : DenseConfig::DefaultVelocity));
        const bool held = DenseSampling::isProvisionalDurationToken(in.duration,
                                                                    modelConfig.inputDuration);
        const int32_t durCs = DenseQuantize::snapDurationCs(in.getRealDuration());
        stage.stream.addControl({in.time, held ? 0 : durCs, pitch, velocity}, held);

        // The UI / filters keep reading the user's notes as instrument-0 dense quads.
        const auto quad = dense(in.time, DenseQuantize::snapDurationCs(in.getRealDuration()),
                                DenseConfig::PianoReductionInputInstrument, pitch, velocity);
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        applied = true;
    }

    TokenUpdate update{};
    bool updated = false;
    while (updatesFromFilter.pull(update)) {
        if (orchestrationUpdatesIncoming != nullptr)
            orchestrationUpdatesIncoming->push(update);
        const int32_t pitch = DenseQuantize::snapPianoPitch(update.oldNote.getPitch());
        const int32_t finalDur = DenseQuantize::snapDurationCs(update.newNote.getRealDuration());
        stage.stream.confirmControl(update.oldNote.time, pitch, finalDur);

        // Keep the UI copy in step: patch the matching instrument-0 quad's duration.
        for (size_t i = 0; i + 3 < inputData.size(); i += kDenseEventWidth) {
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
    stage.stream.commitLongHolds(now);
    return applied;
}

// ---- sampling ----

auto TwoInstrumentAnticipatedReductionTransformer::generateOne(int32_t now) -> std::optional<Token> {
    if (! stage.isLoaded())
        return std::nullopt;

    AnticipationStage::Params params;
    params.now = now;
    // Earliest onset: the event clock, never in the past, raised by the polyphony cap.
    params.lowerAbs = std::max(onsetLowerBoundCs(), now);
    params.temperatures = {getOnsetTemperature(), getDurationTemperature(), getNoteTemperature(),
                           getVelocityTemperature()};
    params.pitchBias = isPitchBiasEnabled();
    params.biases = biasesActiveNow();
    params.pitchLow = RegisterRange::lowPitch();
    params.pitchHigh = RegisterRange::highPitch();
    const auto note = stage.generate(params);
    if (! note.has_value()) {
        if (! stage.lastError.empty())
            reportSamplingError("Anticipated reduction: " + juce::String(stage.lastError));
        return std::nullopt;
    }

    const auto quad = dense(note->onsetCs, note->durationCs, AnticipatedStream::kEventInstrument,
                            note->pitch, note->velocity);
    inputData.insert(inputData.end(), quad.begin(), quad.end());
    notifyInputDataChanged();

    currentTime = note->onsetCs;
    logGeneratedTokenIfResumed(note->onsetCs);

    Token outbound{note->onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + note->durationCs,
                   static_cast<int32_t>(DenseVocab::NoteOffset) + note->pitch, note->velocity};
    pushOutputToken(outbound);

    if (isGeneratedTooFarAhead(note->onsetCs))
        startAheadThrottle();
    return outbound;
}

auto TwoInstrumentAnticipatedReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    if (! stage.stream.isStarted())
        resetStream(now);
    if (const auto cut = consumeModeCut())
        rewindGeneratedFrom(*cut);
    drainInput(now);
    if (isGenerationStopped())
        return false;
    return generateOne(now).has_value();
}

void TwoInstrumentAnticipatedReductionTransformer::threadRun() {
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
    resetStream(nowCs());
    notifyInputDataChanged();

    resetAheadThrottle();
    resetLoopTiming();

    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        finishAheadThrottleIfDue();

        // Leaving Accompaniment: drop the generated notes from the cut on.
        if (const auto cut = consumeModeCut())
            rewindGeneratedFrom(*cut);

        const int32_t now = nowCs();
        const bool inputApplied = drainInput(now);

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
        if (! generateOne(now).has_value())
            wait(5);
        recordThreadLoopMs(loopStartMs);
    }
}

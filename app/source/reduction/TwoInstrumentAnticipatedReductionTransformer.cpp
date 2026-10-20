#include "VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h"

#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"
#include "VirtualOrch/TemporalDensity.h"

#include <algorithm>
#include <limits>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int kDenseEventWidth = 4;
constexpr size_t kGeneratedWindowNotes = 64;

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
    stream.tdTokenProvider = [this] {
        return tdResolver.resolve(tdValueProvider(), static_cast<int32_t>(vocabSize));
    };
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
    backend = std::move(newBackend);
    vocabSize = vocab;
    cache = std::make_unique<StreamingKv::CommittedScratchCache>(*backend);
}

auto TwoInstrumentAnticipatedReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto TwoInstrumentAnticipatedReductionTransformer::samplingForStep(int stepIdx) const
    -> FieldSampling {
    switch (stepIdx % kDenseEventWidth) {
        case 1: return {DenseSampling::DurationTopP, getDurationTemperature()};
        case 2: return {DenseSampling::NoteTopP, getNoteTemperature()};
        case 3: return {DenseSampling::VelocityTopP, getVelocityTemperature()};
        default: return {DenseSampling::OnsetTopP, getOnsetTemperature()};
    }
}

auto TwoInstrumentAnticipatedReductionTransformer::resetStream(int32_t now) -> void {
    stream.reset(now);
    if (cache != nullptr)
        cache->reset();
    generatedDense.clear();
    pitchTauScheduler.reset();
    tdResolver = {};
    currentTime = stream.clock();
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
        stream.addControl({in.time, held ? 0 : durCs, pitch, velocity}, held);

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
        stream.confirmControl(update.oldNote.time, pitch, finalDur);

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
    stream.commitLongHolds(now);
    return applied;
}

// ---- sampling ----

auto TwoInstrumentAnticipatedReductionTransformer::sampleField(
    int field, const std::vector<float> &logits, int32_t lowerModelOnset,
    const DensePitchBias::PitchWindowStats *pitchWindow, float tau, float tauPrime) -> int32_t {
    std::vector<float> raw = logits;
    if (field == 2 && pitchWindow != nullptr)
        DensePitchBias::applyNoteLogitsBias(raw, *pitchWindow, tau, tauPrime);

    std::vector<float> scores(raw.size(), kNegInf);
    const auto allow = [&](int32_t id) {
        if (id >= 0 && static_cast<size_t>(id) < raw.size())
            scores[static_cast<size_t>(id)] = raw[static_cast<size_t>(id)];
    };
    switch (field) {
        case 0:
            for (int32_t t = std::max(lowerModelOnset, 0); t <= AnticipatedStream::kMaxModelTimeCs; ++t)
                allow(t);
            break;
        case 1:
            for (int32_t cs = DenseQuantize::DurationMinCs; cs <= DenseQuantize::DurationMaxCs;
                 cs += DenseQuantize::DurationStepCs)
                allow(static_cast<int32_t>(DenseVocab::DurOffset) + cs);
            break;
        case 2:
            for (int32_t p = DenseQuantize::PianoKeyLow; p <= DenseQuantize::PianoKeyHigh; ++p)
                allow(static_cast<int32_t>(DenseVocab::NoteOffset)
                      + DenseConfig::MaxPitch * AnticipatedStream::kEventInstrument + p);
            break;
        default:
            // v = 0 would be a note-off: only 8..120
            for (int32_t v = DenseQuantize::VelocityStep; v <= DenseQuantize::VelocityMax;
                 v += DenseQuantize::VelocityStep)
                allow(static_cast<int32_t>(DenseVocab::VelocityOffset) + v);
            break;
    }
    const auto sampling = samplingForStep(field);
    return sampleTopP(scores, sampling.topP, sampling.temperature);
}

auto TwoInstrumentAnticipatedReductionTransformer::generateOne(int32_t now) -> std::optional<Token> {
    if (backend == nullptr || cache == nullptr)
        return std::nullopt;

    // Earliest onset: the event clock, never in the past, raised by the polyphony cap.
    const int32_t lower = std::max(onsetLowerBoundCs(), now);
    while (stream.needsRebase(lower))
        stream.rebase();
    stream.insertDue(std::max(stream.clock(), now), now);
    // The cache evicted a note older than 30 s: move the origin, rebuild from the retained window.
    if (stream.rebaseForEviction(cache->evictedTokens(), cache->sinkRows()) > 0)
        cache->reset();

    const auto tokens = stream.tokens();
    if (tokens.empty())
        return std::nullopt; // nothing to continue from yet: wait for the user's first note
    const size_t committed = stream.committedLen();
    const int32_t base = stream.base();
    const int32_t lowerModel = std::max(lower, stream.clock()) - base;

    std::vector<int32_t> partial;
    int32_t onsetModel = 0;
    DensePitchBias::PitchWindowStats pitchWindow;
    const bool pitchBias = isPitchBiasEnabled();
    float tau = DensePitchBias::TauBase;
    float tauPrime = DensePitchBias::TauBase;
    std::array<int32_t, 4> chosen{};

    for (int field = 0; field < kDenseEventWidth; ++field) {
        std::vector<int32_t> desired = tokens;
        desired.insert(desired.end(), partial.begin(), partial.end());
        const auto &logits = cache->logitsAfter(desired, std::min(committed, desired.size()));
        if (logits.empty()) {
            reportSamplingError("Anticipated reduction: KV cache returned no logits");
            return std::nullopt;
        }
        if (logits.size() < static_cast<size_t>(DenseVocab::VocabSize)) {
            reportSamplingError("Anticipated reduction: logits too small ("
                                + juce::String(static_cast<int>(logits.size())) + ")");
            return std::nullopt;
        }

        if (pitchBias && field == 2) {
            pitchWindow = DensePitchBias::collectPitchWindowAt(generatedDense, onsetModel + base, -1);
            const auto scheduled = pitchTauScheduler.evaluate(
                onsetModel + base, pitchWindow.uniquePitches, pitchWindow.uniqueDeltas,
                DensePitchBias::TauBase, DensePitchBias::TauBase, true);
            tau = scheduled.tau;
            tauPrime = scheduled.tauPrime;
        }

        const int32_t token = sampleField(field, logits, lowerModel,
                                          pitchBias && field == 2 ? &pitchWindow : nullptr, tau,
                                          tauPrime);
        chosen[static_cast<size_t>(field)] = token;
        partial.push_back(token);
        if (field == 0)
            onsetModel = token;
    }

    const int32_t onsetAbs = chosen[0] + base;
    const int32_t durCs = chosen[1] - static_cast<int32_t>(DenseVocab::DurOffset);
    const int32_t pitch = chosen[2] - static_cast<int32_t>(DenseVocab::NoteOffset)
                          - DenseConfig::MaxPitch * AnticipatedStream::kEventInstrument;
    const int32_t velocity = chosen[3] - static_cast<int32_t>(DenseVocab::VelocityOffset);

    stream.appendEvent({onsetAbs, durCs, pitch, velocity});

    const auto quad =
        dense(onsetAbs, durCs, AnticipatedStream::kEventInstrument, pitch, velocity);
    generatedDense.insert(generatedDense.end(), quad.begin(), quad.end());
    if (generatedDense.size() > kGeneratedWindowNotes * kDenseEventWidth)
        generatedDense.erase(generatedDense.begin(),
                             generatedDense.end()
                                 - static_cast<std::ptrdiff_t>(kGeneratedWindowNotes * kDenseEventWidth));
    inputData.insert(inputData.end(), quad.begin(), quad.end());
    notifyInputDataChanged();

    currentTime = onsetAbs;
    logGeneratedTokenIfResumed(onsetAbs);

    Token outbound{onsetAbs, static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
                   static_cast<int32_t>(DenseVocab::NoteOffset) + pitch, velocity};
    pushOutputToken(outbound);

    if (isGeneratedTooFarAhead(onsetAbs))
        startAheadThrottle();
    return outbound;
}

auto TwoInstrumentAnticipatedReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    if (! stream.isStarted())
        resetStream(now);
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

        const int32_t now = nowCs();
        const bool inputApplied = drainInput(now);

        if (isGenerationStopped()) {
            maybeEmitGenerationPauseSoftStopClear();
            if (inputApplied && ! generationPause.get() && clock != nullptr) {
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

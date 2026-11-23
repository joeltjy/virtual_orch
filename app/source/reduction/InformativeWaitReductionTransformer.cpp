#include "VirtualOrch/reduction/InformativeWaitReductionTransformer.h"

#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/reduction/DenseTypes.h"

#include <algorithm>

namespace {

constexpr int kEventWidth = 4;

auto denseQuad(int32_t time, int32_t durCs, int32_t instrument, int32_t pitch, int32_t velocity)
    -> std::array<int32_t, 4> {
    return {time, static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * instrument + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity};
}

} // namespace

InformativeWaitReductionTransformer::InformativeWaitReductionTransformer(ModelConfig &modelConfigIn)
    : ReductionTransformer("Reduction Transformer Informative WAIT", modelConfigIn) {
    engine.onEvent = [this](const InformativeWaitEngine::Event &e) {
        if (! SessionLog::isRunning())
            return;
        SessionLog::add("IWAIT", getThreadName() + " | " + juce::String(e.what) + " @" + juce::String(e.atCs) + " "
                                     + juce::String(e.a) + " " + juce::String(e.b)
                                     + (e.text.empty() ? juce::String() : " (" + juce::String(e.text) + ")"));
    };
}

auto InformativeWaitReductionTransformer::modeName(InformativeWaitEngine::Mode mode) -> const char * {
    switch (mode) {
        case InformativeWaitEngine::Mode::LineLegato: return "monophonic_A";
        case InformativeWaitEngine::Mode::LineRests: return "monophonic_B";
        case InformativeWaitEngine::Mode::Unconstrained: return "unconstrained";
    }
    return "unconstrained";
}

auto InformativeWaitReductionTransformer::modeFromName(const juce::String &name) -> InformativeWaitEngine::Mode {
    if (name.equalsIgnoreCase("monophonic_A") || name.equalsIgnoreCase("A") || name.equalsIgnoreCase("legato"))
        return InformativeWaitEngine::Mode::LineLegato;
    if (name.equalsIgnoreCase("monophonic_B") || name.equalsIgnoreCase("B") || name.equalsIgnoreCase("rests"))
        return InformativeWaitEngine::Mode::LineRests;
    return InformativeWaitEngine::Mode::Unconstrained;
}

auto InformativeWaitReductionTransformer::init(const char *modelPath) -> bool {
    lastInitError.clear();
    auto onnx = std::make_unique<StreamingKv::OnnxBackend>();
    std::string error;
    if (! onnx->init(modelPath, error)) {
        lastInitError = error;
        reportSamplingError("Informative WAIT init failed: " + juce::String(error));
        return false;
    }
    if (onnx->vocabSize() != static_cast<size_t>(InformativeWaitEngine::kVocab)) {
        lastInitError = "the informative-WAIT model has vocab 11879 (WAIT 11768.., COUNT 11868.., EXPIRED 11878); got "
                        + std::to_string(onnx->vocabSize());
        reportSamplingError("Informative WAIT: " + juce::String(lastInitError));
        return false;
    }
    if (onnx->keepsKvOnDevice())
        SessionLog::add("IWAIT", "CUDA session: the batched precompute needs host KV, so fires are decoded on the spot");
    initWithBackend(std::move(onnx));
    return true;
}

auto InformativeWaitReductionTransformer::initWithBackend(std::unique_ptr<StreamingKv::Backend> newBackend) -> void {
    backend = std::move(newBackend);
    engine.setBackend(backend.get());
}

auto InformativeWaitReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto InformativeWaitReductionTransformer::applySettings() -> void {
    auto s = engine.settings();
    const auto mode = getMode();
    const bool modeChanged = s.mode != mode;
    s.mode = mode;
    s.bandLow = bandLow.load();
    s.bandHigh = bandHigh.load();
    s.precompute = precompute.load();
    s.onsetTemperature = onsetTemperature.load();
    s.otherTemperature = otherTemperature.load();
    s.minP = minP.load();
    engine.setSettings(s);
    if (modeChanged && streamStarted) {
        SessionLog::add("IWAIT", juce::String("mode ") + modeName(mode) + ": stream restarted");
        resetStream(nowCs());
    }
}

auto InformativeWaitReductionTransformer::resetStream(int32_t now) -> void {
    engine.reset(now);
    streamStarted = true;
    currentTime = now;
    SessionLog::add("IWAIT", getThreadName() + " | stream reset at " + juce::String(now) + ", mode "
                                 + modeName(getMode()));
}

auto InformativeWaitReductionTransformer::drainInput() -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
    }
    bool applied = false;
    Token in{-1, -1, -1};
    while (inputTokenQueue.pull(in)) {
        const int32_t pitch = DenseQuantize::snapPianoPitch(in.getPitch());
        const int32_t velocity = std::clamp(in.velocity > 0 ? in.velocity : DenseConfig::DefaultVelocity, 1, 127);
        const bool held = DenseSampling::isProvisionalDurationToken(in.duration, modelConfig.inputDuration);
        engine.noteOn(in.time, pitch, velocity);
        if (! held)
            engine.noteOff(in.time, pitch, std::max(1, in.getRealDuration()));
        const auto quad = denseQuad(in.time, DenseQuantize::snapDurationCs(in.getRealDuration()),
                                    DenseConfig::PianoReductionInputInstrument, pitch, DenseQuantize::snapVelocity(velocity));
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        applied = true;
    }
    TokenUpdate update{};
    bool updated = false;
    while (updatesFromFilter.pull(update)) {
        if (orchestrationUpdatesIncoming != nullptr)
            orchestrationUpdatesIncoming->push(update);
        const int32_t pitch = DenseQuantize::snapPianoPitch(update.oldNote.getPitch());
        const int32_t finalDur = std::max(1, update.newNote.getRealDuration());
        engine.noteOff(update.oldNote.time, pitch, finalDur);
        for (size_t i = 0; i + 3 < inputData.size(); i += kEventWidth)
            if (inputData[i] == update.oldNote.time
                && DenseQuantize::instrumentOfNoteToken(inputData[i + 2]) == DenseConfig::PianoReductionInputInstrument
                && inputData[i + 2] - static_cast<int32_t>(DenseVocab::NoteOffset) == pitch) {
                inputData[i + 1] = static_cast<int32_t>(DenseVocab::DurOffset) + DenseQuantize::snapDurationCs(finalDur);
                updated = true;
            }
    }
    if (applied || updated)
        notifyInputDataChanged();
    return applied;
}

auto InformativeWaitReductionTransformer::emit(const std::vector<InformativeWaitEngine::OutNote> &notes) -> void {
    const int32_t cancel = engine.takeCancelFrom();
    if (cancel != INT32_MAX) {
        // the piano came first: the generated notes from `cancel` on are gone (queue and history)
        dropOutputHistoryFrom(cancel);
        pushOutputToken(Token{cancel, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
        eraseStridedFrom(inputData, kEventWidth, cancel);
        notifyInputDataChanged();
    }
    if (notes.empty())
        return;
    for (const auto &n: notes) {
        const int32_t dur = std::clamp(n.durationCs, 1, 995);
        const auto quad = denseQuad(n.onsetCs, dur, 1, n.pitch, n.velocity);
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        Token outbound{n.onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + dur,
                       static_cast<int32_t>(DenseVocab::NoteOffset) + n.pitch, n.velocity};
        pushOutputToken(outbound);
        currentTime = std::max(currentTime, n.onsetCs);
        logGeneratedTokenIfResumed(n.onsetCs);
    }
    notifyInputDataChanged();
    std::lock_guard<std::mutex> lock(statsMutex);
    lastStats = engine.stats();
}

auto InformativeWaitReductionTransformer::statsSnapshot() const -> InformativeWaitEngine::Stats {
    std::lock_guard<std::mutex> lock(statsMutex);
    return lastStats;
}

auto InformativeWaitReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    applySettings();
    if (! streamStarted)
        resetStream(now);
    if (clockJumpedBack(now)) {
        clearOutputHistory();
        pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
        inputData.clear();
        resetStream(now);
    }
    if (const auto cut = consumeModeCut())
        resetStream(*cut);
    drainInput();
    if (isGenerationStopped())
        return false;
    const auto out = engine.step(now);
    emit(out);
    return ! out.empty();
}

void InformativeWaitReductionTransformer::threadRun() {
    if (! isModelLoaded())
        return;
    inputData.clear();
    clearInputTokenQueue();
    clearInputConditioningQueue();
    clearOutputTokenQueue();
    clearUpdatesFromFilter();
    clearOutputHistory();
    forgetClock();
    applySettings();
    resetStream(nowCs());
    notifyInputDataChanged();
    resetAheadThrottle();
    resetLoopTiming();

    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        applySettings();
        if (const auto cut = consumeModeCut()) {
            SessionLog::add("IWAIT", "mode cut at " + juce::String(*cut) + ": stream restarted (no rewind in this model)");
            resetStream(*cut);
        }
        const int32_t now = nowCs();
        if (clockJumpedBack(now)) {
            SessionLog::add("CLOCK", getThreadName() + " | the clock jumped back to " + juce::String(now));
            clearOutputHistory();
            pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
            inputData.clear();
            notifyInputDataChanged();
            resetStream(now);
        }
        drainInput();
        if (isGenerationStopped()) {
            maybeEmitGenerationPauseSoftStopClear();
            wait(5);
            recordThreadLoopMs(loopStartMs);
            continue;
        }
        wasGenerationPaused = false;
        const auto out = engine.step(now);
        emit(out);
        if (! engine.lastError.empty()) {
            reportSamplingError("Informative WAIT: " + juce::String(engine.lastError));
            engine.lastError.clear();
        }
        {
            std::lock_guard<std::mutex> lock(statsMutex);
            lastStats = engine.stats();
        }
        if (out.empty())
            wait(2);
        recordThreadLoopMs(loopStartMs);
    }
}

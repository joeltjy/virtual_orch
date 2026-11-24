#include "VirtualOrch/reduction/HybridInformativeWaitReductionTransformer.h"

#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/reduction/DenseTypes.h"

#include <algorithm>
#include <set>

namespace {

constexpr int kEventWidth = 4;

auto denseQuad(int32_t time, int32_t durCs, int32_t instrument, int32_t pitch, int32_t velocity)
    -> std::array<int32_t, 4> {
    return {time, static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * instrument + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity};
}

auto loadOnnx(const char *path, size_t vocab, std::string &error) -> std::unique_ptr<StreamingKv::OnnxBackend> {
    auto onnx = std::make_unique<StreamingKv::OnnxBackend>();
    if (! onnx->init(path, error))
        return nullptr;
    if (onnx->vocabSize() != vocab) {
        error = std::string(path) + ": vocab " + std::to_string(onnx->vocabSize()) + ", expected " + std::to_string(vocab);
        return nullptr;
    }
    return onnx;
}

} // namespace

HybridInformativeWaitReductionTransformer::HybridInformativeWaitReductionTransformer(ModelConfig &modelConfigIn)
    : ReductionTransformer("Reduction Transformer Hybrid v1.1", modelConfigIn) {
    const auto logger = [this](const char *who) {
        return [this, who](const InformativeWaitEngine::Event &e) {
            if (SessionLog::isRunning())
                SessionLog::add("IWAIT", juce::String(who) + " | " + juce::String(e.what) + " @" + juce::String(e.atCs)
                                             + " " + juce::String(e.a) + " " + juce::String(e.b)
                                             + (e.text.empty() ? juce::String() : " (" + juce::String(e.text) + ")"));
        };
    };
    silentEngine.onEvent = logger("m1");
    waitEngine.onEvent = logger("m2");
    tdValueProvider = [] { return TemporalDensity::get(); };
}

auto HybridInformativeWaitReductionTransformer::init(const char *silentModelPath, const char *waitModelPath) -> bool {
    lastInitError.clear();
    std::string error;
    auto silent = loadOnnx(silentModelPath, static_cast<size_t>(InformativeWaitEngine::kWait0), error);
    if (silent == nullptr) {
        lastInitError = "silent-side model (overnight model 1, vocab 11768): " + error;
        reportSamplingError("Hybrid v1.1: " + juce::String(lastInitError));
        return false;
    }
    auto wait = loadOnnx(waitModelPath, static_cast<size_t>(InformativeWaitEngine::kVocab), error);
    if (wait == nullptr) {
        lastInitError = "playing-side model (informative WAIT, vocab 11879): " + error;
        reportSamplingError("Hybrid v1.1: " + juce::String(lastInitError));
        return false;
    }
    silentName = juce::File(silentModelPath).getFileNameWithoutExtension().toStdString();
    waitName = juce::File(waitModelPath).getFileNameWithoutExtension().toStdString();
    initWithBackends(std::move(silent), std::move(wait));
    return true;
}

auto HybridInformativeWaitReductionTransformer::initWithBackends(std::unique_ptr<StreamingKv::Backend> silent,
                                                                 std::unique_ptr<StreamingKv::Backend> wait) -> void {
    silentBackend = std::move(silent);
    waitBackend = std::move(wait);
    silentEngine.setBackend(silentBackend.get());
    waitEngine.setBackend(waitBackend.get());
}

auto HybridInformativeWaitReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto HybridInformativeWaitReductionTransformer::applySettings() -> void {
    auto s = silentEngine.settings();
    s.continuation = true;
    s.mode = InformativeWaitEngine::Mode::Unconstrained;
    s.precompute = false;
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    s.tdLevel = td;
    s.tdGamma = tdGamma(true);
    silentEngine.setSettings(s);
    auto w = waitEngine.settings();
    w.continuation = false;
    w.mode = static_cast<InformativeWaitEngine::Mode>(waitMode.load());
    w.precompute = precompute.load();
    w.seed = 1;
    w.tdLevel = td;
    w.tdGamma = tdGamma(false);
    waitEngine.setSettings(w);
}

auto HybridInformativeWaitReductionTransformer::resetStreams(int32_t now) -> void {
    silentEngine.reset(now);
    waitEngine.reset(now);
    streamsStarted = true;
    tdResetRequested.store(true); // a restart starts from gamma 0, not the last stream's
    currentTime = now;
    playingSinceMs = -1;
    silentActive.store(true);
    reasonValue.store(Reason::Start);
    SessionLog::add("HYBRID", juce::String(silentName) + " from " + juce::String(now) + " (start)");
}

auto HybridInformativeWaitReductionTransformer::drainInput() -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
    }
    bool applied = false;
    Token in{-1, -1, -1};
    while (inputTokenQueue.pull(in)) {
        const int32_t pitch = DenseQuantize::snapPianoPitch(in.getPitch());
        const int32_t velocity = std::clamp(in.velocity > 0 ? in.velocity : DenseConfig::DefaultVelocity, 1, 127);
        const bool held = DenseSampling::isProvisionalDurationToken(in.duration, modelConfig.inputDuration);
        for (auto *e: {&silentEngine, &waitEngine}) {
            e->noteOn(in.time, pitch, velocity);
            if (! held)
                e->noteOff(in.time, pitch, std::max(1, in.getRealDuration()));
        }
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
        silentEngine.noteOff(update.oldNote.time, pitch, finalDur);
        waitEngine.noteOff(update.oldNote.time, pitch, finalDur);
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

auto HybridInformativeWaitReductionTransformer::setSilent(bool silent, Reason why, int32_t now) -> void {
    reasonValue.store(why);
    if (silentActive.exchange(silent) == silent)
        return;
    (silent ? waitEngine : silentEngine).interrupt(); // the model that stops drawing forgets its pending WAIT
    if (! silent)
        waitSinceCs = now;
    static constexpr const char *kWhy[] = {"start", "pedal released", "playing 1 s", "silent 0.5 s",
                                           "2 s of predicted silence"};
    SessionLog::add("HYBRID", juce::String(silent ? silentName : waitName) + " from " + juce::String(now) + " ("
                                  + kWhy[static_cast<int>(why)] + ")");
}

auto HybridInformativeWaitReductionTransformer::cutGeneratedFrom(int32_t now) -> void {
    resetAheadThrottle();
    if (! hasGeneratedFrom(now))
        return;
    dropOutputHistoryFrom(now);
    pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
    silentEngine.dropGeneratedFrom(now);
    waitEngine.dropGeneratedFrom(now);
    eraseStridedFrom(inputData, kEventWidth, now);
    notifyInputDataChanged();
}

auto HybridInformativeWaitReductionTransformer::predictsSilenceFrom(int32_t now) const -> bool {
    int32_t nextOnset = INT32_MAX;
    int32_t lastEnd = waitSinceCs;
    for (const auto &t: getOutputHistory()) {
        if (t.note < static_cast<int32_t>(Vocab::NoteOffset) || t.note >= static_cast<int32_t>(Vocab::Rest))
            continue;
        const int32_t end = t.time + t.getRealDuration();
        if (t.time <= now && now < end)
            return false;
        if (t.time > now)
            nextOnset = std::min(nextOnset, t.time);
        else
            lastEnd = std::max(lastEnd, end);
    }
    if (nextOnset != INT32_MAX)
        return nextOnset - now >= kPredictedSilenceCs;
    return lastEnd != INT32_MIN && now - lastEnd >= kPredictedSilenceCs;
}

auto HybridInformativeWaitReductionTransformer::wallMs() const -> int64_t {
    return wallMsProvider ? wallMsProvider() : juce::Time::currentTimeMillis();
}

auto HybridInformativeWaitReductionTransformer::tdGamma(bool silentModel) const -> std::optional<float> {
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    if (td == TemporalDensity::kAuto)
        return std::nullopt;
    return static_cast<float>((silentModel ? tdControlSilent : tdControlWait).current().gamma);
}

auto HybridInformativeWaitReductionTransformer::tickTdControl() -> void {
    // Before the rate limit, so the next draw already uses gamma 0.
    if (tdResetRequested.exchange(false)) {
        tdControlSilent.reset();
        tdControlWait.reset();
        lastTdTickMs = wallMs(); // the first update after a full tick of new notes
        if (OpLog::on())
            OpLog::emit(OpLog::Src::Rt, "TD_CTRL_RESET", {});
    }
    const int64_t nowMs = wallMs();
    if (lastTdTickMs >= 0 && static_cast<double>(nowMs - lastTdTickMs) < TdController::kTickMs)
        return;
    lastTdTickMs = nowMs;

    // The orchestra's density: distinct onsets of the generated notes in the 3 s ending at the newest one.
    std::optional<double> measured;
    {
        const auto history = getOutputHistorySince(INT32_MIN);
        const auto isNote = [](const Token &t) {
            return t.note >= static_cast<int32_t>(Vocab::NoteOffset) && t.note < static_cast<int32_t>(Vocab::Rest);
        };
        int32_t frontier = INT32_MIN;
        for (const auto &t: history)
            if (isNote(t))
                frontier = std::max(frontier, t.time);
        if (frontier != INT32_MIN) {
            std::set<int32_t> onsets;
            for (const auto &t: history)
                if (isNote(t) && t.time > frontier - TdController::kWindowCs && t.time <= frontier)
                    onsets.insert(t.time);
            measured = TdController::measure(static_cast<int>(onsets.size()));
        }
    }
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    const bool clockRunning = clock == nullptr || clock->isRunning();
    std::vector<TdControlView> views;
    const auto tickOne = [&](TdController &controller, InformativeWaitEngine &engine, const char *model) {
        const auto [sensitivity, draws] = engine.takeSensitivity();
        juce::ignoreUnused(draws);
        // waiting: drew nothing this tick because a WAIT is pending (not "not drawing")
        const bool waiting = ! sensitivity.has_value() && engine.hasPendingWait();
        const auto &state = controller.tick({td, measured, sensitivity, clockRunning, waiting});
        views.push_back({model, state});
        if (OpLog::on() && td != TemporalDensity::kAuto)
            OpLog::emit(OpLog::Src::Rt, "TD_CTRL",
                        {{"model", model}, {"td", td}, {"measured", state.measured}, {"error", state.error},
                         {"gamma", state.gamma}, {"k", state.k}, {"s_hat", state.sensitivity}, {"m", state.m},
                         {"frozen", state.frozen},
                         {"why", state.why.empty() ? OpLog::Value() : OpLog::Value(state.why)}});
    };
    tickOne(tdControlSilent, silentEngine, "m1");
    tickOne(tdControlWait, waitEngine, "m2");
    const std::lock_guard<std::mutex> lock(tdViewMutex);
    tdViews = std::move(views);
}

auto HybridInformativeWaitReductionTransformer::updateHybrid(int32_t now) -> void {
    const int64_t wallMs = this->wallMs();
    const int64_t silentFor = pianoSilentForMs ? pianoSilentForMs() : INT64_MAX;
    const bool playing = silentFor <= kStopMs;
    if (! playing)
        playingSinceMs = -1;
    else if (playingSinceMs < 0)
        playingSinceMs = wallMs;
    if (pedalReleased.exchange(false)) {
        setSilent(true, Reason::PedalReleased, now);
        playingSinceMs = playing ? wallMs : -1;
    }
    if (silentActive.load()) {
        if (! generationPause.get() && playingSinceMs >= 0 && wallMs - playingSinceMs >= kPlayToWaitMs)
            setSilent(false, Reason::Playing, now);
    } else if (! playing) {
        cutGeneratedFrom(now);
        setSilent(true, Reason::Stopped, now);
    } else if (! generationPause.get() && predictsSilenceFrom(now)) {
        cutGeneratedFrom(now);
        setSilent(true, Reason::PredictedSilence, now);
        playingSinceMs = wallMs;
    }
}

auto HybridInformativeWaitReductionTransformer::generate(int32_t now) -> bool {
    auto &active = silentActive.load() ? silentEngine : waitEngine;
    auto &other = silentActive.load() ? waitEngine : silentEngine;
    const auto out = active.step(now);
    other.absorb(now);
    const int32_t cancel = active.takeCancelFrom();
    if (cancel != INT32_MAX) {
        dropOutputHistoryFrom(cancel);
        pushOutputToken(Token{cancel, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
        other.dropGeneratedFrom(cancel);
        eraseStridedFrom(inputData, kEventWidth, cancel);
        notifyInputDataChanged();
    }
    for (const auto &n: out) {
        other.mirrorNote(n);
        const int32_t dur = std::clamp(n.durationCs, 1, 995);
        const auto quad = denseQuad(n.onsetCs, dur, 1, n.pitch, n.velocity);
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        pushOutputToken(Token{n.onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + dur,
                              static_cast<int32_t>(DenseVocab::NoteOffset) + n.pitch, n.velocity});
        currentTime = std::max(currentTime, n.onsetCs);
        logGeneratedTokenIfResumed(n.onsetCs);
    }
    if (! out.empty())
        notifyInputDataChanged();
    if (! active.lastError.empty()) {
        reportSamplingError("Hybrid v1.1: " + juce::String(active.lastError));
        active.lastError.clear();
    }
    return ! out.empty();
}

auto HybridInformativeWaitReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    applySettings();
    if (! streamsStarted)
        resetStreams(now);
    if (const auto cut = consumeModeCut())
        resetStreams(*cut);
    drainInput();
    updateHybrid(now);
    tickTdControl();
    if (isGenerationStopped())
        return false;
    return generate(now);
}

void HybridInformativeWaitReductionTransformer::threadRun() {
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
    resetStreams(nowCs());
    notifyInputDataChanged();
    resetAheadThrottle();
    resetLoopTiming();
    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        applySettings();
        if (const auto cut = consumeModeCut()) {
            SessionLog::add("HYBRID", "mode cut at " + juce::String(*cut) + ": streams restarted");
            resetStreams(*cut);
        }
        const int32_t now = nowCs();
        if (clockJumpedBack(now)) {
            clearOutputHistory();
            pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
            inputData.clear();
            notifyInputDataChanged();
            resetStreams(now);
        }
        drainInput();
        updateHybrid(now);
        tickTdControl();
        if (isGenerationStopped()) {
            maybeEmitGenerationPauseSoftStopClear();
            wait(5);
            recordThreadLoopMs(loopStartMs);
            continue;
        }
        wasGenerationPaused = false;
        if (! generate(now))
            wait(2);
        recordThreadLoopMs(loopStartMs);
    }
}

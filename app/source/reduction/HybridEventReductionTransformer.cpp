#include "VirtualOrch/reduction/HybridEventReductionTransformer.h"

#include "VirtualOrch/PolyphonyCap.h"
#include "VirtualOrch/RegisterMix.h"
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include <algorithm>
#include <random>

namespace {

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

HybridEventReductionTransformer::HybridEventReductionTransformer(ModelConfig &modelConfigIn)
    : EventReductionTransformer("Reduction Transformer Hybrid v1.3", modelConfigIn) {}

auto HybridEventReductionTransformer::init(const char *silentModelPath, const char *waitModelPath) -> bool {
    lastInitError.clear();
    std::string error;
    auto silent = loadOnnx(silentModelPath, EventEngine::kM1eVocab, error);
    if (silent == nullptr) {
        lastInitError = error;
        return false;
    }
    auto wait = loadOnnx(waitModelPath, kM2eVocab, error);
    if (wait == nullptr) {
        lastInitError = error;
        return false;
    }
    initWithBackends(std::move(silent), std::move(wait));
    return true;
}

auto HybridEventReductionTransformer::initWithBackends(std::unique_ptr<StreamingKv::Backend> silent,
                                                       std::unique_ptr<StreamingKv::Backend> wait) -> void {
    silentBackend = std::move(silent);
    waitBackend = std::move(wait);
    silentEngine.setBackend(silentBackend.get());
    waitEngine.setBackend(waitBackend.get());
    streamsStarted = false;
}

auto HybridEventReductionTransformer::loadVoicingBank(const std::string &path) -> bool {
    auto bank = std::make_shared<VoicingBank>();
    std::string error;
    const auto text = juce::File(path).loadFileAsString().toStdString();
    if (text.empty() || ! bank->loadJson(text, error)) {
        SessionLog::add("HYBRID13", "voicing bank not loaded: " + juce::String(path) + " " + juce::String(error));
        voicings = nullptr;
        return false;
    }
    voicings = std::move(bank);
    return true;
}

auto HybridEventReductionTransformer::wallMs() const -> int64_t {
    return wallMsProvider ? wallMsProvider() : juce::Time::currentTimeMillis();
}

auto HybridEventReductionTransformer::applySettings() -> void {
    const int32_t high = RegisterRange::highPitch();
    const int32_t low = PolyphonyCap::lowestPitch(getMaxPolyphony(), RegisterRange::lowPitch(), high); // Mono: above C4
    for (auto *e: {&silentEngine, &waitEngine}) {
        auto s = e->settings();
        s.pitchLow = low;
        s.pitchHigh = high;
        s.maxSounding = getMaxPolyphony();
        s.forceAfterCs = std::max(1, forceAfterMs.load() / 10);
        s.maxHoldCs = kMaxNoteDurationCs;
        // the logit adjustments (biasesActiveNow: the global switch and the piano-silence gate, as v1.1 / v1.2)
        const bool biases = biasesActiveNow();
        const bool silentSide = e == &silentEngine;
        s.pitchBias = biases && (silentSide ? hybridV1PitchBias.load() : isPitchBiasEnabled());
        s.registerMix = biases ? RegisterMix::Target::get() : std::nullopt; // the mood preset's register mix
        s.onsetGapBias = biases && (silentSide ? hybridV1OnsetGapBias.load() : isOnsetGapBiasEnabled());
        s.pitchTau = silentSide ? DensePitchBias::TauBase : getPitchTauBase();
        s.pitchTauPrime = silentSide ? DensePitchBias::TauBase : getPitchTauPrimeBase();
        s.onsetGapTau = silentSide ? DenseOnsetGapBias::TauBase : getOnsetGapTauBase();
        if (e == &silentEngine) {
            s.off0 = EventEngine::kM1eOff0;
            s.vocab = EventEngine::kM1eVocab;
            s.seed = silentSeed;
        } else {
            s.off0 = kM2eOff0;
            s.vocab = kM2eVocab;
            s.seed = waitSeed;
            s.chordTokens = true;
            s.stackOff = kM2eStackOff;
            s.voicings = voicings;
            s.maxLates = 2;
        }
        e->setSettings(s);
    }
}

auto HybridEventReductionTransformer::resetStreams(int32_t now) -> void {
    applySettings();
    silentEngine.reset(now);
    waitEngine.reset(now);
    clearIo();
    startedAtCs = now;
    streamsStarted = true;
    playingSinceMs = -1;
    silentActive.store(true);
    reasonValue.store(Reason::Start);
    SessionLog::add("HYBRID13", "model 1 event from " + juce::String(now) + " (start)");
}

auto HybridEventReductionTransformer::feedPiano(int32_t now) -> void {
    for (const auto &e: drainPianoEvents(now))
        for (auto *engine: {&silentEngine, &waitEngine}) {
            if (e.on)
                engine->pianoOn(e.timeCs, e.pitch, e.velocity);
            else
                engine->pianoOff(e.timeCs, e.pitch);
        }
}

auto HybridEventReductionTransformer::setSilent(bool silent, Reason why, int32_t now) -> void {
    reasonValue.store(why);
    if (silentActive.exchange(silent) == silent)
        return;
    (silent ? waitEngine : silentEngine).interrupt();
    static constexpr const char *kWhy[] = {"start", "pedal released", "playing 1 s", "silent 0.5 s",
                                           "Accompaniment: the pianist is ignored"};
    SessionLog::add("HYBRID13", juce::String(silent ? "model 1 event" : "model 2 event") + " from " + juce::String(now)
                                    + " (" + kWhy[static_cast<int>(why)] + ")");
}

auto HybridEventReductionTransformer::cutGeneratedFrom(int32_t now) -> void {
    resetAheadThrottle();
    if (hasGeneratedFrom(now)) {
        dropOutputHistoryFrom(now);
        pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
    }
    silentEngine.dropGeneratedFrom(now);
    waitEngine.dropGeneratedFrom(now);
    forgetOpenFrom(now);
    currentTime = now;
}

auto HybridEventReductionTransformer::updateHybrid(int32_t now) -> void {
    const int64_t wall = wallMs();
    const int64_t silentFor = pianoSilentForMs ? pianoSilentForMs() : INT64_MAX;
    const bool playing = silentFor <= kStopMs;
    if (! playing)
        playingSinceMs = -1;
    else if (playingSinceMs < 0)
        playingSinceMs = wall;
    if (pedalReleased.exchange(false)) {
        setSilent(true, Reason::PedalReleased, now);
        playingSinceMs = playing ? wall : -1;
    }
    if (keepsQueueOnInput()) {
        if (! silentActive.load() || reasonValue.load() != Reason::Accompaniment)
            setSilent(true, Reason::Accompaniment, now);
        playingSinceMs = -1;
        return;
    }
    if (silentActive.load()) {
        if (! generationPause.get() && playingSinceMs >= 0 && wall - playingSinceMs >= kPlayToWaitMs)
            setSilent(false, Reason::Playing, now);
    } else if (! playing) {
        cutGeneratedFrom(now);
        setSilent(true, Reason::Stopped, now);
    }
}

auto HybridEventReductionTransformer::generate(int32_t now) -> bool {
    const bool silent = silentActive.load();
    if (! silent && now < startedAtCs + startDelayMs.load() / 10)
        return false; // model 2's start delay: the piano only
    auto &active = silent ? silentEngine : waitEngine;
    auto &other = silent ? waitEngine : silentEngine;
    const auto out = active.step(now);
    for (const auto &o: out) {
        if (o.on)
            other.mirrorOn(o.timeCs, o.pitch, o.velocity);
        else
            other.mirrorOff(o.timeCs, o.pitch);
        emitOrchestra(o, silent ? 0 : kSimultaneousVelocityBoost, nextNoteId.fetch_add(1, std::memory_order_relaxed));
    }
    if (! active.lastError.empty()) {
        reportSamplingError(juce::String(label()) + ": " + juce::String(active.lastError));
        active.lastError.clear();
    }
    return ! out.empty();
}

auto HybridEventReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    finishAheadThrottleIfDue();
    if (! streamsStarted)
        resetStreams(now);
    if (const auto cut = consumeModeCut())
        cutGeneratedFrom(*cut);
    applySettings();
    feedPiano(now);
    updateHybrid(now);
    if (isGenerationStopped())
        return false;
    return generate(now);
}

void HybridEventReductionTransformer::threadRun() {
    if (! isModelLoaded())
        return;
    if (voiceSeparation != nullptr)
        voiceSeparation->reset();
    {
        std::random_device rd;
        silentSeed = (static_cast<uint64_t>(rd()) << 32) | rd();
        waitSeed = (static_cast<uint64_t>(rd()) << 32) | rd();
    }
    inputData.clear();
    clearInputTokenQueue();
    clearInputConditioningQueue();
    clearOutputTokenQueue();
    clearUpdatesFromFilter();
    clearOutputHistory();
    forgetClock();
    resetStreams(nowCs());
    notifyInputDataChanged();
    resetAheadThrottle();
    resetLoopTiming();
    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        finishAheadThrottleIfDue();
        if (const auto cut = consumeModeCut()) {
            SessionLog::add("HYBRID13", "mode cut at " + juce::String(*cut));
            cutGeneratedFrom(*cut);
        }
        const int32_t now = nowCs();
        if (clockJumpedBack(now)) {
            SessionLog::add("CLOCK", "jumped back to " + juce::String(now) + ": streams restarted");
            resetAheadThrottle();
            clearOutputHistory();
            pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
            inputData.clear();
            notifyInputDataChanged();
            resetStreams(now);
        }
        applySettings();
        feedPiano(now);
        updateHybrid(now);
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

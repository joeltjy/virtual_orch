#include "VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h"

#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/SessionLog.h"
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
    ModelConfig &modelConfigIn, Variant variantIn)
    : ReductionTransformer(variantIn == Variant::Wait     ? "Reduction Transformer WAIT"
                           : variantIn == Variant::Hybrid ? "Reduction Transformer Hybrid"
                                                          : "Reduction Transformer Anticipated",
                           modelConfigIn),
      variant(variantIn) {
    tdValueProvider = [] { return TemporalDensity::get(); };
    stage.tdValue = [this] { return tdValueProvider(); };
    silentStage.tdValue = [this] { return tdValueProvider(); };
    if (variant == Variant::Wait || variant == Variant::Hybrid) {
        // rope_cache_RT_sampling.txt: WAIT, 5 cs merge, piano first + ascending ties, onset T 1.0.
        stage.waitToken = kWaitToken;
        stage.stream.waitToken = kWaitToken;
        stage.stream.mergeGivenCs = 5;
        stage.ascendingTies = true;
        onsetTemperature.store(1.0f);
        durationTemperature.store(0.5f);
        noteTemperature.store(0.5f);
        velocityTemperature.store(0.5f);
    }
    stage.onWaitEvent = [this](const char *what, int32_t value) {
        SessionLog::add("WAIT", getThreadName() + " | " + what + " " + juce::String(value)
                                    + (std::string(what) == "wait"
                                           ? " P(wait) " + juce::String(stage.sampler.lastWaitProbability, 2)
                                           : juce::String()));
    };
}

auto TwoInstrumentAnticipatedReductionTransformer::loadKvBackend(
    const char *path, std::unique_ptr<StreamingKv::OnnxBackend> &out) -> bool {
    out = std::make_unique<StreamingKv::OnnxBackend>();
    std::string error;
    if (out->init(path, error))
        return true;
    lastInitError = error.rfind("not a KV export", 0) == 0
                        ? error + ": this looks like a full-sequence model, made for V1, V2 or "
                                  "Piano Reduction. Pick that reduction type, or the T3'' KV export."
                        : error;
    reportSamplingError("Anticipated reduction init failed: " + juce::String(lastInitError));
    return false;
}

auto TwoInstrumentAnticipatedReductionTransformer::initHybrid(const char *silentModelPath,
                                                              const char *waitModelPath) -> bool {
    lastInitError.clear();
    std::unique_ptr<StreamingKv::OnnxBackend> silent;
    std::unique_ptr<StreamingKv::OnnxBackend> wait;
    if (! loadKvBackend(silentModelPath, silent) || ! loadKvBackend(waitModelPath, wait))
        return false;
    const size_t silentVocab = silent->vocabSize();
    const size_t waitVocab = wait->vocabSize();
    if (silentVocab > static_cast<size_t>(kWaitToken) || silentVocab < static_cast<size_t>(TdTokens::kDenseFirst + 7)) {
        lastInitError = "the silent-piano model must be T3'' v1 (vocab 11776, no WAIT); got vocab "
                        + std::to_string(silentVocab);
        reportSamplingError("Hybrid: " + juce::String(lastInitError));
        return false;
    }
    if (waitVocab <= static_cast<size_t>(kWaitToken)) {
        lastInitError = "the playing model must be W2/W3 (vocab 11777, WAIT); got vocab "
                        + std::to_string(waitVocab);
        reportSamplingError("Hybrid: " + juce::String(lastInitError));
        return false;
    }
    initHybridWithBackends(std::move(silent), silentVocab, std::move(wait), waitVocab);
    hybridV1Name = juce::File(silentModelPath).getFileNameWithoutExtension().toStdString();
    hybridWaitName = juce::File(waitModelPath).getFileNameWithoutExtension().toStdString();
    return true;
}

auto TwoInstrumentAnticipatedReductionTransformer::initHybridWithBackends(
    std::unique_ptr<StreamingKv::Backend> silentBackend, size_t silentVocab,
    std::unique_ptr<StreamingKv::Backend> waitBackend, size_t waitVocab) -> void {
    silentStage.setBackend(std::move(silentBackend), silentVocab);
    stage.setBackend(std::move(waitBackend), waitVocab);
}

auto TwoInstrumentAnticipatedReductionTransformer::init(const char *modelPath) -> bool {
    lastInitError.clear();
    std::unique_ptr<StreamingKv::OnnxBackend> onnx;
    if (! loadKvBackend(modelPath, onnx))
        return false;
    const size_t vocab = onnx->vocabSize();
    if (variant == Variant::T3 && vocab > static_cast<size_t>(kWaitToken)) {
        lastInitError = "this model has a WAIT token (vocab " + std::to_string(vocab)
                        + "): it is W2/W3. Pick \"Anticipated Reduction + WAIT (W2/W3)\" instead.";
        reportSamplingError("Anticipated reduction: " + juce::String(lastInitError));
        return false;
    }
    if (variant == Variant::Wait && vocab <= static_cast<size_t>(kWaitToken)) {
        lastInitError = "this model has no WAIT token (vocab " + std::to_string(vocab)
                        + ", W2/W3 need 11777). Pick \"Anticipated Piano Reduction (T3'')\" for it.";
        reportSamplingError("WAIT reduction: " + juce::String(lastInitError));
        return false;
    }
    if (vocab < static_cast<size_t>(TdTokens::kDenseFirst + 7)) {
        lastInitError = "unexpected vocab size " + std::to_string(vocab)
                        + " (T3'' needs the dense vocab with TD ids, 11775 or 11776)";
        reportSamplingError("Anticipated reduction: " + juce::String(lastInitError));
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
    forEachStage([now](AnticipationStage &st) { st.reset(now); });
    loggedMain.clear();
    loggedV1.clear();
    SessionLog::add("RT_IN", getThreadName() + " | stream reset at clock " + juce::String(now)
                                 + ", base " + juce::String(stage.stream.base()));
    playingSinceMs = -1;
    hybridPlayingFor.store(-1);
    hybridV1.store(true); // a (re)start begins with v1
    hybridReasonValue.store(HybridReason::Start);
    currentTime = stage.stream.clock();
}

auto TwoInstrumentAnticipatedReductionTransformer::rewindGeneratedFrom(int32_t cutCs) -> void {
    forEachStage([cutCs](AnticipationStage &st) { st.rewindTo(cutCs); });
    rewindInputDataFrom(cutCs);
    currentTime = stage.stream.clock();
}

auto TwoInstrumentAnticipatedReductionTransformer::setHybridModel(bool v1, HybridReason why,
                                                                  int32_t nowCs) -> void {
    hybridReasonValue.store(why);
    if (hybridV1.exchange(v1) == v1)
        return;
    if (! v1)
        hybridWaitSinceCs = nowCs;
    static constexpr const char *kWhy[] = {"start", "pedal released", "playing 1 s", "silent 0.5 s",
                                           "2 s of predicted silence"};
    SessionLog::add("HYBRID", juce::String(v1 ? hybridV1Name : hybridWaitName) + " from "
                                  + juce::String(nowCs) + " (" + kWhy[static_cast<int>(why)] + ")");
}

auto TwoInstrumentAnticipatedReductionTransformer::updateHybrid(int32_t nowCs) -> void {
    if (variant != Variant::Hybrid)
        return;
    const int64_t wallMs = wallMsProvider ? wallMsProvider() : juce::Time::currentTimeMillis();
    const int64_t silentFor = pianoSilentForMs ? pianoSilentForMs() : INT64_MAX;
    const bool playing = silentFor <= kHybridStopMs;
    if (! playing)
        playingSinceMs = -1;
    else if (playingSinceMs < 0)
        playingSinceMs = wallMs;

    if (pedalReleased.exchange(false)) {
        setHybridModel(true, HybridReason::PedalReleased, nowCs);
        playingSinceMs = playing ? wallMs : -1; // the 1 s counts from the release
    }
    hybridPlayingFor.store(playingSinceMs < 0 ? -1 : wallMs - playingSinceMs);

    if (hybridV1.load()) {
        if (! generationPause.get() && playingSinceMs >= 0 && wallMs - playingSinceMs >= kHybridPlayToWaitMs)
            setHybridModel(false, HybridReason::Playing, nowCs);
    } else if (! playing) {
        // Stopped for 0.5 s: W2/W3's future goes, v1 takes over from now.
        cutGeneratedFrom(nowCs);
        setHybridModel(true, HybridReason::Stopped, nowCs);
    } else if (! generationPause.get() && predictsSilenceFrom(nowCs)) {
        // W2/W3 predicts 2 s of silence: v1 fills it (W2/W3 again after 1 more s of playing).
        cutGeneratedFrom(nowCs);
        setHybridModel(true, HybridReason::PredictedSilence, nowCs);
        playingSinceMs = wallMs;
        hybridPlayingFor.store(0);
    }
}

auto TwoInstrumentAnticipatedReductionTransformer::cutGeneratedFrom(int32_t nowCs) -> void {
    resetAheadThrottle();
    if (! hasGeneratedFrom(nowCs))
        return;
    dropOutputHistoryFrom(nowCs);
    pushOutputToken(Token{nowCs, static_cast<int32_t>(DenseVocab::DurOffset),
                          static_cast<int32_t>(Vocab::ClearQueue)});
    rewindGeneratedFrom(nowCs);
}

auto TwoInstrumentAnticipatedReductionTransformer::predictsSilenceFrom(int32_t nowCs) const -> bool {
    int32_t nextOnset = INT32_MAX;
    int32_t lastEnd = hybridWaitSinceCs;
    for (const auto &t: getOutputHistory()) {
        if (t.note < static_cast<int32_t>(Vocab::NoteOffset) || t.note >= static_cast<int32_t>(Vocab::Rest))
            continue;
        const int32_t end = t.time + t.getRealDuration();
        if (t.time <= nowCs && nowCs < end)
            return false; // a generated note is sounding
        if (t.time > nowCs)
            nextOnset = std::min(nextOnset, t.time);
        else
            lastEnd = std::max(lastEnd, end);
    }
    if (nextOnset != INT32_MAX)
        return nextOnset - nowCs >= kHybridPredictedSilenceCs;
    // Nothing queued (W2/W3 keeps waiting): silent since the last note ended or the switch.
    return lastEnd != INT32_MIN && nowCs - lastEnd >= kHybridPredictedSilenceCs;
}

auto TwoInstrumentAnticipatedReductionTransformer::logModelInput(const AnticipationStage &st,
                                                                 std::vector<int32_t> &lastLogged,
                                                                 const char *who) -> void {
    const auto tokens = st.stream.tokens();
    size_t same = 0;
    while (same < tokens.size() && same < lastLogged.size() && tokens[same] == lastLogged[same])
        ++same;
    juce::String text = juce::String(who) + " | base " + juce::String(st.stream.base()) + " | "
                        + (same < lastLogged.size() ? "rewrite from " : "from ") + juce::String(static_cast<int>(same))
                        + " of " + juce::String(static_cast<int>(tokens.size())) + ":";
    for (size_t i = same; i < tokens.size(); ++i)
        text << " " << tokens[i];
    SessionLog::add("RT_IN", text);
    lastLogged = tokens;
}

auto TwoInstrumentAnticipatedReductionTransformer::restartOnClockJumpBack(int32_t now) -> void {
    SessionLog::add("CLOCK", getThreadName() + " | the clock jumped back to " + juce::String(now)
                                 + ": queued notes cleared, stream restarted");
    resetAheadThrottle();
    clearOutputHistory();
    pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset),
                          static_cast<int32_t>(Vocab::ClearQueue)});
    inputData.clear();
    notifyInputDataChanged();
    resetStream(now);
}

auto TwoInstrumentAnticipatedReductionTransformer::cutFutureOnInput(int32_t now) -> void {
    resetAheadThrottle();
    // Edit: the next 2 s play on; the input is reordered in among them (generateOne).
    const int32_t cut = isEditMode() ? now + kEditCutAheadCs : now;
    if (! hasGeneratedFrom(cut))
        return; // nothing generated from the cut on: nothing to cut
    dropOutputHistoryFrom(cut);
    pushOutputToken(Token{cut, static_cast<int32_t>(DenseVocab::DurOffset),
                          static_cast<int32_t>(Vocab::ClearQueue)});
    rewindGeneratedFrom(cut);
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
        forEachStage([&](AnticipationStage &st) {
            st.stream.addControl({in.time, held ? 0 : durCs, pitch, velocity}, held);
        });

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
        forEachStage([&](AnticipationStage &st) {
            st.stream.confirmControl(update.oldNote.time, pitch, finalDur);
        });

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
    forEachStage([now](AnticipationStage &st) { st.stream.commitLongHolds(now); });
    return applied;
}

// ---- sampling ----

auto TwoInstrumentAnticipatedReductionTransformer::generateOne(int32_t now) -> std::optional<Token> {
    if (! isModelLoaded())
        return std::nullopt;

    AnticipationStage::Params params;
    params.now = now;
    // Earliest onset: the event clock, never in the past, raised by the polyphony cap.
    params.lowerAbs = std::max(onsetLowerBoundCs(), now);
    params.temperatures = {getOnsetTemperature(), getDurationTemperature(), getNoteTemperature(),
                           getVelocityTemperature()};
    params.pitchBias = isPitchBiasEnabled();
    params.durationBias = isDurationBiasEnabled();
    params.onsetGapBias = isOnsetGapBiasEnabled();
    params.pitchTau = getPitchTauBase();
    params.pitchTauPrime = getPitchTauPrimeBase();
    params.durationTau = getDurationTauBase();
    params.durationTauPrime = getDurationTauPrimeBase();
    params.onsetGapTau = getOnsetGapTauBase();
    params.onsetOptions.shortGapPenalty = isShortGapPenaltyEnabled();
    params.onsetOptions.logitTd = isLogitTdEnabled();
    // Accompaniment: late input reordered in at most once a second; Edit: at once (the notes
    // generated in the 2 s its input keeps stay, with the input among them in onset order).
    params.reorderLate = keepsQueueOnInput() || isEditMode();
    params.reorderEveryCs = isEditMode() ? 0 : AnticipationStage::kReorderEveryCs;
    params.waitMinPlay = getWaitMinPlay();
    params.biases = biasesActiveNow();
    params.pitchLow = RegisterRange::lowPitch();
    params.pitchHigh = RegisterRange::highPitch();

    // Hybrid: v1 with the T3'' rules (T 0.5) and its own adjustments (all on by default).
    const bool silent = useSilentStage();
    auto &active = silent ? silentStage : stage;
    if (silent) {
        AnticipationStage::Params t3; // base taus
        t3.now = params.now;
        t3.lowerAbs = params.lowerAbs;
        t3.temperatures = {DenseSampling::OnsetTemperature, DenseSampling::DurationTemperature,
                           DenseSampling::NoteTemperature, DenseSampling::OnsetTemperature};
        t3.pitchBias = hybridV1PitchBias.load();
        t3.durationBias = hybridV1DurationBias.load();
        t3.onsetGapBias = hybridV1OnsetGapBias.load();
        t3.onsetOptions.shortGapPenalty = hybridV1ShortGapPenalty.load();
        t3.onsetOptions.logitTd = hybridV1LogitTd.load();
        t3.reorderLate = params.reorderLate;
        t3.reorderEveryCs = params.reorderEveryCs;
        t3.biases = params.biases; // the global switch and the silence gate
        t3.pitchLow = params.pitchLow;
        t3.pitchHigh = params.pitchHigh;
        params = t3;
    }
    const auto note = active.generate(params);
    if (! note.has_value()) {
        if (! active.lastError.empty())
            reportSamplingError("Anticipated reduction: " + juce::String(active.lastError));
        return std::nullopt;
    }
    if (variant == Variant::Hybrid) {
        (silent ? stage : silentStage).mirrorEvent(*note, now, params.reorderLate);
        lastFromSilent = silent;
    }
    if (SessionLog::isRunning())
        logModelInput(active, silent ? loggedV1 : loggedMain,
                      variant == Variant::Hybrid ? (silent ? "v1" : "W") : (variant == Variant::Wait ? "W" : "T3"));

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
    if (clockJumpedBack(now))
        restartOnClockJumpBack(now);
    if (const auto cut = consumeModeCut())
        rewindGeneratedFrom(*cut);
    if (drainInput(now) && ! keepsQueueOnInput() && ! generationPause.get())
        cutFutureOnInput(now);
    updateHybrid(now);
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
    forgetClock();
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
        if (clockJumpedBack(now))
            restartOnClockJumpBack(now);
        const bool inputApplied = drainInput(now);
        // Jam: react to the input now (also ends an ahead throttle). Not under the pedal: its
        // press already cut the future, and the model must stay paused.
        if (inputApplied && ! keepsQueueOnInput() && ! generationPause.get())
            cutFutureOnInput(now);
        updateHybrid(now);

        if (isGenerationStopped()) {
            maybeEmitGenerationPauseSoftStopClear();
            if (! inputApplied) {
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

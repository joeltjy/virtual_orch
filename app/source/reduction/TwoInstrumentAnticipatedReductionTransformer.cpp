#include "VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h"

#include <set>
#include "VirtualOrch/AppTime.h"

#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/PolyphonyCap.h"
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/AppTime.h"

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
        // simultaneous_model_sampling.txt: WAIT, 5 cs merge, piano first + ascending ties, onset T 1.0.
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
        if (OpLog::on()) {
            const bool waitDecision = std::string(what) == "wait" || std::string(what) == "wait-given";
            OpLog::emit(OpLog::Src::Rt, "RT_WAIT",
                        {{"model", "W"}, {"decision", what}, {"at", value},
                         {"p_wait", waitDecision ? OpLog::Value(static_cast<double>(stage.sampler.modelWaitProbability)) : OpLog::Value()},
                         {"p_wait_recorded",
                          waitDecision && stage.sampler.lastWaitProbability != stage.sampler.modelWaitProbability
                              ? OpLog::Value(static_cast<double>(stage.sampler.lastWaitProbability))
                              : OpLog::Value()}});
        }
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
    contextLogMain = {};
    contextLogV1 = {};
    removalReason = "restart";
    SessionLog::add("RT_IN", getThreadName() + " | stream reset at clock " + juce::String(now)
                                 + ", base " + juce::String(stage.stream.base()));
    playingSinceMs = -1;
    hybridPlayingFor.store(-1);
    hybridV1.store(true); // a (re)start begins with v1
    hybridReasonValue.store(HybridReason::Start);
    tdResetRequested.store(true); // a restart starts from gamma 0, not the last session's
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
    if (OpLog::on())
        OpLog::emit(OpLog::Src::Rt, "RT_SWITCH", {{"to", v1 ? "v1" : "W"}, {"reason", kWhy[static_cast<int>(why)]}});
    SessionLog::add("HYBRID", juce::String(v1 ? hybridV1Name : hybridWaitName) + " from "
                                  + juce::String(nowCs) + " (" + kWhy[static_cast<int>(why)] + ")");
}

auto TwoInstrumentAnticipatedReductionTransformer::updateHybrid(int32_t nowCs) -> void {
    if (variant != Variant::Hybrid)
        return;
    const int64_t wallMs = wallMsProvider ? wallMsProvider() : AppTime::nowMs();
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
        cutGeneratedFrom(nowCs, "hybrid: silent 0.5 s");
        setHybridModel(true, HybridReason::Stopped, nowCs);
    } else if (! generationPause.get() && predictsSilenceFrom(nowCs)) {
        // W2/W3 predicts 2 s of silence: v1 fills it (W2/W3 again after 1 more s of playing).
        cutGeneratedFrom(nowCs, "hybrid: 2 s of predicted silence");
        setHybridModel(true, HybridReason::PredictedSilence, nowCs);
        playingSinceMs = wallMs;
        hybridPlayingFor.store(0);
    }
}

auto TwoInstrumentAnticipatedReductionTransformer::cutGeneratedFrom(int32_t nowCs, const std::string &reason) -> void {
    resetAheadThrottle();
    if (! hasGeneratedFrom(nowCs))
        return;
    removalReason = reason;
    dropOutputHistoryFrom(nowCs, reason);
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
    removalReason = "clock jumped back";
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
    std::string reason = "input";
    for (const auto k: drainedKeys)
        reason += " " + OpLog::id('k', k);
    if (isEditMode())
        reason += " (Step-in: next 2 s kept)";
    removalReason = reason;
    dropOutputHistoryFrom(cut, reason);
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
    drainedKeys.clear();
    std::vector<int32_t> releasedKeys;
    Token in{-1, -1, -1};
    while (inputTokenQueue.pull(in)) {
        drainedKeys.push_back(in.keyId);
        if (OpLog::on())
            OpLog::emit(OpLog::Src::Rt, "Q_POP", {{"queue", "input→rt"}, {"item", OpLog::id('k', in.keyId)}});
        const int32_t pitch = DenseQuantize::snapPianoPitch(in.getPitch());
        const int32_t velocity = DenseQuantize::snapVelocity(
            juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                         in.velocity > 0 ? in.velocity : DenseConfig::DefaultVelocity));
        const bool held = DenseSampling::isProvisionalDurationToken(in.duration,
                                                                    modelConfig.inputDuration);
        const int32_t durCs = DenseQuantize::snapDurationCs(in.getRealDuration());
        AnticipatedStream::Note given{in.time, held ? 0 : durCs, pitch, velocity};
        given.id = in.keyId;
        forEachStage([&](AnticipationStage &st) { st.stream.addControl(given, held); });

        // The UI / filters keep reading the user's notes as instrument-0 dense quads.
        const auto quad = dense(in.time, DenseQuantize::snapDurationCs(in.getRealDuration()),
                                DenseConfig::PianoReductionInputInstrument, pitch, velocity);
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        applied = true;
    }

    TokenUpdate update{};
    bool updated = false;
    while (updatesFromFilter.pull(update)) {
        releasedKeys.push_back(update.oldNote.keyId);
        if (OpLog::on())
            OpLog::emit(OpLog::Src::Rt, "Q_POP", {{"queue", "updates→rt"}, {"item", OpLog::id('k', update.oldNote.keyId)}});
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
    if (OpLog::on() && (! drainedKeys.empty() || ! releasedKeys.empty())) {
        OpLog::Strings keys, releases;
        for (const auto k: drainedKeys)
            keys.push_back(OpLog::id('k', k));
        for (const auto k: releasedKeys)
            releases.push_back(OpLog::id('k', k));
        OpLog::emit(OpLog::Src::Rt, "RT_DRAIN", {{"keys", std::move(keys)}, {"releases", std::move(releases)}});
    }
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
    params.onsetOptions.gamma = tdGamma(false);
    // Accompaniment: late input reordered in at most once a second; Edit: at once (the notes
    // generated in the 2 s its input keeps stay, with the input among them in onset order).
    params.reorderLate = keepsQueueOnInput() || isEditMode();
    params.reorderEveryCs = isEditMode() ? 0 : AnticipationStage::kReorderEveryCs;
    params.waitMinPlay = getWaitMinPlay();
    params.biases = biasesActiveNow();
    params.pitchHigh = RegisterRange::highPitch();
    params.pitchLow = PolyphonyCap::lowestPitch(getMaxPolyphony(), RegisterRange::lowPitch(), params.pitchHigh); // Mono: above C4

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
        t3.onsetOptions.gamma = tdGamma(true);
        t3.reorderLate = params.reorderLate;
        t3.reorderEveryCs = params.reorderEveryCs;
        t3.biases = params.biases; // the global switch and the silence gate
        t3.pitchLow = params.pitchLow;
        t3.pitchHigh = params.pitchHigh;
        params = t3;
    }
    auto note = active.generate(params);
    if (! note.has_value()) {
        if (! active.lastError.empty())
            reportSamplingError("Anticipated reduction: " + juce::String(active.lastError));
        return std::nullopt;
    }
    // The note id is assigned now (the same moment the note goes out live), so the context and the
    // operation log can name it before a replay delivers it.
    auto drawn = *note;
    drawn.id = nextNoteId.fetch_add(1, std::memory_order_relaxed);
    active.stream.setLastEventId(drawn.id);
    {
        const std::lock_guard<std::mutex> lock(noteModelsMutex);
        noteModels[drawn.id] = silent ? "v1" : (variant == Variant::Hybrid || variant == Variant::Wait ? "W" : "T3");
    }
    if (OpLog::on()) {
        std::string fields = "[";
        for (size_t i = 0; i < active.sampler.lastDraw.size(); ++i)
            fields += (i ? "," : "") + active.sampler.lastDraw[i].json();
        fields += "]";
        OpLog::emit(OpLog::Src::Rt, "RT_DRAW",
                    {{"model", silent ? "v1" : (variant == Variant::T3 ? "T3" : "W")}, {"r", OpLog::id('r', drawn.id)},
                     {"now", now}, {"floor", params.lowerAbs}, {"base", active.stream.base()},
                     {"context_len", active.stream.tokens().size()}, {"fields", OpLog::Raw{fields}}});
    }
    if (variant == Variant::Hybrid) {
        (silent ? stage : silentStage).mirrorEvent(drawn, now, params.reorderLate);
        lastFromSilent = silent;
    }
    if (SessionLog::isRunning())
        logModelInput(active, silent ? loggedV1 : loggedMain,
                      variant == Variant::Hybrid ? (silent ? "v1" : "W") : (variant == Variant::Wait ? "W" : "T3"));

    // Sympathetic tempo: the output and inputData get the warped note. The anticipation stage's
    // own KV stream already holds the unwarped draw (see README.md).
    applySympatheticWarp(note->onsetCs, note->durationCs);
    const auto quad = dense(note->onsetCs, note->durationCs, AnticipatedStream::kEventInstrument,
                            note->pitch, note->velocity);
    inputData.insert(inputData.end(), quad.begin(), quad.end());
    notifyInputDataChanged();

    currentTime = note->onsetCs;
    logGeneratedTokenIfResumed(note->onsetCs);

    Token outbound{note->onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + note->durationCs,
                   static_cast<int32_t>(DenseVocab::NoteOffset) + note->pitch, note->velocity};
    outbound.noteId = drawn.id;
    pushOutputToken(outbound);

    // The ahead check belongs to the moment the note goes out (a replay may deliver it later).
    if (! lastOutputHeld)
        afterOutputDelivered(outbound);
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
    beginLive();

    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        const auto result = liveStep();
        if (result == LiveStep::Idle || result == LiveStep::NothingGenerated)
            wait(5);
        recordThreadLoopMs(loopStartMs);
    }
}

auto TwoInstrumentAnticipatedReductionTransformer::tdGamma(bool silentModel) const -> std::optional<float> {
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    if (td == TemporalDensity::kAuto)
        return std::nullopt;
    return static_cast<float>((silentModel ? tdControlV1 : tdControlMain).current().gamma);
}

auto TwoInstrumentAnticipatedReductionTransformer::tickTdControl(int32_t now) -> void {
    // Before the tick's rate limit, so the next draw already uses gamma 0.
    if (tdResetRequested.exchange(false)) {
        tdControlMain.reset();
        tdControlV1.reset();
        lastTdTickMs = AppTime::hiResMs(); // the first update after a full tick of new notes
        if (OpLog::on())
            OpLog::emit(OpLog::Src::Rt, "TD_CTRL_RESET", {});
    }
    const double nowMs = AppTime::hiResMs();
    if (lastTdTickMs >= 0.0 && nowMs - lastTdTickMs < TdController::kTickMs)
        return;
    lastTdTickMs = nowMs;
    juce::ignoreUnused(now);

    // The orchestra's density: distinct onsets of the generated notes in the 3 s ending at the
    // newest one (the generation frontier, so what was just drawn counts at once).
    std::optional<double> measured;
    {
        const auto history = getOutputHistorySince(INT32_MIN);
        int32_t frontier = INT32_MIN;
        for (const auto &t: history)
            if (t.note >= static_cast<int32_t>(Vocab::NoteOffset) && t.note < static_cast<int32_t>(Vocab::Rest))
                frontier = std::max(frontier, t.time);
        if (frontier != INT32_MIN) {
            std::set<int32_t> onsets;
            for (const auto &t: history)
                if (t.note >= static_cast<int32_t>(Vocab::NoteOffset) && t.note < static_cast<int32_t>(Vocab::Rest)
                    && t.time > frontier - TdController::kWindowCs && t.time <= frontier)
                    onsets.insert(t.time);
            measured = TdController::measure(static_cast<int>(onsets.size()));
        }
    }
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    const bool clockRunning = clock == nullptr || clock->isRunning();
    std::vector<TdControlView> views;
    const auto tickOne = [&](TdController &controller, AnticipationStage &st, const char *model, bool waiting) {
        const auto [sensitivity, draws] = st.sampler.takeSensitivity();
        juce::ignoreUnused(draws);
        const auto &state = controller.tick({td, measured, sensitivity, clockRunning, waiting});
        views.push_back({model, state});
        if (OpLog::on() && td != TemporalDensity::kAuto)
            OpLog::emit(OpLog::Src::Rt, "TD_CTRL",
                        {{"model", model}, {"td", td}, {"measured", state.measured}, {"error", state.error},
                         {"gamma", state.gamma}, {"k", state.k}, {"s_hat", state.sensitivity}, {"m", state.m},
                         {"frozen", state.frozen},
                         {"why", state.why.empty() ? OpLog::Value() : OpLog::Value(state.why)}});
    };
    tickOne(tdControlMain, stage, variant == Variant::T3 ? "T3" : "W", stage.lastWaited());
    if (variant == Variant::Hybrid)
        tickOne(tdControlV1, silentStage, "v1", false);
    const std::lock_guard<std::mutex> lock(tdViewMutex);
    tdViews = std::move(views);
}

auto TwoInstrumentAnticipatedReductionTransformer::liveStep() -> LiveStep {
    finishAheadThrottleIfDue();

    // Leaving Accompaniment: drop the generated notes from the cut on.
    if (const auto cut = consumeModeCut()) {
        removalReason = lastModeCutReason();
        rewindGeneratedFrom(*cut);
    }

    const int32_t now = nowCs();
    if (clockJumpedBack(now))
        restartOnClockJumpBack(now);
    const bool inputApplied = drainInput(now);
    struct LogAtEnd {
        TwoInstrumentAnticipatedReductionTransformer &t;
        int32_t now;
        ~LogAtEnd() {
            if (! OpLog::on())
                return;
            t.logContexts();
            t.logCheckpoint(now);
        }
    } logAtEnd{*this, now};
    // Jam: react to the input now (also ends an ahead throttle). Not under the pedal: its
    // press already cut the future, and the model must stay paused.
    if (inputApplied && ! keepsQueueOnInput() && ! generationPause.get())
        cutFutureOnInput(now);
    updateHybrid(now);
    tickTdControl(now);

    if (isGenerationStopped()) {
        maybeEmitGenerationPauseSoftStopClear();
        return inputApplied ? LiveStep::StoppedWithInput : LiveStep::Idle;
    }

    wasGenerationPaused = false;
    return generateOne(now).has_value() ? LiveStep::Generated : LiveStep::NothingGenerated;
}

auto TwoInstrumentAnticipatedReductionTransformer::beginLive() -> void {
    // What threadRun does before its loop.
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
}


// ---- operation log

auto TwoInstrumentAnticipatedReductionTransformer::modelOfNote(int32_t noteId) -> std::string {
    const std::lock_guard<std::mutex> lock(noteModelsMutex);
    const auto it = noteModels.find(noteId);
    if (it == noteModels.end())
        return {};
    auto model = it->second;
    noteModels.erase(it);
    return model;
}

namespace {

auto kindName(char k) -> const char * {
    switch (k) {
        case 'g': return "given";
        case 'e': return "gen";
        case 't': return "td";
        case 'w': return "wait";
        case 'r': return "rest";
        default: return "control";
    }
}

auto itemId(const AnticipatedStream::ItemView &v) -> std::string {
    if (v.id < 0)
        return {};
    return OpLog::id(v.kind == 'g' ? 'k' : 'r', v.id);
}

/** The same item (its identity, not its exact tokens: lengths change and a rebase re-encodes onsets). */
auto sameItem(const AnticipatedStream::ItemView &a, const AnticipatedStream::ItemView &b) -> bool {
    if (a.kind != b.kind || a.onsetCs != b.onsetCs)
        return false;
    if (a.kind == 't')
        return a.tokens == b.tokens;
    if (a.kind == 'w')
        return true;
    if (a.id >= 0 || b.id >= 0)
        return a.id == b.id && a.pitch == b.pitch;
    return a.pitch == b.pitch && a.instrument == b.instrument;
}

auto positions(const std::vector<AnticipatedStream::ItemView> &v) -> std::vector<size_t> {
    std::vector<size_t> pos(v.size() + 1, 0);
    for (size_t i = 0; i < v.size(); ++i)
        pos[i + 1] = pos[i] + v[i].tokens.size();
    return pos;
}

auto itemsJson(const std::vector<AnticipatedStream::ItemView> &v) -> std::string {
    const auto pos = positions(v);
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        OpLog::Obj o;
        o.add("pos", static_cast<int64_t>(pos[i])).add("kind", std::string(kindName(v[i].kind)));
        if (v[i].id >= 0)
            o.add("id", itemId(v[i]));
        s += (i ? "," : "") + o.str();
    }
    return s + "]";
}

} // namespace

auto TwoInstrumentAnticipatedReductionTransformer::logContext(const AnticipationStage &st, ContextLog &log,
                                                              const char *model) -> void {
    auto now = st.stream.itemViews();
    const int32_t base = st.stream.base();
    auto &prev = log.prev;
    if (log.base != INT32_MIN && base != log.base) {
        // A rebase drops the oldest items (the cache keeps 4 + 508 rows) and re-encodes onsets.
        size_t dropped = 0;
        if (! now.empty())
            while (dropped < prev.size() && ! sameItem(prev[dropped], now.front()))
                ++dropped;
        else
            dropped = prev.size();
        size_t droppedTokens = 0;
        for (size_t i = 0; i < dropped; ++i)
            droppedTokens += prev[i].tokens.size();
        OpLog::emit(OpLog::Src::Rt, "RT_REBASE",
                    {{"model", model}, {"old_base", log.base}, {"new_base", base},
                     {"dropped", static_cast<int64_t>(droppedTokens)}, {"dropped_items", static_cast<int64_t>(dropped)}});
        prev.erase(prev.begin(), prev.begin() + static_cast<std::ptrdiff_t>(dropped));
        OpLog::requestCheckpoints();
    }
    const auto prevPos = positions(prev);
    const auto nowPos = positions(now);
    size_t a = 0;
    while (a < prev.size() && a < now.size() && sameItem(prev[a], now[a])) {
        const auto &o = prev[a];
        const auto &n = now[a];
        if (o.durationCs != n.durationCs || o.velocity != n.velocity || o.provisional != n.provisional) {
            std::string reason = o.provisional && ! n.provisional
                                     ? "key released: length " + std::to_string(n.durationCs) + " cs"
                                     : (n.provisional ? "held key: length so far " + std::to_string(n.durationCs) + " cs"
                                                      : "length " + std::to_string(n.durationCs) + " cs");
            OpLog::emit(OpLog::Src::Rt, "RT_REWRITE",
                        {{"model", model}, {"pos", static_cast<int64_t>(nowPos[a])}, {"id", itemId(n)},
                         {"old", o.tokens}, {"new", n.tokens}, {"reason", reason}});
        }
        ++a;
    }
    size_t b = 0;
    while (b < prev.size() - a && b < now.size() - a && sameItem(prev[prev.size() - 1 - b], now[now.size() - 1 - b]))
        ++b;
    const size_t removedEnd = prev.size() - b;
    const size_t addedEnd = now.size() - b;
    if (a < removedEnd) {
        std::string items = "[";
        for (size_t i = a; i < removedEnd; ++i) {
            OpLog::Obj o;
            o.add("pos", static_cast<int64_t>(prevPos[i])).add("kind", std::string(kindName(prev[i].kind)));
            if (prev[i].id >= 0)
                o.add("id", itemId(prev[i]));
            items += (i > a ? "," : "") + o.str();
        }
        items += "]";
        OpLog::emit(OpLog::Src::Rt, "RT_REMOVE",
                    {{"model", model}, {"items", OpLog::Raw{items}},
                     {"reason", removalReason.empty() ? std::string("reordered") : removalReason}});
    }
    for (size_t i = a; i < addedEnd; ++i) {
        const auto &n = now[i];
        OpLog::emit(OpLog::Src::Rt, "RT_ADD",
                    {{"model", model}, {"pos", static_cast<int64_t>(nowPos[i])}, {"kind", kindName(n.kind)},
                     {"id", itemId(n)}, {"onset", n.onsetCs},
                     {"dur", n.kind == 'g' || n.kind == 'e' ? OpLog::Value(static_cast<int64_t>(n.durationCs)) : OpLog::Value()},
                     {"pitch", n.kind == 'g' || n.kind == 'e' ? OpLog::Value(static_cast<int64_t>(n.pitch)) : OpLog::Value()},
                     {"vel", n.kind == 'g' || n.kind == 'e' ? OpLog::Value(static_cast<int64_t>(n.velocity)) : OpLog::Value()},
                     {"held", n.provisional ? OpLog::Value(true) : OpLog::Value()}, {"tokens", n.tokens}});
    }
    prev = std::move(now);
    log.base = base;
}

auto TwoInstrumentAnticipatedReductionTransformer::logContexts() -> void {
    logContext(stage, contextLogMain, variant == Variant::T3 ? "T3" : "W");
    if (variant == Variant::Hybrid)
        logContext(silentStage, contextLogV1, "v1");
    removalReason.clear();
}

auto TwoInstrumentAnticipatedReductionTransformer::logCheckpoint(int32_t now) -> void {
    const double t = static_cast<double>(AppTime::hiResMs());
    juce::ignoreUnused(now);
    if (! OpLog::checkpointDue(OpLog::Src::Rt, t))
        return;
    OpLog::markCheckpoint(OpLog::Src::Rt, t);
    const auto part = [](const AnticipationStage &st) {
        OpLog::Obj o;
        auto ids = st.stream.pendingIds();
        OpLog::Strings pending;
        for (const auto k: ids)
            pending.push_back(OpLog::id('k', k));
        const auto views = st.stream.itemViews();
        o.add("base", static_cast<int64_t>(st.stream.base())).add("tokens", st.stream.tokens())
            .add("items", OpLog::Raw{itemsJson(views)}).add("pending", pending);
        return o.str();
    };
    OpLog::Obj models;
    models.add(variant == Variant::T3 ? "T3" : "W", OpLog::Raw{part(stage)});
    if (variant == Variant::Hybrid)
        models.add("v1", OpLog::Raw{part(silentStage)});
    OpLog::Obj state;
    state.add("td", static_cast<int64_t>(TemporalDensity::get())).add("register", static_cast<int64_t>(RegisterRange::get()))
        .add("polyphony", static_cast<int64_t>(getMaxPolyphony())).add("paused", generationPause.get())
        .add("throttled", overflowPause.get()).add("step_in", isEditMode()).add("accompaniment", keepsQueueOnInput())
        .add("hybrid_model", std::string(variant == Variant::Hybrid ? (hybridV1.load() ? "v1" : "W") : ""));
    OpLog::emit(OpLog::Src::State, "CHECKPOINT",
                {{"part", "rt"}, {"rt", models.raw()}, {"state", state.raw()}});
}

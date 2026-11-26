#include "VirtualOrch/reduction/HybridChordTokenReductionTransformer.h"

#include "VirtualOrch/KeyLanding.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/PolyphonyCap.h"
#include "VirtualOrch/RegisterMix.h"
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/reduction/DenseTypes.h"

#include <algorithm>
#include <random>
#include <set>

namespace {

constexpr int kEventWidth = 4;

auto denseQuad(int32_t time, int32_t durCs, int32_t instrument, int32_t pitch, int32_t velocity)
    -> std::array<int32_t, 4> {
    return {time, static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * instrument + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity};
}

auto loadOnnx(const char *path, std::vector<size_t> vocabs, std::string &error)
    -> std::unique_ptr<StreamingKv::OnnxBackend> {
    auto onnx = std::make_unique<StreamingKv::OnnxBackend>();
    if (! onnx->init(path, error))
        return nullptr;
    if (std::find(vocabs.begin(), vocabs.end(), onnx->vocabSize()) == vocabs.end()) {
        error = std::string(path) + ": vocab " + std::to_string(onnx->vocabSize()) + ", expected " + std::to_string(vocabs.front())
                + (vocabs.size() > 1 ? " or " + std::to_string(vocabs.back()) : std::string());
        return nullptr;
    }
    return onnx;
}

} // namespace

HybridChordTokenReductionTransformer::HybridChordTokenReductionTransformer(ModelConfig &modelConfigIn)
    : ReductionTransformer("Reduction Transformer Hybrid v1.2", modelConfigIn) {
    const auto logger = [this](const char *who) {
        return [this, who](const ChordTokenEngine::Event &e) {
            if (SessionLog::isRunning())
                SessionLog::add("IWAIT", juce::String(who) + " | " + juce::String(e.what) + " @" + juce::String(e.atCs)
                                             + " " + juce::String(e.a) + " " + juce::String(e.b)
                                             + (e.text.empty() ? juce::String() : " (" + juce::String(e.text) + ")"));
        };
    };
    silentEngine.onEvent = logger("m1");
    waitEngine.onEvent = [this, log = logger("m2")](const ChordTokenEngine::Event &e) {
        log(e);
        waitTracker.onEvent(e, nowCs());
        logWaitEvent(e);
    };
    tdValueProvider = [] { return TemporalDensity::get(); };
}

auto HybridChordTokenReductionTransformer::init(const char *silentModelPath, const char *waitModelPath) -> bool {
    lastInitError.clear();
    std::string error;
    auto silent = loadOnnx(silentModelPath, {static_cast<size_t>(ChordTokenEngine::kWait0)}, error);
    if (silent == nullptr) {
        lastInitError = "silent-side model (overnight model 1, vocab 11768): " + error;
        reportSamplingError(juce::String(label()) + ": " + juce::String(lastInitError));
        return false;
    }
    // model 2: the chord-token model (vocab 14071)
    auto wait = loadOnnx(waitModelPath, {static_cast<size_t>(ChordTokenEngine::kChordVocab)}, error);
    if (wait == nullptr) {
        lastInitError = "playing-side model (chord-token, vocab 14071): " + error;
        reportSamplingError(juce::String(label()) + ": " + juce::String(lastInitError));
        return false;
    }
    chordModel.store(wait->vocabSize() == static_cast<size_t>(ChordTokenEngine::kChordVocab));
    if (chordModel.load() && (voicings == nullptr || voicings->empty()))
        SessionLog::add("HYBRID", "v1.2: no voicing bank loaded: CHORD blocks are decoded as chord tones stacked from the bass");
    silentName = juce::File(silentModelPath).getFileNameWithoutExtension().toStdString();
    waitName = juce::File(waitModelPath).getFileNameWithoutExtension().toStdString();
    initWithBackends(std::move(silent), std::move(wait));
    return true;
}

auto HybridChordTokenReductionTransformer::loadVoicingBank(const std::string &path) -> bool {
    const juce::File file(path);
    std::string error;
    auto bank = std::make_shared<VoicingBank>();
    if (! file.existsAsFile()) {
        error = "voicing bank not found: " + path;
    } else if (bank->loadJson(file.loadFileAsString().toStdString(), error)) {
        voicings = std::move(bank);
        SessionLog::add("HYBRID", "voicing bank " + juce::String(path) + ": " + juce::String(static_cast<int>(voicings->size())) + " keys");
        return true;
    }
    voicings.reset();
    lastInitError = error;
    SessionLog::add("HYBRID", juce::String(error));
    return false;
}

auto HybridChordTokenReductionTransformer::initWithBackends(std::unique_ptr<StreamingKv::Backend> silent,
                                                                 std::unique_ptr<StreamingKv::Backend> wait) -> void {
    silentBackend = std::move(silent);
    waitBackend = std::move(wait);
    silentEngine.setBackend(silentBackend.get());
    waitEngine.setBackend(waitBackend.get());
}

auto HybridChordTokenReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto HybridChordTokenReductionTransformer::applySettings() -> void {
    auto s = silentEngine.settings();
    s.continuation = true;
    s.mode = ChordTokenEngine::Mode::Unconstrained;
    s.precompute = false;
    // As the legacy Hybrid (Nov 24 2026): its input modes, polyphony cap, held keys rewritten at their
    // release (committed after 10 s), 10-500 cs durations, at most 10 notes a chord, no chord note in the
    // past, the global ahead throttle instead of a per-engine limit; a random seed per run. Sampling
    // (temperatures, min-p, top-p) stays the Python sampler's, as the models were evaluated with.
    const auto inputCut = isEditMode()         ? ChordTokenEngine::Settings::InputCut::Edit
                          : keepsQueueOnInput() ? ChordTokenEngine::Settings::InputCut::Accompaniment
                                                : ChordTokenEngine::Settings::InputCut::Jam;
    const int32_t minOnset = onsetLowerBoundCs();
    for (auto *x: {&s}) {
        x->inputCut = inputCut;
        x->editKeepCs = kEditCutAheadCs;
        x->minOnsetCs = minOnset;
        x->rewriteHeldDurations = true;
        x->holdCommitCs = 1000;
        x->minDurationCs = 10;
        x->maxDurationCs = 500;
        x->maxChordNotes = 10;
        x->chordLateCs = 0;
        x->drawAheadCs = 1 << 20; // no per-engine limit: the ahead throttle (outputMaxAheadSeconds)
    }
    s.ascendingChords = false; // v1's T3'' rules: no order at a shared onset
    s.maxAheadCs = ChordTokenEngine::kOnsetMax; // v1: no per-draw limit
    s.seed = silentSeed;
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    s.tdLevel = td;
    s.tdGamma = tdGamma(true);
    const bool biases = biasesActiveNow(); // BiasSettings: the switch and the piano-silence gate
    const int32_t high = RegisterRange::highPitch();
    const int32_t low = PolyphonyCap::lowestPitch(getMaxPolyphony(), RegisterRange::lowPitch(), high); // Mono: above C4
    // model 1: the silent side's adjustments, with the base taus (the legacy Hybrid's v1)
    s.pitchBias = biases && hybridV1PitchBias.load();
    s.registerMix = biases ? RegisterMix::Target::get() : std::nullopt; // the mood preset's register mix
    s.durationBias = biases && hybridV1DurationBias.load() && durationBiasAtTd(td);
    s.onsetGapBias = biases && hybridV1OnsetGapBias.load();
    s.shortGap = hybridV1ShortGapPenalty.load();
    s.logitTd = hybridV1LogitTd.load();
    // a polyphony cap: the density follows the durations, so they get the TD taper too (Nov 25 2026)
    s.tdDurationTaper = s.logitTd && getMaxPolyphony() > 0 && td > 4;
    s.pitchTau = hybridV1PitchTau.load();
    s.pitchTauPrime = hybridV1PitchTauPrime.load();
    s.durationTau = hybridV1DurationTau.load();
    s.durationTauPrime = hybridV1DurationTauPrime.load();
    s.onsetGapTau = hybridV1OnsetGapTau.load();
    s.pitchLow = low;
    s.pitchHigh = high;
    silentEngine.setSettings(s);
    auto w = waitEngine.settings();
    w.continuation = false;
    w.mode = static_cast<ChordTokenEngine::Mode>(waitMode.load());
    w.precompute = precompute.load();
    w.seed = waitSeed;
    w.inputCut = inputCut;
    w.editKeepCs = kEditCutAheadCs;
    w.minOnsetCs = minOnset;
    w.rewriteHeldDurations = true;
    w.holdCommitCs = 1000;
    w.minDurationCs = 10;
    w.maxDurationCs = 500;
    w.maxChordNotes = 10;
    w.chordLateCs = 0;
    w.drawAheadCs = 1 << 20;
    w.noWaitAfterExpiry = true; // no retry WAITs after an expiry: m2 plays on until the next piano note
    w.fireWindowCs = 6;         // +-6 cs, as in training
    w.waitTopK = 3;             // a piano onset at any of the model's 3 most likely WAIT lengths fires
    w.minWaitCs = minWaitCs.load(); // the checkpoint JSON's "minWaitCs" (0 = any length)
    w.startDelayCs = startDelayMs.load() / 10; // ... and "startDelayMs" (v1.2 default 3000)
    w.shortWaitRedraw = shortWaitRedraw.load(); // ... and "shortWaitRedraw"
    w.expiryGraceCs = 20;       // live input arrives ~12 cs late (p90 ~15): wait for it before expiring
    // a pitch outside the anticipated top 5 is decoded for the note played (not the rank-1 branch's):
    // with the substitution, half the fires in a B-flat/D passage played chords made for other pitches
    w.outsideTopFiveAsRankOne = false;
    w.maxFireNotes = getMaxPolyphony(); // monophonic (cap 1): a fire places one note, not COUNT
    w.ascendingChords = true; // W's rule: pitch >= the previous one at a shared onset
    w.maxAheadCs = 300;       // W: onsets at most 3 s ahead
    w.tdLevel = td;
    w.tdGamma = tdGamma(false);
    // model 2: the main adjustments (all OFF by default) with the widget's taus
    w.pitchBias = biases && isPitchBiasEnabled();
    w.registerMix = biases ? RegisterMix::Target::get() : std::nullopt; // the mood preset's register mix
    w.durationBias = biases && isDurationBiasEnabled() && durationBiasAtTd(td);
    w.onsetGapBias = biases && isOnsetGapBiasEnabled();
    w.shortGap = isShortGapPenaltyEnabled();
    w.logitTd = isLogitTdEnabled();
    w.tdDurationTaper = w.logitTd && getMaxPolyphony() > 0 && td > 4;
    w.pitchTau = getPitchTauBase();
    w.pitchTauPrime = getPitchTauPrimeBase();
    w.durationTau = getDurationTauBase();
    w.durationTauPrime = getDurationTauPrimeBase();
    w.coarseDurationBins = true; // model 2 only: durations biased in coarse bins (τ′ = the preset's durationTauPrime)
    w.onsetGapTau = getOnsetGapTauBase();
    w.pitchLow = low;
    w.pitchHigh = high;
    // v1.2: the chord-token model 2 (B3f): CHORD blocks, [LATE d] (at most 2, then give up), [DUR?]
    w.chordTokens = chordModel.load();
    w.voicings = voicings;
    w.maxLates = 2;
    waitEngine.setSettings(w);
}

auto HybridChordTokenReductionTransformer::resetStreams(int32_t now) -> void {
    silentEngine.reset(now);
    waitEngine.reset(now);
    waitTracker.clear();
    startedAtCs = now;
    streamsStarted = true;
    tdResetRequested.store(true); // a restart starts from gamma 0, not the last stream's
    currentTime = now;
    playingSinceMs = -1;
    silentActive.store(true);
    reasonValue.store(Reason::Start);
    SessionLog::add("HYBRID", juce::String(silentName) + " from " + juce::String(now) + " (start)");
}

auto HybridChordTokenReductionTransformer::drainInput() -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
    }
    bool applied = false;
    Token in{-1, -1, -1};
    // Accompaniment: the pianist is ignored. Their notes never reach either model's context (model 1
    // keeps continuing its own music; see updateHybrid); releases of keys pressed before still apply.
    if (keepsQueueOnInput()) {
        while (inputTokenQueue.pull(in)) {
        }
    }
    while (inputTokenQueue.pull(in)) {
        const int32_t pitch = DenseQuantize::snapPianoPitch(in.getPitch());
        const int32_t velocity = std::clamp(in.velocity > 0 ? in.velocity : DenseConfig::DefaultVelocity, 1, 127);
        const bool held = DenseSampling::isProvisionalDurationToken(in.duration, modelConfig.inputDuration);
        for (auto *e: {&silentEngine, &waitEngine}) {
            e->noteOn(in.time, pitch, velocity, KeyLanding::get(in.keyId));
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

auto HybridChordTokenReductionTransformer::setSilent(bool silent, Reason why, int32_t now) -> void {
    reasonValue.store(why);
    if (silentActive.exchange(silent) == silent)
        return;
    (silent ? waitEngine : silentEngine).interrupt(); // the model that stops drawing forgets its pending WAIT
    if (silent)
        waitTracker.cancel();
    if (! silent)
        waitSinceCs = now;
    static constexpr const char *kWhy[] = {"start", "pedal released", "playing 1 s", "silent 0.5 s",
                                           "2 s of predicted silence", "Accompaniment: the pianist is ignored"};
    SessionLog::add("HYBRID", juce::String(silent ? silentName : waitName) + " from " + juce::String(now) + " ("
                                  + kWhy[static_cast<int>(why)] + ")");
}

auto HybridChordTokenReductionTransformer::cutGeneratedFrom(int32_t now) -> void {
    resetAheadThrottle();
    if (! hasGeneratedFrom(now)) {
        resetGenerationClock(now); // also with nothing to cut: the bound may still be far ahead
        return;
    }
    dropOutputHistoryFrom(now);
    pushOutputToken(Token{now, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
    silentEngine.dropGeneratedFrom(now);
    waitEngine.dropGeneratedFrom(now);
    eraseStridedFrom(inputData, kEventWidth, now);
    notifyInputDataChanged();
    resetGenerationClock(now); // the generation clock goes back with the cut (see generate)
}

auto HybridChordTokenReductionTransformer::predictsSilenceFrom(int32_t now) const -> bool {
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

auto HybridChordTokenReductionTransformer::wallMs() const -> int64_t {
    return wallMsProvider ? wallMsProvider() : juce::Time::currentTimeMillis();
}

auto HybridChordTokenReductionTransformer::tdGamma(bool silentModel) const -> std::optional<float> {
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    if (td == TemporalDensity::kAuto)
        return std::nullopt;
    return static_cast<float>((silentModel ? tdControlSilent : tdControlWait).current().gamma);
}

auto HybridChordTokenReductionTransformer::tickTdControl() -> void {
    // Before the rate limit, so the next draw already uses gamma 0.
    if (tdResetRequested.exchange(false)) {
        tdControlSilent.reset();
        tdControlWait.reset();
        lastTdTickMs = wallMs(); // the first update after a full tick of new notes
        if (OpLog::on())
            OpLog::emit(OpLog::Src::Rt, "TD_CTRL_RESET", {});
    }
    if (tdSoftResetRequested.exchange(false)) {
        tdControlSilent.softReset();
        tdControlWait.softReset();
        if (OpLog::on())
            OpLog::emit(OpLog::Src::Rt, "TD_CTRL_SOFT_RESET",
                        {{"gamma_m1", tdControlSilent.current().gamma}, {"gamma_m2", tdControlWait.current().gamma}});
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
    // the orchestra after the OT, when it has a fresh reading (tremolo counts 8 a second there)
    bool measuredAfterOt = false;
    if (measuredTdProvider)
        if (const auto ot = measuredTdProvider()) {
            measured = *ot;
            measuredAfterOt = true;
        }
    const int32_t td = tdValueProvider ? tdValueProvider() : TemporalDensity::kAuto;
    const bool clockRunning = clock == nullptr || clock->isRunning();
    std::vector<TdControlView> views;
    const auto tickOne = [&](TdController &controller, ChordTokenEngine &engine, const char *model) {
        const auto [sensitivity, draws] = engine.takeSensitivity();
        juce::ignoreUnused(draws);
        // waiting: drew nothing this tick because a WAIT is pending (not "not drawing")
        const bool waiting = ! sensitivity.has_value() && engine.hasPendingWait();
        const auto &state = controller.tick({td, measured, sensitivity, clockRunning, waiting});
        views.push_back({model, state});
        if (OpLog::on()) // td 0 = AUTO (no target: the controller holds)
            OpLog::emit(OpLog::Src::Rt, "TD_CTRL",
                        {{"model", model}, {"td", td}, {"measured", state.measured},
                         {"measured_by", measuredAfterOt ? "ot" : "rt"}, {"error", state.error},
                         {"gamma", state.gamma}, {"k", state.k}, {"s_hat", state.sensitivity}, {"m", state.m},
                         {"frozen", state.frozen},
                         {"why", state.why.empty() ? OpLog::Value() : OpLog::Value(state.why)}});
    };
    tickOne(tdControlSilent, silentEngine, "m1");
    tickOne(tdControlWait, waitEngine, "m2");
    const std::lock_guard<std::mutex> lock(tdViewMutex);
    tdViews = std::move(views);
}

auto HybridChordTokenReductionTransformer::updateHybrid(int32_t now) -> void {
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
    playingForValue.store(playingSinceMs >= 0 ? wallMs - playingSinceMs : -1);
    if (keepsQueueOnInput()) {
        // Accompaniment: model 1 only, whatever the pianist does (model 2 follows the pianist). Its
        // notes so far stay (Accompaniment keeps the future); model 2's pending WAIT is dropped.
        if (! silentActive.load() || reasonValue.load() != Reason::Accompaniment)
            setSilent(true, Reason::Accompaniment, now);
        playingSinceMs = -1;
        playingForValue.store(-1);
        return;
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

auto HybridChordTokenReductionTransformer::generate(int32_t now) -> bool {
    // The settings again, after the mode cut and the hybrid switch of this iteration: the lower bound
    // (onsetLowerBoundCs) must be the one after their cuts, as the legacy Hybrid computes its draw
    // params right before drawing. With the one from the top of the loop, model 1 drew past the notes
    // just cut, seconds ahead, and model 2 was switched out at once (Nov 24 2026).
    applySettings();
    if (silentActive.load() && now < startedAtCs + silentStartDelayMs.load() / 10) {
        silentEngine.absorb(now); // model 1's start delay: it listens, draws nothing
        waitEngine.absorb(now);
        return false;
    }
    auto &active = silentActive.load() ? silentEngine : waitEngine;
    auto &other = silentActive.load() ? waitEngine : silentEngine;
    const bool logging = OpLog::on();
    silentEngine.logDraws = logging;
    waitEngine.logDraws = logging;
    const auto out = active.step(now);
    other.absorb(now);
    const int32_t cancel = active.takeCancelFrom();
    if (cancel != INT32_MAX) {
        dropOutputHistoryFrom(cancel);
        pushOutputToken(Token{cancel, static_cast<int32_t>(DenseVocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)});
        other.dropGeneratedFrom(cancel);
        eraseStridedFrom(inputData, kEventWidth, cancel);
        resetAheadThrottle(); // as the legacy cut on input
        // the generation clock (and so the polyphony lower bound, onsetLowerBoundCs) goes back with the cut:
        // left at the furthest onset ever drawn, every later note of model 1 was forced past it, tens of
        // seconds ahead, and model 2 was switched out at once for "2 s of predicted silence" (Nov 24 2026)
        resetGenerationClock(nowCs());
        notifyInputDataChanged();
    }
    for (const auto &n: out) {
        other.mirrorNote(n);
        const int32_t dur = std::clamp(n.durationCs, 1, 995);
        const auto quad = denseQuad(n.onsetCs, dur, 1, n.pitch, n.velocity);
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        // Model 2 plays while the pianist does: its notes go out louder so the orchestra is heard over
        // the piano (the context keeps the model's own velocity). Both models give 120 almost always,
        // so this is at most +7, to 127.
        const int32_t outVelocity = &active == &waitEngine ? std::min(127, n.velocity + kSimultaneousVelocityBoost) : n.velocity;
        Token token{n.onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + dur,
                    static_cast<int32_t>(DenseVocab::NoteOffset) + n.pitch, outVelocity};
        token.noteId = nextNoteId.fetch_add(1, std::memory_order_relaxed);
        if (logging)
            logDraw(active, &active == &silentEngine ? "m1" : "m2", n, token.noteId, now);
        pushOutputToken(token);
        afterOutputDelivered(token); // more than outputMaxAheadSeconds ahead: generation pauses (the legacy throttle)
        currentTime = std::max(currentTime, n.onsetCs);
        logGeneratedTokenIfResumed(n.onsetCs);
    }
    if (! out.empty())
        notifyInputDataChanged();
    if (! active.lastError.empty()) {
        reportSamplingError(juce::String(label()) + ": " + juce::String(active.lastError));
        active.lastError.clear();
    }
    return ! out.empty();
}

auto HybridChordTokenReductionTransformer::resetGenerationClock(int32_t now) -> void {
    // After a cut: the newest generated note left, or now (the legacy stream clock after a rewind).
    int32_t latest = INT32_MIN;
    for (const auto &t: getOutputHistory())
        if (t.note >= static_cast<int32_t>(Vocab::NoteOffset) && t.note < static_cast<int32_t>(Vocab::Rest))
            latest = std::max(latest, t.time);
    currentTime = std::max(latest, now);
}

auto HybridChordTokenReductionTransformer::rewindFrom(int32_t cut) -> void {
    // A mode cut (pedal up, TD / register press, preset, leaving Accompaniment): as the legacy Hybrid's
    // rewind, only the generated notes from the cut on go (consumeModeCut already cleared them
    // downstream); the context stays, so the next pedal press or TD change continues from it. It used
    // to restart both streams, and an empty stream waits for the pianist.
    for (auto *e: {&silentEngine, &waitEngine}) {
        e->interrupt(); // a pending WAIT, its anticipation and any stack being decoded
        e->dropGeneratedFrom(cut);
    }
    waitTracker.cancel();
    eraseStridedFrom(inputData, kEventWidth, cut);
    resetGenerationClock(std::min(nowCs(), cut));
    notifyInputDataChanged();
    resetAheadThrottle();
}

auto HybridChordTokenReductionTransformer::absorbWhileStopped(int32_t now) -> void {
    // Paused (pedal up) or stopped: the pianist's notes still go into both contexts as they
    // arrive, as the legacy Hybrid writes its input at once. Left queued, they were applied in
    // one go at the pedal press, possibly many seconds later, and model 1 then drew 5 cs
    // velocity-120 runs from that context (Nov 23 2026).
    silentEngine.absorb(now);
    waitEngine.absorb(now);
}

auto HybridChordTokenReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    finishAheadThrottleIfDue();
    applySettings();
    if (! streamsStarted)
        resetStreams(now);
    if (const auto cut = consumeModeCut())
        rewindFrom(*cut);
    drainInput();
    updateHybrid(now);
    tickTdControl();
    if (isGenerationStopped()) {
        absorbWhileStopped(now);
        return false;
    }
    return generate(now);
}

void HybridChordTokenReductionTransformer::threadRun() {
    if (! isModelLoaded())
        return;
    if (voiceSeparation != nullptr)
        voiceSeparation->reset();
    {
        std::random_device rd; // a different sequence every run, as the legacy samplers
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
    applySettings();
    resetStreams(nowCs());
    notifyInputDataChanged();
    resetAheadThrottle();
    resetLoopTiming();
    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        finishAheadThrottleIfDue();
        applySettings();
        if (const auto cut = consumeModeCut()) {
            SessionLog::add("HYBRID", "mode cut at " + juce::String(*cut) + ": generated notes from it removed, context kept");
            rewindFrom(*cut);
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
        drainInput();
        updateHybrid(now);
        tickTdControl();
        if (isGenerationStopped()) {
            absorbWhileStopped(now);
            maybeEmitGenerationPauseSoftStopClear();
            wait(5);
            recordThreadLoopMs(loopStartMs);
            continue;
        }
        wasGenerationPaused = false;
        // no note this step: nap 2 ms (a key or update wakes the thread at once, InputFilter::onReductionInput),
        // but not between the precompute's model calls: the naps added ~2 ms a call (Nov 24 2026)
        if (! generate(now) && ! (! silentActive.load() && waitEngine.precomputing()))
            wait(2);
        recordThreadLoopMs(loopStartMs);
    }
}

auto HybridChordTokenReductionTransformer::logDraw(const ChordTokenEngine &engine, const char *model,
                                                        const ChordTokenEngine::OutNote &n, int32_t noteId,
                                                        int32_t now) -> void {
    // As the legacy RT_DRAW: the note's draws (onset, dur, pitch, vel; tokens in model 2's vocab, onsets
    // relative to `base`). A note from a precomputed branch was sampled in the batched precompute and a
    // stacked note of a fire has no onset draw (the pianist's onset); "source" says which.
    std::string fields = "[";
    for (size_t i = 0; i < n.draws.size(); ++i)
        fields += (i ? "," : "") + n.draws[i].json();
    fields += "]";
    const char *source = n.precomputed ? "precomputed" : n.draws.empty() ? "unlogged" : "sampled";
    OpLog::emit(OpLog::Src::Rt, "RT_DRAW",
                {{"model", model}, {"r", OpLog::id('r', noteId)}, {"now", now}, {"floor", onsetLowerBoundCs()},
                 {"base", engine.base()}, {"context_len", engine.tokens().size()}, {"source", source},
                 {"onset", n.onsetCs}, {"fields", OpLog::Raw{fields}}, {"taus", OpLog::Raw{engine.liveTaus().json()}}});
}

auto HybridChordTokenReductionTransformer::logWaitEvent(const ChordTokenEngine::Event &e) -> void {
    // Model 2's WAITs, as RT_WAIT records: "wait" (drawn: d, COUNT k, the onset-step and COUNT draws),
    // then what became of it: "fire", "snap" (a top-3 length), "early" (a piano onset before / between
    // the windows: ordinary input), "expired", "hit" / "miss" (the precomputed branch or not), "short".
    static const std::set<std::string> kinds{"wait", "fire", "snap", "early", "expired", "hit", "miss", "short", "late", "chord"};
    if (! OpLog::on() || kinds.count(e.what) == 0)
        return;
    std::vector<OpLog::Field> f{{"model", "m2"}, {"decision", e.what}, {"at", e.atCs}, {"a", e.a}, {"b", e.b}};
    if (! e.text.empty())
        f.push_back({"text", e.text});
    if (std::string(e.what) == "wait") {
        f.push_back({"d", e.a});
        f.push_back({"k", e.b});
        f.push_back({"target", e.atCs + e.a});
        std::string fields = "[";
        const auto &draws = waitEngine.lastWaitDraws();
        for (size_t i = 0; i < draws.size(); ++i)
            fields += (i ? "," : "") + draws[i].json();
        fields += "]";
        f.push_back({"fields", OpLog::Raw{fields}});
    }
    OpLog::emit(OpLog::Src::Rt, "RT_WAIT", f);
}

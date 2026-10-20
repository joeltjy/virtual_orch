#include "VirtualOrch/reduction/MelodyFirstReductionTransformer.h"

#include "VirtualOrch/PolyphonyCap.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include <algorithm>
#include <limits>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int kEventWidth = 4;
constexpr size_t kGeneratedWindowNotes = 64;
constexpr size_t kMelodyEventsKept = 64;
constexpr int kMaxOnsetResamples = 256;
constexpr int32_t kMelodyVocabSize = 55285;

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
    reductionStream.tdTokenProvider = [this] {
        return tdResolver.resolve(tdValueProvider(), static_cast<int32_t>(reductionVocab));
    };
}

auto MelodyFirstReductionTransformer::init(const char *reductionModelPath,
                                           const char *melodyModelPath) -> bool {
    auto da = std::make_unique<StreamingKv::OnnxBackend>();
    auto mb = std::make_unique<StreamingKv::OnnxBackend>();
    std::string error;
    if (! da->init(reductionModelPath, error)) {
        reportSamplingError("Melody-first reduction (DA) init failed: " + juce::String(error));
        return false;
    }
    if (! mb->init(melodyModelPath, error)) {
        reportSamplingError("Melody-first reduction (mb) init failed: " + juce::String(error));
        return false;
    }
    if (! mb->usesPositionIds()) {
        reportSamplingError("Melody-first reduction: the mb export needs position_ids "
                            "(learned positions), got a RoPE model");
        return false;
    }
    const size_t daVocab = da->vocabSize();
    const size_t mbVocab = mb->vocabSize();
    if (daVocab < static_cast<size_t>(TdTokens::kDenseFirst + 7)
        || mbVocab < static_cast<size_t>(kMelodyVocabSize)) {
        reportSamplingError("Melody-first reduction: unexpected vocab sizes "
                            + juce::String(static_cast<int>(daVocab)) + " / "
                            + juce::String(static_cast<int>(mbVocab)));
        return false;
    }
    mb->setRowLimit(MelodyStream::kMaxTokens + 32);
    initWithBackends(std::move(mb), mbVocab, std::move(da), daVocab);
    return true;
}

auto MelodyFirstReductionTransformer::initWithBackends(
    std::unique_ptr<StreamingKv::Backend> melodyBackendIn, size_t melodyVocabIn,
    std::unique_ptr<StreamingKv::Backend> reductionBackendIn, size_t reductionVocabIn) -> void {
    melodyBackend = std::move(melodyBackendIn);
    reductionBackend = std::move(reductionBackendIn);
    melodyVocab = melodyVocabIn;
    reductionVocab = reductionVocabIn;
    melodyCache = std::make_unique<StreamingKv::CommittedScratchCache>(*melodyBackend);
    reductionCache = std::make_unique<StreamingKv::CommittedScratchCache>(*reductionBackend);
}

auto MelodyFirstReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto MelodyFirstReductionTransformer::samplingForStep(int stepIdx) const -> FieldSampling {
    switch (stepIdx % kEventWidth) {
        case 1: return {DenseSampling::DurationTopP, getDurationTemperature()};
        case 2: return {DenseSampling::NoteTopP, getNoteTemperature()};
        case 3: return {DenseSampling::VelocityTopP, getVelocityTemperature()};
        default: return {DenseSampling::OnsetTopP, getOnsetTemperature()};
    }
}

auto MelodyFirstReductionTransformer::resetAll(int32_t now) -> void {
    state.reset();
    melodyStream.reset(now);
    reductionStream.reset(now);
    if (melodyCache != nullptr)
        melodyCache->reset();
    if (reductionCache != nullptr)
        reductionCache->reset();
    tdResolver = {};
    aDirty = false;
    userHasPlayed = false;
    melodyEvents.clear();
    generatedDense.clear();
    pitchTauScheduler.reset();
    latestMelodyOnset = melodyStream.base();
    latestReductionOnset = reductionStream.base();
    currentTime = reductionStream.clock();
}

auto MelodyFirstReductionTransformer::melodyIsTooFarAhead() const -> bool {
    return latestMelodyOnset - latestReductionOnset > kMaxMelodyAheadCs;
}

auto MelodyFirstReductionTransformer::rewindGeneratedFrom(int32_t cutCs) -> void {
    // Both KV caches see a changed committed prefix and rebuild by themselves.
    melodyStream.rewindTo(cutCs);
    reductionStream.rewindTo(cutCs, /*drop the melody's M controls*/ 2);
    std::erase_if(melodyEvents, [cutCs](const Token &t) { return t.time >= cutCs; });
    eraseStridedFrom(generatedDense, kEventWidth, cutCs);
    rewindInputDataFrom(cutCs);
    pitchTauScheduler.reset();

    latestMelodyOnset = melodyStream.base();
    for (const auto &t: melodyEvents)
        latestMelodyOnset = std::max(latestMelodyOnset, t.time);
    latestReductionOnset = reductionStream.clock();
    currentTime = reductionStream.clock();
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
        reductionStream.addControl({in.time, held ? 0 : durCs, pitch, velocity}, held,
                                   /*instrument*/ 0, /*delta*/ 0, /*priority*/ 1);

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
        reductionStream.confirmControl(update.oldNote.time, pitch, finalDur, 0);
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
    reductionStream.commitLongHolds(now);
    melodyStream.commitLongHolds(now);
    return applied;
}

// ---- melody (T8) ----

auto MelodyFirstReductionTransformer::sampleMelodyField(int field, const std::vector<float> &logits,
                                                        int32_t lowerModelOnset) -> int32_t {
    std::vector<float> scores(logits.size(), kNegInf);
    const auto allow = [&](int32_t id) {
        if (id >= 0 && static_cast<size_t>(id) < logits.size())
            scores[static_cast<size_t>(id)] = logits[static_cast<size_t>(id)];
    };
    switch (field) {
        case 0:
            for (int32_t t = std::max(lowerModelOnset, 0); t <= MelodyStream::kMaxModelTimeCs; ++t)
                allow(t);
            break;
        case 1:
            for (int32_t cs = DenseQuantize::DurationMinCs; cs <= DenseQuantize::DurationMaxCs;
                 cs += DenseQuantize::DurationStepCs)
                allow(10000 + cs);
            break;
        case 2:
            for (int32_t p = DenseQuantize::PianoKeyLow; p <= DenseQuantize::PianoKeyHigh; ++p)
                allow(11000 + 128 * 1 + p);
            break;
        default:
            for (int32_t v = DenseQuantize::VelocityStep; v <= DenseQuantize::VelocityMax;
                 v += DenseQuantize::VelocityStep)
                allow(MelodyStream::kVelEvent + v);
            break;
    }
    const auto sampling = samplingForStep(field);
    return sampleTopP(scores, sampling.topP, sampling.temperature);
}

auto MelodyFirstReductionTransformer::generateMelodyNote(int32_t now)
    -> std::optional<MelodyStream::Note> {
    // Visible rest quads while the clock catches up with the app clock (silence keeps moving).
    while (melodyStream.needsRebase(std::max(melodyStream.clock(), now)))
        melodyStream.rebase();
    while (melodyStream.clock() + MelodyStream::kRestStepCs <= now) {
        melodyStream.appendRest(melodyStream.clock() + MelodyStream::kRestStepCs);
        melodyStream.insertDue(melodyStream.clock());
    }
    melodyStream.insertDue(melodyStream.clock());
    if (melodyStream.trimIfNeeded())
        juce::Logger::writeToLog("[melody-first] mb stream cut to 128 tokens, cache rebuilt");

    const int cap = getMaxPolyphony();
    int32_t lower = melodyStream.clock();
    if (cap > 0)
        lower = std::max(lower, PolyphonyCap::earliestOnsetCs(melodyEvents, melodyStream.clock(), cap));

    int32_t lowerModelOnset = lower - melodyStream.base();
    std::vector<int32_t> partial;
    std::array<int32_t, 4> chosen{};

    for (int resample = 0; resample < kMaxOnsetResamples; ++resample) {
        while (melodyStream.needsRebase(std::max(melodyStream.clock(), lower))) {
            melodyStream.rebase();
            lowerModelOnset = lower - melodyStream.base();
        }
        const auto tokens = melodyStream.tokens();
        const auto &logits = melodyCache->logitsAfter(tokens, melodyStream.committedLen());
        if (logits.empty() || logits.size() < static_cast<size_t>(kMelodyVocabSize)) {
            reportSamplingError("Melody-first: mb model returned no logits");
            return std::nullopt;
        }
        const int32_t onsetModel = sampleMelodyField(0, logits, lowerModelOnset);
        const int32_t onsetAbs = onsetModel + melodyStream.base();

        // Visible rest steps: every 1 s step before the proposed onset is written, then the
        // onset is sampled again after it.
        const int32_t nextRest = melodyStream.clock() + MelodyStream::kRestStepCs;
        if (nextRest < onsetAbs) {
            melodyStream.appendRest(nextRest);
            melodyStream.insertDue(nextRest);
            lower = std::max(lower, nextRest + 1);
            lowerModelOnset = lower - melodyStream.base();
            continue;
        }
        chosen[0] = onsetModel;
        partial = {onsetModel};
        break;
    }
    if (partial.empty()) {
        reportSamplingError("Melody-first: no onset found");
        return std::nullopt;
    }

    for (int field = 1; field < kEventWidth; ++field) {
        auto desired = melodyStream.tokens();
        desired.insert(desired.end(), partial.begin(), partial.end());
        const auto &logits = melodyCache->logitsAfter(
            desired, std::min(melodyStream.committedLen(), desired.size()));
        if (logits.size() < static_cast<size_t>(kMelodyVocabSize)) {
            reportSamplingError("Melody-first: mb model returned no logits");
            return std::nullopt;
        }
        const int32_t token = sampleMelodyField(field, logits, 0);
        chosen[static_cast<size_t>(field)] = token;
        partial.push_back(token);
    }

    MelodyStream::Note note;
    note.onsetCs = chosen[0] + melodyStream.base();
    note.durationCs = chosen[1] - 10000;
    note.pitch = chosen[2] - 11000 - 128;
    note.velocity = chosen[3] - MelodyStream::kVelEvent;
    melodyStream.appendEvent(note);

    Token event{note.onsetCs, static_cast<int32_t>(DenseVocab::DurOffset) + note.durationCs,
                static_cast<int32_t>(Vocab::NoteOffset) + note.pitch, note.velocity};
    melodyEvents.push_back(event);
    if (melodyEvents.size() > kMelodyEventsKept)
        melodyEvents.erase(melodyEvents.begin());
    latestMelodyOnset = std::max(latestMelodyOnset, note.onsetCs);

    // The generated melody is the M control (instrument 2, delta 3 s) of the reduction model.
    reductionStream.addControl({note.onsetCs, note.durationCs, note.pitch, note.velocity}, false,
                               /*instrument*/ 2, kMelodyDeltaCs, /*priority*/ 0);
    return note;
}

// ---- reduction (DA) ----

auto MelodyFirstReductionTransformer::sampleReductionField(
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
            for (int32_t v = DenseQuantize::VelocityStep; v <= DenseQuantize::VelocityMax;
                 v += DenseQuantize::VelocityStep)
                allow(static_cast<int32_t>(DenseVocab::VelocityOffset) + v);
            break;
    }
    const auto sampling = samplingForStep(field);
    return sampleTopP(scores, sampling.topP, sampling.temperature);
}

auto MelodyFirstReductionTransformer::generateReductionNote(int32_t now) -> std::optional<Token> {
    const int32_t clock0 = std::max(reductionStream.clock(), now);
    int32_t lower = std::max(onsetLowerBoundCs(), clock0);
    while (reductionStream.needsRebase(lower))
        reductionStream.rebase();
    reductionStream.insertDue(clock0, now);

    int32_t restBase = clock0;
    int32_t onsetModel = 0;
    bool sampled = false;
    for (int resample = 0; resample < kMaxOnsetResamples; ++resample) {
        const auto tokens = reductionStream.tokens();
        if (tokens.empty())
            return std::nullopt;
        const auto &logits = reductionCache->logitsAfter(tokens, reductionStream.committedLen());
        if (logits.empty() || logits.size() < static_cast<size_t>(DenseVocab::VocabSize)) {
            reportSamplingError("Melody-first: reduction model returned no logits");
            return std::nullopt;
        }
        onsetModel = sampleReductionField(0, logits, lower - reductionStream.base(), nullptr, 0, 0);
        const int32_t onsetAbs = onsetModel + reductionStream.base();

        // Hidden 1 s rest steps: at the first step before the proposed onset where a pending
        // control becomes due, drop the onset, move the clock there, insert, sample again.
        bool jumped = false;
        for (int32_t r = restBase + 100; r < onsetAbs; r += 100) {
            if (! reductionStream.hasPendingDueBy(r))
                continue;
            reductionStream.insertDue(r, now);
            lower = std::max(lower, r + 1);
            restBase = r;
            jumped = true;
            break;
        }
        if (jumped)
            continue;
        sampled = true;
        break;
    }
    if (! sampled) {
        reportSamplingError("Melody-first: no reduction onset found");
        return std::nullopt;
    }

    const int32_t base = reductionStream.base();
    std::vector<int32_t> partial{onsetModel};
    std::array<int32_t, 4> chosen{onsetModel, 0, 0, 0};
    DensePitchBias::PitchWindowStats pitchWindow;
    const bool pitchBias = isPitchBiasEnabled();
    float tau = DensePitchBias::TauBase;
    float tauPrime = DensePitchBias::TauBase;

    for (int field = 1; field < kEventWidth; ++field) {
        auto desired = reductionStream.tokens();
        desired.insert(desired.end(), partial.begin(), partial.end());
        const auto &logits = reductionCache->logitsAfter(
            desired, std::min(reductionStream.committedLen(), desired.size()));
        if (logits.size() < static_cast<size_t>(DenseVocab::VocabSize)) {
            reportSamplingError("Melody-first: reduction model returned no logits");
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
        const int32_t token = sampleReductionField(
            field, logits, 0, pitchBias && field == 2 ? &pitchWindow : nullptr, tau, tauPrime);
        chosen[static_cast<size_t>(field)] = token;
        partial.push_back(token);
    }

    const int32_t onsetAbs = chosen[0] + base;
    const int32_t durCs = chosen[1] - static_cast<int32_t>(DenseVocab::DurOffset);
    const int32_t pitch = chosen[2] - static_cast<int32_t>(DenseVocab::NoteOffset)
                          - DenseConfig::MaxPitch * AnticipatedStream::kEventInstrument;
    const int32_t velocity = chosen[3] - static_cast<int32_t>(DenseVocab::VelocityOffset);
    reductionStream.appendEvent({onsetAbs, durCs, pitch, velocity});

    const auto quad = dense(onsetAbs, durCs, AnticipatedStream::kEventInstrument, pitch, velocity);
    generatedDense.insert(generatedDense.end(), quad.begin(), quad.end());
    if (generatedDense.size() > kGeneratedWindowNotes * kEventWidth)
        generatedDense.erase(generatedDense.begin(),
                             generatedDense.end()
                                 - static_cast<std::ptrdiff_t>(kGeneratedWindowNotes * kEventWidth));
    inputData.insert(inputData.end(), quad.begin(), quad.end());
    notifyInputDataChanged();

    currentTime = onsetAbs;
    latestReductionOnset = std::max(latestReductionOnset, onsetAbs);
    logGeneratedTokenIfResumed(onsetAbs);

    Token outbound{onsetAbs, static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
                   static_cast<int32_t>(DenseVocab::NoteOffset) + pitch, velocity};
    pushOutputToken(outbound);
    if (isGeneratedTooFarAhead(onsetAbs))
        startAheadThrottle();
    return outbound;
}

// ---- loop ----

auto MelodyFirstReductionTransformer::stepOnce() -> bool {
    const int32_t now = nowCs();
    if (! melodyStream.isStarted())
        resetAll(now);
    if (const auto cut = consumeModeCut())
        rewindGeneratedFrom(*cut);
    drainInput(now);
    if (aDirty) {
        state.runWindow(now);
        melodyStream.syncControls(state.getPieces());
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
            melodyStream.syncControls(state.getPieces());
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

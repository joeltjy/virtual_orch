#include "VirtualOrch/reduction/ReductionTransformerV2.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/SamplingRelativeTime.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include <algorithm>
#include <limits>

namespace {

constexpr int denseEventWidth = 4;

} // namespace

ReductionTransformerV2::ReductionTransformerV2(ModelConfig &modelConfigIn)
    : ReductionTransformer("Reduction Transformer V2", modelConfigIn) {
}

void ReductionTransformerV2::init(const char *modelPath) {
    Ort::SessionOptions sessionOptions;

#ifdef __linux__
    OrtCUDAProviderOptions cudaOptions{};
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
#endif

    session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, sessionOptions);

    allocatedInputNames.clear();
    allocatedOutputNames.clear();
    const Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i)
        allocatedInputNames.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session->GetOutputCount(); ++i)
        allocatedOutputNames.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
}

void ReductionTransformerV2::threadInit() {
}

auto ReductionTransformerV2::refreshPitchTauSchedule() -> void {
    // Same now for π/s_y window and τ schedule: max(clock, last onset). Matches
    // amt_causal's prefix-relative window when generation is ahead of the clock.
    const int32_t clockNow =
        clock != nullptr ? static_cast<int32_t>(clock->getTime()) : static_cast<int32_t>(-1);
    const auto inputSnapshot = getInputData();

    const juce::ScopedLock lock(pitchBiasLock);
    lastPitchWindow = DensePitchBias::collectPitchWindow(inputSnapshot, clockNow);
    int32_t scheduleNow = 0;
    for (size_t i = 0; i + 3 < inputSnapshot.size(); i += denseEventWidth)
        scheduleNow = inputSnapshot[i];
    if (clockNow >= 0)
        scheduleNow = std::max(scheduleNow, clockNow);
    const auto scheduled =
        pitchTauScheduler.evaluate(scheduleNow, lastPitchWindow.uniquePitches,
                                   lastPitchWindow.uniqueDeltas);
    pitchTau.store(scheduled.tau, std::memory_order_relaxed);
    pitchTauPrime.store(scheduled.tauPrime, std::memory_order_relaxed);
}

void ReductionTransformerV2::threadRun() {
    if (session == nullptr)
        return;

    if (voiceSeparation != nullptr)
        voiceSeparation->reset();

    inputData.clear();
    clearInputTokenQueue();
    clearInputConditioningQueue();
    clearOutputTokenQueue();
    clearUpdatesFromFilter();
    clearOutputHistory();

    {
        const juce::ScopedLock lock(pitchBiasLock);
        pitchTauScheduler.reset();
        lastPitchWindow = {};
        pitchTau.store(DensePitchBias::TauBase, std::memory_order_relaxed);
        pitchTauPrime.store(DensePitchBias::TauBase, std::memory_order_relaxed);
    }

    currentTime = modelConfig.outputStartTime;

    int32_t forceAtTime = -1;
    if (modelConfig.outputForceStartTime)
        forceAtTime = modelConfig.outputStartTime;

    if (modelConfig.inputInitialData.length() > 0) {
        juce::StringArray parts;
        parts.addTokens(modelConfig.inputInitialData, " ", "");
        for (int i = 0; i + 3 < parts.size(); i += denseEventWidth) {
            inputData.push_back(parts[i].getIntValue());
            inputData.push_back(parts[i + 1].getIntValue());
            inputData.push_back(parts[i + 2].getIntValue());
            inputData.push_back(parts[i + 3].getIntValue());
        }
    }
    notifyInputDataChanged();

    bool inputApplied = false;
    resetAheadThrottle();
    resetLoopTiming();

    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        ReductionLoopStepMs step{};
        loopAccum = {};

        finishAheadThrottleIfDue();

        if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput
            && directInputBlock.value) {
            wait(5);
            recordThreadLoopMs(loopStartMs);
            continue;
        }

        {
            const ScopedMs drainMs(&step.drain);
            inputApplied = applyQueuedInputToInputData();
            applyUpdatesFromFilter();
        }

        if (isGenerationStopped()) {
            maybeEmitGenerationPauseSoftStopClear();
            // overflowPause + live input: soft-stop clear (now + 1 s), same as manual
            // generationPause — a ClearQueue at time 0 wiped Out: RT including notes that
            // were just seeded from outputHistory, leaving silence until throttle ends.
            if (inputApplied && ! generationPause.get() && clock != nullptr) {
                const auto clearAt =
                    static_cast<int32_t>(clock->getTime()) + Config::TimeResolution;
                const Token clearToken{clearAt, static_cast<int32_t>(DenseVocab::DurOffset),
                                       static_cast<int32_t>(Vocab::ClearQueue)};
                pushOutputToken(clearToken);
            } else if (! inputApplied) {
                wait(5);
            }
            inputApplied = false;
            recordThreadLoopMs(loopStartMs);
            continue;
        }

        wasGenerationPaused = false;

        Token newToken = generateNewToken(forceAtTime);
        forceAtTime = -1;
        step.prep = loopAccum.prep;
        step.onnx = loopAccum.onnx;
        step.logits = loopAccum.logits;
        step.mask = loopAccum.mask;
        step.sample = loopAccum.sample;
        step.contextTokens = loopAccum.contextTokens;
        step.onnxCalls = loopAccum.onnxCalls;

        if (newToken.time < 0) {
            wait(5);
            recordThreadLoopMs(loopStartMs);
            continue;
        }

        {
            const ScopedMs pushMs(&step.push);
            inputData.push_back(newToken.time);
            inputData.push_back(DenseQuantize::snapDurationToken(newToken.duration));
            inputData.push_back(DenseQuantize::snapNoteToken(newToken.note));
            inputData.push_back(DenseQuantize::snapVelocityToken(static_cast<int32_t>(
                DenseVocab::VelocityOffset + newToken.velocity)));
            notifyInputDataChanged();

            currentTime = newToken.time;

            if (inputApplied) {
                const Token clearToken{static_cast<int32_t>(DenseVocab::TimeOffset),
                                       static_cast<int32_t>(DenseVocab::DurOffset),
                                       static_cast<int32_t>(Vocab::ClearQueue)};
                pushOutputToken(clearToken);
                inputApplied = false;
            }

            pushOutputToken(newToken);

            if (isGeneratedTooFarAhead(newToken.time))
                startAheadThrottle();
        }

        step.total =
            static_cast<float>(juce::Time::getMillisecondCounterHiRes() - loopStartMs);
        recordThreadLoopMs(loopStartMs);
        publishBusyReductionProfile(step);
    }
}

auto ReductionTransformerV2::applyQueuedInputToInputData() -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
    }

    bool inputApplied = false;
    Token inputToken{-1, -1, -1};

    while (inputTokenQueue.pull(inputToken)) {
        if (! inputApplied) {
            for (size_t i = 0; i + 3 < inputData.size(); i += denseEventWidth) {
                if (inputData[i] > inputToken.time) {
                    inputData.erase(inputData.begin() + static_cast<std::ptrdiff_t>(i), inputData.end());
                    break;
                }
            }
        }

        const int32_t velocity = DenseQuantize::snapVelocity(
            juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                         inputToken.velocity > 0 ? inputToken.velocity : DenseConfig::DefaultVelocity));

        // Dense vocab MaxInstr == 5; always ingest as instrument 0 + snapped A0–C8 pitch.
        const int32_t denseNote = static_cast<int32_t>(
            DenseVocab::NoteOffset + DenseQuantize::snapPianoPitch(inputToken.getPitch()));
        const int32_t denseDur = DenseQuantize::snapDurationToken(inputToken.duration);

        if (modelConfig.inputMode != InputMode::Buffer) {
            inputData.push_back(inputToken.time);
            inputData.push_back(denseDur);
            inputData.push_back(denseNote);
            inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
        } else {
            const int32_t pitchClass = DenseQuantize::snapPianoPitch(inputToken.getPitch()) % 12;
            const int32_t instrBase =
                static_cast<int32_t>(DenseVocab::NoteOffset
                                    + DenseConfig::MaxPitch
                                          * InstrumentConstants::kReductionInputLocalInstrumentId);
            if (! inputApplied) {
                inputData.push_back(inputToken.time);
                inputData.push_back(denseDur);
                inputData.push_back(instrBase + 36 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
                inputData.push_back(inputToken.time);
                inputData.push_back(denseDur);
                inputData.push_back(instrBase + 48 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
            } else {
                inputData.push_back(inputToken.time);
                inputData.push_back(denseDur);
                inputData.push_back(instrBase + 60 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
            }
        }

        currentTime = inputToken.time;
        inputApplied = true;
    }

    if (inputApplied)
        notifyInputDataChanged();

    return inputApplied;
}

auto ReductionTransformerV2::applyUpdatesFromFilter() -> bool {
    TokenUpdate update{};
    if (! updatesFromFilter.pull(update))
        return false;

    std::vector<Token> history;
    history.reserve(inputData.size() / denseEventWidth);
    for (size_t i = 0; i + 3 < inputData.size(); i += denseEventWidth) {
        const int32_t velToken = inputData[i + 3];
        const int32_t velocity =
            juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                         velToken - static_cast<int32_t>(DenseVocab::VelocityOffset));
        history.push_back(Token{inputData[i], inputData[i + 1], inputData[i + 2], velocity});
    }

    bool changed = false;
    do {
        if (orchestrationUpdatesIncoming != nullptr)
            orchestrationUpdatesIncoming->push(update);
        if (applyTokenUpdateToHistory(history, update))
            changed = true;
    } while (updatesFromFilter.pull(update));

    if (! changed)
        return false;

    inputData.clear();
    inputData.reserve(history.size() * denseEventWidth);
    for (const auto &t : history) {
        inputData.push_back(t.time);
        inputData.push_back(DenseQuantize::snapDurationToken(t.duration));
        inputData.push_back(DenseQuantize::snapNoteToken(t.note));
        inputData.push_back(DenseQuantize::snapVelocityToken(static_cast<int32_t>(
            DenseVocab::VelocityOffset
            + juce::jlimit(0, DenseConfig::MaxVelocity - 1, t.velocity))));
    }
    notifyInputDataChanged();
    return true;
}

void ReductionTransformerV2::threadStop() {
}

void ReductionTransformerV2::futureLogits(std::vector<float> &logits, const int curTime,
                                         int32_t forceAtTime) {
    if (forceAtTime != -1 && forceAtTime < DenseConfig::MaxTime) {
        std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset), forceAtTime,
                    -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset + forceAtTime + 1),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                  -std::numeric_limits<float>::infinity());
        return;
    }

    // V2: past-onset ban only (no maximumFuture clamp).
    if (curTime > 0) {
        std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset), curTime,
                    -std::numeric_limits<float>::infinity());
    }
}

auto ReductionTransformerV2::samplingForStep(int stepIdx) const -> FieldSampling {
    if (stepIdx % denseEventWidth == 1)
        return {DenseSampling::DurationTopP, getDurationTemperature()};
    if (stepIdx % denseEventWidth == 2)
        return {DenseSampling::NoteTopP, getNoteTemperature()};
    if (stepIdx % denseEventWidth == 3)
        return {DenseSampling::VelocityTopP, getVelocityTemperature()};
    return {DenseSampling::OnsetTopP, getOnsetTemperature()};
}

void ReductionTransformerV2::velocityLogits(std::vector<float> &logits) {
    juce::ignoreUnused(logits);
}

void ReductionTransformerV2::safeLogits(std::vector<float> &logits, size_t stepIdx) {
    if (logits.size() < DenseVocab::VocabSize) {
        const auto detail = "logits size " + juce::String(static_cast<int>(logits.size()))
                            + " < VocabSize "
                            + juce::String(static_cast<int>(DenseVocab::VocabSize));
        DBG("ReductionTransformerV2::safeLogits: " + detail);
        reportSamplingError(detail);
        return;
    }

    // Mask anything past model vocab (if ORT returns a larger last dim).
    if (logits.size() > DenseVocab::VocabSize) {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VocabSize), logits.end(),
                  -std::numeric_limits<float>::infinity());
    }

    const auto maskTime = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                  -std::numeric_limits<float>::infinity());
    };
    const auto maskDur = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
                  -std::numeric_limits<float>::infinity());
    };
    const auto maskNote = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
                  -std::numeric_limits<float>::infinity());
    };
    const auto maskVel = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VocabSize),
                  -std::numeric_limits<float>::infinity());
    };

    switch (stepIdx % denseEventWidth) {
        case 0: // time
            maskDur();
            maskNote();
            maskVel();
            break;
        case 1: // duration — 5 cs grid
            maskTime();
            maskNote();
            maskVel();
            DenseQuantize::maskIllegalDurationLogits(logits);
            break;
        case 2: // note — A0–C8 per instrument band
            maskTime();
            maskDur();
            maskVel();
            DenseQuantize::maskIllegalPianoNoteLogits(logits);
            break;
        case 3: // velocity — step-8 grid
            maskTime();
            maskDur();
            maskNote();
            DenseQuantize::maskIllegalVelocityLogits(logits);
            break;
        default:
            break;
    }
}

std::vector<float> ReductionTransformerV2::runModelAndGetLogits(std::vector<int32_t> &tokens) {
    if (session == nullptr || tokens.empty())
        return {};

    const std::vector input_shape{1, static_cast<int64_t>(tokens.size())};

    // Prefer int64 input_ids (common for exported dense stubs); fall back to int32.
    std::vector<int64_t> tokenIds(tokens.begin(), tokens.end());
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo, tokenIds.data(), tokenIds.size(), input_shape.data(), input_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    std::vector<const char *> inputNames;
    for (auto &name : allocatedInputNames)
        inputNames.push_back(name.c_str());

    // Only logits are consumed; asking for KV-cache outputs is wasted work (and on
    // checkpoint-36500 that is dozens of tensors per call).
    std::vector<const char *> outputNames;
    {
        const char *logitsName = nullptr;
        for (const auto &name : allocatedOutputNames) {
            if (name == "logits") {
                logitsName = name.c_str();
                break;
            }
        }
        if (logitsName == nullptr && ! allocatedOutputNames.empty())
            logitsName = allocatedOutputNames.front().c_str();
        if (logitsName == nullptr)
            return {};
        outputNames.push_back(logitsName);
    }

    try {
        std::vector<Ort::Value> output_tensors;
        {
            const ScopedMs onnxMs(&loopAccum.onnx);
            output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                                          input_tensors.data(), input_tensors.size(),
                                          outputNames.data(), outputNames.size());
        }
        ++loopAccum.onnxCalls;

        Ort::Value &logits_tensor = output_tensors.front();
        const auto shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.empty())
            return {};

        const auto vocab = static_cast<size_t>(shape.back());
        if (vocab == 0)
            return {};

        const size_t seq = tokens.size();
        const auto elementCount = logits_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        if (seq == 0 || vocab * seq > elementCount)
            return {};

        const ScopedMs logitsMs(&loopAccum.logits);
        const float *data = logits_tensor.GetTensorMutableData<float>();
        const float *last = data + vocab * (seq - 1);
        return {last, last + vocab};
    } catch (const Ort::Exception &exception) {
        const auto detail = juce::String(exception.what());
        DBG("ReductionTransformerV2 ORT error: " + detail);
        reportSamplingError(detail);
    }

    return {};
}

auto ReductionTransformerV2::denseMinTime(std::vector<int32_t> &tokens) -> int32_t {
    return SamplingRelativeTime::minStridedOnset(tokens, static_cast<size_t>(denseEventWidth));
}

Token ReductionTransformerV2::generateNewToken(int32_t forceAtTime) {
    if (inputData.size() % denseEventWidth != 0)
        throw std::runtime_error("ReductionTransformerV2 inputData must be a multiple of 4");

    if (inputData.empty())
        return {-1, -1, -1, DenseConfig::DefaultVelocity};

    // UI / silence path: hybrid now. Note-field bias is refreshed again after onset
    // is sampled (amt_causal: τ/π from the prefix that already contains the new onset).
    refreshPitchTauSchedule();

    DensePitchBias::PitchWindowStats pitchWindow;
    float tau = DensePitchBias::TauBase;
    float tauPrime = DensePitchBias::TauBase;
    {
        const juce::ScopedLock lock(pitchBiasLock);
        pitchWindow = lastPitchWindow;
        tau = pitchTau.load(std::memory_order_relaxed);
        tauPrime = pitchTauPrime.load(std::memory_order_relaxed);
    }

    std::vector<int32_t> history;
    int32_t offset = 0;
    {
        const ScopedMs prepMs(&loopAccum.prep);
        const int lookbackInts = DenseSampling::ContextNotes * denseEventWidth;
        const int lookback = std::max(static_cast<int>(inputData.size()) - lookbackInts, 0);
        history.assign(inputData.begin() + lookback, inputData.end());
        DenseQuantize::snapPackedEvents(history, static_cast<size_t>(denseEventWidth));
        NoteWindow::sortStridedByOnset(history, static_cast<size_t>(denseEventWidth));
        offset = history.empty() ? 0 : denseMinTime(history);
        SamplingRelativeTime::relativizeStridedOnsets(history, static_cast<size_t>(denseEventWidth),
                                                       offset);
        loopAccum.contextTokens =
            juce::jmax(loopAccum.contextTokens, static_cast<int>(history.size()));
    }

    Token newToken{-1, -1, -1, DenseConfig::DefaultVelocity};

    for (int i = 0; i < denseEventWidth; ++i) {
        // After onset is known, rebuild π / s_y / τ for the note-field Menon adjust so
        // they track this prefix step (inputs + prior outputs + new onset), matching
        // offline amt_causal (in-sample τ after onset+duration are on the prefix).
        if (i == 2 && newToken.time >= 0) {
            const juce::ScopedLock lock(pitchBiasLock);
            pitchWindow = DensePitchBias::collectPitchWindowAt(inputData, newToken.time);
            lastPitchWindow = pitchWindow;
            const auto scheduled =
                pitchTauScheduler.evaluate(newToken.time, pitchWindow.uniquePitches,
                                           pitchWindow.uniqueDeltas);
            tau = scheduled.tau;
            tauPrime = scheduled.tauPrime;
            pitchTau.store(tau, std::memory_order_relaxed);
            pitchTauPrime.store(tauPrime, std::memory_order_relaxed);
        }

        std::vector<float> scores = runModelAndGetLogits(history);
        if (scores.size() < DenseVocab::VocabSize) {
            const auto detail =
                "unexpected logits size " + juce::String(static_cast<int>(scores.size()))
                + " (historyInts=" + juce::String(static_cast<int>(history.size())) + ")";
            DBG("ReductionTransformerV2::generateNewToken: " + detail);
            reportSamplingError(detail);
            return {-1, -1, -1, DenseConfig::DefaultVelocity};
        }

        {
            const ScopedMs maskMs(&loopAccum.mask);
            safeLogits(scores, static_cast<size_t>(i));
            if (i == 0)
                futureLogits(scores, currentTime - offset,
                             forceAtTime != -1 ? forceAtTime - offset : -1);
            else if (i == 2)
                DensePitchBias::applyNoteLogitsBias(scores, pitchWindow, tau, tauPrime);
            else if (i == 3)
                velocityLogits(scores);
        }

        const auto fieldSampling = samplingForStep(i);
        int32_t token = 0;
        {
            const ScopedMs sampleMs(&loopAccum.sample);
            token = sampleTopP(scores, fieldSampling.topP, fieldSampling.temperature);
        }
        history.push_back(token);

        if (i == 0)
            newToken.time = token + offset;
        else if (i == 1)
            newToken.duration = token;
        else if (i == 2)
            newToken.note = token;
        else
            newToken.velocity =
                juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                             token - static_cast<int32_t>(DenseVocab::VelocityOffset));
    }

    return newToken;
}

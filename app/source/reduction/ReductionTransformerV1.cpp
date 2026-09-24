#include "VirtualOrch/reduction/ReductionTransformerV1.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/SamplingRelativeTime.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace {

constexpr int denseEventWidth = 4;

/**
 * Nearest-rank percentile of `values` (ceil(q * n)-th smallest); 0 when empty.
 * e.g. q = 0.75 over 15 values is the 12th smallest.
 */
auto percentileOf(std::vector<int32_t> values, float quantile) -> int32_t {
    if (values.empty())
        return 0;

    const auto count = static_cast<float>(values.size());
    const auto rank = static_cast<size_t>(std::ceil(juce::jlimit(0.0f, 1.0f, quantile) * count));
    const size_t index = juce::jlimit<size_t>(0, values.size() - 1, rank == 0 ? 0 : rank - 1);

    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

} // namespace

ReductionTransformerV1::ReductionTransformerV1(ModelConfig &modelConfigIn)
    : ReductionTransformer("Reduction Transformer V1", modelConfigIn) {
}

void ReductionTransformerV1::init(const char *modelPath) {
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

void ReductionTransformerV1::threadInit() {
}

void ReductionTransformerV1::threadRun() {
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
            // generationPause — avoids wiping Out: RT schedule with ClearQueue at time 0.
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

        // Pedal up (or no pause) but live MIDI still at provisional inputDuration —
        // wait for note-off updates to snap real durations before sampling.
        if (hasUnresolvedProvisionalInputNotes()) {
            wait(5);
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
            inputData.push_back(DenseSampling::snapGeneratedDurationToken(
                newToken.duration, modelConfig.inputDuration));
            inputData.push_back(newToken.note);
            inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + newToken.velocity));
            notifyInputDataChanged();

            currentTime = newToken.time;
            logGeneratedTokenIfResumed(newToken.time);

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

auto ReductionTransformerV1::applyQueuedInputToInputData() -> bool {
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

        const int32_t velocity =
            juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                         inputToken.velocity > 0 ? inputToken.velocity : DenseConfig::DefaultVelocity);

        // Dense vocab MaxInstr == 5; always ingest as instrument 0 + pitch.
        const int32_t denseNote = static_cast<int32_t>(
            DenseVocab::NoteOffset + inputToken.getPitch());

        if (modelConfig.inputMode != InputMode::Buffer) {
            inputData.push_back(inputToken.time);
            inputData.push_back(inputToken.duration);
            inputData.push_back(denseNote);
            inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
        } else {
            const int32_t pitchClass = inputToken.getPitch() % 12;
            const int32_t instrBase =
                static_cast<int32_t>(DenseVocab::NoteOffset
                                    + DenseConfig::MaxPitch
                                          * InstrumentConstants::kReductionInputLocalInstrumentId);
            if (! inputApplied) {
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(instrBase + 36 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(instrBase + 48 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
            } else {
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
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

auto ReductionTransformerV1::applyUpdatesFromFilter() -> bool {
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
        inputData.push_back(t.duration);
        inputData.push_back(t.note);
        inputData.push_back(static_cast<int32_t>(
            DenseVocab::VelocityOffset
            + juce::jlimit(0, DenseConfig::MaxVelocity - 1, t.velocity)));
    }
    notifyInputDataChanged();
    return true;
}

void ReductionTransformerV1::threadStop() {
}

void ReductionTransformerV1::instrLogits(std::vector<float> &logits) {
    auto it = modelConfig.sortedActiveOutputInstruments.begin();
    auto end = modelConfig.sortedActiveOutputInstruments.end();

    if (it == end) {
        // No active instruments configured: allow all dense instruments 0..MaxInstr-1.
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
                  logits.end(),
                  -std::numeric_limits<float>::infinity());
        return;
    }

    // Clamp to dense instrument range.
    while (it != end && *it >= DenseConfig::MaxInstr)
        ++it;
    if (it == end) {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
                  logits.end(),
                  -std::numeric_limits<float>::infinity());
        return;
    }

    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
              logits.begin()
                  + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                                + DenseConfig::MaxPitch * (*it)
                                                + modelConfig.outputInstruments[*it].low),
              -std::numeric_limits<float>::infinity());

    auto prev_it = it;
    ++it;
    for (; it != end; ++it, ++prev_it) {
        if (*it >= DenseConfig::MaxInstr)
            break;
        std::fill(
            logits.begin()
                + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                              + DenseConfig::MaxPitch * (*prev_it)
                                              + modelConfig.outputInstruments[*prev_it].high),
            logits.begin()
                + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                              + DenseConfig::MaxPitch * (*it)
                                              + modelConfig.outputInstruments[*it].low),
            -std::numeric_limits<float>::infinity());
    }

    std::fill(
        logits.begin()
            + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                          + DenseConfig::MaxPitch * (*prev_it)
                                          + modelConfig.outputInstruments[*prev_it].high),
        logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
        -std::numeric_limits<float>::infinity());

    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
              logits.end(),
              -std::numeric_limits<float>::infinity());
}

void ReductionTransformerV1::futureLogits(std::vector<float> &logits, const int curTime,
                                         int32_t forceAtTime) {
    if (forceAtTime != -1 && forceAtTime < DenseConfig::MaxTime) {
        std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset), forceAtTime,
                    -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset + forceAtTime + 1),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                  -std::numeric_limits<float>::infinity());
        return;
    }

    if (curTime > 0) {
        std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset), curTime,
                    -std::numeric_limits<float>::infinity());
    }

    if (curTime < (DenseConfig::MaxTime - maximumFuture)) {
        std::fill(
            logits.begin()
                + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset + curTime + maximumFuture),
            logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
            -std::numeric_limits<float>::infinity());
    }
}

auto ReductionTransformerV1::samplingForStep(int stepIdx) const -> FieldSampling {
    if (stepIdx % denseEventWidth == 1)
        return {DenseSampling::DurationTopP, getDurationTemperature()};
    if (stepIdx % denseEventWidth == 2)
        return {DenseSampling::NoteTopP, getNoteTemperature()};
    if (stepIdx % denseEventWidth == 3)
        return {DenseSampling::VelocityTopP, getVelocityTemperature()};
    return {DenseSampling::OnsetTopP, getOnsetTemperature()};
}

auto ReductionTransformerV1::durationCeilingCs(const std::vector<int32_t> &denseInputData)
    -> int32_t {
    const size_t noteCount = denseInputData.size() / static_cast<size_t>(denseEventWidth);
    const size_t take =
        std::min(noteCount, static_cast<size_t>(DenseSampling::CeilingRecentNotes));

    std::vector<int32_t> durations;
    durations.reserve(take);
    for (size_t note = noteCount - take; note < noteCount; ++note) {
        const size_t base = note * static_cast<size_t>(denseEventWidth);
        durations.push_back(juce::jmax(0, denseInputData[base + 1]
                                              - static_cast<int32_t>(DenseVocab::DurOffset)));
    }

    const int32_t recentDuration =
        juce::jmax(percentileOf(std::move(durations), DenseSampling::CeilingDurationPercentile),
                   DenseSampling::CeilingMinDurationCs);

    return DenseSampling::CeilingMultiplier * recentDuration;
}

void ReductionTransformerV1::durLogits(std::vector<float> &logits) {
    const int32_t lowest =
        juce::jlimit(0, DenseConfig::MaxDur - 1, modelConfig.outputMinimumDuration);
    const int32_t highest =
        juce::jlimit(lowest, DenseConfig::MaxDur - 1,
                     juce::jmin(modelConfig.outputMaximumDuration,
                                durationCeilingCs(inputData)));

    std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                lowest,
                -std::numeric_limits<float>::infinity());
    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset + highest + 1),
              logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
              -std::numeric_limits<float>::infinity());
}

void ReductionTransformerV1::velocityLogits(std::vector<float> &logits) {
    // Allow full velocity band; mask everything else via safeLogits.
    juce::ignoreUnused(logits);
}

void ReductionTransformerV1::safeLogits(std::vector<float> &logits, size_t stepIdx) {
    if (logits.size() < DenseVocab::VocabSize) {
        const auto detail = "logits size " + juce::String(static_cast<int>(logits.size()))
                            + " < VocabSize "
                            + juce::String(static_cast<int>(DenseVocab::VocabSize));
        DBG("ReductionTransformerV1::safeLogits: " + detail);
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
        case 1: // duration
            maskTime();
            maskNote();
            maskVel();
            break;
        case 2: // note
            maskTime();
            maskDur();
            maskVel();
            break;
        case 3: // velocity
            maskTime();
            maskDur();
            maskNote();
            break;
        default:
            break;
    }
}

std::vector<float> ReductionTransformerV1::runModelAndGetLogits(std::vector<int32_t> &tokens) {
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
        DBG("ReductionTransformerV1 ORT error: " + detail);
        reportSamplingError(detail);
    }

    return {};
}

auto ReductionTransformerV1::denseMinTime(std::vector<int32_t> &tokens) -> int32_t {
    return SamplingRelativeTime::minStridedOnset(tokens, static_cast<size_t>(denseEventWidth));
}

Token ReductionTransformerV1::generateNewToken(int32_t forceAtTime) {
    if (inputData.size() % denseEventWidth != 0)
        throw std::runtime_error("ReductionTransformerV1 inputData must be a multiple of 4");

    if (inputData.empty())
        return {-1, -1, -1, DenseConfig::DefaultVelocity};

    std::vector<int32_t> history;
    int32_t offset = 0;
    {
        const ScopedMs prepMs(&loopAccum.prep);
        history = DenseSampling::copyDenseContextSkippingProvisional(
            inputData, DenseSampling::ContextNotes, modelConfig.inputDuration, denseEventWidth);
        if (history.empty())
            return {-1, -1, -1, DenseConfig::DefaultVelocity};
        NoteWindow::sortStridedByOnset(history, static_cast<size_t>(denseEventWidth));
        offset = denseMinTime(history);
        SamplingRelativeTime::relativizeStridedOnsets(history, static_cast<size_t>(denseEventWidth),
                                                       offset);
        loopAccum.contextTokens =
            juce::jmax(loopAccum.contextTokens, static_cast<int>(history.size()));
    }

    Token newToken{-1, -1, -1, DenseConfig::DefaultVelocity};

    for (int i = 0; i < denseEventWidth; ++i) {
        std::vector<float> scores = runModelAndGetLogits(history);
        if (scores.size() < DenseVocab::VocabSize) {
            const auto detail =
                "unexpected logits size " + juce::String(static_cast<int>(scores.size()))
                + " (historyInts=" + juce::String(static_cast<int>(history.size())) + ")";
            DBG("ReductionTransformerV1::generateNewToken: " + detail);
            reportSamplingError(detail);
            return {-1, -1, -1, DenseConfig::DefaultVelocity};
        }

        {
            const ScopedMs maskMs(&loopAccum.mask);
            safeLogits(scores, static_cast<size_t>(i));
            if (i == 0)
                futureLogits(scores, currentTime - offset,
                             forceAtTime != -1 ? forceAtTime - offset : -1);
            else if (i == 1)
                durLogits(scores);
            else if (i == 2)
                instrLogits(scores);
            else
                velocityLogits(scores);
        }

        const auto fieldSampling = samplingForStep(i);
        int32_t token = 0;
        {
            const ScopedMs sampleMs(&loopAccum.sample);
            token = sampleTopP(scores, fieldSampling.topP, fieldSampling.temperature);
        }
        if (i == 1)
            publishDurationLogits(scores, token);
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

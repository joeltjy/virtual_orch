#pragma once

#include <JuceHeader.h>

/** One OT iteration's step timings (ms) and counts. */
struct OrchLoopStepMs {
    float drain = 0.0f;
    float snap = 0.0f;
    float hist = 0.0f;
    float model = 0.0f;
    float prep = 0.0f;
    float onnx = 0.0f;
    float logits = 0.0f;
    float mask = 0.0f;
    float sample = 0.0f;
    float decode = 0.0f;
    float pack = 0.0f;
    float upd = 0.0f;
    float total = 0.0f;
    int midiIn = 0;
    int reductionIn = 0;
    int getOutputCalls = 0;
    int contextTokens = 0;
};

/** Last busy reduction iteration + running max (reset when reduction threadRun starts). */
struct ReductionLoopStepMs {
    float drain = 0.0f;
    float prep = 0.0f;
    float onnx = 0.0f;
    float logits = 0.0f;
    float mask = 0.0f;
    float sample = 0.0f;
    float push = 0.0f;
    float total = 0.0f;
    int contextTokens = 0;
    int onnxCalls = 0;
};

struct ReductionLoopProfile {
    ReductionLoopStepMs current;
    ReductionLoopStepMs maxMs;

    auto publishBusy(const ReductionLoopStepMs &step) -> void {
        current = step;
        maxMs.drain = juce::jmax(maxMs.drain, step.drain);
        maxMs.prep = juce::jmax(maxMs.prep, step.prep);
        maxMs.onnx = juce::jmax(maxMs.onnx, step.onnx);
        maxMs.logits = juce::jmax(maxMs.logits, step.logits);
        maxMs.mask = juce::jmax(maxMs.mask, step.mask);
        maxMs.sample = juce::jmax(maxMs.sample, step.sample);
        maxMs.push = juce::jmax(maxMs.push, step.push);
        maxMs.total = juce::jmax(maxMs.total, step.total);
        maxMs.contextTokens = juce::jmax(maxMs.contextTokens, step.contextTokens);
        maxMs.onnxCalls = juce::jmax(maxMs.onnxCalls, step.onnxCalls);
    }
};

/** One vocsep call's step timings (ms) and counts. */
struct VocsepLoopStepMs {
    float graph = 0.0f;
    float onnx = 0.0f;
    float hungarian = 0.0f;
    float total = 0.0f;
    int nodes = 0;
    int edges = 0;
};

/** Last vocsep call + running max (reset when reduction generation starts). */
struct VocsepLoopProfile {
    VocsepLoopStepMs current;
    VocsepLoopStepMs maxMs;

    auto publishBusy(const VocsepLoopStepMs &step) -> void {
        current = step;
        maxMs.graph = juce::jmax(maxMs.graph, step.graph);
        maxMs.onnx = juce::jmax(maxMs.onnx, step.onnx);
        maxMs.hungarian = juce::jmax(maxMs.hungarian, step.hungarian);
        maxMs.total = juce::jmax(maxMs.total, step.total);
        maxMs.nodes = juce::jmax(maxMs.nodes, step.nodes);
        maxMs.edges = juce::jmax(maxMs.edges, step.edges);
    }
};

/** Last busy OT iteration + running max (reset when OT threadRun starts). */
struct OrchLoopProfile {
    OrchLoopStepMs current;
    OrchLoopStepMs maxMs;

    auto publishBusy(const OrchLoopStepMs &step) -> void {
        current = step;
        maxMs.drain = juce::jmax(maxMs.drain, step.drain);
        maxMs.snap = juce::jmax(maxMs.snap, step.snap);
        maxMs.hist = juce::jmax(maxMs.hist, step.hist);
        maxMs.model = juce::jmax(maxMs.model, step.model);
        maxMs.prep = juce::jmax(maxMs.prep, step.prep);
        maxMs.onnx = juce::jmax(maxMs.onnx, step.onnx);
        maxMs.logits = juce::jmax(maxMs.logits, step.logits);
        maxMs.mask = juce::jmax(maxMs.mask, step.mask);
        maxMs.sample = juce::jmax(maxMs.sample, step.sample);
        maxMs.decode = juce::jmax(maxMs.decode, step.decode);
        maxMs.pack = juce::jmax(maxMs.pack, step.pack);
        maxMs.upd = juce::jmax(maxMs.upd, step.upd);
        maxMs.total = juce::jmax(maxMs.total, step.total);
        maxMs.midiIn = juce::jmax(maxMs.midiIn, step.midiIn);
        maxMs.reductionIn = juce::jmax(maxMs.reductionIn, step.reductionIn);
        maxMs.getOutputCalls = juce::jmax(maxMs.getOutputCalls, step.getOutputCalls);
        maxMs.contextTokens = juce::jmax(maxMs.contextTokens, step.contextTokens);
    }
};

/** Adds elapsed ms into `out` if non-null (OT / getOutput step timing). */
struct ScopedMs {
    float *out = nullptr;
    double start = 0.0;

    explicit ScopedMs(float *field)
        : out(field),
          start(field != nullptr ? juce::Time::getMillisecondCounterHiRes() : 0.0) {}

    ~ScopedMs() {
        if (out != nullptr)
            *out += static_cast<float>(juce::Time::getMillisecondCounterHiRes() - start);
    }

    ScopedMs(const ScopedMs &) = delete;
    auto operator=(const ScopedMs &) -> ScopedMs & = delete;
};

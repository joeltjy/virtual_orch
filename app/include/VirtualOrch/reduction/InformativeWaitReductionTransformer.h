#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/InformativeWaitEngine.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * Piano -> orchestra reduction with the informative-WAIT model (overnight model 2, 2026-11-23,
 * KV + StreamingLLM export, vocab 11879). The sampling, the WAIT / COUNT / EXPIRED state machine,
 * the three modes and the batched top-5 anticipation live in InformativeWaitEngine; this class
 * connects it to the app: the user's notes (note-on now, note-off when the key goes up) go in, the
 * generated notes go to the output queue, and a note the piano cut (it came first) is removed with a
 * ClearQueue token stamped at its onset.
 *
 * Mode, pitch band and precompute come from the checkpoint JSON ("iwMode": "unconstrained" |
 * "monophonic_A" | "monophonic_B", "bandLow", "bandHigh", "precompute") and can be changed live.
 * Not supported (logged): Accompaniment reordering of late input and the Edit-mode 2 s keep (a
 * mode cut restarts the stream at now), the logit-adjustment biases and LogitTD of T3''/W3.
 */
class InformativeWaitReductionTransformer : public ReductionTransformer {
public:
    explicit InformativeWaitReductionTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }
    void threadInit() {}
    void threadRun();
    void threadStop() {}

    /** Load the KV export (vocab 11879). False + lastInitError if it cannot be used. */
    auto init(const char *modelPath) -> bool;
    /** Tests: another backend (the transformer owns it). */
    auto initWithBackend(std::unique_ptr<StreamingKv::Backend> backend) -> void;
    std::string lastInitError;

    [[nodiscard]] auto isModelLoaded() const -> bool override { return backend != nullptr; }
    [[nodiscard]] auto inputEventWidth() const -> size_t override { return 4; }

    auto setMode(InformativeWaitEngine::Mode mode) -> void { modeValue.store(static_cast<int>(mode)); }
    [[nodiscard]] auto getMode() const -> InformativeWaitEngine::Mode {
        return static_cast<InformativeWaitEngine::Mode>(modeValue.load());
    }
    auto setBand(int32_t low, int32_t high) -> void {
        bandLow.store(low);
        bandHigh.store(high);
    }
    auto setPrecompute(bool on) -> void { precompute.store(on); }
    [[nodiscard]] auto getPrecompute() const -> bool { return precompute.load(); }
    auto setOnsetTemperature(float t) -> void { onsetTemperature.store(t); }
    auto setOtherTemperature(float t) -> void { otherTemperature.store(t); }
    auto setMinP(float p) -> void { minP.store(p); }
    [[nodiscard]] auto getOnsetTemperature() const -> float { return onsetTemperature.load(); }
    [[nodiscard]] auto getOtherTemperature() const -> float { return otherTemperature.load(); }
    [[nodiscard]] auto getMinP() const -> float { return minP.load(); }

    /** The engine's counters (copied under a lock; any thread). */
    [[nodiscard]] auto statsSnapshot() const -> InformativeWaitEngine::Stats;
    [[nodiscard]] static auto modeName(InformativeWaitEngine::Mode mode) -> const char *;
    [[nodiscard]] static auto modeFromName(const juce::String &name) -> InformativeWaitEngine::Mode;

    /** App clock in cs (defaults to the Clock set by AppSession); tests replace it. */
    std::function<int32_t()> nowCsProvider;
    /** One loop iteration without waiting (tests). Returns true when notes were produced. */
    auto stepOnce() -> bool;

private:
    auto nowCs() const -> int32_t;
    auto applySettings() -> void;
    auto resetStream(int32_t now) -> void;
    auto drainInput() -> bool;
    auto emit(const std::vector<InformativeWaitEngine::OutNote> &notes) -> void;

    std::unique_ptr<StreamingKv::Backend> backend;
    InformativeWaitEngine engine;
    mutable std::mutex statsMutex;
    InformativeWaitEngine::Stats lastStats;

    std::atomic<int> modeValue{0};
    std::atomic<int32_t> bandLow{55};
    std::atomic<int32_t> bandHigh{84};
    std::atomic<bool> precompute{true};
    std::atomic<float> onsetTemperature{1.0f};
    std::atomic<float> otherTemperature{0.5f};
    std::atomic<float> minP{0.25f};
    bool streamStarted = false;
};

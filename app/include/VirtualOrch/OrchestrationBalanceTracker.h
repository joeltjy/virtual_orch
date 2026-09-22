#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include <JuceHeader.h>

#include "VirtualOrch/InstrumentConstants.h"

/**
 * Per-stream (user or model) orchestration balance: note proportions p_i and a
 * decaying diversity weight τ_i used to bias IC family-combo logits.
 *
 * Thread-safe: Launchpad (MIDI) may add/remove while OT records notes / builds bias views.
 */
class OrchestrationBalanceTracker {
public:
    static constexpr size_t kNumInstruments = InstrumentConstants::kInstrumentMappings.size();
    static constexpr float initialTau = 2.0f;
    static constexpr float decayPerSecond = 0.1f;
    static constexpr float logFloor = 1.0e-6f;

    /** Read-only snapshot for logit bias (built on the OT thread before getOutput). */
    struct BiasView {
        bool apply = false;
        uint64_t totalNotes = 0;
        std::array<float, kNumInstruments> tau{};
        std::array<float, kNumInstruments> p{};

        /** Σ_{i∈combo} τ_i · log(max(p_i, ε)); 0 when bias inactive or no history. */
        [[nodiscard]] auto comboPenalty(const std::vector<int32_t> &instrumentsInCombo) const
            -> float {
            if (! apply || totalNotes == 0 || instrumentsInCombo.empty())
                return 0.0f;
            float sum = 0.0f;
            for (const int32_t id: instrumentsInCombo) {
                if (id < 0 || static_cast<size_t>(id) >= kNumInstruments)
                    continue;
                const float pClamped = std::max(p[static_cast<size_t>(id)], logFloor);
                sum += tau[static_cast<size_t>(id)] * std::log(pClamped);
            }
            return sum;
        }
    };

    auto reset() -> void {
        const juce::ScopedLock lock(mutex);
        totalNotes = 0;
        counts.fill(0);
        active.fill(false);
        addedAtMs.fill(0);
    }

    /** Mark each id in `activeIds` as just-added at nowMs (after reset at generation start). */
    auto syncActive(const std::set<int32_t> &activeIds,
                    uint32_t nowMs = juce::Time::getMillisecondCounter()) -> void {
        const juce::ScopedLock lock(mutex);
        for (const int32_t id: activeIds)
            onInstrumentAddedUnlocked(id, nowMs);
    }

    auto onInstrumentAdded(int32_t localInstrumentId,
                           uint32_t nowMs = juce::Time::getMillisecondCounter()) -> void {
        const juce::ScopedLock lock(mutex);
        onInstrumentAddedUnlocked(localInstrumentId, nowMs);
    }

    auto onInstrumentRemoved(int32_t localInstrumentId) -> void {
        const juce::ScopedLock lock(mutex);
        if (! InstrumentConstants::isValidLocalInstrumentId(localInstrumentId))
            return;
        const auto idx = static_cast<size_t>(localInstrumentId);
        active[idx] = false;
        addedAtMs[idx] = 0;
    }

    [[nodiscard]] auto tauFor(int32_t localInstrumentId,
                              uint32_t nowMs = juce::Time::getMillisecondCounter()) const -> float {
        const juce::ScopedLock lock(mutex);
        return tauForUnlocked(localInstrumentId, nowMs);
    }

    [[nodiscard]] auto proportion(int32_t localInstrumentId) const -> float {
        const juce::ScopedLock lock(mutex);
        return proportionUnlocked(localInstrumentId);
    }

    [[nodiscard]] auto getTotalNotes() const -> uint64_t {
        const juce::ScopedLock lock(mutex);
        return totalNotes;
    }

    /** One incoming note: denom += 1; each distinct instrument on that note += 1. */
    auto recordNote(const std::vector<int32_t> &instrumentsOnThisNote) -> void {
        const juce::ScopedLock lock(mutex);
        ++totalNotes;
        std::array<bool, kNumInstruments> seen{};
        for (const int32_t id: instrumentsOnThisNote) {
            if (! InstrumentConstants::isValidLocalInstrumentId(id))
                continue;
            const auto idx = static_cast<size_t>(id);
            if (seen[idx])
                continue;
            seen[idx] = true;
            ++counts[idx];
        }
    }

    [[nodiscard]] auto makeBiasView(bool applyBias,
                                    uint32_t nowMs = juce::Time::getMillisecondCounter()) const
        -> BiasView {
        const juce::ScopedLock lock(mutex);
        BiasView view;
        view.apply = applyBias;
        view.totalNotes = totalNotes;
        for (size_t i = 0; i < kNumInstruments; ++i) {
            view.tau[i] = tauForUnlocked(static_cast<int32_t>(i), nowMs);
            view.p[i] = proportionUnlocked(static_cast<int32_t>(i));
        }
        return view;
    }

    /**
     * Jam union: proportions from `countsSource`; τ from user tracker if the id is in
     * userInstruments, else from model tracker.
     */
    [[nodiscard]] static auto makeJamBiasView(const OrchestrationBalanceTracker &user,
                                              const OrchestrationBalanceTracker &model,
                                              const OrchestrationBalanceTracker &countsSource,
                                              const std::set<int32_t> &userInstruments,
                                              const std::set<int32_t> &modelInstruments,
                                              bool applyBias,
                                              uint32_t nowMs = juce::Time::getMillisecondCounter())
        -> BiasView {
        BiasView view;
        view.apply = applyBias;
        view.totalNotes = countsSource.getTotalNotes();
        for (size_t i = 0; i < kNumInstruments; ++i) {
            const auto id = static_cast<int32_t>(i);
            view.p[i] = countsSource.proportion(id);
            if (userInstruments.count(id) != 0)
                view.tau[i] = user.tauFor(id, nowMs);
            else if (modelInstruments.count(id) != 0)
                view.tau[i] = model.tauFor(id, nowMs);
            else
                view.tau[i] = 0.0f;
        }
        return view;
    }

private:
    auto onInstrumentAddedUnlocked(int32_t localInstrumentId, uint32_t nowMs) -> void {
        if (! InstrumentConstants::isValidLocalInstrumentId(localInstrumentId))
            return;
        const auto idx = static_cast<size_t>(localInstrumentId);
        active[idx] = true;
        addedAtMs[idx] = nowMs;
    }

    [[nodiscard]] auto tauForUnlocked(int32_t localInstrumentId, uint32_t nowMs) const -> float {
        if (! InstrumentConstants::isValidLocalInstrumentId(localInstrumentId))
            return 0.0f;
        const auto idx = static_cast<size_t>(localInstrumentId);
        if (! active[idx])
            return 0.0f;
        const auto elapsedMs = nowMs - addedAtMs[idx];
        const auto seconds = static_cast<float>(elapsedMs / 1000u);
        return std::max(0.0f, initialTau - decayPerSecond * seconds);
    }

    [[nodiscard]] auto proportionUnlocked(int32_t localInstrumentId) const -> float {
        if (! InstrumentConstants::isValidLocalInstrumentId(localInstrumentId) || totalNotes == 0)
            return 0.0f;
        return static_cast<float>(counts[static_cast<size_t>(localInstrumentId)])
               / static_cast<float>(totalNotes);
    }

    mutable juce::CriticalSection mutex;
    uint64_t totalNotes = 0;
    std::array<uint64_t, kNumInstruments> counts{};
    std::array<bool, kNumInstruments> active{};
    std::array<uint32_t, kNumInstruments> addedAtMs{};
};

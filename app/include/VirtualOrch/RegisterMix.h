#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"

/**
 * Register mix (Nov 25 2026): the share of the reduction's notes in three bands, steered by logit offsets, like
 * the other logit adjustments. Set only by the mood presets (MoodPresets::Preset::registerMix); there is no direct
 * control.
 *
 *   band 0  below C3       pitch < 48
 *   band 1  C3 up to C5    48 <= pitch < 72
 *   band 2  C5 and above   pitch >= 72
 *
 * Controller (one per engine, like the tau schedulers): measured = the shares of the last kNotes committed
 * orchestra notes. At every newly committed note (none while only precomputed branches are drawn, so an
 * unplayed branch never counts), with at least kMinNotes in the window,
 *   e_i       = clamp(ln((target_i + eps) / (measured_i + eps)), -kMaxError, kMaxError)
 *   offset_i += kGain x e_i,  clamped to [-kMaxOffset, kMaxOffset], then shifted to mean 0
 * Every orchestra pitch draw adds offset_band(pitch) to the pitch logits, keeping the drawn block's total mass (at
 * the draw's temperature) as it was: only which register changes, not whether a pitch is drawn. A new target
 * starts from offsets 0. A band with target 0 is masked outright (as a register pad masks), unless that would leave
 * no pitch to draw.
 *
 * An option (Target::presetsUseMix, off by default): off, the presets set LOW / MEDIUM / HIGH / AUTO as before;
 * on, they set AUTO (C4 +/- 2.5 octaves) and their mix.
 */
namespace RegisterMix {

using Mix = std::array<float, 3>;

inline constexpr int32_t kC3 = 48;
inline constexpr int32_t kC5 = 72;
inline constexpr int kNotes = 16;
inline constexpr int kMinNotes = 4;
inline constexpr float kGain = 0.15f;
inline constexpr float kMaxError = 2.0f;
inline constexpr float kMaxOffset = 3.0f;
inline constexpr float kEps = 0.02f;

[[nodiscard]] inline auto band(int32_t pitch) -> int { return pitch < kC3 ? 0 : (pitch < kC5 ? 1 : 2); }

/** The target the presets set (LaunchpadGrid writes, the RTs read). Empty = no register mix. */
class Target {
public:
    [[nodiscard]] static auto get() -> std::optional<Mix> {
        if (! on.load(std::memory_order_acquire))
            return std::nullopt;
        return Mix{share[0].load(std::memory_order_relaxed), share[1].load(std::memory_order_relaxed),
                   share[2].load(std::memory_order_relaxed)};
    }
    /** The option: the mood presets set their mix (true) or their LOW / MEDIUM / HIGH register (false). */
    [[nodiscard]] static auto presetsUseMix() -> bool { return useMix.load(std::memory_order_relaxed); }
    static auto setPresetsUseMix(bool on) -> void {
        useMix.store(on, std::memory_order_relaxed);
        if (! on)
            set(std::nullopt);
    }

    static auto set(std::optional<Mix> mix) -> void {
        if (! mix.has_value()) {
            on.store(false, std::memory_order_release);
            return;
        }
        float total = 0.0f;
        for (const float x: *mix)
            total += std::max(x, 0.0f);
        if (total <= 0.0f) {
            on.store(false, std::memory_order_release);
            return;
        }
        for (size_t i = 0; i < 3; ++i)
            share[i].store(std::max((*mix)[i], 0.0f) / total, std::memory_order_relaxed);
        on.store(true, std::memory_order_release);
    }

private:
    static inline std::atomic<bool> on{false};
    static inline std::atomic<bool> useMix{false};
    static inline std::array<std::atomic<float>, 3> share{};
};

/** The pitch of an orchestra (instrument 1) note token of a dense quad, or -1. */
[[nodiscard]] inline auto orchestraPitch(int32_t noteToken) -> int32_t {
    const int32_t rel = noteToken - static_cast<int32_t>(DenseVocab::NoteOffset);
    if (rel < DenseConfig::MaxPitch || rel >= 2 * DenseConfig::MaxPitch)
        return -1;
    return rel - DenseConfig::MaxPitch;
}

/** The shares of the last kNotes orchestra notes of dense quads (onset, duration, note, velocity), and their count. */
[[nodiscard]] inline auto measure(const std::vector<int32_t> &quads) -> std::pair<Mix, int> {
    Mix counts{};
    int n = 0;
    for (size_t i = quads.size() / 4; i-- > 0 && n < kNotes;) {
        const int32_t pitch = orchestraPitch(quads[4 * i + 2]);
        if (pitch < 0)
            continue;
        counts[static_cast<size_t>(band(pitch))] += 1.0f;
        ++n;
    }
    if (n > 0)
        for (auto &c: counts)
            c /= static_cast<float>(n);
    return {counts, n};
}

class Controller {
public:
    /**
     * Steps once per newly committed orchestra note of `quads` (the committed history), then returns the offsets.
     * Calling it again with the same history (a precomputed branch, a second pick) changes nothing.
     */
    auto update(const std::vector<int32_t> &quads, const Mix &target) -> const Mix & {
        if (! lastTarget.has_value() || *lastTarget != target) {
            offsetsValue = Mix{};
            lastTarget = target;
        }
        int total = 0;
        int32_t lastOnset = std::numeric_limits<int32_t>::min();
        for (size_t i = 0; i + 3 < quads.size(); i += 4)
            if (orchestraPitch(quads[i + 2]) >= 0) {
                ++total;
                lastOnset = quads[i];
            }
        if (total == seenCount && lastOnset == seenLastOnset)
            return offsetsValue;
        const bool newNote = total > seenCount || lastOnset > seenLastOnset;
        seenCount = total;
        seenLastOnset = lastOnset;
        if (! newNote)
            return offsetsValue; // the history shrank (a cut, a context rebuild): nothing new to learn from
        const auto [measured, n] = measure(quads);
        if (n < kMinNotes)
            return offsetsValue;
        float mean = 0.0f;
        for (size_t i = 0; i < 3; ++i) {
            const float e = std::clamp(std::log((target[i] + kEps) / (measured[i] + kEps)), -kMaxError, kMaxError);
            offsetsValue[i] = std::clamp(offsetsValue[i] + kGain * e, -kMaxOffset, kMaxOffset);
            mean += offsetsValue[i] / 3.0f;
        }
        for (auto &o: offsetsValue)
            o -= mean;
        return offsetsValue;
    }

    [[nodiscard]] auto offsets() const -> const Mix & { return offsetsValue; }
    auto reset() -> void { *this = Controller{}; }

private:
    Mix offsetsValue{};
    std::optional<Mix> lastTarget;
    int seenCount = -1;
    int32_t seenLastOnset = std::numeric_limits<int32_t>::min();
};

/**
 * Adds offsets[band(p)] to the logits of the ids orch0 + p (p in [0, 127], finite only), masks the bands whose
 * target share is 0 (unless no finite pitch would be left), then shifts them so their total mass at `temperature`
 * is what it was.
 */
inline auto apply(std::vector<float> &logits, int32_t orch0, const Mix &offsets, float temperature,
                  const Mix &target) -> void {
    const float temp = std::max(temperature, 1e-4f);
    const auto finite = [&](int32_t p) {
        const auto id = static_cast<size_t>(orch0 + p);
        return id < logits.size() && std::isfinite(logits[id]);
    };
    bool anyKept = false;
    bool anyZeroed = false;
    for (int32_t p = 0; p < 128; ++p)
        if (finite(p))
            (target[static_cast<size_t>(band(p))] > 0.0f ? anyKept : anyZeroed) = true;
    // the total mass before any change: a masked band's mass goes to the others
    const auto massBefore = [&] {
        float peak = -std::numeric_limits<float>::infinity();
        for (int32_t p = 0; p < 128; ++p)
            if (finite(p))
                peak = std::max(peak, logits[static_cast<size_t>(orch0 + p)]);
        if (! std::isfinite(peak))
            return peak;
        double sum = 0.0;
        for (int32_t p = 0; p < 128; ++p)
            if (finite(p))
                sum += std::exp(static_cast<double>((logits[static_cast<size_t>(orch0 + p)] - peak) / temp));
        return peak / temp + static_cast<float>(std::log(sum));
    }();
    if (anyKept && anyZeroed)
        for (int32_t p = 0; p < 128; ++p)
            if (finite(p) && target[static_cast<size_t>(band(p))] <= 0.0f)
                logits[static_cast<size_t>(orch0 + p)] = -std::numeric_limits<float>::infinity();
    const auto lse = [&](bool withOffsets) {
        float peak = -std::numeric_limits<float>::infinity();
        for (int32_t p = 0; p < 128; ++p) {
            const auto id = static_cast<size_t>(orch0 + p);
            if (id < logits.size() && std::isfinite(logits[id]))
                peak = std::max(peak, logits[id] + (withOffsets ? offsets[static_cast<size_t>(band(p))] : 0.0f));
        }
        if (! std::isfinite(peak))
            return peak;
        double sum = 0.0;
        for (int32_t p = 0; p < 128; ++p) {
            const auto id = static_cast<size_t>(orch0 + p);
            if (id < logits.size() && std::isfinite(logits[id]))
                sum += std::exp(static_cast<double>(
                    (logits[id] + (withOffsets ? offsets[static_cast<size_t>(band(p))] : 0.0f) - peak) / temp));
        }
        return peak / temp + static_cast<float>(std::log(sum));
    };
    const float before = massBefore;
    const float after = lse(true);
    if (! std::isfinite(before) || ! std::isfinite(after))
        return;
    const float restore = temp * (before - after);
    for (int32_t p = 0; p < 128; ++p) {
        const auto id = static_cast<size_t>(orch0 + p);
        if (id < logits.size() && std::isfinite(logits[id]))
            logits[id] += offsets[static_cast<size_t>(band(p))] + restore;
    }
}

} // namespace RegisterMix

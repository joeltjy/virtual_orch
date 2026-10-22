#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <limits>
#include <vector>

#include "VirtualOrch/TemporalDensity.h"

/**
 * LogitTD (a temporary hack): taper long onset gaps when a high TD level is chosen on the
 * Launchpad, leaving short ones alone. With t the earliest legal onset, the logit of onset t + c
 * (c in cs) is shifted by
 *
 *   f(k, c) = -(a_k / 4) * ln(1 + (c / g_k)^4),   g_k = kWindowCs / k,   a_k = s * log2(k) / 6
 *
 * for TD level k (1..64), s = strength() (the Logit Adjustments slider). g_k is the mean gap of k
 * notes per 2 s TD window: below it f is nearly 0 (at c = g_k / 2 it is -0.016 a_k), around it f
 * bends smoothly, and beyond it the probability falls like (c / g_k)^-a_k. TD AUTO and level 1
 * leave the logits unchanged. Written by the Logit Adjustments widget, read by every reduction
 * transformer when it draws an onset.
 *
 * The shift only reshapes the FORWARD gaps (c > 0): afterwards a constant is added to all of them
 * so that, at the onset temperature, P(c = 0) (another note on the same onset, a chord) is what
 * the model gave. Without this, c = 0 is never penalised while every c > 0 is, so above some TD
 * the model stops moving forward and stacks dozens of notes on one onset. The chord is then
 * shaped on its own: with n generated notes already at t, the c = 0 logit gets
 *
 *   h(k, n) = a_k * ln(kChordTargetNotes / n)   (n >= 1; 0 for n = 0, i.e. t is past the last onset)
 *
 * so the odds of stacking are multiplied by (4 / n)^a_k: a push for small chords, neutral at 4
 * notes, a growing brake beyond, which counters the model's own pull toward ever bigger chords.
 *
 * Short forward gaps are discouraged smoothly, LogitTD on or off: c >= 1 gets
 * p(c) = -ln(1 + (kShortGapHalfCs / c)^4), which keeps 0.08% of gap 1, 6% of gap 3, half of gap 6
 * and 89% of gap 10 (c = 0, a chord, is untouched).
 *
 * A chord holds at most kMaxChordNotes generated notes: with that many already at t, c = 0 is
 * masked (LogitTD on or off), unless it is the only legal onset (a forced onset).
 *
 * Every onset draw also leaves a Snapshot of the gap distribution (softmax of the onset logits over
 * c >= 0, before and after the shift) for the Onset Gaps widget.
 */
/**
 * Per-model switches for TdOnsetBias::apply (T3'' and W2/W3 keep their own, default off). Unset
 * `logitTd` follows the global LogitTD switch (V1 / V2 / Piano Reduction).
 */
struct TdOnsetOptions {
    bool shortGapPenalty = true;
    std::optional<bool> logitTd;
};

class TdOnsetBias {
public:
    static constexpr float kDefaultStrength = 6.0f;
    static constexpr float kMaxStrength = 20.0f;
    static constexpr float kWindowCs = 200.0f;
    static constexpr float kShortGapHalfCs = 6.0f;
    static constexpr float kChordTargetNotes = 4.0f;
    static constexpr int32_t kMaxChordNotes = 10;

    [[nodiscard]] static auto enabled() -> bool { return enabledFlag.load(std::memory_order_relaxed); }
    static auto setEnabled(bool on) -> void { enabledFlag.store(on, std::memory_order_relaxed); }

    [[nodiscard]] static auto strength() -> float { return strengthValue.load(std::memory_order_relaxed); }
    static auto setStrength(float s) -> void {
        strengthValue.store(std::clamp(s, 0.0f, kMaxStrength), std::memory_order_relaxed);
    }

    /** f(k, c); 0 for AUTO, k <= 1 or c <= 0. */
    [[nodiscard]] static auto shift(int32_t tdLevel, int32_t gapCs) -> float {
        if (tdLevel <= 1 || gapCs <= 0)
            return 0.0f;
        const float k = static_cast<float>(tdLevel);
        const float exponent = strength() * std::log2(k) / 6.0f;
        const float x = static_cast<float>(gapCs) * k / kWindowCs; // c / g_k
        const float x2 = x * x;
        return -(exponent / 4.0f) * std::log1p(x2 * x2);
    }

    /** p(c): the short-gap penalty, 0 for c <= 0. */
    [[nodiscard]] static auto shortGapPenalty(int32_t gapCs) -> float {
        if (gapCs <= 0)
            return 0.0f;
        const float r = kShortGapHalfCs / static_cast<float>(gapCs);
        return -std::log1p(r * r * r * r);
    }

    /** h(k, n): the chord shift with n generated notes already at t; 0 for AUTO, k <= 1, n <= 0. */
    [[nodiscard]] static auto chordShift(int32_t tdLevel, int32_t chordNotes) -> float {
        if (tdLevel <= 1 || chordNotes <= 0)
            return 0.0f;
        const float exponent = strength() * std::log2(static_cast<float>(tdLevel)) / 6.0f;
        return exponent * std::log(kChordTargetNotes / static_cast<float>(chordNotes));
    }

    using Options = TdOnsetOptions;

    /** Gap bins shown by the widget: c = 0..kSnapshotGaps-1 cs; the rest is the tail. */
    static constexpr int32_t kSnapshotGaps = 400;

    struct Snapshot {
        uint64_t sequence = 0; // 0 = no onset drawn yet
        int32_t tdLevel = TemporalDensity::kAuto;
        bool shifted = false;  // LogitTD changed the logits for this draw
        int32_t chordNotes = 0; // generated notes already at t
        std::array<float, kSnapshotGaps> before{}; // P(gap = c), softmax before the shift
        std::array<float, kSnapshotGaps> after{};  // the same after it
        float beforeTail = 0.0f; // P(gap >= kSnapshotGaps)
        float afterTail = 0.0f;
    };

    [[nodiscard]] static auto latestSnapshot() -> Snapshot {
        const std::lock_guard<std::mutex> lock(snapshotMutex());
        return snapshotStore();
    }

    /**
     * Onset logits logits[timeOffset + t] (t = 0..maxTime-1, model time), c = t - fromModelTime:
     * add p(c) to every c >= 1; then (LogitTD on, TD > 1) add f(tdLevel, c) to every c > 0,
     * renormalise the c > 0 mass back to what it was, and add h(tdLevel, chordNotes) to c = 0.
     * `temperature` is the onset field's: the renormalisation and the snapshot use it, so they
     * match what is sampled (top-p 0.98 aside). Masked (-inf) and earlier onsets are left.
     */
    static auto apply(std::vector<float> &logits, int32_t tdLevel, int32_t fromModelTime,
                      size_t timeOffset, int32_t maxTime, float temperature = 1.0f,
                      int32_t chordNotes = 0, const Options &options = {}) -> void {
        const float temp = temperature > 1e-4f ? temperature : 1e-4f;
        const int32_t from = std::max(fromModelTime, 0);
        const int32_t end = std::min(maxTime, static_cast<int32_t>(logits.size()) - static_cast<int32_t>(timeOffset));
        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        if (options.shortGapPenalty)
            for (int32_t t = std::max(from, fromModelTime + 1); t < end; ++t) {
                float &logit = logits[timeOffset + static_cast<size_t>(t)];
                if (std::isfinite(logit))
                    logit += shortGapPenalty(t - fromModelTime);
            }
        if (chordNotes >= kMaxChordNotes && fromModelTime >= from && fromModelTime < end) {
            bool otherLegal = false;
            for (int32_t t = fromModelTime + 1; t < end && ! otherLegal; ++t)
                otherLegal = std::isfinite(logits[timeOffset + static_cast<size_t>(t)]);
            if (otherLegal)
                logits[timeOffset + static_cast<size_t>(fromModelTime)] = kNegInf;
        }

        Snapshot snap;
        snap.tdLevel = tdLevel;
        snap.shifted = options.logitTd.value_or(enabled()) && tdLevel != TemporalDensity::kAuto
                       && tdLevel > 1;
        snap.chordNotes = chordNotes;
        snap.beforeTail = distribution(logits, from, end, timeOffset, fromModelTime, temp, snap.before);
        if (snap.shifted) {
            // log-sum-exp of the forward gaps (c > 0) at the temperature, before and after
            const int32_t firstForward = std::max(from, fromModelTime + 1);
            const auto forwardLse = [&] {
                float peak = kNegInf;
                for (int32_t t = firstForward; t < end; ++t)
                    peak = std::max(peak, logits[timeOffset + static_cast<size_t>(t)]);
                if (! std::isfinite(peak))
                    return kNegInf;
                double sum = 0.0;
                for (int32_t t = firstForward; t < end; ++t) {
                    const float l = logits[timeOffset + static_cast<size_t>(t)];
                    if (std::isfinite(l))
                        sum += std::exp(static_cast<double>((l - peak) / temp));
                }
                return peak / temp + static_cast<float>(std::log(sum));
            };
            const float lseBefore = forwardLse();
            for (int32_t t = firstForward; t < end; ++t) {
                float &logit = logits[timeOffset + static_cast<size_t>(t)];
                if (std::isfinite(logit))
                    logit += shift(tdLevel, t - fromModelTime);
            }
            const float lseAfter = forwardLse();
            if (std::isfinite(lseBefore) && std::isfinite(lseAfter)) {
                const float restore = temp * (lseBefore - lseAfter);
                for (int32_t t = firstForward; t < end; ++t) {
                    float &logit = logits[timeOffset + static_cast<size_t>(t)];
                    if (std::isfinite(logit))
                        logit += restore;
                }
            }
            if (fromModelTime >= from && fromModelTime < end) {
                float &chord = logits[timeOffset + static_cast<size_t>(fromModelTime)];
                if (std::isfinite(chord))
                    chord += chordShift(tdLevel, chordNotes);
            }
        }
        snap.afterTail = snap.shifted
                             ? distribution(logits, from, end, timeOffset, fromModelTime, temp, snap.after)
                             : snap.beforeTail;
        if (! snap.shifted)
            snap.after = snap.before;
        const std::lock_guard<std::mutex> lock(snapshotMutex());
        snap.sequence = snapshotStore().sequence + 1;
        snapshotStore() = snap;
    }

private:
    /** Softmax over the finite onset logits from `from` on, binned by gap; returns the tail mass. */
    static auto distribution(const std::vector<float> &logits, int32_t from, int32_t end,
                             size_t timeOffset, int32_t fromModelTime, float temperature,
                             std::array<float, kSnapshotGaps> &bins) -> float {
        bins.fill(0.0f);
        float peak = -std::numeric_limits<float>::infinity();
        for (int32_t t = from; t < end; ++t)
            peak = std::max(peak, logits[timeOffset + static_cast<size_t>(t)]);
        if (! std::isfinite(peak))
            return 0.0f;
        double total = 0.0;
        double tail = 0.0;
        for (int32_t t = from; t < end; ++t) {
            const float logit = logits[timeOffset + static_cast<size_t>(t)];
            if (! std::isfinite(logit))
                continue;
            const double p = std::exp(static_cast<double>((logit - peak) / temperature));
            total += p;
            const int32_t gap = t - fromModelTime;
            if (gap >= 0 && gap < kSnapshotGaps)
                bins[static_cast<size_t>(gap)] += static_cast<float>(p);
            else
                tail += p;
        }
        for (auto &b: bins)
            b = static_cast<float>(b / total);
        return static_cast<float>(tail / total);
    }

    static auto snapshotMutex() -> std::mutex & {
        static std::mutex m;
        return m;
    }
    static auto snapshotStore() -> Snapshot & {
        static Snapshot s;
        return s;
    }

    static inline std::atomic<bool> enabledFlag{true};
    static inline std::atomic<float> strengthValue{kDefaultStrength};
};

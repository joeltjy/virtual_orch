#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "VirtualOrch/TemporalDensity.h"

/**
 * Dynamic LogitTD: steers one reduction model's LogitTD strength gamma so the orchestra's
 * measured temporal density follows the TD chosen on the Launchpad. One controller per model
 * (Hybrid: v1 and W), each updated only while its model draws.
 *
 * Every tick (kTickMs):
 *   e     = clamp(log2(target / measured), -kMaxError, kMaxError)       (TD steps)
 *   gamma = gamma + K e,     K = (kTickMs / kTauMs) / max(S, kMinSensitivity) x N / (N + kNoiseN) x m
 *
 *  - measured: the orchestra's onsets in the 3 s ending at its newest generated note, in TD units
 *    (onsets per 2 s; at least one onset counted, so silence is a finite error).
 *  - S: the model's sensitivity d log2(density) / d gamma, the mean over its onset draws in the
 *    tick (TdOnsetBias::sensitivity), smoothed.
 *  - N = 1.5 x target: the onsets expected in 3 s at the target. Counting noise is about 1 / sqrt(N),
 *    so K shrinks by N / (N + 4) and errors below 1 / sqrt(N) (the dead zone) leave gamma alone.
 *  - m: every kCheckTicks updates, halved if the error changed sign at any update since the
 *    last check (overshoot), x 1.5 (up to kMaxM) if it kept its sign and shrank by less than a
 *    quarter (too slow).
 *  - gamma stays in [kMinGamma, kMaxGamma] and does not integrate further into a limit.
 *  - Frozen (gamma kept, not updated) at TD AUTO, while the clock is stopped, while the model
 *    waits (W2/W3 WAIT), and in a tick in which it drew nothing. A new target resets m.
 */
class TdController {
public:
    static constexpr double kTickMs = 250.0;
    static constexpr double kTauMs = 2500.0;
    static constexpr double kMinSensitivity = 0.05;
    static constexpr double kNoiseN = 4.0;
    static constexpr double kMaxError = 2.0;
    static constexpr double kMinGamma = -10.0;
    static constexpr double kMaxGamma = 20.0;
    static constexpr int kCheckTicks = static_cast<int>(kTauMs / kTickMs);
    static constexpr double kMaxM = 2.0;
    static constexpr double kMinM = 0.125;
    static constexpr double kSmoothing = 0.3; // weight of a new tick's sensitivity
    static constexpr int32_t kWindowCs = 300;  // measured over 3 s
    /** The floor of the measured density: one onset in the window, in TD units. */
    static constexpr double kMinMeasured = 200.0 / kWindowCs;

    struct Input {
        int32_t target = TemporalDensity::kAuto; // the Launchpad's TD
        std::optional<double> measured;          // TD units; empty = no notes in the window
        std::optional<double> sensitivity;       // this tick's mean S; empty = the model drew nothing
        bool clockRunning = true;
        bool waiting = false;
    };

    struct State {
        int32_t target = TemporalDensity::kAuto;
        double measured = 0.0;
        double error = 0.0;
        double gamma = 0.0;
        double k = 0.0;           // the K of the last update (0 while frozen)
        double sensitivity = 0.0; // smoothed S
        double m = 1.0;
        bool frozen = true;
        std::string why = "TD AUTO"; // why it is frozen, or "dead zone" / "at limit" / ""
    };

    /** The density measured from onsets: distinct onsets in the window, in onsets per 2 s. */
    [[nodiscard]] static auto measure(int onsetsInWindow) -> double {
        return static_cast<double>(onsetsInWindow) * 200.0 / static_cast<double>(kWindowCs);
    }

    auto tick(const Input &in) -> const State & {
        if (in.target != state.target) {
            state.m = 1.0;
            checkCount = 0;
            lastCheckError.reset();
            lastUpdateError.reset();
            signChanged = false;
        }
        state.target = in.target;
        state.measured = in.measured.value_or(0.0);
        if (in.sensitivity.has_value())
            state.sensitivity = haveSensitivity ? (1.0 - kSmoothing) * state.sensitivity + kSmoothing * *in.sensitivity
                                                : *in.sensitivity;
        haveSensitivity = haveSensitivity || in.sensitivity.has_value();
        state.k = 0.0;
        const auto freeze = [&](const char *why) -> const State & {
            state.frozen = true;
            state.why = why;
            return state;
        };
        if (in.target == TemporalDensity::kAuto || in.target < 1)
            return freeze("TD AUTO");
        const double target = static_cast<double>(in.target);
        state.error = std::clamp(std::log2(target / std::max(state.measured, kMinMeasured)), -kMaxError, kMaxError);
        if (! in.clockRunning)
            return freeze("clock stopped");
        if (in.waiting)
            return freeze("waiting");
        if (! in.sensitivity.has_value())
            return freeze("not drawing");
        state.frozen = false;

        const double n = 1.5 * target;
        if (std::abs(state.error) < 1.0 / std::sqrt(n)) {
            state.why = "dead zone";
            return state;
        }
        state.k = (kTickMs / kTauMs) / std::max(state.sensitivity, kMinSensitivity) * n / (n + kNoiseN) * state.m;
        const double next = state.gamma + state.k * state.error;
        if ((state.gamma >= kMaxGamma && state.error > 0) || (state.gamma <= kMinGamma && state.error < 0)) {
            state.why = "at limit"; // no further into the limit
        } else {
            state.gamma = std::clamp(next, kMinGamma, kMaxGamma);
            state.why = "";
        }

        if (lastUpdateError.has_value() && (state.error > 0) != (*lastUpdateError > 0))
            signChanged = true;
        lastUpdateError = state.error;
        if (++checkCount >= kCheckTicks) {
            checkCount = 0;
            if (lastCheckError.has_value()) {
                if (signChanged)
                    state.m = std::max(kMinM, state.m * 0.5);
                else if (std::abs(state.error) > 0.75 * std::abs(*lastCheckError))
                    state.m = std::min(kMaxM, state.m * 1.5);
            }
            lastCheckError = state.error;
            signChanged = false;
        }
        return state;
    }

    [[nodiscard]] auto current() const -> const State & { return state; }

private:
    State state;
    bool haveSensitivity = false;
    int checkCount = 0;
    std::optional<double> lastCheckError;
    std::optional<double> lastUpdateError;
    bool signChanged = false;
};

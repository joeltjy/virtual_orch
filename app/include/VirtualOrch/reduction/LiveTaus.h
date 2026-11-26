#pragma once

#include <cstdio>
#include <string>

/**
 * The bias strengths an engine last drew with (the tau schedulers' output, Nov 26 2026): logged with each
 * RT_DRAW as "taus" so the sample viewer can plot them over time. -1 = that bias has not been applied yet.
 */
struct LiveTaus {
    float pitchTau = -1.0f;
    float pitchTauPrime = -1.0f;
    float durationTau = -1.0f;
    float durationTauPrime = -1.0f;
    float onsetGapTau = -1.0f;
    /** The register mix at the latest pitch draw (RegisterMix.h): target and measured shares, offsets; per band. */
    bool hasMix = false;
    float mixTarget[3] = {0, 0, 0};
    float mixMeasured[3] = {0, 0, 0};
    float mixOffset[3] = {0, 0, 0};
    int mixNotes = 0;

    [[nodiscard]] auto json() const -> std::string {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "{\"pitch_tau\":%.4g,\"pitch_tau_prime\":%.4g,\"dur_tau\":%.4g,\"dur_tau_prime\":%.4g,"
                      "\"gap_tau\":%.4g}",
                      pitchTau, pitchTauPrime, durationTau, durationTauPrime, onsetGapTau);
        std::string s = buf;
        if (hasMix) {
            char m[300];
            std::snprintf(m, sizeof(m),
                          ",\"mix\":{\"target\":[%.3f,%.3f,%.3f],\"measured\":[%.3f,%.3f,%.3f],"
                          "\"offset\":[%.3f,%.3f,%.3f],\"notes\":%d}",
                          mixTarget[0], mixTarget[1], mixTarget[2], mixMeasured[0], mixMeasured[1], mixMeasured[2],
                          mixOffset[0], mixOffset[1], mixOffset[2], mixNotes);
            s.insert(s.size() - 1, m);
        }
        return s;
    }

    /** Records the register mix of a pitch draw (target empty: none). */
    template <typename Mix>
    auto recordMix(const Mix *target, const Mix &measured, int notes, const Mix &offsets) -> void {
        hasMix = target != nullptr;
        if (! hasMix)
            return;
        for (int i = 0; i < 3; ++i) {
            mixTarget[i] = (*target)[static_cast<size_t>(i)];
            mixMeasured[i] = measured[static_cast<size_t>(i)];
            mixOffset[i] = offsets[static_cast<size_t>(i)];
        }
        mixNotes = notes;
    }
};

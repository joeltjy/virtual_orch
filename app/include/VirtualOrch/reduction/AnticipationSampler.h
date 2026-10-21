#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/DenseOnsetGapBias.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/DenseTypes.h"

/**
 * Field sampling shared by the three anticipation-style models (T3'', T8'', T7''): for each of the
 * four fields of an instrument-1 event quad (onset, duration, note, velocity)
 *
 *   1. the app's biases on the raw logits (onset-gap 0.5, duration 0.5 / 0, pitch 0.2 / 0.2; no
 *      per-instrument adjustment), each from the history of the notes this sampler generated;
 *   2. masks: everything but the legal ids of the field is -inf (onset >= the lower bound,
 *      durations 10..995 step 5, instrument 1 pitches A0..C8, velocities 8..120 step 8; given,
 *      control, TD and special ids are never legal);
 *   3. top-p 0.98, then the temperature, then a multinomial draw.
 *
 * Event ids are the same in the dense vocab (11776) and the AMT-native vocab (55293): onset
 * 0..9999, duration 10000 + d, note 11000 + 128 * instr + pitch; only the velocity base differs
 * (dense 11640, AMT 55028).
 */
class AnticipationSampler {
public:
    static constexpr float kTopP = 0.98f;
    static constexpr int32_t kEventInstrument = 1;
    static constexpr int32_t kDenseVelocityBase = 11640;
    static constexpr int32_t kAmtVelocityBase = 55028;

    struct Context {
        int32_t base = 0;           // origin: model time = absolute - base
        int32_t lowerOnsetModel = 0; // onset ids below this are masked
        int32_t onsetAbs = 0;       // the onset already chosen (fields 1..3)
        int32_t clockAbs = 0;       // max(app clock, last generated onset): the bias "now"
        int32_t pitchLow = DenseQuantize::PianoKeyLow;   // register limits (inclusive)
        int32_t pitchHigh = DenseQuantize::PianoKeyHigh;
    };

    explicit AnticipationSampler(int32_t velocityBaseIn) : velocityBase(velocityBaseIn) {}

    bool pitchBias = true;
    bool durationBias = true;
    bool onsetGapBias = true;
    float pitchTau = DensePitchBias::TauBase;
    float pitchTauPrime = DensePitchBias::TauBase;
    float durationTau = DenseDurationBias::TauBase;
    float durationTauPrime = DenseDurationBias::TauPrimeBase;
    float onsetGapTau = DenseOnsetGapBias::TauBase;

    auto reset() -> void {
        generated.clear();
        pitchScheduler.reset();
        durationScheduler.reset();
    }

    /** A note this stage generated (absolute onset) joins the bias history. */
    auto recordGenerated(int32_t onsetAbs, int32_t durCs, int32_t pitch, int32_t velocity) -> void {
        generated.push_back(onsetAbs);
        generated.push_back(static_cast<int32_t>(DenseVocab::DurOffset) + durCs);
        generated.push_back(static_cast<int32_t>(DenseVocab::NoteOffset)
                            + DenseConfig::MaxPitch * kEventInstrument + pitch);
        generated.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity);
        constexpr size_t kKeep = 64 * 4;
        if (generated.size() > kKeep)
            generated.erase(generated.begin(),
                            generated.end() - static_cast<std::ptrdiff_t>(kKeep));
    }

    /** Sample one field (0 onset, 1 duration, 2 note, 3 velocity) from the model's logits. */
    [[nodiscard]] auto sample(int field, const std::vector<float> &logits, const Context &ctx,
                              float temperature, float topP = kTopP) -> int32_t {
        std::vector<float> raw = logits;
        applyBiases(field, raw, ctx);
        auto scores = masked(field, raw, ctx);
        const auto hasFinite = [](const std::vector<float> &v) {
            return std::any_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
        };
        // Biases must never leave nothing to sample (sampleTopP would return token 0, an
        // illegal id): fall back to the model's own logits.
        if (! hasFinite(scores))
            scores = masked(field, logits, ctx);
        return sampleTopP(scores, topP, temperature);
    }

    /** The legal-id mask of a field applied to `logits` (illegal ids -> -inf). */
    [[nodiscard]] auto masked(int field, const std::vector<float> &logits, const Context &ctx) const
        -> std::vector<float> {
        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(logits.size(), kNegInf);
        const auto allow = [&](int32_t id) {
            if (id >= 0 && static_cast<size_t>(id) < logits.size())
                scores[static_cast<size_t>(id)] = logits[static_cast<size_t>(id)];
        };
        switch (field) {
            case 0:
                for (int32_t t = std::max(ctx.lowerOnsetModel, 0); t <= 9999; ++t)
                    allow(t);
                break;
            case 1:
                for (int32_t cs = DenseQuantize::DurationMinCs; cs <= DenseQuantize::DurationMaxCs;
                     cs += DenseQuantize::DurationStepCs)
                    allow(static_cast<int32_t>(DenseVocab::DurOffset) + cs);
                break;
            case 2: {
                // piano range, narrowed to the register; an empty intersection keeps the piano range
                int32_t lo = std::max(DenseQuantize::PianoKeyLow, ctx.pitchLow);
                int32_t hi = std::min(DenseQuantize::PianoKeyHigh, ctx.pitchHigh);
                if (lo > hi) {
                    lo = DenseQuantize::PianoKeyLow;
                    hi = DenseQuantize::PianoKeyHigh;
                }
                for (int32_t p = lo; p <= hi; ++p)
                    allow(static_cast<int32_t>(DenseVocab::NoteOffset)
                          + DenseConfig::MaxPitch * kEventInstrument + p);
                break;
            }
            default:
                // v = 0 would read as a note-off: only 8..120
                for (int32_t v = DenseQuantize::VelocityStep; v <= DenseQuantize::VelocityMax;
                     v += DenseQuantize::VelocityStep)
                    allow(velocityBase + v);
                break;
        }
        return scores;
    }

private:
    auto applyBiases(int field, std::vector<float> &raw, const Context &ctx) -> void {
        if (field == 0 && onsetGapBias && ! generated.empty()) {
            const auto window = DenseOnsetGapBias::collectOnsetGapWindow(generated, ctx.clockAbs, -1);
            DenseOnsetGapBias::applyOnsetLogitsBias(raw, window, onsetGapTau, ctx.base);
        } else if (field == 1 && durationBias) {
            const auto window = DenseDurationBias::collectDurationWindowAt(
                generated, ctx.onsetAbs, -1, kEventInstrument);
            const auto scheduled = durationScheduler.evaluate(
                ctx.onsetAbs, window.uniqueDurations, window.uniqueDeltas, durationTau,
                durationTauPrime, true);
            DenseDurationBias::applyDurationLogitsBias(raw, window, scheduled.tau,
                                                       scheduled.tauPrime);
        } else if (field == 2 && pitchBias) {
            const auto window = DensePitchBias::collectPitchWindowAt(generated, ctx.onsetAbs, -1);
            const auto scheduled = pitchScheduler.evaluate(
                ctx.onsetAbs, window.uniquePitches, window.uniqueDeltas, pitchTau, pitchTauPrime,
                true);
            DensePitchBias::applyNoteLogitsBias(raw, window, scheduled.tau, scheduled.tauPrime);
        }
    }

    int32_t velocityBase;
    std::vector<int32_t> generated; // dense quads, absolute onsets
    DensePitchBias::PitchTauScheduler pitchScheduler;
    DenseDurationBias::DurationTauScheduler durationScheduler;
};

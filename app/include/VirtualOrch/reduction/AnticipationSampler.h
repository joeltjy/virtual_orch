#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/DenseOnsetGapBias.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/TdOnsetBias.h"

/**
 * Field sampling shared by the three anticipation-style models (T3'', T8'', T7''): for each of the
 * four fields of an instrument-1 event quad (onset, duration, note, velocity)
 *
 *   1. the app's biases on the raw logits (onset-gap 0.5, duration 0.5 / 0, pitch 0.2 / 0.2; no
 *      per-instrument adjustment), each from the history of the notes this sampler generated;
 *   2. masks: everything but the legal ids of the field is -inf (onset >= the lower bound,
 *      durations 10..995 step 5, instrument 1 pitches A0..C8, velocities 8..120 step 8; given,
 *      control, TD and special ids are never legal);
 *      the onset also gets the LogitTD shift (TdOnsetBias) from the lower bound on;
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
        int32_t upperOnsetModel = INT32_MAX; // ... and above this (W2/W3: at most 3 s ahead)
        int32_t onsetAbs = 0;       // the onset already chosen (fields 1..3)
        int32_t clockAbs = 0;       // max(app clock, last generated onset): the bias "now"
        int32_t pitchLow = DenseQuantize::PianoKeyLow;   // register limits (inclusive)
        int32_t pitchHigh = DenseQuantize::PianoKeyHigh;
        /** W2/W3 tie rule: a generated note at the onset of the previous one is not lower. */
        int32_t pitchFloor = INT32_MIN;
    };

    /** Longest duration drawn (5 s, as playback caps every reduction note). */
    static constexpr int32_t kMaxDurationCs = 500;

    explicit AnticipationSampler(int32_t velocityBaseIn) : velocityBase(velocityBaseIn) {}

    bool pitchBias = true;
    bool durationBias = true;
    bool onsetGapBias = true;
    float pitchTau = DensePitchBias::TauBase;
    float pitchTauPrime = DensePitchBias::TauBase;
    float durationTau = DenseDurationBias::TauBase;
    float durationTauPrime = DenseDurationBias::TauPrimeBase;
    float onsetGapTau = DenseOnsetGapBias::TauBase;
    /** Launchpad TD value for the LogitTD onset shift (TemporalDensity::kAuto = none). */
    int32_t logitTdLevel = TemporalDensity::kAuto;
    /** Short-gap penalty and LogitTD for this model (TdOnsetBias::Options). */
    TdOnsetBias::Options onsetOptions;

    /**
     * W2/W3 threshold rule: P(WAIT) over the onset step's masked distribution (onset ids from the
     * lower bound, plus waitId) at temperature 1, before top-p, after this model's adjustments.
     * True = WAIT: P(play) = 1 - P(WAIT) is below `minPlayProbability`.
     */
    [[nodiscard]] auto decideWait(const std::vector<float> &logits, const Context &ctx, int32_t waitId,
                                  float minPlayProbability) -> bool {
        if (waitId < 0 || static_cast<size_t>(waitId) >= logits.size())
            return false;
        std::vector<float> raw = logits;
        applyBiases(0, raw, ctx);
        TdOnsetBias::apply(raw, logitTdLevel, ctx.lowerOnsetModel, 0, DenseConfig::MaxTime, 1.0f,
                           chordNotesAt(ctx.lowerOnsetModel + ctx.base), onsetOptions);
        const auto onset = masked(0, raw, ctx);
        const float waitLogit = logits[static_cast<size_t>(waitId)];
        if (! std::isfinite(waitLogit))
            return false;
        float peak = waitLogit;
        for (const float l: onset)
            if (std::isfinite(l))
                peak = std::max(peak, l);
        double play = 0.0;
        for (const float l: onset)
            if (std::isfinite(l))
                play += std::exp(static_cast<double>(l - peak));
        const double wait = std::exp(static_cast<double>(waitLogit - peak));
        lastWaitProbability = static_cast<float>(wait / (wait + play));
        return 1.0f - lastWaitProbability < minPlayProbability;
    }
    /** P(WAIT) at the last decideWait (for the log / UI). */
    float lastWaitProbability = 0.0f;

    auto reset() -> void {
        generated.clear();
        pitchScheduler.reset();
        durationScheduler.reset();
    }

    /** Leaving Accompaniment: the generated notes from cutCs on leave the bias history. */
    auto forgetGeneratedFrom(int32_t cutCs) -> void {
        std::vector<int32_t> kept;
        kept.reserve(generated.size());
        for (size_t i = 0; i + 4 <= generated.size(); i += 4)
            if (generated[i] < cutCs)
                kept.insert(kept.end(), generated.begin() + static_cast<std::ptrdiff_t>(i),
                            generated.begin() + static_cast<std::ptrdiff_t>(i + 4));
        generated = std::move(kept);
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
        if (field == 0)
            TdOnsetBias::apply(raw, logitTdLevel, ctx.lowerOnsetModel, 0, DenseConfig::MaxTime,
                               temperature, chordNotesAt(ctx.lowerOnsetModel + ctx.base), onsetOptions);
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
                for (int32_t t = std::max(ctx.lowerOnsetModel, 0);
                     t <= std::min(9999, std::max(ctx.upperOnsetModel, ctx.lowerOnsetModel)); ++t)
                    allow(t);
                break;
            case 1:
                // at most ReductionTransformer::kMaxNoteDurationCs (5 s)
                for (int32_t cs = DenseQuantize::DurationMinCs; cs <= std::min(DenseQuantize::DurationMaxCs, kMaxDurationCs);
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
                if (ctx.pitchFloor > lo && ctx.pitchFloor <= hi)
                    lo = ctx.pitchFloor; // ascending order at a shared onset (W2/W3)
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
    /** Notes this sampler generated with absolute onset exactly onsetAbs. */
    [[nodiscard]] auto chordNotesAt(int32_t onsetAbs) const -> int32_t {
        int32_t n = 0;
        for (size_t i = 0; i + 4 <= generated.size(); i += 4)
            if (generated[i] == onsetAbs)
                ++n;
        return n;
    }

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

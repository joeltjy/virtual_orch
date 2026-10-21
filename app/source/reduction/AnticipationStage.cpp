#include "VirtualOrch/reduction/AnticipationStage.h"

#include "VirtualOrch/TemporalDensity.h"

#include <algorithm>

auto AnticipationStage::generate(const Params &params) -> std::optional<AnticipatedStream::Note> {
    lastError.clear();
    if (cache == nullptr)
        return std::nullopt;
    const int32_t now = params.now;

    // Earliest onset: never in the past, raised by the polyphony cap; a rest moves it up.
    int32_t floorAbs = std::max(params.lowerAbs, now);
    while (stream.needsRebase(floorAbs))
        stream.rebase();

    if (visibleRests) {
        // Silence keeps moving: one visible rest per second until the clock is within a step of now.
        while (stream.clock() + kRestStepCs <= now) {
            stream.appendRest(stream.clock() + kRestStepCs);
            stream.insertDue(stream.clock(), now);
        }
    }
    const int32_t clock0 = visibleRests ? stream.clock() : std::max(stream.clock(), now);
    const bool tieFirst = tieFirstGenerated;
    // Given notes the clock has reached go in first. With the T3'' tie order a given note exactly
    // at the clock waits for the first generated note at that onset, except that an empty stream
    // starts with the given note (the model needs a first token to condition on).
    stream.insertDue(clock0, now, ! tieFirst || stream.eventEmittedAt(clock0) || stream.isEmpty());
    // The cache evicted a note older than 30 s: move the origin, rebuild from the retained window.
    if (stream.rebaseForEviction(cache->evictedTokens(), cache->sinkRows()) > 0)
        cache->reset();

    sampler.pitchBias = params.pitchBias && params.biases;
    sampler.durationBias = params.biases;
    sampler.onsetGapBias = params.biases;
    AnticipationSampler::Context ctx;
    int32_t onsetModel = 0;
    bool drawn = false;
    for (int redraw = 0; redraw < kMaxRedraws; ++redraw) {
        const auto tokens = stream.tokens();
        if (stream.isEmpty())
            return std::nullopt; // nothing to continue from yet
        const auto &logits = cache->logitsAfter(tokens, stream.committedLen());
        if (logits.empty() || logits.size() < static_cast<size_t>(DenseVocab::VocabSize)) {
            lastError = "KV cache returned no usable logits";
            return std::nullopt;
        }
        ctx.base = stream.base();
        ctx.lowerOnsetModel = std::max(floorAbs, stream.clock()) - ctx.base;
        ctx.clockAbs = std::max(now, stream.clock());
        onsetModel = sampler.sample(0, logits, ctx, params.temperatures[0]);
        const int32_t onsetAbs = onsetModel + ctx.base;

        // Next given note (the AMT melody controls are released by the clock alone) and next rest.
        const int32_t nextGiven =
            visibleRests ? stream.nextPendingGivenActual() : stream.nextPendingActual();
        bool needGiven = nextGiven < onsetAbs
                         || (nextGiven == onsetAbs
                             && (! tieFirst || stream.eventEmittedAt(onsetAbs) || stream.isEmpty()));
        const int32_t nextRest = stream.clock() + kRestStepCs;
        bool needRest = visibleRests && nextRest < onsetAbs;
        if (needGiven && needRest) {
            if (nextGiven <= nextRest)
                needRest = false;
            else
                needGiven = false;
        }
        if (needGiven && stream.insertNext(now, /*givenOnly*/ visibleRests))
            continue;
        if (needRest) {
            stream.appendRest(nextRest);
            stream.insertDue(nextRest, now);
            floorAbs = std::max(floorAbs, nextRest + 1);
            continue;
        }
        drawn = true;
        break;
    }
    if (! drawn) {
        lastError = "no onset found";
        return std::nullopt;
    }

    const auto tokens = stream.tokens();
    const size_t committed = stream.committedLen();
    const int32_t base = stream.base();
    ctx.base = base;
    ctx.onsetAbs = onsetModel + base;
    ctx.lowerOnsetModel = 0;
    ctx.pitchLow = params.pitchLow;
    ctx.pitchHigh = params.pitchHigh;
    std::vector<int32_t> partial{onsetModel};
    std::array<int32_t, 4> chosen{onsetModel, 0, 0, 0};
    for (int field = 1; field < 4; ++field) {
        std::vector<int32_t> desired = tokens;
        desired.insert(desired.end(), partial.begin(), partial.end());
        const auto &logits = cache->logitsAfter(desired, std::min(committed, desired.size()));
        if (logits.empty() || logits.size() < static_cast<size_t>(DenseVocab::VocabSize)) {
            lastError = "KV cache returned no usable logits";
            return std::nullopt;
        }
        const int32_t token = sampler.sample(field, logits, ctx, params.temperatures[static_cast<size_t>(field)]);
        chosen[static_cast<size_t>(field)] = token;
        partial.push_back(token);
    }

    const int32_t velocityBase = stream.vocabKind() == AnticipatedStream::Vocab::Amt
                                     ? AnticipationSampler::kAmtVelocityBase
                                     : AnticipationSampler::kDenseVelocityBase;
    AnticipatedStream::Note note;
    note.onsetCs = chosen[0] + base;
    note.durationCs = chosen[1] - static_cast<int32_t>(DenseVocab::DurOffset);
    note.pitch = chosen[2] - static_cast<int32_t>(DenseVocab::NoteOffset)
                 - DenseConfig::MaxPitch * AnticipatedStream::kEventInstrument;
    note.velocity = chosen[3] - velocityBase;
    stream.appendEvent(note);
    sampler.recordGenerated(note.onsetCs, note.durationCs, note.pitch, note.velocity);
    return note;
}

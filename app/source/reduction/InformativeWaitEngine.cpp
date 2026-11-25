#include "VirtualOrch/reduction/InformativeWaitEngine.h"

#include "VirtualOrch/reduction/ChordTheory.h"
#include "VirtualOrch/reduction/TdOnsetBias.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

namespace {

auto markChord(std::vector<InformativeWaitEngine::OutNote> notes) -> std::vector<InformativeWaitEngine::OutNote> {
    for (auto &n: notes)
        n.fromChord = true;
    return notes;
}


constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int32_t kPianoLow = 21;
constexpr int32_t kPianoHigh = 108;
constexpr int32_t kMaxChord = 12;

struct Range {
    int32_t lo;
    int32_t hi; // exclusive
    int32_t step;
};

auto msSince(std::chrono::steady_clock::time_point t0) -> double {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

auto argmaxIn(const float *logits, int32_t lo, int32_t hi, int32_t step) -> int32_t {
    int32_t best = lo;
    float bestV = kNegInf;
    for (int32_t t = lo; t < hi; t += step)
        if (logits[t] > bestV) {
            bestV = logits[t];
            best = t;
        }
    return best;
}

} // namespace

auto InformativeWaitEngine::q5(int32_t d) -> int32_t {
    const int32_t q = 5 * static_cast<int32_t>(std::lround(static_cast<double>(d) / 5.0));
    return std::clamp(q, 5, 995);
}

auto InformativeWaitEngine::q8(int32_t v) -> int32_t {
    const int32_t q = 8 * static_cast<int32_t>(std::lround(static_cast<double>(v) / 8.0));
    return std::clamp(q, 0, 120);
}

InformativeWaitEngine::InformativeWaitEngine() : rng(0) {}
InformativeWaitEngine::~InformativeWaitEngine() = default;

auto InformativeWaitEngine::setBackend(StreamingKv::Backend *backendIn) -> void {
    backend = backendIn;
    cache = backend != nullptr ? std::make_unique<StreamingKv::CommittedScratchCache>(*backend) : nullptr;
    if (cache != nullptr)
        cache->reuseScratchOnCommit = true;
}

auto InformativeWaitEngine::setSettings(const Settings &s) -> void {
    const bool reseed = s.seed != cfg.seed;
    cfg = s;
    if (reseed)
        rng.seed(cfg.seed);
}

auto InformativeWaitEngine::reset(int32_t startCs) -> void {
    pitchScheduler.reset();
    durationScheduler.reset();
    items.clear();
    before.clear();
    cur = {};
    baseCs = 0;
    nowCs = startCs;
    started = true;
    pending.reset();
    group.reset();
    inbox.clear();
    offs.clear();
    pre.reset();
    cancelFrom = INT32_MAX;
    releasedDurations.clear();
    fireLeft.reset();
    mirrored.clear();
    st = {};
    rng.seed(cfg.seed);
    if (cache != nullptr)
        cache->reset();
}

auto InformativeWaitEngine::noteOn(int32_t onsetCs, int32_t pitch, int32_t velocity, int64_t landedUs) -> void {
    lastPianoVelocity = std::clamp(velocity, 1, 127);
    inbox.push_back({onsetCs, std::clamp(pitch, kPianoLow, kPianoHigh), std::clamp(velocity, 1, 127), -1, landedUs});
}

auto InformativeWaitEngine::noteOff(int32_t onsetCs, int32_t pitch, int32_t durationCs) -> void {
    offs.push_back({onsetCs, std::clamp(pitch, kPianoLow, kPianoHigh), std::max(durationCs, 1)});
}

auto InformativeWaitEngine::takeCancelFrom() -> int32_t {
    const int32_t c = cancelFrom;
    cancelFrom = INT32_MAX;
    return c;
}

auto InformativeWaitEngine::rel(int32_t absCs) const -> int32_t {
    return std::clamp(absCs - baseCs, 0, kOnsetMax);
}

auto InformativeWaitEngine::event(const char *what, int32_t at, int32_t a, int32_t b, std::string text) -> void {
    if (onEvent)
        onEvent(Event{what, at, a, b, std::move(text)});
}

auto InformativeWaitEngine::tokens() const -> std::vector<int32_t> {
    std::vector<int32_t> out;
    out.reserve(items.size() * 4);
    for (const auto &it: items) {
        switch (it.kind) {
            case Kind::Piano:
                out.insert(out.end(), {rel(it.onsetCs), it.durUnknown ? kDurUnknown : kDur0 + q5(it.durationCs),
                                       kPiano0 + it.pitch, kVel0 + q8(it.velocity)});
                break;
            case Kind::Chord:
                out.insert(out.end(), {it.token, kDur0 + it.durationCs, kOrch0 + it.pitch, kVel0 + it.velocity});
                break;
            case Kind::Late:
                out.push_back(it.token);
                break;
            case Kind::Orch:
                out.insert(out.end(), {rel(it.onsetCs), kDur0 + it.durationCs, kOrch0 + it.pitch, kVel0 + it.velocity});
                break;
            case Kind::Wait:
            case Kind::Count:
                out.push_back(it.token);
                break;
            case Kind::Expired:
                out.push_back(expiredToken());
                break;
        }
    }
    return out;
}

auto InformativeWaitEngine::committedLen() const -> size_t {
    // Everything before the first item that may still change: a held piano note (its duration),
    // a provisional WAIT (removed at expiry) or a generated note that has not started (a piano onset
    // may cut it).
    size_t len = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto &it = items[i];
        const bool mutableItem = (it.kind == Kind::Piano && it.held) || it.provisional
                                 || (it.kind == Kind::Orch && it.onsetCs > nowCs)
                                 || (cfg.waitTopK > 1 && pending.has_value() && i >= pending->waitItem);
        if (mutableItem)
            return len;
        len += width(it.kind);
    }
    return len;
}

auto InformativeWaitEngine::logitsFor(const std::vector<int32_t> &desired) -> const std::vector<float> * {
    if (cache == nullptr || desired.empty())
        return nullptr;
    const auto &logits = cache->logitsAfter(desired, std::min(committedLen(), desired.size()));
    if (logits.size() < static_cast<size_t>(vocab())) {
        lastError = "model returned no usable logits (vocab " + std::to_string(logits.size()) + ")";
        return nullptr;
    }
    return &logits;
}

auto InformativeWaitEngine::pickFrom(std::vector<float> scores, float temperature) -> int32_t {
    // temperature -> top-p -> draw (the Python sampler's order after the mask and min-p)
    const float t = std::max(temperature, 1e-4f);
    float peak = kNegInf;
    for (const float s: scores)
        peak = std::max(peak, s);
    if (! std::isfinite(peak))
        return -1;
    std::vector<std::pair<double, int32_t>> probs;
    double total = 0;
    for (size_t i = 0; i < scores.size(); ++i)
        if (std::isfinite(scores[i])) {
            const double p = std::exp(static_cast<double>((scores[i] - peak) / t));
            probs.emplace_back(p, static_cast<int32_t>(i));
            total += p;
        }
    std::sort(probs.begin(), probs.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    double cum = 0;
    size_t keep = 0;
    for (; keep < probs.size(); ++keep) {
        const double p = probs[keep].first / total;
        if (cum > static_cast<double>(cfg.topP))
            break;
        cum += p;
    }
    keep = std::max<size_t>(keep, 1);
    double kept = 0;
    for (size_t i = 0; i < keep; ++i)
        kept += probs[i].first;
    std::uniform_real_distribution<double> u(0.0, kept);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        r -= probs[i].first;
        if (r <= 0)
            return probs[i].second;
    }
    return probs[keep - 1].second;
}

auto InformativeWaitEngine::samplingProbs(const std::vector<float> &scores, float temperature) const
    -> std::vector<float> {
    // as pickFrom: softmax at the temperature, then the smallest top set whose mass exceeds top-p
    std::vector<float> out(scores.size(), 0.0f);
    const float t = std::max(temperature, 1e-4f);
    float peak = kNegInf;
    for (const float x: scores)
        peak = std::max(peak, x);
    if (! std::isfinite(peak))
        return out;
    std::vector<std::pair<double, int32_t>> probs;
    double total = 0;
    for (size_t i = 0; i < scores.size(); ++i)
        if (std::isfinite(scores[i])) {
            const double p = std::exp(static_cast<double>((scores[i] - peak) / t));
            probs.emplace_back(p, static_cast<int32_t>(i));
            total += p;
        }
    std::sort(probs.begin(), probs.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    double cum = 0;
    size_t keep = 0;
    for (; keep < probs.size(); ++keep) {
        if (cum > static_cast<double>(cfg.topP))
            break;
        cum += probs[keep].first / total;
    }
    keep = std::max<size_t>(keep, 1);
    double kept = 0;
    for (size_t i = 0; i < keep; ++i)
        kept += probs[i].first;
    for (size_t i = 0; i < keep; ++i)
        out[static_cast<size_t>(probs[i].second)] = static_cast<float>(probs[i].first / kept);
    return out;
}

auto InformativeWaitEngine::pick(const std::vector<float> &logits,
                                 const std::vector<std::pair<int32_t, int32_t>> &ranges, float temperature,
                                 float minP, int32_t step) -> int32_t {
    std::vector<float> scores(logits.size(), kNegInf);
    float peak = kNegInf;
    for (const auto &[lo, hi]: ranges)
        for (int32_t t = std::max(lo, 0); t < std::min(hi, static_cast<int32_t>(logits.size())); t += step) {
            scores[static_cast<size_t>(t)] = logits[static_cast<size_t>(t)];
            peak = std::max(peak, logits[static_cast<size_t>(t)]);
        }
    if (minP > 0 && std::isfinite(peak)) {
        // min-p on the raw (masked) distribution: p_i >= minP * p_max  <=>  l_i >= l_max + ln(minP)
        const float cut = peak + std::log(minP);
        for (auto &s: scores)
            if (s < cut)
                s = kNegInf;
    }
    if (! logDraws || ! recordDraws)
        return pickFrom(std::move(scores), temperature);
    const auto ps = samplingProbs(scores, temperature);
    const int32_t token = pickFrom(scores, temperature);
    const int32_t first = ranges.empty() ? 0 : ranges.front().first;
    const char *name = chordMode() && first >= kChord0                      ? "chord"
                       : chordMode() && first >= kLate0 && first < kDurUnknown ? "late"
                       : chordMode() && first >= kK0 && first < kChordExpired  ? "k"
                       : first >= kCount0 && first < kExpired ? "count"
                       : first >= kVel0 && first < kWait0   ? "vel"
                       : first >= kOrch0 && first < kVel0   ? "pitch"
                       : first >= kDur0 && first < kPiano0  ? "dur"
                                                            : "onset";
    const auto pm = OpLogDraw::softmax(logits.data(), logits.size());
    OpLogDraw::Draw d;
    d.name = name;
    d.chosen = token;
    d.p = token >= 0 ? ps[static_cast<size_t>(token)] : 0.0f;
    d.pModel = token >= 0 ? pm[static_cast<size_t>(token)] : 0.0f;
    std::vector<int32_t> idx;
    for (size_t i = 0; i < ps.size(); ++i)
        if (ps[i] > 0.0f)
            idx.push_back(static_cast<int32_t>(i));
    const size_t k = std::min<size_t>(10, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](int32_t a, int32_t b) { return ps[static_cast<size_t>(a)] > ps[static_cast<size_t>(b)]; });
    for (size_t j = 0; j < k; ++j)
        d.top.push_back({idx[j], ps[static_cast<size_t>(idx[j])], pm[static_cast<size_t>(idx[j])]});
    drawLog.push_back(std::move(d));
    return token;
}

auto InformativeWaitEngine::applyTdGamma(std::vector<float> &logits, int32_t loRel, int32_t hiRel) -> void {
    const float gamma = *cfg.tdGamma;
    const float temp = std::max(cfg.onsetTemperature, 1e-4f);
    const int32_t first = std::max(loRel, 0);
    const int32_t last = std::min(hiRel, static_cast<int32_t>(logits.size()) - 1);
    if (first > last)
        return;
    const int32_t ref = loRel - 1; // c = t - ref >= 1: the earliest legal new onset is one step on
    const auto lse = [&] {
        float peak = kNegInf;
        for (int32_t t = first; t <= last; ++t)
            peak = std::max(peak, logits[static_cast<size_t>(t)]);
        if (! std::isfinite(peak))
            return kNegInf;
        double sum = 0.0;
        for (int32_t t = first; t <= last; ++t)
            if (std::isfinite(logits[static_cast<size_t>(t)]))
                sum += std::exp(static_cast<double>((logits[static_cast<size_t>(t)] - peak) / temp));
        return peak / temp + static_cast<float>(std::log(sum));
    };
    const float before = lse();
    if (! std::isfinite(before))
        return;
    for (int32_t t = first; t <= last; ++t)
        logits[static_cast<size_t>(t)] += gamma * TdOnsetBias::kernelFor(gamma, cfg.tdLevel, t - ref);
    const float after = lse();
    if (std::isfinite(after)) {
        const float restore = temp * (before - after);
        for (int32_t t = first; t <= last; ++t)
            logits[static_cast<size_t>(t)] += restore;
    }

    // Sensitivity over these onsets, gaps measured from the last orchestra onset before them.
    int32_t lastOrch = INT32_MIN;
    for (const auto &it: items)
        if (it.kind == Kind::Orch && it.onsetCs < loRel + baseCs)
            lastOrch = std::max(lastOrch, it.onsetCs);
    const int32_t offset = lastOrch == INT32_MIN ? 0 : std::max(0, ref + baseCs - lastOrch);
    std::vector<float> scores(static_cast<size_t>(last - ref + 1), kNegInf); // index c; c = 0 left out
    for (int32_t t = first; t <= last; ++t)
        scores[static_cast<size_t>(t - ref)] = logits[static_cast<size_t>(t)];
    if (const auto s = TdOnsetBias::sensitivity(scores, 0, offset, cfg.tdLevel, temp, gamma)) {
        sensitivitySum += *s;
        ++sensitivityDraws;
    }
}

auto InformativeWaitEngine::orchestraHistory() const -> std::vector<int32_t> {
    std::vector<int32_t> quads;
    quads.reserve(items.size() * 4);
    for (const auto &it: items)
        if (it.kind == Kind::Orch || it.kind == Kind::Chord) { // chord mode: a chord block counts as its bass note
            quads.push_back(it.onsetCs);
            quads.push_back(static_cast<int32_t>(DenseVocab::DurOffset) + std::clamp(it.durationCs, 1, 995));
            quads.push_back(static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * 1 + it.pitch);
            quads.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + it.velocity);
        }
    return quads;
}

auto InformativeWaitEngine::clampToRegister(int32_t &lo, int32_t &hi) const -> bool {
    lo = std::max(lo, cfg.pitchLow);
    hi = std::min(hi, cfg.pitchHigh);
    return lo <= hi;
}

auto InformativeWaitEngine::takeSensitivity() -> std::pair<std::optional<double>, int> {
    const int draws = sensitivityDraws;
    const std::optional<double> mean = draws > 0 ? std::optional<double>(sensitivitySum / draws) : std::nullopt;
    sensitivitySum = 0.0;
    sensitivityDraws = 0;
    return {mean, draws};
}

// ---------------------------------------------------------------- stream

auto InformativeWaitEngine::append(const Item &item) -> void {
    // chord mode ([DUR?] rule, B3f 3b): a model-written unit after a piano quad fixes its [DUR?]
    if (chordMode() && item.kind != Kind::Piano && item.kind != Kind::Expired)
        freezeUnknownDurations();
    before.push_back(cur);
    items.push_back(item);
}

auto InformativeWaitEngine::freezeUnknownDurations() -> void {
    for (auto &it: items)
        if (it.kind == Kind::Piano && it.held && it.durUnknown)
            it.held = false;
}

auto InformativeWaitEngine::truncateTo(size_t index) -> void {
    if (index >= items.size())
        return;
    cur = before[index];
    items.resize(index);
    before.resize(index);
    if (pending.has_value() && pending->waitItem >= index) {
        pending.reset();
        pre.reset();
    }
}

auto InformativeWaitEngine::reappend(std::vector<Item> tail) -> void {
    std::stable_sort(tail.begin(), tail.end(), [](const Item &a, const Item &b) {
        if (a.onsetCs != b.onsetCs)
            return a.onsetCs < b.onsetCs;
        return a.kind == Kind::Piano && b.kind != Kind::Piano;
    });
    int32_t pianoOnset = INT32_MIN;
    int pianoAtOnset = 0;
    for (const auto &it: tail) {
        if (it.kind == Kind::Piano) {
            append(it);
            pianoAtOnset = it.onsetCs == pianoOnset ? pianoAtOnset + 1 : 1;
            pianoOnset = it.onsetCs;
            cur.clockCs = std::max(cur.clockCs, it.onsetCs);
            cur.floorCs = std::max(cur.floorCs, it.onsetCs);
            cur.exclHiCs = cfg.continuation ? it.onsetCs - 1
                         : chordMode()      ? (pianoAtOnset >= 2 ? it.onsetCs : it.onsetCs - 1) // as closeGroup
                                            : it.onsetCs + kMergeCs;
            cur.chordOnCs = INT32_MIN;
            cur.chordCount = 0;
            cur.lastPianoCs = std::max(cur.lastPianoCs, it.onsetCs);
            cur.afterExpired = false;
            cur.chainK = 0; // a piano onset ends a re-wait chain
        } else if (it.kind == Kind::Chord) {
            // a chord block: nothing else is drawn at its onset (its notes are all in the block)
            append(it);
            cur.clockCs = std::max(cur.clockCs, it.onsetCs);
            cur.floorCs = std::max(cur.floorCs, it.onsetCs);
            cur.chordOnCs = INT32_MIN;
            cur.chordCount = 0;
            cur.exclHiCs = it.onsetCs + kMergeCs;
        } else if (it.kind == Kind::Orch) {
            append(it);
            const bool same = cur.chordOnCs == it.onsetCs;
            cur.chordCount = same ? cur.chordCount + 1 : 1;
            cur.lastOrchPitch = it.pitch;
            cur.clockCs = std::max(cur.clockCs, it.onsetCs);
            cur.floorCs = std::max(cur.floorCs, it.onsetCs);
            cur.chordOnCs = it.onsetCs;
            cur.exclHiCs = it.onsetCs + gap();
        }
    }
}

auto InformativeWaitEngine::takeAfter(int32_t onsetCs) -> std::vector<Item> {
    for (size_t i = 0; i < items.size(); ++i)
        if (isEvent(items[i].kind) && items[i].onsetCs > onsetCs) {
            std::vector<Item> tail(items.begin() + static_cast<std::ptrdiff_t>(i), items.end());
            truncateTo(i);
            return tail;
        }
    return {};
}

auto InformativeWaitEngine::cutFuture(int32_t pianoOnsetCs) -> void {
    // Jam: a generated note that has not started and lies at or after (piano onset - 5) goes, with
    // everything written after it (the sampler's "the piano goes first, redraw" rule). Edit: only those
    // editKeepCs or more after the onset. Accompaniment: none.
    if (cfg.inputCut == Settings::InputCut::Accompaniment)
        return;
    const int32_t from = cfg.inputCut == Settings::InputCut::Edit ? pianoOnsetCs + cfg.editKeepCs : pianoOnsetCs - gap();
    mirrored.erase(std::remove_if(mirrored.begin(), mirrored.end(), [&](const auto &n) { return n.onsetCs > nowCs && n.onsetCs >= from; }),
                   mirrored.end());
    for (size_t i = 0; i < items.size(); ++i) {
        const auto &it = items[i];
        if (it.kind == Kind::Orch && it.onsetCs > nowCs && it.onsetCs >= from) {
            int cut = 0;
            for (size_t j = i; j < items.size(); ++j)
                cut += items[j].kind == Kind::Orch ? 1 : 0;
            cancelFrom = std::min(cancelFrom, it.onsetCs);
            st.cutNotes += cut;
            st.orchestraNotes -= cut;
            event("cut", nowCs, it.onsetCs, cut, "piano at " + std::to_string(pianoOnsetCs) + " came first");
            truncateTo(i);
            // the transformer's lower bound (the newest generated onset) was set before this cut: until it
            // is recomputed, no later than now, or the next draw lands where the cut notes were
            cfg.minOnsetCs = std::min(cfg.minOnsetCs, nowCs);
            return;
        }
    }
}

auto InformativeWaitEngine::writePianoNotes(std::vector<PianoNote> notes) -> void {
    if (notes.empty())
        return;
    const int32_t head = notes.front().onsetCs;
    std::stable_sort(notes.begin(), notes.end(), [](const auto &a, const auto &b) { return a.pitch < b.pitch; });
    if (items.empty())
        baseCs = head; // training lines start at onset 0
    for (const auto &n: notes) {
        Item it;
        it.kind = Kind::Piano;
        it.onsetCs = head;
        it.pitch = n.pitch;
        it.velocity = n.velocity;
        const bool down = n.durationCs < 0;
        if (chordMode()) {
            // [DUR?] at key-down; the release may still replace it (applyInput) until a model-written unit follows
            it.durUnknown = down;
            it.held = down;
            it.durationCs = down ? 0 : std::max(5, n.onsetCs + n.durationCs - head);
            append(it);
            continue;
        }
        if (! down)
            it.durationCs = std::max(5, n.onsetCs + n.durationCs - head);
        else if (cfg.rewriteHeldDurations)
            it.durationCs = estimatedDuration(nowCs - n.onsetCs); // a typical length until its note-off
        else
            it.durationCs = estimatedDuration(nowCs - n.onsetCs);
        it.held = down && cfg.rewriteHeldDurations;
        append(it);
    }
}

auto InformativeWaitEngine::estimatedDuration(int32_t heldSoFar) const -> int32_t {
    int32_t est = 30;
    if (! releasedDurations.empty()) {
        auto d = releasedDurations;
        std::nth_element(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2), d.end());
        est = d[d.size() / 2];
    }
    return std::clamp(std::max(est, heldSoFar), 5, 995);
}

auto InformativeWaitEngine::commitLongHolds() -> void {
    if (chordMode())
        return; // [DUR?] replaces the hold commit (B3f 3b)
    // a key still down after holdCommitCs is committed with the time it has been down (it stops being
    // rewritten, so the KV cache's scratch tail stays short)
    for (auto &it: items)
        if (it.kind == Kind::Piano && it.held && nowCs - it.onsetCs > cfg.holdCommitCs) {
            it.held = false;
            it.durationCs = std::max(it.durationCs, nowCs - it.onsetCs);
        }
}

// ---------------------------------------------------------------- input

auto InformativeWaitEngine::applyInput() -> std::vector<OutNote> {
    std::vector<OutNote> out;
    // A key released before its note was applied (the input came in a backlog): the release
    // goes onto the waiting note itself. Otherwise it found nothing below and was dropped, and
    // the note was written as held for as long as it had been waiting (up to 995 cs).
    for (auto off = offs.begin(); off != offs.end();) {
        const auto &[onset, pitch, dur] = *off;
        auto waiting = std::find_if(inbox.begin(), inbox.end(), [&](const PianoNote &n) {
            return n.durationCs < 0 && n.pitch == pitch && n.onsetCs == onset;
        });
        if (waiting == inbox.end()) {
            ++off;
            continue;
        }
        waiting->durationCs = dur;
        releasedDurations.push_back(dur);
        if (releasedDurations.size() > 16)
            releasedDurations.erase(releasedDurations.begin());
        off = offs.erase(off);
    }
    for (const auto &[onset, pitch, dur]: offs) {
        releasedDurations.push_back(dur);
        if (releasedDurations.size() > 16)
            releasedDurations.erase(releasedDurations.begin());
        bool done = false;
        if (group.has_value())
            for (auto &n: group->notes)
                if (! done && n.durationCs < 0 && n.pitch == pitch && n.onsetCs == onset) {
                    n.durationCs = dur;
                    done = true;
                }
        for (auto it = items.rbegin(); ! done && it != items.rend(); ++it)
            if (it->kind == Kind::Piano && it->held && it->pitch == pitch && onset >= it->onsetCs
                && onset - it->onsetCs <= kMergeCs) {
                it->durationCs = std::max(5, onset + dur - it->onsetCs);
                it->held = false;
                it->durUnknown = false; // chord mode: nothing model-written after it yet, the real duration goes in
                done = true;
            }
    }
    offs.clear();

    auto arrived = std::move(inbox);
    inbox.clear();
    // (notes of a new group that arrive while a fire's stack is decoded are kept in the inbox)
    std::stable_sort(arrived.begin(), arrived.end(), [](const auto &a, const auto &b) { return a.onsetCs < b.onsetCs; });
    for (const auto &n: arrived) {
        if (group.has_value() && n.onsetCs - group->headCs <= kMergeCs) {
            group->notes.push_back(n);
            continue;
        }
        if (group.has_value() && group->firePending) {
            auto f = markChord(fireChord()); // chord mode: every piano note of the onset that came in is written first
            out.insert(out.end(), f.begin(), f.end());
        }
        if (fireLeft.has_value()) {
            inbox.push_back(n); // the stack of the current fire is finished first (training order)
            continue;
        }
        if (group.has_value()) {
            auto closed = closeGroup();
            out.insert(out.end(), closed.begin(), closed.end());
        }
        auto fired = onGroupHead(n);
        out.insert(out.end(), fired.begin(), fired.end());
    }
    if (group.has_value() && group->firePending) {
        auto f = markChord(fireChord());
        out.insert(out.end(), f.begin(), f.end());
    }
    return out;
}

auto InformativeWaitEngine::onGroupHead(const PianoNote &head) -> std::vector<OutNote> {
    std::vector<OutNote> out;
    cutFuture(head.onsetCs);
    if (pending.has_value() && head.onsetCs > pending->hi(pending->provisional)) {
        auto e = expire(head.onsetCs); // its window closed before this onset
        out.insert(out.end(), e.begin(), e.end());
        cutFuture(head.onsetCs); // a legato note written at the expiry may now be in the future
    }
    group = OpenGroup{head.onsetCs, {head}, false};
    if (pending.has_value()) {
        const auto length = pending->provisional ? std::optional<int32_t>(pending->d) : pending->match(head.onsetCs);
        if (head.onsetCs < pending->lo() || ! length.has_value()) {
            ++st.earlyPiano;
            event("early", head.onsetCs, head.pitch, pending->lo(),
                  head.onsetCs < pending->lo() ? "piano before the WAIT window: ordinary input"
                                               : "piano between the likely WAIT lengths: ordinary input");
        } else {
            if (*length != pending->d) {
                // one of the model's most likely lengths: the WAIT is rewritten to it, then fires
                event("snap", head.onsetCs, pending->d, *length, "WAIT rewritten to a top-" + std::to_string(cfg.waitTopK) + " length");
                items[pending->waitItem].token = waitToken(*length);
                pending->d = *length;
                ++st.waitSnaps;
            }
            group->fired = true;
            if (chordMode()) {
                group->firePending = true; // the block goes after all the onset's piano notes (applyInput)
            } else {
                group->written = 1;
                auto f = fire(head);
                out.insert(out.end(), f.begin(), f.end());
            }
        }
    }
    return out;
}

auto InformativeWaitEngine::closeGroup() -> std::vector<OutNote> {
    std::vector<OutNote> out;
    if (! group.has_value())
        return out;
    auto g = std::move(*group);
    group.reset();
    ++st.pianoGroups;
    if (g.fired) {
        // the fired piano notes and the stack are written; the rest of the chord follows them
        std::vector<PianoNote> rest(g.notes.begin() + static_cast<std::ptrdiff_t>(std::min(g.written, g.notes.size())),
                                    g.notes.end());
        if (! rest.empty()) {
            for (auto &n: rest)
                n.onsetCs = g.headCs;
            writePianoNotes(rest);
        }
        if (isLine() && ! items.empty()) {
            // the note fired onto the piano onset gets its provisional WAIT after the chord
            for (auto it = items.rbegin(); it != items.rend(); ++it)
                if (it->kind == Kind::Orch) {
                    addProvisionalWait(OutNote{it->onsetCs, it->durationCs, it->pitch, it->velocity});
                    break;
                }
        }
        event("piano", g.headCs, static_cast<int32_t>(g.notes.size()), 1, "rest of the fired chord");
        return out;
    }
    std::vector<PianoNote> notes = g.notes;
    for (auto &n: notes) {
        if (n.durationCs >= 0)
            n.durationCs = n.onsetCs + n.durationCs - g.headCs; // end time kept
        n.onsetCs = g.headCs;
    }
    // Notes kept after this onset (Edit / Accompaniment, or the other model's) are written after it.
    const auto later = takeAfter(g.headCs);
    writePianoNotes(notes);
    cur.clockCs = std::max(cur.clockCs, g.headCs);
    cur.floorCs = std::max(cur.floorCs, g.headCs);
    // model 2: nothing within 5 cs after a piano onset; model 1 and the chord-token model 2 (B3f 1g: joint onsets
    // without a WAIT are in its training data): the orchestra may share it
    // chord mode: a single piano note's onset may be shared (training: orchestra quads after the FIRST piano quad of a
    // joint onset without a WAIT); after a group of 2+ notes, the next plain onset is later (training never has
    // orchestra quads after several piano quads of one onset; with them the model stacked 8-12-note clusters)
    cur.exclHiCs = cfg.continuation ? g.headCs - 1
                 : chordMode()      ? (notes.size() >= 2 ? g.headCs : g.headCs - 1)
                                    : g.headCs + kMergeCs;
    cur.chordOnCs = INT32_MIN;
    cur.chordCount = 0;
    cur.lastPianoCs = g.headCs;
    cur.afterExpired = false;
    cur.chainK = 0; // a piano onset ends a re-wait chain
    reappend(later);
    event("piano", g.headCs, static_cast<int32_t>(notes.size()));
    // A pending WAIT's anticipation is NOT restarted by this (early) piano group: live, restarting on every
    // group of a dense passage never lets it finish; branches from a slightly older context still serve.
    return out;
}

// ---------------------------------------------------------------- WAIT state machine

auto InformativeWaitEngine::fire(const PianoNote &head) -> std::vector<OutNote> {
    std::vector<OutNote> out;
    const auto p = *pending;
    ++st.fires;
    scoreAnticipation(head.pitch);
    const auto t0 = std::chrono::steady_clock::now();
    fireFromKey = head.landedUs >= 0;
    fireFromUs = fireFromKey ? head.landedUs
                             : std::chrono::duration_cast<std::chrono::microseconds>(t0.time_since_epoch()).count();

    PianoNote first = head;
    writePianoNotes({first});
    if (p.provisional && p.waitItem + 1 < items.size()) {
        items[p.waitItem].provisional = false; // a fired WAIT stays
        items[p.waitItem + 1].provisional = false;
    }

    // A branch for this pitch serves the notes it has finished (all k when the precompute is complete);
    // the rest are decoded now, after them.
    // A pitch outside the top 5 takes the rank-1 branch (branches are in rank order), as if the
    // pianist had played the most likely note: its notes are ready, nothing is decoded on the spot.
    const Branch *branch = nullptr;
    bool substituted = false;
    if (pre.has_value() && ! pre->failed) {
        for (const auto &b: pre->branches)
            if (b.pitch == head.pitch)
                branch = &b;
        if (branch == nullptr && cfg.outsideTopFiveAsRankOne && ! pre->branches.empty()) {
            branch = &pre->branches.front();
            substituted = true;
        }
    }
    // a polyphony cap (monophonic: 1) limits the stack, whatever COUNT said
    const int32_t k = cfg.maxFireNotes > 0 ? std::min(p.k, cfg.maxFireNotes) : p.k;
    int32_t ready = 0;
    if (branch != nullptr)
        for (const auto &n: branch->notes)
            if (n[1] > 0 && n[2] > 0 && ready < k)
                ++ready;

    cur.clockCs = cur.floorCs = head.onsetCs;
    int32_t minPitch = isLine() ? cfg.bandLow : kPianoLow;
    for (int32_t j = 0; j < ready; ++j) {
        const auto &[dur, pitch, vel] = branch->notes[static_cast<size_t>(j)];
        Item it;
        it.kind = Kind::Orch;
        it.onsetCs = head.onsetCs;
        it.durationCs = dur;
        it.pitch = pitch;
        it.velocity = vel;
        append(it);
        out.push_back({head.onsetCs, dur, pitch, vel, true});
        minPitch = std::min(pitch + 1, kPianoHigh);
    }
    countCommittedForBias(head.onsetCs, ready); // the branch's notes are committed now
    if (ready == k) {
        ++st.precomputeHits;
        event("hit", head.onsetCs, head.pitch, p.k,
              substituted ? "not in the top 5: the rank-1 branch (" + std::to_string(branch->pitch) + ") played"
                          : "precomputed branch played");
    } else {
        if (ready > 0)
            ++st.precomputePartial;
        // the remaining notes are decoded one per step (each is played as soon as it exists; input is
        // handled in between); the rest of the chord is written after the last of them
        fireLeft = FireLeft{head.onsetCs, k - ready, minPitch};
        ++st.fireDecoded;
        event("miss", head.onsetCs, head.pitch, p.k,
              branch == nullptr ? (pre.has_value() && ! pre->pianoProbs.empty() ? "not in the top 5" : "no anticipation yet")
                                : "branch had " + std::to_string(ready) + " of " + std::to_string(p.k) + " notes");
    }
    st.orchestraNotes += static_cast<int>(out.size());
    event("fire", head.onsetCs, p.k, static_cast<int32_t>(out.size()));
    firePlaced = static_cast<int32_t>(out.size());
    if (! fireLeft.has_value())
        firePlacedEvent(head.onsetCs, ready == k);
    cur.exclHiCs = head.onsetCs + kMergeCs;
    cur.chordOnCs = INT32_MIN;
    cur.chordCount = 0;
    cur.lastOrchPitch = out.empty() ? -1 : out.back().pitch;
    cur.lastPianoCs = head.onsetCs;
    cur.afterExpired = false;
    cur.chainK = 0; // a piano onset ends a re-wait chain
    pending.reset();
    pre.reset();
    return out;
}

auto InformativeWaitEngine::firePlacedEvent(int32_t onsetCs, bool precomputed) -> void {
    const int64_t nowUs =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t us = std::max<int64_t>(0, nowUs - fireFromUs);
    event("placed", onsetCs, firePlaced, static_cast<int32_t>(std::min<int64_t>(us, INT32_MAX)),
          std::string(precomputed ? "precomputed" : "decoded") + (fireFromKey ? "|key" : "|fire"));
}

auto InformativeWaitEngine::expire(std::optional<int32_t> arrivalCs) -> std::vector<OutNote> {
    std::vector<OutNote> out;
    if (! pending.has_value())
        return out;
    if (chordMode() && ! pending->provisional)
        return closeChordWindow(arrivalCs);
    const auto p = *pending;
    pending.reset();
    pre.reset();
    if (p.provisional) {
        // line modes: the provisional WAIT / COUNT leave the context (no EXPIRED)
        const int32_t end = p.refCs + p.d;
        removeProvisionalWait(p.waitItem);
        ++st.removed;
        event("removed", end, p.d, 0, "no piano by the note's end");
        if (cfg.mode == Mode::LineLegato && end >= nowCs - cfg.lagGuardCs)
            return noteAt(end);
        if (cfg.mode == Mode::LineLegato) {
            // the engine is behind (the note ended lagGuardCs ago): no fill in the past, the line goes on from now
            ++st.lagSkips;
            event("lag", nowCs, end, nowCs - end, "legato fill skipped: the engine is behind");
            cur.freeFloorCs = nowCs;
            return out;
        }
        cur.freeFloorCs = end;
        return out;
    }
    const int32_t at = p.hi(false);
    Item it;
    it.kind = Kind::Expired;
    it.onsetCs = at;
    append(it);
    ++st.expired;
    cur.clockCs = std::max(cur.clockCs, at);
    cur.floorCs = std::max(cur.floorCs, at);
    cur.exclHiCs = cur.floorCs - 1;
    cur.chordOnCs = INT32_MIN;
    cur.freeFloorCs = at;
    cur.afterExpired = true;
    event("expired", at, p.d, p.k);
    return out;
}

auto InformativeWaitEngine::removeProvisionalWait(size_t index) -> void {
    if (index + 1 >= items.size() || ! items[index].provisional)
        return;
    items.erase(items.begin() + static_cast<std::ptrdiff_t>(index),
                items.begin() + static_cast<std::ptrdiff_t>(index + 2));
    before.erase(before.begin() + static_cast<std::ptrdiff_t>(index),
                 before.begin() + static_cast<std::ptrdiff_t>(index + 2));
}

auto InformativeWaitEngine::addProvisionalWait(const OutNote &note) -> void {
    const int32_t x = quantizeWait(note.durationCs);
    Item w;
    w.kind = Kind::Wait;
    w.onsetCs = note.onsetCs;
    w.token = waitToken(x);
    w.provisional = true;
    const size_t at = items.size();
    append(w);
    Item c;
    c.kind = Kind::Count;
    c.onsetCs = note.onsetCs;
    c.token = countToken(1);
    c.provisional = true;
    append(c);
    pending = Pending{note.onsetCs, x, 1, true, at};
    ++st.waitsProvisional;
    event("provisional", note.onsetCs, x, 1);
    startPrecompute();
}

auto InformativeWaitEngine::applyDurationBias(std::vector<float> &logits, const std::vector<int32_t> &history,
                                             int32_t onsetCs, DenseDurationBias::DurationTauScheduler &scheduler) const
    -> void {
    if (cfg.coarseDurationBins) {
        const auto window = DenseDurationBias::collectCoarseDurationWindowAt(history, onsetCs, 1);
        const auto taus = scheduler.evaluate(onsetCs, window.uniqueBins, window.uniqueMoves, cfg.durationTau,
                                             cfg.durationTauPrime, true);
        DenseDurationBias::applyCoarseDurationLogitsBias(logits, window, taus.tau, taus.tauPrime);
        return;
    }
    const auto window = DenseDurationBias::collectDurationWindowAt(history, onsetCs, -1, 1);
    const auto taus = scheduler.evaluate(onsetCs, window.uniqueDurations, window.uniqueDeltas, cfg.durationTau,
                                         cfg.durationTauPrime, true);
    DenseDurationBias::applyDurationLogitsBias(logits, window, taus.tau, taus.tauPrime);
}

auto InformativeWaitEngine::applyPitchBias(std::vector<float> &logits, const std::vector<int32_t> &history,
                                          int32_t onsetCs, DensePitchBias::PitchTauScheduler &scheduler) const
    -> void {
    const auto window = DensePitchBias::collectPitchWindowAt(history, onsetCs, -1);
    const auto taus = scheduler.evaluate(onsetCs, window.uniquePitches, window.uniqueDeltas, cfg.pitchTau,
                                         cfg.pitchTauPrime, true);
    DensePitchBias::applyNoteLogitsBias(logits, window, taus.tau, taus.tauPrime);
}

auto InformativeWaitEngine::countCommittedForBias(int32_t onsetCs, int notes) -> void {
    if (! cfg.durationBias && ! cfg.pitchBias)
        return;
    // as drawBody: each note's τ is evaluated on the history before it, then the note is committed
    const auto history = orchestraHistory();
    const size_t committed = history.size() / 4;
    for (int j = 0; j < notes && static_cast<size_t>(j) < committed; ++j) {
        const std::vector<int32_t> before(history.begin(),
                                          history.end() - static_cast<std::ptrdiff_t>(4 * (notes - j)));
        if (cfg.durationBias) {
            std::vector<float> none; // the τ only: applyDurationBias on no logits is a no-op past the evaluate
            applyDurationBias(none, before, onsetCs, durationScheduler);
        }
        if (cfg.pitchBias) {
            const auto w = DensePitchBias::collectPitchWindowAt(before, onsetCs, -1);
            (void) pitchScheduler.evaluate(onsetCs, w.uniquePitches, w.uniqueDeltas, cfg.pitchTau, cfg.pitchTauPrime,
                                           true);
        }
    }
}

auto InformativeWaitEngine::drawBody(int32_t onsetCs, int32_t minPitch, bool pianoAllowed) -> std::optional<OutNote> {
    drewPianoPitch = false;
    auto desired = tokens();
    desired.push_back(rel(onsetCs));
    const auto *l1 = logitsFor(desired);
    if (l1 == nullptr)
        return std::nullopt;
    std::vector<int32_t> history;
    if (cfg.durationBias || cfg.pitchBias)
        history = orchestraHistory();
    std::vector<float> biased;
    if (cfg.durationBias) {
        biased = *l1;
        applyDurationBias(biased, history, onsetCs, durationScheduler);
        l1 = &biased;
    }
    const int32_t dur = pick(*l1, {{kDur0 + cfg.minDurationCs, kDur0 + cfg.maxDurationCs + 1}}, cfg.otherTemperature, cfg.minP, 5);
    if (dur < 0)
        return std::nullopt;
    desired.push_back(dur);
    int32_t lo = std::max(minPitch, kPianoLow);
    int32_t hi = kPianoHigh;
    if (isLine()) {
        lo = std::max(lo, cfg.bandLow);
        hi = std::min(hi, cfg.bandHigh);
    }
    if (! clampToRegister(lo, hi))
        return std::nullopt; // a chord stacked up to the top of the register: no further note
    const auto *l2 = logitsFor(desired);
    if (l2 == nullptr)
        return std::nullopt;
    if (cfg.pitchBias) {
        biased = *l2;
        applyPitchBias(biased, history, onsetCs, pitchScheduler);
        l2 = &biased;
    }
    std::vector<std::pair<int32_t, int32_t>> pitchRanges{{kOrch0 + lo, kOrch0 + hi + 1}};
    if (pianoAllowed)
        pitchRanges.push_back({kPiano0 + kPianoLow, kPiano0 + kPianoHigh + 1}); // the model may want a piano note
    const int32_t note = pick(*l2, pitchRanges, cfg.otherTemperature, cfg.minP);
    if (note < 0)
        return std::nullopt;
    if (note >= kPiano0 && note < kOrch0) {
        drewPianoPitch = true; // nothing is written (the caller drops the onset and duration)
        return std::nullopt;
    }
    desired.push_back(note);
    const auto *l3 = logitsFor(desired);
    if (l3 == nullptr)
        return std::nullopt;
    // velocities 8..120 (0 would be a note-off in the app)
    const int32_t vel = pick(*l3, {{kVel0 + 8, kVel0 + 121}}, cfg.otherTemperature, cfg.minP, 8);
    if (vel < 0)
        return std::nullopt;
    return OutNote{onsetCs, dur - kDur0, note - kOrch0, vel - kVel0, false};
}

auto InformativeWaitEngine::noteAt(int32_t onsetCs) -> std::vector<OutNote> {
    std::vector<OutNote> out;
    auto note = drawBody(onsetCs, cfg.bandLow);
    if (! note.has_value())
        return out;
    note->draws = std::move(drawLog);
    drawLog.clear();
    Item it;
    it.kind = Kind::Orch;
    it.onsetCs = onsetCs;
    it.durationCs = note->durationCs;
    it.pitch = note->pitch;
    it.velocity = note->velocity;
    append(it);
    ++st.orchestraNotes;
    cur.clockCs = std::max(cur.clockCs, onsetCs);
    cur.freeFloorCs = INT32_MIN;
    out.push_back(*note);
    event("orch", onsetCs, note->pitch, note->durationCs);
    addProvisionalWait(*note);
    return out;
}

auto InformativeWaitEngine::drawEvent() -> std::vector<OutNote> {
    std::vector<OutNote> out;
    drawLog.clear();
    const bool line = isLine();
    int32_t lo = 0;
    if (line)
        lo = std::max({cur.clockCs, cur.freeFloorCs, cur.lastPianoCs == INT32_MIN ? INT32_MIN : cur.lastPianoCs + kMergeCs + 1, nowCs});
    else
        lo = std::max({cur.floorCs, cur.exclHiCs == INT32_MIN ? INT32_MIN : cur.exclHiCs + 1, nowCs});
    lo = std::max(lo, cfg.minOnsetCs); // the polyphony cap
    if (lo - baseCs > kOnsetMax)
        return out; // past the end of the onset window: wait until the base moves on (maybeShrinkContext);
                    // clamped, every further note was stacked on onset 9999
    const int32_t hi = std::max(lo, nowCs + cfg.maxAheadCs);
    const int32_t loRel = rel(lo);
    const int32_t hiRel = std::min(kOnsetMax, std::max(loRel, hi - baseCs));
    std::vector<std::pair<int32_t, int32_t>> ranges{{loRel, hiRel + 1}};
    const bool chordOk = ! line && cur.chordOnCs != INT32_MIN && cur.chordOnCs >= nowCs - cfg.chordLateCs
                         && cur.chordOnCs >= cfg.minOnsetCs && cur.chordOnCs < lo
                         && (! cfg.ascendingChords || cur.lastOrchPitch <= std::min(kPianoHigh, cfg.pitchHigh))
                         && cur.chordCount < std::min(cfg.maxChordNotes, kMaxChord);
    if (chordOk)
        ranges.push_back({rel(cur.chordOnCs), rel(cur.chordOnCs) + 1});
    // chord mode: [EXPIRED] then a WAIT was drawn, so the next token is that WAIT (with the same K, as in training)
    bool chain = chordMode() && cur.chainK != 0;
    const bool waitOk = cfg.mode != Mode::LineLegato && ! cfg.continuation
                        && (chain || ! (cfg.noWaitAfterExpiry && cur.afterExpired));
    if (waitOk) {
        // only WAITs whose window has not already passed: ref + d + 6 > now (chord mode: and >= the 6 cs floor)
        int32_t dMin = std::max(kGrid, nowCs + 1 - cur.clockCs - cfg.fireWindowCs);
        if (chordMode())
            dMin = std::max(dMin, chain && plusOf(cur.chainK) ? kWaitFloorPlus : kWaitFloor);
        if (dMin <= kMaxWait) {
            if (chain)
                ranges.clear(); // the chain: a WAIT only
            ranges.push_back({waitToken(((dMin + kGrid - 1) / kGrid) * kGrid), kCount0});
        } else if (chain) {
            cur.chainK = 0; // the engine is too far behind for any chained WAIT: the model draws freely
            chain = false;
        }
    }
    const auto toks = tokens();
    const auto *logits = logitsFor(toks);
    if (logits == nullptr)
        return out;
    std::vector<float> shifted;
    const bool gammaOn = cfg.tdGamma.has_value() && cfg.tdLevel >= 1;
    const bool chordShiftOn = cfg.logitTd && chordOk && cfg.tdLevel > 1;
    if (gammaOn || cfg.onsetGapBias || cfg.shortGap || chordShiftOn) {
        shifted = *logits;
        if (cfg.onsetGapBias) {
            const auto window = DenseOnsetGapBias::collectOnsetGapWindow(orchestraHistory(), std::max(nowCs, cur.clockCs), -1);
            DenseOnsetGapBias::applyOnsetLogitsBias(shifted, window, cfg.onsetGapTau, baseCs);
        }
        if (cfg.shortGap)
            for (int32_t t = std::max(loRel, 0); t <= hiRel && t < static_cast<int32_t>(shifted.size()); ++t)
                if (std::isfinite(shifted[static_cast<size_t>(t)]))
                    shifted[static_cast<size_t>(t)] += TdOnsetBias::shortGapPenalty(t - (loRel - 1));
        if (gammaOn)
            applyTdGamma(shifted, loRel, hiRel);
        if (chordShiftOn) {
            auto &chord = shifted[static_cast<size_t>(rel(cur.chordOnCs))];
            if (std::isfinite(chord))
                chord += TdOnsetBias::chordShift(cfg.tdLevel, cur.chordCount);
        }
        logits = &shifted;
    }
    int32_t t = pick(*logits, ranges, cfg.onsetTemperature, 0.0f);
    if (cfg.minWaitCs > 0 && waitOk && t >= kWait0 && t < kCount0 && waitValue(t) < cfg.minWaitCs) {
        // A WAIT under the minimum: the orchestra plays instead. The longer WAITs keep the probability the
        // model gave them. The note goes at the onset the WAIT predicted for the pianist (the event before it
        // + d, within the onsets allowed now; only duration, pitch and velocity are drawn), or, with
        // shortWaitRedraw, at an onset redrawn from the orchestra onsets (ranges.back() is the WAIT range).
        const int32_t d = waitValue(t);
        ++st.shortWaitsReplaced;
        if (cfg.shortWaitRedraw) {
            auto onsets = ranges;
            onsets.pop_back();
            t = onsets.empty() ? -1 : pick(*logits, onsets, cfg.onsetTemperature, 0.0f);
        } else {
            t = std::clamp(rel(cur.clockCs + d), loRel, hiRel);
        }
        event("short", cur.clockCs, d, t < 0 ? -1 : t + baseCs,
              cfg.shortWaitRedraw ? "WAIT under the minimum: an orchestra onset redrawn instead"
                                  : "WAIT under the minimum: an orchestra note at its predicted onset instead");
    }
    if (t < 0) {
        lastError = "nothing to draw at the onset step";
        return out;
    }
    if (t >= kWait0 && t < kCount0) {
        const int32_t d = waitValue(t);
        // the model's most likely WAIT lengths (of those still possible), read before the COUNT step
        // reuses the logits buffer: the top K (the drawn one aside) are also accepted, and the top 3
        // with their probabilities go to the log
        std::vector<int32_t> alts;
        std::string topText;
        {
            const auto [wLo, wHi] = ranges.back();
            std::vector<std::pair<float, int32_t>> byP;
            for (int32_t x = wLo; x < wHi && x < static_cast<int32_t>(logits->size()); ++x)
                if (std::isfinite((*logits)[static_cast<size_t>(x)]) && waitValue(x) >= cfg.minWaitCs)
                    byP.push_back({(*logits)[static_cast<size_t>(x)], x});
            std::sort(byP.begin(), byP.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
            if (cfg.waitTopK > 1)
                for (size_t r = 0; r < byP.size() && r < static_cast<size_t>(cfg.waitTopK); ++r)
                    if (waitValue(byP[r].second) != d)
                        alts.push_back(waitValue(byP[r].second));
            double total = 0;
            for (const auto &[lg, x]: byP)
                total += std::exp(static_cast<double>(lg - byP.front().first));
            topText = "top 3:";
            for (size_t r = 0; r < byP.size() && r < 3; ++r) {
                const double pr = std::exp(static_cast<double>(byP[r].first - byP.front().first)) / total;
                topText += " " + std::to_string(waitValue(byP[r].second)) + " ("
                           + std::to_string(static_cast<int>(std::lround(pr * 100))) + "%)";
            }
        }
        const size_t at = items.size();
        Item w;
        w.kind = Kind::Wait;
        w.onsetCs = cur.clockCs;
        w.token = t;
        append(w);
        int32_t k = 1;
        if (line) {
            Item c; // line B: COUNT forced to 1
            c.kind = Kind::Count;
            c.onsetCs = cur.clockCs;
            c.token = countToken(1);
            append(c);
        } else if (chordMode()) {
            // [K k,+1]: the chain repeats its K; otherwise drawn, with the leftover (+1) masked under its 12 cs floor
            int32_t c = cur.chainK;
            if (c == 0) {
                const auto *lc = logitsFor(tokens());
                if (lc == nullptr)
                    return out;
                c = d < kWaitFloorPlus ? pick(*lc, {{kK0, kK0 + 8}}, cfg.onsetTemperature, 0.0f, 2)
                                       : pick(*lc, {{kK0, kK0 + 8}}, cfg.onsetTemperature, 0.0f);
                if (c < 0)
                    return out;
            } else {
                ++st.chains;
            }
            cur.chainK = 0;
            k = kOf(c);
            Item ci;
            ci.kind = Kind::Count;
            ci.onsetCs = cur.clockCs;
            ci.token = c;
            append(ci);
            const int32_t floor = plusOf(c) ? kWaitFloorPlus : kWaitFloor;
            alts.erase(std::remove_if(alts.begin(), alts.end(), [floor](int32_t a) { return a < floor; }), alts.end());
        } else {
            auto desired = tokens();
            const auto *lc = logitsFor(desired);
            if (lc == nullptr)
                return out;
            const int32_t c = pick(*lc, {{kCount0, kExpired}}, cfg.onsetTemperature, 0.0f);
            k = c - kCount0 + 1;
            Item ci;
            ci.kind = Kind::Count;
            ci.onsetCs = cur.clockCs;
            ci.token = c;
            append(ci);
        }
        pending = Pending{cur.clockCs, d, k, false, at};
        pending->alts = std::move(alts);
        pending->win = cfg.fireWindowCs;
        if (chordMode()) {
            pending->kToken = items.back().token;
            pending->plus = plusOf(pending->kToken);
        }
        ++st.waitsDrawn;
        if (d >= 10)
            ++st.waitsGe10;
        waitDraws = std::move(drawLog);
        drawLog.clear();
        event("wait", cur.clockCs, d, k, topText);
        startPrecompute();
        return out;
    }
    const int32_t onset = t + baseCs;
    if (line)
        return noteAt(onset);
    const bool same = chordOk && onset == cur.chordOnCs;
    // chord mode: a plain note's pitch step also allows piano pitches. r2b's training order for joint onsets without
    // a WAIT is [first piano quad][orchestra quads][other piano quads], so after a piano group the model often wants
    // another piano note; masked to orchestra pitches (ascending within the onset) it walked up the scale into a
    // 12-note cluster. A piano pitch drops the onset and duration, and the next plain onset is strictly later.
    auto note = drawBody(onset, same && cfg.ascendingChords ? cur.lastOrchPitch : kPianoLow, chordMode());
    if (! note.has_value()) {
        if (drewPianoPitch) {
            drawLog.clear();
            ++st.pianoPitchDrops;
            cur.exclHiCs = std::max(cur.exclHiCs, onset);
            if (cur.chordOnCs != INT32_MIN && cur.chordOnCs <= onset) {
                cur.chordOnCs = INT32_MIN; // no stacking onto this (or an earlier) onset either
                cur.chordCount = 0;
            }
            event("drop", onset, 0, 0, "piano pitch drawn for a plain note: nothing written");
        }
        return out;
    }
    note->draws = std::move(drawLog);
    drawLog.clear();
    Item it;
    it.kind = Kind::Orch;
    it.onsetCs = onset;
    it.durationCs = note->durationCs;
    it.pitch = note->pitch;
    it.velocity = note->velocity;
    append(it);
    ++st.orchestraNotes;
    cur.chordCount = same ? cur.chordCount + 1 : 1;
    cur.lastOrchPitch = note->pitch;
    cur.clockCs = cur.floorCs = cur.chordOnCs = onset;
    cur.exclHiCs = onset + gap();
    event("orch", onset, note->pitch, note->durationCs);
    out.push_back(*note);
    return out;
}

// ---------------------------------------------------------------- chord-token mode (m2c)

auto InformativeWaitEngine::chordRanges(int32_t k) -> std::vector<std::pair<int32_t, int32_t>> {
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t type = 0; type < kChordTypes; ++type)
        if (VoicingBank::pitchClassCount(type) == k)
            r.push_back({chordToken(type, 0, 1), chordToken(type, 11, 10) + 1}); // the type's 12 roots x 10 n
    return r;
}

auto InformativeWaitEngine::bassRanges(int32_t chordTok, int32_t lo, int32_t hi) -> std::vector<std::pair<int32_t, int32_t>> {
    // training: BASS is the lowest band note, so its pitch class is always a chord tone
    const auto cv = chordValue(chordTok);
    const auto mask = ChordTheory::chordMask(cv.type, cv.root);
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t p = lo; p <= hi; ++p)
        if ((mask >> (((p % 12) + 12) % 12)) & 1u)
            r.push_back({kOrch0 + p, kOrch0 + p + 1});
    if (r.empty())
        r.push_back({kOrch0 + lo, kOrch0 + hi + 1}); // no chord tone in the register: any pitch rather than none
    return r;
}

auto InformativeWaitEngine::decodeBlock(std::vector<int32_t> desired, int32_t k, int32_t onsetCs, int32_t firstChord,
                                       std::vector<std::pair<int32_t, int32_t>> chordMask) -> std::vector<int32_t> {
    std::vector<int32_t> block;
    if (firstChord >= 0) {
        block.push_back(firstChord); // already in the context (a give-up's CHORD, drawn after [EXPIRED])
        desired.push_back(firstChord);
    }
    // SDUR and BASS take the duration / pitch bias as a decoded note's duration and pitch (the engine's schedulers:
    // this block is committed)
    const auto history = cfg.durationBias || cfg.pitchBias ? orchestraHistory() : std::vector<int32_t>{};
    const auto next = [&](const std::vector<std::pair<int32_t, int32_t>> &ranges, int32_t step, int bias = 0) -> bool {
        const auto *l = logitsFor(desired);
        if (l == nullptr)
            return false;
        std::vector<float> biased;
        if ((bias == 1 && cfg.durationBias) || (bias == 2 && cfg.pitchBias)) {
            biased = *l;
            if (bias == 1)
                applyDurationBias(biased, history, onsetCs, durationScheduler);
            else
                applyPitchBias(biased, history, onsetCs, pitchScheduler);
            l = &biased;
        }
        const int32_t t = pick(*l, ranges, cfg.otherTemperature, cfg.minP, step);
        if (t < 0)
            return false;
        block.push_back(t);
        desired.push_back(t);
        return true;
    };
    int32_t lo = kPianoLow;
    int32_t hi = kPianoHigh;
    clampToRegister(lo, hi);
    lo = std::min(lo, hi);
    if ((firstChord < 0 && ! next(chordMask.empty() ? chordRanges(k) : chordMask, 1)) || ! next({{kDur0 + cfg.minDurationCs, kDur0 + cfg.maxDurationCs + 1}}, 5, 1)
        || ! next(bassRanges(block.front(), lo, hi), 1, 2) || ! next({{kVel0 + 8, kVel0 + 121}}, 8))
        return {};
    return block;
}

auto InformativeWaitEngine::writeBlock(const std::vector<int32_t> &block, int32_t onsetCs, int32_t playCs,
                                       int32_t already, bool precomputed) -> std::vector<OutNote> {
    std::vector<OutNote> out;
    if (block.size() < 4)
        return out;
    Item it;
    it.kind = Kind::Chord;
    it.onsetCs = onsetCs;
    it.token = block[0];
    it.durationCs = block[1] - kDur0;
    it.pitch = block[2] - kOrch0;
    it.velocity = block[3] - kVel0;
    append(it);
    ++st.chordBlocks;
    // nothing plain at the block's onset (fire, inference or give-up): only its leftover sits at T in training
    cur.exclHiCs = std::max(cur.exclHiCs, onsetCs + kMergeCs);
    cur.chordOnCs = INT32_MIN;
    cur.chordCount = 0;
    const auto cv = chordValue(block[0]);
    VoicingBank::Decoded dec;
    if (cfg.voicings != nullptr && ! cfg.voicings->empty()) {
        dec = cfg.voicings->decode(cv.type, cv.root, cv.n, it.pitch, rng);
    } else {
        static const VoicingBank none;
        dec = none.decode(cv.type, cv.root, cv.n, it.pitch, rng); // empty bank: the chord tones stacked
    }
    if (dec.source != VoicingBank::Decoded::Source::Bank)
        ++st.voicingFallbacks;
    // the Launchpad register: notes above it go down an octave while they stay above the bass, else they go
    std::vector<int32_t> pitches;
    for (int32_t p: dec.pitches) {
        while (p > cfg.pitchHigh && p - 12 >= it.pitch)
            p -= 12;
        if (p <= cfg.pitchHigh && std::find(pitches.begin(), pitches.end(), p) == pitches.end())
            pitches.push_back(p);
    }
    std::sort(pitches.begin(), pitches.end());
    std::string text = VoicingBank::key(cv.type, cv.n, ((it.pitch - cv.root) % 12 + 12) % 12) + ":";
    for (const int32_t p: pitches) {
        text += " " + std::to_string(p);
        if (cfg.maxFireNotes > 0 && already + static_cast<int32_t>(out.size()) >= cfg.maxFireNotes)
            continue; // the polyphony cap (monophonic: the bass only)
        OutNote n{playCs, it.durationCs, p, std::clamp(it.velocity, 8, 120), precomputed};
        if (out.empty())
            n.draws = std::move(drawLog);
        out.push_back(std::move(n));
    }
    drawLog.clear();
    st.orchestraNotes += static_cast<int>(out.size());
    event("chord", onsetCs, static_cast<int32_t>(out.size()), cv.n, text);
    cur.clockCs = std::max(cur.clockCs, onsetCs);
    cur.floorCs = std::max(cur.floorCs, onsetCs);
    cur.chordOnCs = INT32_MIN;
    cur.chordCount = 0;
    cur.exclHiCs = onsetCs + kMergeCs;
    cur.lastOrchPitch = pitches.empty() ? -1 : pitches.back();
    return out;
}

auto InformativeWaitEngine::keyWeights(int32_t atCs) const -> std::array<double, 12> {
    std::array<double, 12> w{};
    constexpr int32_t kKeyWindowCs = 800; // the last ~8 s
    for (const auto &it: items) {
        if (it.onsetCs < atCs - kKeyWindowCs || it.onsetCs > atCs)
            continue;
        if (it.kind != Kind::Piano && it.kind != Kind::Orch && it.kind != Kind::Chord)
            continue;
        const int32_t dur = it.durUnknown ? std::max(5, nowCs - it.onsetCs) : std::max(5, it.durationCs);
        const double weight = std::min(dur, kKeyWindowCs);
        if (it.kind == Kind::Chord) {
            const auto cv = chordValue(it.token); // a chord block: each of its pitch classes
            const auto m = ChordTheory::chordMask(cv.type, cv.root);
            for (int pc = 0; pc < 12; ++pc)
                if ((m >> pc) & 1u)
                    w[static_cast<size_t>(pc)] += weight;
        } else {
            w[static_cast<size_t>(((it.pitch % 12) + 12) % 12)] += weight;
        }
    }
    return w;
}

auto InformativeWaitEngine::fireChord() -> std::vector<OutNote> {
    std::vector<OutNote> out;
    const auto p = *pending;
    auto &g = *group;
    g.firePending = false;
    g.written = g.notes.size();
    const PianoNote head = g.notes.front();
    ++st.fires;
    scoreAnticipation(head.pitch);
    const auto t0 = std::chrono::steady_clock::now();
    fireFromKey = head.landedUs >= 0;
    fireFromUs = fireFromKey ? head.landedUs
                             : std::chrono::duration_cast<std::chrono::microseconds>(t0.time_since_epoch()).count();

    // every piano note of the onset, each [DUR?] (training: they still sound at the chord block), frozen
    drawLog.clear();
    std::vector<PianoNote> notes = g.notes;
    std::vector<int32_t> pitches;
    int32_t lowest = kPianoHigh;
    for (auto &n: notes) {
        n.onsetCs = g.headCs;
        n.durationCs = -1;
        pitches.push_back(n.pitch);
        lowest = std::min(lowest, n.pitch);
    }
    const size_t from = items.size();
    writePianoNotes(notes);
    for (size_t i = from; i < items.size(); ++i)
        items[i].held = false;
    const auto piano = ChordTheory::maskOf(pitches);

    // MATCH RULE: the piano pitch classes with a candidate's form a triad, a seventh or < 3 pitch classes
    const Branch *branch = nullptr;
    if (pre.has_value() && pre->complete)
        for (const auto &b: pre->branches) {
            if (b.block.size() != 4)
                continue;
            const auto cv = chordValue(b.block[0]);
            if (ChordTheory::recognized(static_cast<ChordTheory::Pcs>(piano | ChordTheory::chordMask(cv.type, cv.root)))
                && (branch == nullptr || b.score > branch->score))
                branch = &b;
        }
    cur.clockCs = cur.floorCs = head.onsetCs;
    fireNotesOut = 0;
    firePlaced = 0;
    std::string names;
    for (const int32_t x: pitches)
        names += " " + std::to_string(x);
    if (branch != nullptr) {
        auto block = writeBlock(branch->block, head.onsetCs, head.onsetCs, 0, true);
        fireNotesOut += static_cast<int32_t>(block.size());
        out.insert(out.end(), block.begin(), block.end());
        if (p.plus && ! branch->notes.empty()) {
            const auto &[dur, pitch, vel] = branch->notes.front();
            Item it;
            it.kind = Kind::Orch;
            it.onsetCs = head.onsetCs;
            it.durationCs = dur;
            it.pitch = pitch;
            it.velocity = vel;
            append(it);
            if (cfg.maxFireNotes <= 0 || fireNotesOut < cfg.maxFireNotes) {
                out.push_back({head.onsetCs, dur, pitch, vel, true});
                ++fireNotesOut;
                ++st.orchestraNotes;
            }
        }
        countCommittedForBias(head.onsetCs, p.plus && ! branch->notes.empty() ? 2 : 1); // block (+ leftover) committed
        ++st.precomputeHits;
        ++st.chordMatched;
        const auto cv = chordValue(branch->block[0]);
        event("hit", head.onsetCs, head.pitch, p.k,
              "piano" + names + ": candidate " + VoicingBank::kTypeNames[static_cast<size_t>(cv.type)] + " root "
                  + std::to_string(cv.root) + " matches");
    } else {
        // INFERENCE RULE: CHORD's type and root from the piano notes; the model picks n, then the rest is decoded
        const auto inf = ChordTheory::infer(piano, lowest, p.k, keyWeights(head.onsetCs));
        ++st.inferSteps[static_cast<size_t>(std::clamp(inf.step, 0, 4))];
        fireLeft = FireLeft{head.onsetCs, 0, kPianoLow, true, true, p.plus, p.k, false, -1, inf.chord.type, inf.chord.root};
        ++st.fireDecoded;
        event("miss", head.onsetCs, head.pitch, p.k,
              std::string(pre.has_value() && pre->complete ? "no candidate matches" : "precompute not finished")
                  + "; piano" + names + ": inferred " + VoicingBank::kTypeNames[static_cast<size_t>(inf.chord.type)]
                  + " root " + std::to_string(inf.chord.root) + " (step " + std::to_string(inf.step) + ")");
    }
    event("fire", head.onsetCs, p.k, static_cast<int32_t>(out.size()));
    firePlaced = static_cast<int32_t>(out.size());
    if (! fireLeft.has_value())
        firePlacedEvent(head.onsetCs, true);
    cur.exclHiCs = head.onsetCs + kMergeCs; // everything of this onset is in the block
    cur.chordOnCs = INT32_MIN;
    cur.chordCount = 0;
    cur.lastPianoCs = head.onsetCs;
    cur.afterExpired = false;
    cur.chainK = 0;
    pending.reset();
    pre.reset();
    return out;
}

auto InformativeWaitEngine::fireChordStep() -> std::vector<OutNote> {
    std::vector<OutNote> out;
    auto &f = *fireLeft;
    const auto t0 = std::chrono::steady_clock::now();
    drawLog.clear();
    const int32_t playCs = f.noPiano ? std::max(nowCs, f.onsetCs) : f.onsetCs;
    bool failed = false;
    if (f.needBlock) {
        std::vector<std::pair<int32_t, int32_t>> mask;
        if (f.inferType >= 0)
            mask.push_back({chordToken(f.inferType, f.inferRoot, 1), chordToken(f.inferType, f.inferRoot, 10) + 1});
        const auto block = decodeBlock(tokens(), f.k, f.onsetCs, f.firstChord, mask);
        if (block.empty()) {
            failed = true;
        } else {
            out = writeBlock(block, f.onsetCs, playCs, fireNotesOut, false);
            fireNotesOut += static_cast<int32_t>(out.size());
        }
        f.needBlock = false;
    } else if (f.needLeftover) {
        // the leftover quad: onset forced to the block's onset, then duration, pitch, velocity
        auto note = drawBody(f.onsetCs, kPianoLow);
        if (note.has_value()) {
            note->draws = std::move(drawLog);
            drawLog.clear();
            Item it;
            it.kind = Kind::Orch;
            it.onsetCs = f.onsetCs;
            it.durationCs = note->durationCs;
            it.pitch = note->pitch;
            it.velocity = note->velocity;
            append(it);
            if (cfg.maxFireNotes <= 0 || fireNotesOut < cfg.maxFireNotes) {
                note->onsetCs = playCs;
                out.push_back(*note);
                ++fireNotesOut;
                ++st.orchestraNotes;
            }
        } else {
            failed = true;
        }
        f.needLeftover = false;
    }
    st.fireDecodeMsTotal += msSince(t0);
    firePlaced += static_cast<int32_t>(out.size());
    if (failed || (! f.needBlock && ! f.needLeftover)) {
        if (! f.noPiano)
            firePlacedEvent(f.onsetCs, false);
        fireLeft.reset();
    }
    return out;
}

auto InformativeWaitEngine::closeChordWindow(std::optional<int32_t> arrivalCs) -> std::vector<OutNote> {
    std::vector<OutNote> out;
    auto p = *pending;
    const int32_t close = p.hi(false);
    const auto writeExpired = [&] {
        Item it;
        it.kind = Kind::Expired;
        it.onsetCs = close;
        append(it);
        ++st.expired;
        cur.clockCs = std::max(cur.clockCs, close);
        cur.floorCs = std::max(cur.floorCs, close);
        cur.exclHiCs = cur.floorCs - 1;
        cur.chordOnCs = INT32_MIN;
        cur.freeFloorCs = close;
    };
    // 1. [LATE d] or [EXPIRED], drawn by the model (no LATE after maxLates). LATE d = the time from the close to the
    // piano's arrival: a piano onset already known to be past the close sets it (as training's true remaining
    // delay); otherwise the LATEs whose window has not passed yet (close + d + win > now) are allowed.
    int32_t tok = kChordExpired;
    bool forced = false;
    bool chose = false;
    if (p.lates < cfg.maxLates) {
        if (arrivalCs.has_value() && *arrivalCs - close <= kLateMax + p.win) {
            tok = lateToken(std::max(kGrid, *arrivalCs - close));
            forced = true;
        } else {
            std::vector<std::pair<int32_t, int32_t>> ranges{{kChordExpired, kChordExpired + 1}};
            const int32_t dMin = std::max(kGrid, nowCs + 1 - close - p.win);
            if (dMin <= kLateMax)
                ranges.push_back({lateToken(((dMin + kGrid - 1) / kGrid) * kGrid), kLate0 + kLateMax / kGrid});
            const auto *l = logitsFor(tokens());
            if (l == nullptr)
                return out;
            tok = pick(*l, ranges, cfg.onsetTemperature, 0.0f);
            if (tok < 0)
                tok = kChordExpired;
            chose = true;
        }
    }
    if (tok != kChordExpired) {
        pre.reset();
        Item it;
        it.kind = Kind::Late;
        it.onsetCs = close;
        it.token = tok;
        append(it);
        ++st.lates;
        st.lateForced += forced ? 1 : 0;
        p.refCs = close;
        p.d = lateValue(tok);
        p.alts.clear();
        p.lates += 1;
        p.isLate = true;
        p.floorLo = close; // the window starts no earlier than the close
        pending = p;
        event("late", close, p.d, p.k, forced ? "d from the piano onset already in" : "d sampled");
        startPrecompute(); // the context gained the LATE and the target moved
        return out;
    }
    pending.reset();
    pre.reset();
    writeExpired();
    st.expiredChosen += chose ? 1 : 0;
    // 2. after [EXPIRED]: a WAIT (re-wait, same K) or a CHORD masked to k (give up, B3f 2e). Only the choice is
    // taken here: on a WAIT, drawEvent draws the WAIT from the same logits within the WAIT range (the same
    // distribution as this draw given a WAIT), with the WAITs whose window has passed by then masked.
    auto ranges = chordRanges(p.k);
    const int32_t waitFloor = p.plus ? kWaitFloorPlus : kWaitFloor;
    const int32_t wMin = std::max({kGrid, waitFloor, nowCs + 1 - close - cfg.fireWindowCs});
    if (wMin <= kMaxWait)
        ranges.push_back({waitToken(((wMin + kGrid - 1) / kGrid) * kGrid), kCount0});
    int32_t next = -1;
    if (const auto *l = logitsFor(tokens()); l != nullptr)
        next = pick(*l, ranges, cfg.onsetTemperature, 0.0f);
    if (next >= kWait0 && next < kCount0) {
        cur.chainK = p.kToken;
        event("expired", close, p.d, p.k,
              "after " + std::to_string(p.lates) + " LATE: WAIT drawn, re-wait with the same K");
        return out;
    }
    // give up: the CHORD drawn (or, if the draw failed, drawn again by decodeBlock), then SDUR/BASS/SVEL (+ leftover)
    cur.afterExpired = true;
    ++st.giveUps;
    event("expired", close, p.d, p.k, "after " + std::to_string(p.lates) + " LATE: CHORD drawn, the chord plays alone");
    fireNotesOut = 0;
    firePlaced = 0;
    fireLeft = FireLeft{close, 0, kPianoLow, true, true, p.plus, p.k, true, next >= kChord0 ? next : -1};
    return out;
}

// ---------------------------------------------------------------- anticipation

auto InformativeWaitEngine::startPrecompute() -> void {
    if (! pending.has_value())
        return;
    pre = Precompute{};
    pre->targetCs = pending->refCs + pending->d;
    pre->k = pending->k;
    pre->plus = pending->plus;
    ++st.precomputeStarted;
}

auto InformativeWaitEngine::precomputeComplete() const -> bool { return pre.has_value() && pre->complete; }

auto InformativeWaitEngine::anticipatedPitches() const -> std::vector<int32_t> {
    std::vector<int32_t> out;
    if (pre.has_value())
        for (const auto &b: pre->branches)
            out.push_back(b.pitch);
    return out;
}

auto InformativeWaitEngine::scoreAnticipation(int32_t pitch) -> void {
    if (! pre.has_value() || pre->pianoProbs.size() != 88 || pitch < kPianoLow || pitch > kPianoHigh)
        return;
    const float pa = pre->pianoProbs[static_cast<size_t>(pitch - kPianoLow)];
    int rank = 1;
    for (const float x: pre->pianoProbs)
        rank += x > pa ? 1 : 0;
    ++st.anticipated;
    st.antTop1 += rank == 1 ? 1 : 0;
    st.antTop5 += rank <= 5 ? 1 : 0;
    event("anticipate", nowCs, pitch, rank);
}

auto InformativeWaitEngine::precomputeStep() -> void {
    // the branches' draws are not the notes that will play (a fire places one branch): not logged
    struct NoDrawLog {
        bool &flag;
        explicit NoDrawLog(bool &f) : flag(f) { flag = false; }
        ~NoDrawLog() { flag = true; }
    } noDrawLog(recordDraws);
    auto &pc = *pre;
    const auto t0 = std::chrono::steady_clock::now();
    const int32_t T = rel(pc.targetCs);
    auto finish = [&](bool ok) {
        pc.ms += msSince(t0);
        pc.complete = ok;
        pc.failed = ! ok;
        if (ok) {
            ++st.precomputeCompleted;
            st.precomputeMsTotal += pc.ms;
        }
        char text[96];
        std::snprintf(text, sizeof text, "%s: %d %s x 5 branches, %d calls, %.1f ms inference, %.1f ms wall",
                      ok ? "precompute done" : "precompute failed", pc.k,
                      chordMode() ? (pc.plus ? "pc chord block + leftover" : "pc chord block") : "notes", pc.calls, pc.ms,
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pc.started).count());
        event("precomputed", pc.targetCs, pc.k, ok ? 1 : 0, text);
    };
    const auto dims = backend->dims();
    if (pc.phase == 0) {
        // One batch-1 call on a copy of the KV state (the cache is left as it is): the target onset T and the
        // duration the piano note will be written with (the held-key estimate) -> the 88 piano keys.
        const auto toks = tokens();
        const auto *state = cache->stateAfter(toks, std::min(committedLen(), toks.size()));
        if (state == nullptr)
            return finish(false);
        StreamingKv::State s = *state;
        pc.durationCs = estimatedDuration(0);
        StreamingKv::evictForIncoming(s, dims, 2);
        std::vector<float> lv;
        ++pc.calls;
        // chord mode: the target piano quad is always written with [DUR?] (a chord block follows it at once)
        const int32_t durTok = chordMode() ? kDurUnknown : kDur0 + q5(pc.durationCs);
        if (! backend->run(s, {T, durTok}, pc.state, lv) || lv.size() < static_cast<size_t>(vocab()))
            return finish(false);
        float peak = kNegInf;
        for (int32_t p = kPianoLow; p <= kPianoHigh; ++p)
            peak = std::max(peak, lv[static_cast<size_t>(kPiano0 + p)]);
        pc.pianoProbs.assign(88, 0.0f);
        double total = 0;
        for (int32_t p = kPianoLow; p <= kPianoHigh; ++p) {
            const double e = std::exp(static_cast<double>(lv[static_cast<size_t>(kPiano0 + p)] - peak));
            pc.pianoProbs[static_cast<size_t>(p - kPianoLow)] = static_cast<float>(e);
            total += e;
        }
        for (auto &x: pc.pianoProbs)
            x = static_cast<float>(x / total);
        std::vector<int32_t> order(88);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + kBranches, order.end(), [&](int32_t a, int32_t b) {
            return pc.pianoProbs[static_cast<size_t>(a)] > pc.pianoProbs[static_cast<size_t>(b)];
        });
        pc.branches.clear();
        std::string text;
        for (int i = 0; i < kBranches; ++i) {
            pc.branches.push_back(Branch{kPianoLow + order[static_cast<size_t>(i)], q8(lastPianoVelocity), {}});
            text += std::to_string(pc.branches.back().pitch) + " ";
        }
        event("anticipate", pc.targetCs, pc.durationCs, 0, "top 5: " + text);
        ++pc.phase;
        pc.ms += msSince(t0);
        if (! cfg.precompute)
            pc.failed = true; // anticipation only: no branches decoded
        return;
    }
    // Batched branches, one call per step:
    //   phase 1: tile, feed [piano pitch, piano velocity (the pianist's last), T]  -> duration
    //   then per orchestra note: [duration] -> pitch, [pitch] -> velocity, and [velocity, T] -> next duration
    std::vector<int64_t> feed;
    if (pc.phase == 1) {
        const bool tiled = StreamingKv::tileState(pc.state, dims, kBranches, pc.kv);
        if (! chordMode())
            pc.state = {}; // chord mode tiles it again for the chord candidates
        if (! tiled)
            return finish(false);
        for (const auto &b: pc.branches) {
            if (chordMode())
                feed.insert(feed.end(), {kPiano0 + b.pitch, kVel0 + b.velocity}); // -> CHORD
            else
                feed.insert(feed.end(), {kPiano0 + b.pitch, kVel0 + b.velocity, T});
        }
    } else if (chordMode()) {
        // chord mode, what the last logits predict: 0 CHORD, 1 SDUR, 2 BASS, 3 SVEL, then the leftover quad's
        // 4 duration, 5 pitch, 6 velocity (fed [SVEL, T] so the leftover's onset is T)
        const auto V = static_cast<size_t>(vocab());
        if (pc.logits.size() < V * kBranches)
            return finish(false);
        const int r = pc.phase - 2;
        if (r == 0) {
            // CHORD BRANCHES (A iii): per pitch b, p(CHORD | b) masked to the types with k pitch classes; chord
            // type + root (n summed) scored by sum_b p(b) p(chord | b); the top 5 become the branches, each decoded
            // after its largest contributor, which also picks its n. The batch is tiled again from [T, DUR?].
            std::vector<double> score(static_cast<size_t>(kChordTypes * 12), 0.0);
            std::vector<double> bestPart(score.size(), -1.0);
            std::vector<int> contributor(score.size(), 0);
            const auto ranges = chordRanges(pc.k);
            for (int b = 0; b < kBranches; ++b) {
                const auto *lg = pc.logits.data() + static_cast<size_t>(b) * V;
                float peak = kNegInf;
                for (const auto &[lo, hi]: ranges)
                    for (int32_t t = lo; t < hi; ++t)
                        peak = std::max(peak, lg[t]);
                double total = 0;
                for (const auto &[lo, hi]: ranges)
                    for (int32_t t = lo; t < hi; ++t)
                        total += std::isfinite(lg[t]) ? std::exp(static_cast<double>(lg[t] - peak)) : 0.0;
                if (! (total > 0))
                    continue;
                const double pb = pc.pianoProbs[static_cast<size_t>(pc.branches[static_cast<size_t>(b)].pitch - kPianoLow)];
                std::vector<double> part(score.size(), 0.0);
                for (const auto &[lo, hi]: ranges)
                    for (int32_t t = lo; t < hi; ++t)
                        if (std::isfinite(lg[t]))
                            part[static_cast<size_t>((t - kChord0) / 10)] += std::exp(static_cast<double>(lg[t] - peak)) / total;
                for (size_t c = 0; c < score.size(); ++c) {
                    score[c] += pb * part[c];
                    if (pb * part[c] > bestPart[c]) {
                        bestPart[c] = pb * part[c];
                        contributor[c] = b;
                    }
                }
            }
            std::vector<int> order;
            for (const auto &[lo, hi]: ranges)
                for (int32_t t = lo; t < hi; t += 10)
                    order.push_back((t - kChord0) / 10);
            if (static_cast<int>(order.size()) < kBranches)
                return finish(false);
            std::partial_sort(order.begin(), order.begin() + kBranches, order.end(),
                              [&](int a, int b) { return score[static_cast<size_t>(a)] > score[static_cast<size_t>(b)]; });
            std::vector<Branch> cands;
            std::string text;
            for (int i = 0; i < kBranches; ++i) {
                const int c = order[static_cast<size_t>(i)];
                const auto &src = pc.branches[static_cast<size_t>(contributor[static_cast<size_t>(c)])];
                const std::vector<float> lg(pc.logits.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(contributor[static_cast<size_t>(c)]) * V),
                                            pc.logits.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(contributor[static_cast<size_t>(c)] + 1) * V));
                const int32_t tok = pick(lg, {{kChord0 + c * 10, kChord0 + c * 10 + 10}}, cfg.otherTemperature, cfg.minP);
                if (tok < 0)
                    return finish(false);
                Branch br{src.pitch, src.velocity, {}, {tok}, score[static_cast<size_t>(c)]};
                cands.push_back(br);
                char buf[64];
                std::snprintf(buf, sizeof buf, "%s/%d (%.2f, after %d) ", VoicingBank::kTypeNames[static_cast<size_t>(c / 12)],
                              c % 12, score[static_cast<size_t>(c)], src.pitch);
                text += buf;
                feed.insert(feed.end(), {kPiano0 + br.pitch, kVel0 + br.velocity, tok});
            }
            pc.branches = std::move(cands);
            event("candidates", pc.targetCs, pc.k, 0, text);
            if (! StreamingKv::tileState(pc.state, dims, kBranches, pc.kv))
                return finish(false);
        }
        // SDUR / the leftover's duration and BASS / the leftover's pitch: the duration / pitch biases from the
        // committed notes only, with copies of the schedulers (countCommittedForBias counts them at a match)
        const bool biasDur = (r == 1 || r == 4) && cfg.durationBias;
        const bool biasPitch = (r == 2 || r == 5) && cfg.pitchBias;
        const auto history = biasDur || biasPitch ? orchestraHistory() : std::vector<int32_t>{};
        for (int b = 0; b < kBranches && r > 0; ++b) {
            auto &br = pc.branches[static_cast<size_t>(b)];
            std::vector<float> lg(pc.logits.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(b) * V),
                                  pc.logits.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(b + 1) * V));
            if (biasDur) {
                auto scheduler = durationScheduler;
                applyDurationBias(lg, history, pc.targetCs, scheduler);
            } else if (biasPitch) {
                auto scheduler = pitchScheduler;
                applyPitchBias(lg, history, pc.targetCs, scheduler);
            }
            int32_t tok = -1;
            if (r == 1 || r == 4) {
                tok = pick(lg, {{kDur0 + cfg.minDurationCs, kDur0 + cfg.maxDurationCs + 1}}, cfg.otherTemperature, cfg.minP, 5);
            } else if (r == 2 || r == 5) {
                int32_t lo = kPianoLow;
                int32_t hi = kPianoHigh;
                clampToRegister(lo, hi);
                lo = std::min(lo, hi);
                // BASS: the chord's tones only; the leftover's pitch (r 5): any
                tok = pick(lg, r == 2 ? bassRanges(br.block.front(), lo, hi) : std::vector<std::pair<int32_t, int32_t>>{{kOrch0 + lo, kOrch0 + hi + 1}},
                           cfg.otherTemperature, cfg.minP);
            } else {
                tok = pick(lg, {{kVel0 + 8, kVel0 + 121}}, cfg.otherTemperature, cfg.minP, 8);
            }
            if (tok < 0)
                return finish(false);
            if (r <= 3) {
                br.block.push_back(tok);
            } else if (r == 4) {
                br.notes.push_back({tok - kDur0, 0, 0});
            } else if (r == 5) {
                br.notes.back()[1] = tok - kOrch0;
            } else {
                br.notes.back()[2] = tok - kVel0;
            }
            if (r == 3 && pc.plus)
                feed.insert(feed.end(), {tok, T});
            else
                feed.push_back(tok);
        }
        if ((r == 3 && ! pc.plus) || r == 6)
            return finish(true); // every branch has its block (and leftover): nothing more to feed
    } else {
        const auto V = static_cast<size_t>(kVocab);
        if (pc.logits.size() < V * kBranches)
            return finish(false);
        const int r = (pc.phase - 2) % 3; // what the last logits predict: 0 duration, 1 pitch, 2 velocity
        // The duration / pitch biases as at a decoded fire, from the committed orchestra notes only: a branch's
        // own notes are not committed (most branches are never played), so they change neither the
        // statistics nor the τ schedulers (copies here; countCommittedForBias counts them at the fire).
        const bool biasDur = r == 0 && cfg.durationBias;
        const bool biasPitch = r == 1 && cfg.pitchBias;
        std::vector<int32_t> history;
        if (biasDur || biasPitch)
            history = orchestraHistory();
        for (int b = 0; b < kBranches; ++b) {
            auto &br = pc.branches[static_cast<size_t>(b)];
            const auto row = static_cast<size_t>(b);
            std::vector<float> lg(pc.logits.begin() + static_cast<std::ptrdiff_t>(row * V),
                                  pc.logits.begin() + static_cast<std::ptrdiff_t>((row + 1) * V));
            if (biasDur) {
                auto scheduler = durationScheduler;
                applyDurationBias(lg, history, pc.targetCs, scheduler);
            } else if (biasPitch) {
                auto scheduler = pitchScheduler;
                applyPitchBias(lg, history, pc.targetCs, scheduler);
            }
            if (r == 0) {
                const int32_t tok = pick(lg, {{kDur0 + cfg.minDurationCs, kDur0 + cfg.maxDurationCs + 1}}, cfg.otherTemperature, cfg.minP, 5);
                br.notes.push_back({tok - kDur0, 0, 0});
                feed.push_back(tok);
            } else if (r == 1) {
                int32_t lo = br.notes.size() > 1 ? std::min(br.notes[br.notes.size() - 2][1] + 1, kPianoHigh)
                                                 : (isLine() ? cfg.bandLow : kPianoLow);
                int32_t hi = kPianoHigh;
                if (isLine()) {
                    lo = std::max(lo, cfg.bandLow);
                    hi = std::min(hi, cfg.bandHigh);
                }
                clampToRegister(lo, hi);
                lo = std::min(lo, hi);
                const int32_t tok = pick(lg, {{kOrch0 + lo, kOrch0 + hi + 1}}, cfg.otherTemperature, cfg.minP);
                br.notes.back()[1] = tok - kOrch0;
                feed.push_back(tok);
            } else {
                const int32_t tok = pick(lg, {{kVel0 + 8, kVel0 + 121}}, cfg.otherTemperature, cfg.minP, 8);
                br.notes.back()[2] = tok - kVel0;
                feed.insert(feed.end(), {tok, T});
            }
        }
        if (r == 2 && static_cast<int32_t>(pc.branches.front().notes.size()) >= pc.k)
            return finish(true); // the last velocity completes every branch: nothing more to feed
    }
    const int perBranch = static_cast<int>(feed.size()) / kBranches;
    StreamingKv::evictBatchForIncoming(pc.kv, dims, perBranch);
    StreamingKv::BatchState next;
    ++pc.calls;
    if (! backend->runBatched(pc.kv, feed, next, pc.logits))
        return finish(false);
    pc.kv = std::move(next);
    ++pc.phase;
    pc.ms += msSince(t0);
}

// ---------------------------------------------------------------- context window

auto InformativeWaitEngine::maybeShrinkContext() -> void {
    // Onsets are written relative to baseCs and must stay below kOnsetMax: when the clock nears that
    // (a sparse context, a long pause), the base moves on and events older than 60 s go, as the legacy
    // stream rebases. Without it, onsets were clamped to 9999 about 100 s after the base.
    if (! items.empty() && nowCs + std::min(cfg.maxAheadCs, 1000) - baseCs > kOnsetMax - 200) {
        size_t cut = items.size();
        for (size_t i = 0; i < items.size(); ++i)
            if (isEvent(items[i].kind) && items[i].onsetCs >= nowCs - 6000) {
                cut = i;
                break;
            }
        if (pending.has_value() && cut > pending->waitItem) {
            pending.reset();
            pre.reset();
        }
        if (cut >= items.size()) {
            items.clear();
            before.clear();
            baseCs = nowCs;
        } else if (cut > 0) {
            baseCs = items[cut].onsetCs;
            items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(cut));
            before.erase(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(cut));
            if (pending.has_value())
                pending->waitItem -= cut;
        } else {
            baseCs = items.front().onsetCs; // everything is recent: the base is the oldest event
        }
        if (cache != nullptr)
            cache->reset();
        ++st.contextRebuilds;
        return;
    }
    size_t total = 0;
    for (const auto &it: items)
        total += width(it.kind);
    if (total <= cfg.maxTokens)
        return;
    // the prefill that follows a cut is the costliest call: not while a WAIT is pending (unless the context
    // is 25% over the limit)
    if (pending.has_value() && total <= cfg.maxTokens + cfg.maxTokens / 4)
        return;
    // drop the oldest items: keep <= keepTokens, starting on an event (piano / orchestra quad), never
    // past the pending WAIT (then the latest event boundary before it)
    const size_t limit = pending.has_value() ? std::min(pending->waitItem, items.size()) : items.size();
    size_t keep = total;
    size_t cut = 0;
    size_t fallback = 0;
    for (size_t i = 0; i < limit; ++i) {
        const auto &it = items[i];
        const bool boundary = isEvent(it.kind);
        if (boundary && i > 0) {
            if (keep <= cfg.keepTokens) {
                cut = i;
                break;
            }
            fallback = i;
        }
        keep -= width(it.kind);
    }
    if (cut == 0)
        cut = fallback;
    if (cut == 0)
        return;
    baseCs = items[cut].onsetCs;
    items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(cut));
    before.erase(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(cut));
    if (pending.has_value())
        pending->waitItem -= cut;
    if (cache != nullptr)
        cache->reset();
    ++st.contextRebuilds;
    if (pre.has_value())
        startPrecompute(); // its context changed
}

// ---------------------------------------------------------------- hybrid support

auto InformativeWaitEngine::mirrorNote(const OutNote &note) -> void { mirrored.push_back(note); }

auto InformativeWaitEngine::applyMirrored() -> void {
    if (mirrored.empty() || items.empty())
        return;
    // the other model's notes go in at their onsets, among what is written (the legacy Hybrid's order)
    int32_t first = INT32_MAX;
    std::vector<Item> add;
    for (const auto &n: mirrored) {
        Item it;
        it.kind = Kind::Orch;
        it.onsetCs = n.onsetCs;
        it.durationCs = q5(n.durationCs);
        it.pitch = std::clamp(n.pitch, kPianoLow, kPianoHigh);
        it.velocity = std::clamp(8 * ((n.velocity + 4) / 8), 8, 120);
        add.push_back(it);
        first = std::min(first, n.onsetCs);
    }
    mirrored.clear();
    auto tail = takeAfter(first);
    tail.insert(tail.end(), add.begin(), add.end());
    reappend(std::move(tail));
}

auto InformativeWaitEngine::interrupt() -> void {
    // The pending WAIT / COUNT leave the context: neither fired nor EXPIRED, they would stay unanswered,
    // which no training line has (a model-drawn one too, not only a line mode's provisional one).
    if (pending.has_value() && chordMode()) {
        // the WAIT, its K and any LATEs written for it (piano notes between them stay)
        for (size_t i = items.size(); i-- > pending->waitItem;)
            if (items[i].kind == Kind::Wait || items[i].kind == Kind::Count || items[i].kind == Kind::Late) {
                items.erase(items.begin() + static_cast<std::ptrdiff_t>(i));
                before.erase(before.begin() + static_cast<std::ptrdiff_t>(i));
            }
    } else if (pending.has_value()) {
        const size_t i = pending->waitItem;
        if (i + 1 < items.size() && items[i].kind == Kind::Wait && items[i + 1].kind == Kind::Count) {
            items.erase(items.begin() + static_cast<std::ptrdiff_t>(i), items.begin() + static_cast<std::ptrdiff_t>(i + 2));
            before.erase(before.begin() + static_cast<std::ptrdiff_t>(i), before.begin() + static_cast<std::ptrdiff_t>(i + 2));
        }
    }
    pending.reset();
    pre.reset();
    fireLeft.reset();
}

auto InformativeWaitEngine::dropGeneratedFrom(int32_t cutCs) -> void {
    mirrored.erase(std::remove_if(mirrored.begin(), mirrored.end(), [cutCs](const auto &n) { return n.onsetCs >= cutCs; }),
                   mirrored.end());
    // Only the orchestra notes from the cut on go; piano notes after it are written back (the legacy rewind).
    const auto generated = [cutCs](const Item &it) {
        return (it.kind == Kind::Orch || it.kind == Kind::Chord) && it.onsetCs >= cutCs;
    };
    for (size_t i = 0; i < items.size(); ++i)
        if (generated(items[i])) {
            std::vector<Item> tail(items.begin() + static_cast<std::ptrdiff_t>(i), items.end());
            truncateTo(i);
            tail.erase(std::remove_if(tail.begin(), tail.end(), generated), tail.end());
            reappend(std::move(tail));
            return;
        }
}

auto InformativeWaitEngine::absorb(int32_t stepNowCs) -> void {
    nowCs = stepNowCs;
    if (! started)
        return;
    applyInput();
    applyMirrored();
    commitLongHolds();
    if (group.has_value() && nowCs >= group->headCs + kMergeCs)
        closeGroup();
    maybeShrinkContext();
    cancelFrom = INT32_MAX; // the drawing model reports cuts
}

// ---------------------------------------------------------------- main step

auto InformativeWaitEngine::step(int32_t stepNowCs) -> std::vector<OutNote> {
    nowCs = stepNowCs;
    std::vector<OutNote> out;
    if (! started || backend == nullptr)
        return out;
    out = applyInput();
    applyMirrored();
    commitLongHolds();
    if (fireLeft.has_value() && fireLeft->chord && out.empty())
        return markChord(fireChordStep()); // the chord block, then the leftover, of a fire or a give-up
    if (fireLeft.has_value() && out.empty()) {
        // the next stacked note of a fire
        const auto t0 = std::chrono::steady_clock::now();
        drawLog.clear();
        auto note = drawBody(fireLeft->onsetCs, fireLeft->minPitch);
        st.fireDecodeMsTotal += msSince(t0);
        if (note.has_value()) {
            note->draws = std::move(drawLog);
            drawLog.clear();
            Item it;
            it.kind = Kind::Orch;
            it.onsetCs = fireLeft->onsetCs;
            it.durationCs = note->durationCs;
            it.pitch = note->pitch;
            it.velocity = note->velocity;
            append(it);
            ++st.orchestraNotes;
            cur.lastOrchPitch = note->pitch;
            fireLeft->minPitch = std::min(note->pitch + 1, kPianoHigh);
            out.push_back(*note);
        }
        if (note.has_value())
            ++firePlaced;
        if (! note.has_value() || --fireLeft->left <= 0) {
            firePlacedEvent(fireLeft->onsetCs, false);
            fireLeft.reset();
        }
        return out;
    }
    if (group.has_value() && ! fireLeft.has_value() && nowCs >= group->headCs + kMergeCs) {
        auto c = closeGroup();
        out.insert(out.end(), c.begin(), c.end());
    }
    if (pending.has_value() && ! group.has_value()
        && nowCs > pending->hi(pending->provisional) + (pending->provisional ? 0 : cfg.expiryGraceCs)) {
        auto e = expire();
        out.insert(out.end(), e.begin(), e.end());
        if (fireLeft.has_value() && fireLeft->chord) {
            // chord mode, given up: the chord block (and leftover) right away, before anything else is drawn
            auto c = markChord(fireChordStep());
            out.insert(out.end(), c.begin(), c.end());
            maybeShrinkContext();
            return out;
        }
    }
    if (! out.empty()) {
        maybeShrinkContext();
        return out;
    }
    if (pending.has_value()) {
        if (pre.has_value() && ! pre->complete && ! pre->failed)
            precomputeStep();
        return out;
    }
    if (group.has_value() || items.empty())
        return out; // a piano group is still being collected / no context yet
    if (! isLine() && cur.clockCs > nowCs + cfg.drawAheadCs)
        return out; // far enough ahead
    out = drawEvent();
    maybeShrinkContext();
    return out;
}

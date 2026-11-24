#include "VirtualOrch/reduction/InformativeWaitEngine.h"

#include "VirtualOrch/reduction/TdOnsetBias.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace {

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
                out.insert(out.end(), {rel(it.onsetCs), kDur0 + q5(it.durationCs), kPiano0 + it.pitch,
                                       kVel0 + q8(it.velocity)});
                break;
            case Kind::Orch:
                out.insert(out.end(), {rel(it.onsetCs), kDur0 + it.durationCs, kOrch0 + it.pitch, kVel0 + it.velocity});
                break;
            case Kind::Wait:
            case Kind::Count:
                out.push_back(it.token);
                break;
            case Kind::Expired:
                out.push_back(kExpired);
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
    for (const auto &it: items) {
        const bool mutableItem = (it.kind == Kind::Piano && it.held) || it.provisional
                                 || (it.kind == Kind::Orch && it.onsetCs > nowCs);
        if (mutableItem)
            return len;
        len += (it.kind == Kind::Piano || it.kind == Kind::Orch) ? 4 : 1;
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
    return pickFrom(std::move(scores), temperature);
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
        if (it.kind == Kind::Orch) {
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
    before.push_back(cur);
    items.push_back(item);
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
    for (const auto &it: tail) {
        if (it.kind == Kind::Piano) {
            append(it);
            cur.clockCs = std::max(cur.clockCs, it.onsetCs);
            cur.floorCs = std::max(cur.floorCs, it.onsetCs);
            cur.exclHiCs = cfg.continuation ? it.onsetCs - 1 : it.onsetCs + kMergeCs;
            cur.chordOnCs = INT32_MIN;
            cur.chordCount = 0;
            cur.lastPianoCs = std::max(cur.lastPianoCs, it.onsetCs);
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
        if ((items[i].kind == Kind::Piano || items[i].kind == Kind::Orch) && items[i].onsetCs > onsetCs) {
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
        if (! down)
            it.durationCs = std::max(5, n.onsetCs + n.durationCs - head);
        else if (cfg.rewriteHeldDurations)
            it.durationCs = std::max(5, nowCs - n.onsetCs); // frozen until its note-off
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
    return out;
}

auto InformativeWaitEngine::onGroupHead(const PianoNote &head) -> std::vector<OutNote> {
    std::vector<OutNote> out;
    cutFuture(head.onsetCs);
    if (pending.has_value() && head.onsetCs > pending->hi(pending->provisional)) {
        auto e = expire(); // its window closed before this onset
        out.insert(out.end(), e.begin(), e.end());
        cutFuture(head.onsetCs); // a legato note written at the expiry may now be in the future
    }
    group = OpenGroup{head.onsetCs, {head}, false};
    if (pending.has_value()) {
        if (head.onsetCs < pending->lo()) {
            ++st.earlyPiano;
            event("early", head.onsetCs, head.pitch, pending->lo(), "piano before the WAIT window: ordinary input");
        } else {
            group->fired = true;
            auto f = fire(head);
            out.insert(out.end(), f.begin(), f.end());
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
        // first piano note and the stack are written; the rest of the chord follows them
        std::vector<PianoNote> rest(g.notes.begin() + 1, g.notes.end());
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
    // model 2: nothing within 5 cs after a piano onset; model 1: the orchestra may share it
    cur.exclHiCs = cfg.continuation ? g.headCs - 1 : g.headCs + kMergeCs;
    cur.chordOnCs = INT32_MIN;
    cur.chordCount = 0;
    cur.lastPianoCs = g.headCs;
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
    const Branch *branch = nullptr;
    if (pre.has_value() && ! pre->failed)
        for (const auto &b: pre->branches)
            if (b.pitch == head.pitch)
                branch = &b;
    int32_t ready = 0;
    if (branch != nullptr)
        for (const auto &n: branch->notes)
            if (n[1] > 0 && n[2] > 0 && ready < p.k)
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
    if (ready == p.k) {
        ++st.precomputeHits;
        event("hit", head.onsetCs, head.pitch, p.k, "precomputed branch played");
    } else {
        if (ready > 0)
            ++st.precomputePartial;
        // the remaining notes are decoded one per step (each is played as soon as it exists; input is
        // handled in between); the rest of the chord is written after the last of them
        fireLeft = FireLeft{head.onsetCs, p.k - ready, minPitch};
        ++st.fireDecoded;
        event("miss", head.onsetCs, head.pitch, p.k,
              branch == nullptr ? (pre.has_value() && ! pre->pianoProbs.empty() ? "not in the top 5" : "no anticipation yet")
                                : "branch had " + std::to_string(ready) + " of " + std::to_string(p.k) + " notes");
    }
    st.orchestraNotes += static_cast<int>(out.size());
    event("fire", head.onsetCs, p.k, static_cast<int32_t>(out.size()));
    firePlaced = static_cast<int32_t>(out.size());
    if (! fireLeft.has_value())
        firePlacedEvent(head.onsetCs, ready == p.k);
    cur.exclHiCs = head.onsetCs + kMergeCs;
    cur.chordOnCs = INT32_MIN;
    cur.chordCount = 0;
    cur.lastOrchPitch = out.empty() ? -1 : out.back().pitch;
    cur.lastPianoCs = head.onsetCs;
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

auto InformativeWaitEngine::expire() -> std::vector<OutNote> {
    std::vector<OutNote> out;
    if (! pending.has_value())
        return out;
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

auto InformativeWaitEngine::drawBody(int32_t onsetCs, int32_t minPitch) -> std::optional<OutNote> {
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
        const auto window = DenseDurationBias::collectDurationWindowAt(history, onsetCs, -1, 1);
        const auto taus = durationScheduler.evaluate(onsetCs, window.uniqueDurations, window.uniqueDeltas,
                                                     cfg.durationTau, cfg.durationTauPrime, true);
        DenseDurationBias::applyDurationLogitsBias(biased, window, taus.tau, taus.tauPrime);
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
        const auto window = DensePitchBias::collectPitchWindowAt(history, onsetCs, -1);
        const auto taus = pitchScheduler.evaluate(onsetCs, window.uniquePitches, window.uniqueDeltas, cfg.pitchTau,
                                                  cfg.pitchTauPrime, true);
        DensePitchBias::applyNoteLogitsBias(biased, window, taus.tau, taus.tauPrime);
        l2 = &biased;
    }
    const int32_t note = pick(*l2, {{kOrch0 + lo, kOrch0 + hi + 1}}, cfg.otherTemperature, cfg.minP);
    if (note < 0)
        return std::nullopt;
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
    const auto note = drawBody(onsetCs, cfg.bandLow);
    if (! note.has_value())
        return out;
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
    const bool waitOk = cfg.mode != Mode::LineLegato && ! cfg.continuation;
    if (waitOk) {
        // only WAITs whose window has not already passed: ref + d + 6 > now
        const int32_t dMin = std::max(kGrid, nowCs + 1 - cur.clockCs - kEps);
        if (dMin <= kMaxWait)
            ranges.push_back({waitToken(((dMin + kGrid - 1) / kGrid) * kGrid), kCount0});
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
    const int32_t t = pick(*logits, ranges, cfg.onsetTemperature, 0.0f);
    if (t < 0) {
        lastError = "nothing to draw at the onset step";
        return out;
    }
    if (t >= kWait0 && t < kCount0) {
        const int32_t d = waitValue(t);
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
        ++st.waitsDrawn;
        if (d >= 10)
            ++st.waitsGe10;
        event("wait", cur.clockCs, d, k);
        startPrecompute();
        return out;
    }
    const int32_t onset = t + baseCs;
    if (line)
        return noteAt(onset);
    const bool same = chordOk && onset == cur.chordOnCs;
    const auto note = drawBody(onset, same && cfg.ascendingChords ? cur.lastOrchPitch : kPianoLow);
    if (! note.has_value())
        return out;
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

// ---------------------------------------------------------------- anticipation

auto InformativeWaitEngine::startPrecompute() -> void {
    if (! pending.has_value())
        return;
    pre = Precompute{};
    pre->targetCs = pending->refCs + pending->d;
    pre->k = pending->k;
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
        if (! backend->run(s, {T, kDur0 + q5(pc.durationCs)}, pc.state, lv) || lv.size() < static_cast<size_t>(kVocab))
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
        pc.state = {};
        if (! tiled)
            return finish(false);
        for (const auto &b: pc.branches)
            feed.insert(feed.end(), {kPiano0 + b.pitch, kVel0 + b.velocity, T});
    } else {
        const auto V = static_cast<size_t>(kVocab);
        if (pc.logits.size() < V * kBranches)
            return finish(false);
        const int r = (pc.phase - 2) % 3; // what the last logits predict: 0 duration, 1 pitch, 2 velocity
        for (int b = 0; b < kBranches; ++b) {
            auto &br = pc.branches[static_cast<size_t>(b)];
            const auto row = static_cast<size_t>(b);
            const std::vector<float> lg(pc.logits.begin() + static_cast<std::ptrdiff_t>(row * V),
                                        pc.logits.begin() + static_cast<std::ptrdiff_t>((row + 1) * V));
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
            if ((items[i].kind == Kind::Piano || items[i].kind == Kind::Orch) && items[i].onsetCs >= nowCs - 6000) {
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
        total += (it.kind == Kind::Piano || it.kind == Kind::Orch) ? 4 : 1;
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
        const bool boundary = it.kind == Kind::Piano || it.kind == Kind::Orch;
        if (boundary && i > 0) {
            if (keep <= cfg.keepTokens) {
                cut = i;
                break;
            }
            fallback = i;
        }
        keep -= boundary ? 4 : 1;
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
    if (pending.has_value() && pending->provisional)
        removeProvisionalWait(pending->waitItem);
    pending.reset();
    pre.reset();
    fireLeft.reset();
}

auto InformativeWaitEngine::dropGeneratedFrom(int32_t cutCs) -> void {
    mirrored.erase(std::remove_if(mirrored.begin(), mirrored.end(), [cutCs](const auto &n) { return n.onsetCs >= cutCs; }),
                   mirrored.end());
    // Only the orchestra notes from the cut on go; piano notes after it are written back (the legacy rewind).
    for (size_t i = 0; i < items.size(); ++i)
        if (items[i].kind == Kind::Orch && items[i].onsetCs >= cutCs) {
            std::vector<Item> tail(items.begin() + static_cast<std::ptrdiff_t>(i), items.end());
            truncateTo(i);
            tail.erase(std::remove_if(tail.begin(), tail.end(),
                                      [cutCs](const Item &it) { return it.kind == Kind::Orch && it.onsetCs >= cutCs; }),
                       tail.end());
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
    if (fireLeft.has_value() && out.empty()) {
        // the next stacked note of a fire
        const auto t0 = std::chrono::steady_clock::now();
        const auto note = drawBody(fireLeft->onsetCs, fireLeft->minPitch);
        st.fireDecodeMsTotal += msSince(t0);
        if (note.has_value()) {
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
    if (pending.has_value() && ! group.has_value() && nowCs > pending->hi(pending->provisional)) {
        auto e = expire();
        out.insert(out.end(), e.begin(), e.end());
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

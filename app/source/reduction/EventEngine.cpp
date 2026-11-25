#include "VirtualOrch/reduction/EventEngine.h"
#include "VirtualOrch/reduction/DenseOnsetGapBias.h"
#include "VirtualOrch/reduction/DenseTypes.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

auto isOn(EventEngine::Kind k) -> bool { return k == EventEngine::Kind::PianoOn || k == EventEngine::Kind::OrchOn; }
auto isOrch(EventEngine::Kind k) -> bool { return k == EventEngine::Kind::OrchOn || k == EventEngine::Kind::OrchOff; }
auto widthOf(const EventEngine::Event &e) -> size_t {
    if (e.kind == EventEngine::Kind::Special)
        return e.toks.size();
    return isOn(e.kind) ? 3 : 2;
}
auto qd(int32_t d, int32_t hi) -> int32_t {
    const int32_t g = EventEngine::kGrid;
    return std::clamp(g * static_cast<int32_t>(std::lround(static_cast<double>(d) / g)), g, hi);
}
auto pcMask(int type, int root) -> uint16_t {
    static const std::vector<std::vector<int>> iv = {{0},          {0, 1},       {0, 2},       {0, 3},       {0, 4},
                                                     {0, 5},       {0, 6},       {0, 4, 7},    {0, 3, 7},    {0, 3, 6},
                                                     {0, 4, 8},    {0, 4, 7, 10}, {0, 3, 7, 10}, {0, 4, 7, 11}, {0, 3, 7, 11},
                                                     {0, 3, 6, 10}, {0, 3, 6, 9}, {0, 4, 8, 11}};
    uint16_t m = 0;
    if (type >= 0 && type < static_cast<int>(iv.size()))
        for (const int i: iv[static_cast<size_t>(type)])
            m = static_cast<uint16_t>(m | (1u << ((root + i) % 12)));
    return m;
}

} // namespace

EventEngine::EventEngine() : rng(1) {}

auto EventEngine::setBackend(StreamingKv::Backend *backendIn) -> void {
    backend = backendIn;
    cache = backend != nullptr ? std::make_unique<StreamingKv::CommittedScratchCache>(*backend) : nullptr;
    if (cache != nullptr)
        cache->reuseScratchOnCommit = true; // a committed draw is the head of the scratch it was drawn on
}

auto EventEngine::reset(int32_t now) -> void {
    items.clear();
    baseCs = now;
    nowCs = now;
    floorCs = INT32_MIN;
    lastOrchCs = now; // the force-after-hold timer starts at the reset
    pending.reset();
    wait.reset();
    fire.reset();
    stacks.clear();
    st = {};
    pitchScheduler.reset();
    rng.seed(cfg.seed);
    if (cache != nullptr)
        cache->reset();
}

auto EventEngine::noteToken(Kind kind, int32_t pitch) const -> int32_t {
    if (kind == Kind::StackOff)
        return cfg.stackOff;
    const int32_t note = (isOrch(kind) ? kOrch0 : kNote0) + std::clamp(pitch, 0, 127);
    return isOn(kind) ? note : cfg.off0 + (note - kNote0);
}

auto EventEngine::tokens() const -> std::vector<int32_t> {
    std::vector<int32_t> out;
    out.reserve(items.size() * 3);
    for (const auto &e: items) {
        if (e.kind == Kind::Special) {
            out.insert(out.end(), e.toks.begin(), e.toks.end());
            continue;
        }
        out.push_back(std::clamp(timeToken(e.timeCs), 0, kMaxTime));
        out.push_back(noteToken(e.kind, e.pitch));
        if (isOn(e.kind))
            out.push_back(kVel0 + std::clamp(e.velocity, 0, 127));
    }
    return out;
}

auto EventEngine::sounding() const -> std::vector<std::pair<int32_t, int32_t>> {
    std::vector<std::pair<int32_t, int32_t>> open;
    for (const auto &e: items) {
        if (e.kind == Kind::OrchOn) {
            open.erase(std::remove_if(open.begin(), open.end(), [&](const auto &o) { return o.second == e.pitch; }),
                       open.end());
            open.emplace_back(e.timeCs, e.pitch);
        } else if (e.kind == Kind::OrchOff) {
            open.erase(std::remove_if(open.begin(), open.end(), [&](const auto &o) { return o.second == e.pitch; }),
                       open.end());
        }
    }
    return open;
}

auto EventEngine::soundingAll() const -> std::vector<int32_t> {
    std::vector<int32_t> out;
    for (const auto &[onset, p]: sounding())
        out.push_back(p);
    for (const auto &s: stacks)
        out.insert(out.end(), s.pitches.begin(), s.pitches.end());
    return out;
}

auto EventEngine::append(const Event &e) -> void { items.push_back(e); }

auto EventEngine::special(int32_t timeCs, std::vector<int32_t> toks) -> void {
    Event e;
    e.kind = Kind::Special;
    e.timeCs = std::max(timeCs, lastTime());
    e.generated = true;
    e.toks = std::move(toks);
    append(e);
}

auto EventEngine::insertInput(Event e) -> void {
    // Live input is never ahead of the clock; a drawn event not yet due is drawn again after it. Input that comes
    // late (MIDI latency) behind an orchestra event already written keeps the context in time order: its time is
    // raised to that event's.
    if (pending.has_value() && pending->timeCs >= e.timeCs) {
        pending.reset();
        ++st.discarded;
    }
    e.timeCs = std::max(e.timeCs, lastTime());
    maybeRebuild(e.timeCs);
    if (chordMode() && e.kind == Kind::PianoOn)
        onPianoOnChord(e);
    append(e);
}

auto EventEngine::pianoOn(int32_t timeCs, int32_t pitch, int32_t velocity) -> void {
    insertInput({Kind::PianoOn, timeCs, std::clamp(pitch, kPianoLow, kPianoHigh), std::clamp(velocity, 1, 127), false, {}});
}

auto EventEngine::pianoOff(int32_t timeCs, int32_t pitch) -> void {
    insertInput({Kind::PianoOff, timeCs, std::clamp(pitch, kPianoLow, kPianoHigh), 0, false, {}});
}

auto EventEngine::mirrorOn(int32_t timeCs, int32_t pitch, int32_t velocity) -> void {
    insertInput({Kind::OrchOn, timeCs, pitch, velocity, false, {}});
    lastOrchCs = std::max(lastOrchCs, timeCs);
}

auto EventEngine::mirrorOff(int32_t timeCs, int32_t pitch) -> void {
    insertInput({Kind::OrchOff, timeCs, pitch, 0, false, {}});
}

auto EventEngine::dropGeneratedFrom(int32_t cutCs) -> void {
    pending.reset();
    wait.reset();
    fire.reset();
    items.erase(std::remove_if(items.begin(), items.end(), [&](const Event &e) { return e.generated && e.timeCs >= cutCs; }),
                items.end());
    stacks.erase(std::remove_if(stacks.begin(), stacks.end(), [&](const Stack &s) { return s.onsetCs >= cutCs; }),
                 stacks.end());
    floorCs = std::min(floorCs, cutCs);
}

auto EventEngine::maybeRebuild(int32_t floor) -> void {
    if (floor - baseCs <= cfg.maxSpanCs || wait.has_value() || fire.has_value())
        return; // (not in the middle of a WAIT: its reference is in the context)
    // keep the newest events: at most keepTokens tokens, and a span that leaves room for the next draws
    size_t tokensKept = 0;
    size_t first = items.size();
    while (first > 0) {
        const auto &e = items[first - 1];
        if (tokensKept + widthOf(e) > cfg.keepTokens || floor - e.timeCs > cfg.maxSpanCs / 2)
            break;
        tokensKept += widthOf(e);
        --first;
    }
    items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(first));
    baseCs = items.empty() ? floor : items.front().timeCs;
    pending.reset();
    if (cache != nullptr)
        cache->reset();
    ++st.rebuilds;
}

auto EventEngine::logitsFor(const std::vector<int32_t> &desired) -> const std::vector<float> * {
    if (cache == nullptr || desired.empty())
        return nullptr;
    const size_t committed = tokens().size();
    const auto &l = cache->logitsAfter(desired, std::min(committed, desired.size()));
    if (l.size() < static_cast<size_t>(cfg.vocab)) {
        lastError = "event model returned no usable logits (" + std::to_string(l.size()) + ")";
        return nullptr;
    }
    return &l;
}

auto EventEngine::pick(const std::vector<float> &logits, const std::vector<std::pair<int32_t, int32_t>> &ranges,
                       float temperature) -> int32_t {
    float peak = kNegInf;
    for (const auto &[lo, hi]: ranges)
        for (int32_t t = std::max(lo, 0); t < std::min(hi, static_cast<int32_t>(logits.size())); ++t)
            peak = std::max(peak, logits[static_cast<size_t>(t)]);
    if (! std::isfinite(peak))
        return -1;
    const float cut = cfg.minP > 0 ? peak + std::log(cfg.minP) : kNegInf;
    const float tau = std::max(temperature, 1e-4f);
    std::vector<std::pair<double, int32_t>> p;
    double total = 0;
    for (const auto &[lo, hi]: ranges)
        for (int32_t t = std::max(lo, 0); t < std::min(hi, static_cast<int32_t>(logits.size())); ++t) {
            const float l = logits[static_cast<size_t>(t)];
            if (! std::isfinite(l) || l < cut)
                continue;
            const double w = std::exp(static_cast<double>((l - peak) / tau));
            p.emplace_back(w, t);
            total += w;
        }
    if (p.empty() || ! (total > 0))
        return -1;
    double u = std::uniform_real_distribution<double>(0.0, total)(rng);
    for (const auto &[w, t]: p) {
        u -= w;
        if (u <= 0)
            return t;
    }
    return p.back().second;
}

// ---------------------------------------------------------------- model 2 event: WAIT family and chord blocks

auto EventEngine::waitRanges(int32_t refCs, int32_t now) const -> std::vector<std::pair<int32_t, int32_t>> {
    // WAIT d: d = 3 .. 300 on the 3 cs grid, at least the 6 cs floor, and a target still ahead of the clock
    const int32_t dMin = std::max(kFloor, now + 1 - refCs);
    const int32_t first = std::max(1, (dMin + kGrid - 1) / kGrid - 1);
    const int32_t last = kMaxWait / kGrid - 1;
    if (first > last)
        return {};
    return {{kWait0 + first, kWait0 + last + 1}};
}

auto EventEngine::chordRanges(int32_t k) const -> std::vector<std::pair<int32_t, int32_t>> {
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t type = 0; type < 18; ++type)
        if (VoicingBank::pitchClassCount(type) == k)
            r.push_back({kChord0 + type * 120, kChord0 + (type + 1) * 120}); // the type's 12 roots x 10 n
    if (r.empty())
        r.push_back({kChord0, kChordEnd});
    return r;
}

auto EventEngine::drawWaitK(std::vector<int32_t> desired, int32_t waitTok, int32_t refCs, int32_t sameKTok) -> bool {
    const int32_t d = (waitTok - kWait0 + 1) * kGrid;
    int32_t kTok = sameKTok;
    if (kTok < 0) {
        desired.push_back(waitTok);
        const auto *l = logitsFor(desired);
        if (l == nullptr)
            return false;
        std::vector<std::pair<int32_t, int32_t>> ranges;
        for (int32_t i = 0; i < 8; ++i)
            if (i % 2 == 0 || d >= kFloorPlus) // +1 only from 12 cs
                ranges.push_back({kK0 + i, kK0 + i + 1});
        kTok = pick(*l, ranges, cfg.otherTemperature);
        if (kTok < 0)
            return false;
    }
    special(refCs, {waitTok, kTok});
    Wait w;
    w.refCs = refCs;
    w.d = d;
    w.kTok = kTok;
    w.k = (kTok - kK0) / 2 + 1;
    w.plus = (kTok - kK0) % 2 == 1;
    w.closeCs = refCs + d + kEps;
    wait = w;
    ++st.waits;
    return true;
}

auto EventEngine::onPianoOnChord(Event &e) -> void {
    if (fire.has_value()) {
        if (! fire->noPiano && e.timeCs <= fire->onsetCs + kMergeCs)
            e.timeCs = fire->onsetCs; // a piano ON of the fired onset: written at its time, before the block
        return;
    }
    if (! wait.has_value() || e.timeCs < wait->refCs + wait->d - kWinEarly)
        return; // no WAIT, or before its window: ordinary input
    if (e.timeCs > wait->closeCs && wait->lates < cfg.maxLates) {
        // the piano came after the close before the engine closed it: the LATE it came at
        const int32_t dl = std::clamp(qd(e.timeCs - wait->closeCs, kLateMax), kGrid, kLateMax);
        special(wait->closeCs, {kLate0 + dl / kGrid - 1});
        ++wait->lates;
        ++st.lates;
        ++st.latesForced;
    }
    Fire f;
    f.onsetCs = e.timeCs;
    f.readyCs = e.timeCs + kMergeCs;
    f.k = wait->k;
    f.plus = wait->plus;
    fire = f;
    wait.reset();
    ++st.fires;
}

auto EventEngine::closeWindow(int32_t now) -> void {
    auto &w = *wait;
    auto desired = tokens();
    const auto *l = logitsFor(desired);
    if (l == nullptr) {
        wait.reset();
        return;
    }
    std::vector<std::pair<int32_t, int32_t>> ranges{{kExpired, kExpired + 1}};
    if (w.lates < cfg.maxLates)
        ranges.push_back({kLate0, kLate0 + kLateMax / kGrid});
    const int32_t tok = pick(*l, ranges, cfg.otherTemperature);
    if (tok >= kLate0 && tok < kLate0 + kLateMax / kGrid) {
        const int32_t dl = (tok - kLate0 + 1) * kGrid;
        special(w.closeCs, {tok});
        w.closeCs += qd(dl, kLateMax) + kEps;
        ++w.lates;
        ++st.lates;
        return;
    }
    const int32_t close = w.closeCs;
    special(close, {kExpired});
    ++st.expired;
    desired = tokens();
    const auto *l2 = logitsFor(desired);
    const Wait was = w;
    wait.reset();
    if (l2 == nullptr)
        return;
    auto next = waitRanges(close, now);
    const auto chords = chordRanges(was.k);
    next.insert(next.end(), chords.begin(), chords.end());
    const int32_t t2 = pick(*l2, next, cfg.otherTemperature);
    if (t2 >= kWait0 && t2 < kK0) {
        if (drawWaitK(desired, t2, close, was.kTok)) // re-wait with the same K
            ++st.reWaits;
        return;
    }
    if (t2 >= kChord0 && t2 < kChordEnd) {
        Fire f; // give up: the block plays alone at the close
        f.onsetCs = close;
        f.readyCs = now;
        f.noPiano = true;
        f.firstChord = t2;
        f.k = was.k;
        f.plus = was.plus;
        fire = f;
        ++st.giveUps;
    }
}

auto EventEngine::writeBlock(int32_t /*now*/) -> std::vector<Out> {
    std::vector<Out> out;
    const Fire f = *fire;
    fire.reset();
    const int32_t t = std::max(f.onsetCs, lastTime());
    auto desired = tokens();
    int32_t chord = f.firstChord;
    if (chord < 0) {
        const auto *l = logitsFor(desired);
        if (l == nullptr)
            return out;
        chord = pick(*l, chordRanges(f.k), cfg.otherTemperature);
        if (chord < 0)
            return out;
    }
    desired.push_back(chord);
    const int32_t ci = chord - kChord0;
    const int type = ci / 120;
    const int root = (ci / 10) % 12;
    const int n = ci % 10 + 1;
    const uint16_t mask = pcMask(type, root);
    // BASS: a chord tone in the register
    const auto *lb = logitsFor(desired);
    if (lb == nullptr)
        return out;
    std::vector<std::pair<int32_t, int32_t>> bassR;
    for (int32_t p = std::max(cfg.pitchLow, 0); p <= std::min(cfg.pitchHigh, 127); ++p)
        if ((mask >> (p % 12)) & 1u)
            bassR.push_back({kOrch0 + p, kOrch0 + p + 1});
    if (bassR.empty())
        bassR.push_back({kOrch0 + std::max(cfg.pitchLow, 0), kOrch0 + std::min(cfg.pitchHigh, 127) + 1});
    const int32_t bassTok = pick(*lb, bassR, cfg.otherTemperature);
    if (bassTok < 0)
        return out;
    desired.push_back(bassTok);
    const auto *lv = logitsFor(desired);
    if (lv == nullptr)
        return out;
    const int32_t svel = pick(*lv, {{kVel0 + 1, kVel0 + 128}}, cfg.otherTemperature);
    if (svel < 0)
        return out;
    special(t, {chord, bassTok, svel});
    // the band notes: the voicing bank (an empty bank stacks the chord tones), folded into the register
    const int32_t bass = bassTok - kOrch0;
    VoicingBank::Decoded dec;
    if (cfg.voicings != nullptr && ! cfg.voicings->empty()) {
        dec = cfg.voicings->decode(type, root, n, bass, rng);
    } else {
        static const VoicingBank none;
        dec = none.decode(type, root, n, bass, rng);
    }
    if (dec.source != VoicingBank::Decoded::Source::Bank)
        ++st.voicingFallbacks;
    const auto busy = soundingAll();
    std::vector<int32_t> pitches;
    for (int32_t p: dec.pitches) {
        while (p > cfg.pitchHigh && p - 12 >= bass)
            p -= 12;
        if (p <= cfg.pitchHigh && p >= 0 && std::find(pitches.begin(), pitches.end(), p) == pitches.end()
            && std::find(busy.begin(), busy.end(), p) == busy.end())
            pitches.push_back(p);
    }
    std::sort(pitches.begin(), pitches.end());
    if (cfg.maxSounding > 0) {
        const auto room = static_cast<size_t>(std::max<int32_t>(0, cfg.maxSounding - static_cast<int32_t>(busy.size())));
        if (pitches.size() > room)
            pitches.resize(room); // the polyphony cap (Mono: the bass)
    }
    if (! pitches.empty())
        stacks.push_back({t, pitches});
    for (const int32_t p: pitches)
        out.push_back(Out{true, t, p, svel - kVel0, false, true});
    lastOrchCs = std::max(lastOrchCs, t);
    floorCs = std::max(floorCs, t);
    // the leftover ON: at the block's onset
    if (f.plus) {
        desired = tokens();
        desired.push_back(timeToken(t));
        const auto *lo = logitsFor(desired);
        if (lo == nullptr)
            return out;
        const auto busyNow = soundingAll();
        std::vector<std::pair<int32_t, int32_t>> r;
        if (cfg.maxSounding <= 0 || static_cast<int32_t>(busyNow.size()) < cfg.maxSounding)
            for (int32_t p = std::max(cfg.pitchLow, 0); p <= std::min(cfg.pitchHigh, 127); ++p)
                if (std::find(busyNow.begin(), busyNow.end(), p) == busyNow.end())
                    r.push_back({kOrch0 + p, kOrch0 + p + 1});
        const int32_t note = r.empty() ? -1 : pick(*lo, r, cfg.otherTemperature);
        if (note >= 0) {
            desired.push_back(note);
            const auto *lv2 = logitsFor(desired);
            const int32_t vel = lv2 == nullptr ? -1 : pick(*lv2, {{kVel0 + 1, kVel0 + 128}}, cfg.otherTemperature);
            if (vel >= 0) {
                append({Kind::OrchOn, t, note - kOrch0, vel - kVel0, true, {}});
                ++st.ons;
                out.push_back(Out{true, t, note - kOrch0, vel - kVel0, false, true});
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------- drawing

auto EventEngine::orchestraQuads() const -> std::vector<int32_t> {
    // the bias windows read onsets and pitches only; the duration is a placeholder (events carry none)
    std::vector<int32_t> quads;
    quads.reserve(items.size() * 4);
    for (const auto &e: items)
        if (e.kind == Kind::OrchOn) {
            quads.push_back(e.timeCs);
            quads.push_back(static_cast<int32_t>(DenseVocab::DurOffset) + 1);
            quads.push_back(kOrch0 + std::clamp(e.pitch, 0, 127));
            quads.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + std::clamp(e.velocity, 0, 127));
        }
    return quads;
}

auto EventEngine::applyPitchBias(std::vector<float> &logits, const std::vector<std::pair<int32_t, int32_t>> &ranges,
                                 int32_t timeCs) -> void {
    const auto history = orchestraQuads();
    const auto window = DensePitchBias::collectPitchWindowAt(history, timeCs, -1);
    if (window.noteCount <= 0)
        return;
    const auto taus = pitchScheduler.evaluate(timeCs, window.uniquePitches, window.uniqueDeltas, cfg.pitchTau,
                                              cfg.pitchTauPrime, true);
    // the orchestra ON ids the pick allows
    std::vector<int32_t> ons;
    for (const auto &[lo, hi]: ranges)
        for (int32_t id = std::max(lo, kOrch0); id < std::min(hi, kOrch0 + 128); ++id)
            if (static_cast<size_t>(id) < logits.size() && std::isfinite(logits[static_cast<size_t>(id)]))
                ons.push_back(id);
    if (ons.empty())
        return;
    const float temp = std::max(cfg.otherTemperature, 1e-4f);
    const auto lse = [&](const std::vector<float> &l) {
        float peak = -std::numeric_limits<float>::infinity();
        for (const int32_t id: ons)
            peak = std::max(peak, l[static_cast<size_t>(id)]);
        double sum = 0.0;
        for (const int32_t id: ons)
            sum += std::exp(static_cast<double>((l[static_cast<size_t>(id)] - peak) / temp));
        return peak / temp + static_cast<float>(std::log(sum));
    };
    std::vector<float> biased = logits;
    DensePitchBias::applyNoteLogitsBias(biased, window, taus.tau, taus.tauPrime);
    // keep the ONs' total mass at the temperature: only which pitch changes, not ON vs OFF vs a piano no-op
    const float restore = temp * (lse(logits) - lse(biased));
    for (const int32_t id: ons)
        logits[static_cast<size_t>(id)] = biased[static_cast<size_t>(id)] + restore;
}

auto EventEngine::draw(int32_t now) -> void {
    const int32_t floor = std::max({lastTime(), now, floorCs});
    maybeRebuild(floor);
    int32_t hiTime = std::min(floor + cfg.maxAheadCs, baseCs + kMaxTime);
    if (wait.has_value())
        hiTime = std::min(hiTime, wait->refCs + wait->d - 1); // releases only, before the target (Q9)
    auto desired = tokens();
    const auto *l = logitsFor(desired);
    if (l == nullptr)
        return;
    std::vector<std::pair<int32_t, int32_t>> first;
    if (hiTime >= floor)
        first.push_back({timeToken(floor), timeToken(hiTime) + 1});
    if (chordMode() && ! wait.has_value()) {
        const auto w = waitRanges(lastTime(), now);
        first.insert(first.end(), w.begin(), w.end());
    }
    if (first.empty())
        return;
    std::vector<float> timeBiased;
    if (cfg.onsetGapBias) {
        // the orchestra's onset gaps on the time tokens (DenseOnsetGapBias touches only 0..9999, not the WAIT family)
        timeBiased = *l;
        const auto window = DenseOnsetGapBias::collectOnsetGapWindow(orchestraQuads(), now, -1);
        DenseOnsetGapBias::applyOnsetLogitsBias(timeBiased, window, cfg.onsetGapTau, baseCs);
        l = &timeBiased;
    }
    const int32_t firstTok = pick(*l, first, cfg.onsetTemperature);
    if (firstTok < 0)
        return;
    if (chordMode() && firstTok >= kWait0 && firstTok < kK0) {
        drawWaitK(desired, firstTok, lastTime(), -1);
        return;
    }
    const int32_t timeTok = firstTok;
    const int32_t t = baseCs + timeTok;

    // what happens at t: an orchestra ON / OFF (STACK_OFF), or a piano event (the model expects the pianist)
    const auto open = sounding();
    const auto busy = soundingAll();
    const Event *last = items.empty() ? nullptr : &items.back();
    const bool sameTime = last != nullptr && last->kind != Kind::Special && last->timeCs == t;
    const int32_t lastId = sameTime ? noteToken(last->kind, last->pitch) : -1;
    std::vector<std::pair<int32_t, int32_t>> ranges;
    const bool onsAllowed = ! wait.has_value() && t >= cfg.minOnsetCs
                            && (cfg.maxSounding <= 0 || static_cast<int32_t>(busy.size()) < cfg.maxSounding);
    if (onsAllowed) {
        int32_t lo = kOrch0 + std::max(cfg.pitchLow, 0);
        if (sameTime && isOn(last->kind))
            lo = std::max(lo, lastId + 1); // at one time: ONs by note id
        for (int32_t id = lo; id <= kOrch0 + std::min(cfg.pitchHigh, 127); ++id) {
            const int32_t p = id - kOrch0;
            if (std::find(busy.begin(), busy.end(), p) == busy.end())
                ranges.push_back({id, id + 1});
        }
    }
    const bool offsAllowed = ! sameTime || ! isOn(last->kind); // at one time every OFF comes before every ON
    if (offsAllowed) {
        for (const auto &[onset, p]: open) {
            const int32_t id = noteToken(Kind::OrchOff, p);
            if (onset < t && (! sameTime || last->kind == Kind::StackOff || id > lastId))
                ranges.push_back({id, id + 1});
        }
        if (chordMode() && cfg.stackOff >= 0 && ! stacks.empty() && stacks.front().onsetCs < t
            && ! (sameTime && last->kind == Kind::StackOff))
            ranges.push_back({cfg.stackOff, cfg.stackOff + 1});
    }
    const bool forced = t - lastOrchCs >= cfg.forceAfterCs && ! wait.has_value();
    const size_t orchRanges = ranges.size();
    if (! forced) {
        ranges.push_back({kNote0, kOrch0});                         // piano ONs
        ranges.push_back({cfg.off0, cfg.off0 + (kOrch0 - kNote0)}); // piano OFFs
    }
    desired.push_back(timeTok);
    const auto *l2 = logitsFor(desired);
    if (l2 == nullptr)
        return;
    std::vector<float> whatBiased;
    if (cfg.pitchBias && orchRanges > 0) {
        whatBiased = *l2;
        applyPitchBias(whatBiased, ranges, t);
        l2 = &whatBiased;
    }
    const int32_t what = ranges.empty() ? -1 : pick(*l2, ranges, cfg.otherTemperature);
    const bool piano = (what >= kNote0 && what < kOrch0) || (what >= cfg.off0 && what < cfg.off0 + (kOrch0 - kNote0));
    if (what < 0 || piano) {
        if (piano)
            ++st.pianoMeant;
        floorCs = t + 1; // nothing happens; the next time is later
        return;
    }
    if (forced && orchRanges > 0)
        ++st.forced;
    Pending pd;
    pd.timeCs = t;
    pd.toks = {timeTok, what};
    if (what >= kOrch0 && what < kVel0) {
        pd.on = true;
        pd.pitch = what - kOrch0;
        desired.push_back(what);
        const auto *l3 = logitsFor(desired);
        if (l3 == nullptr)
            return;
        const int32_t vel = pick(*l3, {{kVel0 + 1, kVel0 + 128}}, cfg.otherTemperature);
        if (vel < 0)
            return;
        pd.velocity = vel - kVel0;
        pd.toks.push_back(vel);
    } else if (chordMode() && what == cfg.stackOff) {
        pd.on = false;
        pd.pitch = -1; // the oldest sounding chord block
    } else {
        pd.on = false;
        pd.pitch = what - cfg.off0 - (kOrch0 - kNote0);
    }
    pending = pd;
}

auto EventEngine::commitPending() -> std::vector<Out> {
    const auto pd = *pending;
    pending.reset();
    lastOrchCs = std::max(lastOrchCs, pd.timeCs);
    floorCs = std::max(floorCs, pd.timeCs);
    if (pd.pitch < 0) { // STACK_OFF: every band note of the oldest block
        std::vector<Out> out;
        append({Kind::StackOff, pd.timeCs, 0, 0, true, {}});
        ++st.stackOffs;
        if (! stacks.empty()) {
            for (const int32_t p: stacks.front().pitches)
                out.push_back(Out{false, pd.timeCs, p, 0, false, true});
            stacks.erase(stacks.begin());
        }
        return out;
    }
    append({pd.on ? Kind::OrchOn : Kind::OrchOff, pd.timeCs, pd.pitch, pd.velocity, true, {}});
    if (pd.on)
        ++st.ons;
    else
        ++st.offs;
    return {Out{pd.on, pd.timeCs, pd.pitch, pd.velocity, false, false}};
}

auto EventEngine::releaseLongHolds(int32_t now) -> std::vector<Out> {
    std::vector<Out> out;
    if (cfg.maxHoldCs <= 0)
        return out;
    const auto dropPending = [&] {
        if (pending.has_value()) {
            pending.reset();
            ++st.discarded;
        }
    };
    for (const auto &[onset, p]: sounding()) {
        if (now - onset < cfg.maxHoldCs)
            continue;
        dropPending();
        const int32_t t = std::max(onset + cfg.maxHoldCs, lastTime());
        append({Kind::OrchOff, t, p, 0, true, {}});
        ++st.forcedReleases;
        out.push_back(Out{false, t, p, 0, true, false});
    }
    while (! stacks.empty() && now - stacks.front().onsetCs >= cfg.maxHoldCs && cfg.stackOff >= 0) {
        dropPending();
        const int32_t t = std::max(stacks.front().onsetCs + cfg.maxHoldCs, lastTime());
        append({Kind::StackOff, t, 0, 0, true, {}});
        ++st.forcedReleases;
        for (const int32_t p: stacks.front().pitches)
            out.push_back(Out{false, t, p, 0, true, true});
        stacks.erase(stacks.begin());
    }
    return out;
}

auto EventEngine::step(int32_t now) -> std::vector<Out> {
    nowCs = now;
    auto out = releaseLongHolds(now);
    if (backend == nullptr)
        return out;
    if (chordMode()) {
        if (fire.has_value() && now >= fire->readyCs) {
            auto b = writeBlock(now);
            out.insert(out.end(), b.begin(), b.end());
            return out;
        }
        if (fire.has_value())
            return out; // the onset's piano notes are still coming
        if (wait.has_value() && now > wait->closeCs + cfg.expiryGraceCs) {
            pending.reset();
            closeWindow(now);
            if (fire.has_value()) {
                auto b = writeBlock(now);
                out.insert(out.end(), b.begin(), b.end());
            }
            return out;
        }
    }
    if (! pending.has_value())
        draw(now);
    if (pending.has_value() && pending->timeCs <= now) {
        auto c = commitPending();
        out.insert(out.end(), c.begin(), c.end());
    }
    return out;
}

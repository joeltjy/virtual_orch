#include "VirtualOrch/reduction/EventEngine.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

auto isOn(EventEngine::Kind k) -> bool { return k == EventEngine::Kind::PianoOn || k == EventEngine::Kind::OrchOn; }
auto isOrch(EventEngine::Kind k) -> bool { return k == EventEngine::Kind::OrchOn || k == EventEngine::Kind::OrchOff; }
auto width(EventEngine::Kind k) -> size_t { return isOn(k) ? 3 : 2; }

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
    st = {};
    rng.seed(cfg.seed);
    if (cache != nullptr)
        cache->reset();
}

auto EventEngine::noteToken(Kind kind, int32_t pitch) const -> int32_t {
    const int32_t note = (isOrch(kind) ? kOrch0 : kNote0) + std::clamp(pitch, 0, 127);
    return isOn(kind) ? note : cfg.off0 + (note - kNote0);
}

auto EventEngine::tokens() const -> std::vector<int32_t> {
    std::vector<int32_t> out;
    out.reserve(items.size() * 3);
    for (const auto &e: items) {
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

auto EventEngine::append(const Event &e) -> void { items.push_back(e); }

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
    append(e);
}

auto EventEngine::pianoOn(int32_t timeCs, int32_t pitch, int32_t velocity) -> void {
    insertInput({Kind::PianoOn, timeCs, std::clamp(pitch, kPianoLow, kPianoHigh), std::clamp(velocity, 1, 127), false});
}

auto EventEngine::pianoOff(int32_t timeCs, int32_t pitch) -> void {
    insertInput({Kind::PianoOff, timeCs, std::clamp(pitch, kPianoLow, kPianoHigh), 0, false});
}

auto EventEngine::mirrorOn(int32_t timeCs, int32_t pitch, int32_t velocity) -> void {
    insertInput({Kind::OrchOn, timeCs, pitch, velocity, false});
    lastOrchCs = std::max(lastOrchCs, timeCs);
}

auto EventEngine::mirrorOff(int32_t timeCs, int32_t pitch) -> void {
    insertInput({Kind::OrchOff, timeCs, pitch, 0, false});
}

auto EventEngine::dropGeneratedFrom(int32_t cutCs) -> void {
    pending.reset();
    items.erase(std::remove_if(items.begin(), items.end(), [&](const Event &e) { return e.generated && e.timeCs >= cutCs; }),
                items.end());
    floorCs = std::min(floorCs, cutCs);
}

auto EventEngine::maybeRebuild(int32_t floor) -> void {
    if (floor - baseCs <= cfg.maxSpanCs)
        return;
    // keep the newest events: at most keepTokens tokens, and a span that leaves room for the next draws
    size_t tokensKept = 0;
    size_t first = items.size();
    while (first > 0) {
        const auto &e = items[first - 1];
        if (tokensKept + width(e.kind) > cfg.keepTokens || floor - e.timeCs > cfg.maxSpanCs / 2)
            break;
        tokensKept += width(e.kind);
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

auto EventEngine::draw(int32_t now) -> void {
    const int32_t floor = std::max({lastTime(), now, floorCs});
    maybeRebuild(floor);
    const int32_t hiTime = std::min(floor + cfg.maxAheadCs, baseCs + kMaxTime);
    if (hiTime < floor)
        return;
    auto desired = tokens();
    const auto *l = logitsFor(desired);
    if (l == nullptr)
        return;
    const int32_t timeTok = pick(*l, {{timeToken(floor), timeToken(hiTime) + 1}}, cfg.onsetTemperature);
    if (timeTok < 0)
        return;
    const int32_t t = baseCs + timeTok;

    // what happens at t: an orchestra ON / OFF, or a piano event (the model expects the pianist)
    const auto open = sounding();
    const Event *last = items.empty() ? nullptr : &items.back();
    const bool sameTime = last != nullptr && last->timeCs == t;
    const int32_t lastId = sameTime ? noteToken(last->kind, last->pitch) : -1;
    std::vector<std::pair<int32_t, int32_t>> ranges;
    const bool onsAllowed = t >= cfg.minOnsetCs && (cfg.maxSounding <= 0 || static_cast<int32_t>(open.size()) < cfg.maxSounding);
    if (onsAllowed) {
        int32_t lo = kOrch0 + std::max(cfg.pitchLow, 0);
        if (sameTime && isOn(last->kind))
            lo = std::max(lo, lastId + 1); // at one time: ONs by note id
        for (int32_t id = lo; id <= kOrch0 + std::min(cfg.pitchHigh, 127); ++id) {
            const int32_t p = id - kOrch0;
            if (std::none_of(open.begin(), open.end(), [p](const auto &o) { return o.second == p; }))
                ranges.push_back({id, id + 1});
        }
    }
    if (! sameTime || ! isOn(last->kind)) // at one time every OFF comes before every ON
        for (const auto &[onset, p]: open) {
            const int32_t id = noteToken(Kind::OrchOff, p);
            if (onset < t && (! sameTime || id > lastId))
                ranges.push_back({id, id + 1});
        }
    const bool forced = t - lastOrchCs >= cfg.forceAfterCs;
    const size_t orchRanges = ranges.size();
    if (! forced) {
        ranges.push_back({kNote0, kOrch0});                    // piano ONs
        ranges.push_back({cfg.off0, cfg.off0 + (kOrch0 - kNote0)}); // piano OFFs
    }
    desired.push_back(timeTok);
    const auto *l2 = logitsFor(desired);
    if (l2 == nullptr)
        return;
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
    } else {
        pd.on = false;
        pd.pitch = what - cfg.off0 - (kOrch0 - kNote0);
    }
    pending = pd;
}

auto EventEngine::commitPending() -> Out {
    const auto pd = *pending;
    pending.reset();
    append({pd.on ? Kind::OrchOn : Kind::OrchOff, pd.timeCs, pd.pitch, pd.velocity, true});
    lastOrchCs = std::max(lastOrchCs, pd.timeCs);
    floorCs = std::max(floorCs, pd.timeCs);
    if (pd.on)
        ++st.ons;
    else
        ++st.offs;
    return Out{pd.on, pd.timeCs, pd.pitch, pd.velocity, false};
}

auto EventEngine::releaseLongHolds(int32_t now) -> std::vector<Out> {
    std::vector<Out> out;
    if (cfg.maxHoldCs <= 0)
        return out;
    for (const auto &[onset, p]: sounding()) {
        if (now - onset < cfg.maxHoldCs)
            continue;
        if (pending.has_value()) {
            pending.reset();
            ++st.discarded;
        }
        const int32_t t = std::max(onset + cfg.maxHoldCs, lastTime());
        append({Kind::OrchOff, t, p, 0, true});
        ++st.forcedReleases;
        out.push_back(Out{false, t, p, 0, true});
    }
    return out;
}

auto EventEngine::step(int32_t now) -> std::vector<Out> {
    nowCs = now;
    auto out = releaseLongHolds(now);
    if (backend == nullptr)
        return out;
    if (! pending.has_value())
        draw(now);
    if (pending.has_value() && pending->timeCs <= now)
        out.push_back(commitPending());
    return out;
}

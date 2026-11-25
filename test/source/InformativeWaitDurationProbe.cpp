#include <gtest/gtest.h>

#include <JuceHeader.h>

#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/InformativeWaitEngine.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

// Opt-in (IW_DUR_PROBE=<key file>, "onset offset pitch velocity" per line in cs; IW_DUR_PROBE_FROM / _SECONDS
// pick the stretch): model 2 as Hybrid v1.1 runs it in Mono, on real keys, with the duration bias in 5 cs
// steps (tau' 0) and in coarse bins (tau' 0.3). Prints how much the orchestra's durations vary note to note.
TEST(InformativeWaitDurationProbe, fineVersusCoarseDurationBias) {
    const auto keyPath = juce::SystemStats::getEnvironmentVariable("IW_DUR_PROBE", "");
    if (keyPath.isEmpty())
        GTEST_SKIP() << "set IW_DUR_PROBE=<key file>";
    const int32_t from = juce::SystemStats::getEnvironmentVariable("IW_DUR_PROBE_FROM", "0").getIntValue();
    const int32_t seconds = juce::SystemStats::getEnvironmentVariable("IW_DUR_PROBE_SECONDS", "60").getIntValue();
    struct Key { int32_t on, off, pitch, vel; };
    std::vector<Key> keys;
    juce::StringArray lines;
    juce::File(keyPath).readLines(lines);
    for (const auto &line: lines) {
        juce::StringArray f;
        f.addTokens(line, " ", "");
        if (f.size() >= 4 && f[0].getIntValue() >= from && f[0].getIntValue() < from + seconds * 100)
            keys.push_back({f[0].getIntValue(), f[1].getIntValue(), f[2].getIntValue(), f[3].getIntValue()});
    }
    ASSERT_FALSE(keys.empty());

    StreamingKv::OnnxBackend backend;
    std::string error;
    ASSERT_TRUE(backend.init(modelsDir().getChildFile("rt_simultaneous_latest.onnx").getFullPathName().toRawUTF8(), error))
        << error;

    struct Variant { const char *name; bool coarse; float tauPrime; };
    for (const auto &v: {Variant{"5 cs steps, tau' 0  ", false, 0.0f}, Variant{"coarse bins, tau' 0.3", true, 0.3f}}) {
        for (const uint64_t seed: {11ull, 12ull, 13ull}) {
            InformativeWaitEngine engine;
            engine.setBackend(&backend);
            auto w = engine.settings();
            w.mode = InformativeWaitEngine::Mode::Unconstrained;
            w.precompute = true;
            w.seed = seed;
            w.rewriteHeldDurations = true;
            w.holdCommitCs = 1000;
            w.minDurationCs = 10;
            w.maxDurationCs = 500;
            w.maxChordNotes = 10;
            w.chordLateCs = 0;
            w.drawAheadCs = 1 << 20;
            w.noWaitAfterExpiry = true;
            w.fireWindowCs = 6;
            w.waitTopK = 3;
            w.openEndedWait = true;
            w.expiryGraceCs = 20;
            w.pitchClassBranches = true;
            w.maxFireNotes = 1; // Mono
            w.ascendingChords = true;
            w.maxAheadCs = 300;
            w.pitchLow = 61; // Mono: above C4
            w.pitchBias = true;
            w.durationBias = true;
            w.onsetGapBias = true;
            w.shortGap = true;
            w.coarseDurationBins = v.coarse;
            w.durationTauPrime = v.tauPrime;
            engine.setSettings(w);
            engine.reset(keys.front().on - 50);
            std::vector<std::pair<int32_t, int32_t>> notes; // (onset, duration) emitted
            std::multimap<int32_t, int32_t> offs;
            size_t next = 0;
            int32_t floorSet = INT32_MIN;
            for (int32_t now = keys.front().on - 50; now < keys.back().off + 100; ++now) {
                while (next < keys.size() && keys[next].on <= now) {
                    engine.noteOn(now, keys[next].pitch, keys[next].vel);
                    offs.emplace(keys[next].off, keys[next].pitch);
                    ++next;
                }
                while (! offs.empty() && offs.begin()->first <= now) {
                    engine.noteOff(now, offs.begin()->second, 64);
                    offs.erase(offs.begin());
                }
                // the Mono floor (PolyphonyCap, cap 1): no new onset before the sounding note ends
                int32_t floor = now;
                for (auto it = notes.rbegin(); it != notes.rend() && it - notes.rbegin() < 8; ++it)
                    if (it->first <= now && now < it->first + it->second)
                        floor = std::max(floor, it->first + it->second);
                if (floor != floorSet) {
                    auto s = engine.settings();
                    s.minOnsetCs = floor;
                    engine.setSettings(s);
                    floorSet = floor;
                }
                for (const auto &n: engine.step(now))
                    notes.emplace_back(n.onsetCs, n.durationCs);
            }
            std::sort(notes.begin(), notes.end());
            std::vector<int32_t> d;
            int same = 0, within10 = 0, sameBin = 0, pairs = 0;
            std::vector<int32_t> heard;
            for (size_t i = 0; i < notes.size(); ++i) {
                d.push_back(notes[i].second);
                if (i + 1 < notes.size() && notes[i + 1].first > notes[i].first)
                    heard.push_back(std::min(notes[i].second, notes[i + 1].first - notes[i].first));
                if (i > 0) {
                    const int32_t a = notes[i - 1].second, b = notes[i].second;
                    ++pairs;
                    same += a == b ? 1 : 0;
                    within10 += std::abs(a - b) <= 10 ? 1 : 0;
                    sameBin += DenseDurationBias::coarseBin(a) == DenseDurationBias::coarseBin(b) ? 1 : 0;
                }
            }
            std::sort(d.begin(), d.end());
            std::sort(heard.begin(), heard.end());
            const auto pct = [](const std::vector<int32_t> &x, double q) { return x.empty() ? 0 : x[static_cast<size_t>(q * (x.size() - 1))]; };
            std::printf("%s seed %llu: %zu notes | same as last %.0f%%, within 10 cs %.0f%%, same bin %.0f%% | "
                        "drawn p10/50/90 %d/%d/%d cs, heard %d/%d/%d cs\n",
                        v.name, static_cast<unsigned long long>(seed), notes.size(), 100.0 * same / std::max(pairs, 1),
                        100.0 * within10 / std::max(pairs, 1), 100.0 * sameBin / std::max(pairs, 1), pct(d, 0.1),
                        pct(d, 0.5), pct(d, 0.9), pct(heard, 0.1), pct(heard, 0.5), pct(heard, 0.9));
        }
    }
}

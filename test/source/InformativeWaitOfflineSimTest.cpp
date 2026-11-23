#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>

#include <VirtualOrch/reduction/InformativeWaitEngine.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>

/**
 * Offline simulation of the informative-WAIT engine on the MAESTRO fixtures (input.mid, first 15 s),
 * with the real ONNX export. Time is simulated but honest: every step() costs the wall time it took
 * (at least 1 cs per step), so a fire that has to be decoded on the spot is late by its decode time.
 * Writes, per mode and fixture, generated.mid (piano + orchestra) and summary.json (engine stats,
 * fire latency, top-5 hit rate). Runs only when IW_FIXTURES (fixtures dir) and IW_OUT are set:
 *
 *   IW_FIXTURES=~/orchestration/samples/maestro_fixtures_10 IW_OUT=/tmp/iw_sim \
 *   [IW_MODEL=...onnx] [IW_MODES=unconstrained,monophonic_A,monophonic_B] [IW_SECONDS=15] \
 *   ./VirtualOrchTest --gtest_filter='InformativeWaitOfflineSim.*'
 */
namespace {

using Engine = InformativeWaitEngine;

/**
 * Forwards to the ONNX backend and charges each call to a virtual clock: msPerCall for a 1-token
 * batch-1 call (longer feeds: + msPerCall / 16 per extra token), msPerBatched for a batch-5 call.
 * Calibrated to Joel's laptop (40 ms per batch-1 step) and the cluster's measured batch-5 / batch-1
 * ratio (262 / 107 ms = 2.45), so the simulated latency does not depend on this machine's load.
 */
class CostedBackend : public StreamingKv::Backend {
public:
    explicit CostedBackend(StreamingKv::Backend &innerIn) : inner(innerIn) {}
    [[nodiscard]] auto dims() const -> StreamingKv::Dims override { return inner.dims(); }
    auto run(const StreamingKv::State &past, const std::vector<int64_t> &ids, StreamingKv::State &present,
             std::vector<float> &last) -> bool override {
        if (verbose)
            std::cout << "      run n=" << ids.size() << " past rows " << past.rows << std::endl;
        virtualMs += msPerCall * (1.0 + static_cast<double>(ids.size() - 1) / 16.0);
        ++calls;
        return inner.run(past, ids, present, last);
    }
    auto runBatched(const StreamingKv::BatchState &past, const std::vector<int64_t> &ids,
                    StreamingKv::BatchState &present, std::vector<float> &logits) -> bool override {
        virtualMs += msPerBatched;
        ++batchedCalls;
        return inner.runBatched(past, ids, present, logits);
    }
    StreamingKv::Backend &inner;
    double msPerCall = 40.0;
    double msPerBatched = 98.0;
    double virtualMs = 0;
    long calls = 0;
    long batchedCalls = 0;
    bool verbose = false;
};

struct PianoNote {
    int32_t onset;
    int32_t duration;
    int32_t pitch;
    int32_t velocity;
};

auto loadPiano(const juce::File &mid, int32_t endCs) -> std::vector<PianoNote> {
    juce::FileInputStream in(mid);
    juce::MidiFile file;
    if (! in.openedOk() || ! file.readFrom(in))
        return {};
    file.convertTimestampTicksToSeconds();
    std::vector<PianoNote> out;
    for (int t = 0; t < file.getNumTracks(); ++t) {
        juce::MidiMessageSequence seq(*file.getTrack(t));
        seq.updateMatchedPairs();
        for (int i = 0; i < seq.getNumEvents(); ++i) {
            const auto *e = seq.getEventPointer(i);
            if (! e->message.isNoteOn())
                continue;
            const auto on = static_cast<int32_t>(std::lround(e->message.getTimeStamp() * 100.0));
            const double offT = e->noteOffObject != nullptr ? e->noteOffObject->message.getTimeStamp()
                                                            : e->message.getTimeStamp() + 0.5;
            const auto off = static_cast<int32_t>(std::lround(offT * 100.0));
            if (on < endCs)
                out.push_back({on, std::max(1, off - on), e->message.getNoteNumber(), e->message.getVelocity()});
        }
    }
    std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.onset < b.onset; });
    return out;
}

auto writeMidi(const juce::File &path, const std::vector<PianoNote> &piano,
               const std::vector<std::pair<Engine::OutNote, int32_t>> &orch) -> void {
    juce::MidiFile file;
    file.setTicksPerQuarterNote(480); // 120 bpm: 1 s = 960 ticks
    const auto ticks = [](int32_t cs) { return cs * 9.6; };
    juce::MidiMessageSequence p, o;
    p.addEvent(juce::MidiMessage::programChange(1, 0), 0);
    for (const auto &n: piano) {
        p.addEvent(juce::MidiMessage::noteOn(1, n.pitch, static_cast<juce::uint8>(n.velocity)), ticks(n.onset));
        p.addEvent(juce::MidiMessage::noteOff(1, n.pitch), ticks(n.onset + n.duration));
    }
    o.addEvent(juce::MidiMessage::programChange(2, 48), 0);
    for (const auto &[n, playedAt]: orch) {
        const int32_t start = std::max(n.onsetCs, playedAt); // a late note sounds when it was produced
        o.addEvent(juce::MidiMessage::noteOn(2, n.pitch, static_cast<juce::uint8>(std::max(1, n.velocity))), ticks(start));
        o.addEvent(juce::MidiMessage::noteOff(2, n.pitch), ticks(start + std::max(1, n.durationCs)));
    }
    p.updateMatchedPairs();
    o.sort();
    o.updateMatchedPairs();
    file.addTrack(p);
    file.addTrack(o);
    path.deleteFile();
    juce::FileOutputStream out(path);
    file.writeTo(out);
}

auto envOr(const char *name, const juce::String &fallback) -> juce::String {
    const char *v = std::getenv(name);
    return v != nullptr ? juce::String(v) : fallback;
}

TEST(InformativeWaitOfflineSim, Fixtures) {
    const auto fixtures = envOr("IW_FIXTURES", {});
    const auto outRoot = envOr("IW_OUT", {});
    if (fixtures.isEmpty() || outRoot.isEmpty())
        GTEST_SKIP() << "set IW_FIXTURES and IW_OUT";
    const auto model = envOr("IW_MODEL", juce::File::getSpecialLocation(juce::File::userHomeDirectory)
                                             .getChildFile("Documents/virtual_orch_models/simultaneous_model_v3.onnx")
                                             .getFullPathName());
    const int32_t endCs = envOr("IW_SECONDS", "15").getIntValue() * 100;
    const bool precompute = envOr("IW_PRECOMPUTE", "1") != "0";
    StreamingKv::OnnxBackend onnx;
    std::string err;
    ASSERT_TRUE(onnx.init(model.toRawUTF8(), err)) << err;
    CostedBackend backend(onnx);
    backend.msPerCall = envOr("IW_MS_PER_CALL", "40").getDoubleValue();
    backend.msPerBatched = envOr("IW_MS_PER_BATCHED", "98").getDoubleValue();
    const int only = envOr("IW_ONLY", "-1").getIntValue();
    backend.verbose = envOr("IW_TRACE", "0") == "1";

    juce::StringArray modes;
    modes.addTokens(envOr("IW_MODES", "unconstrained,monophonic_A,monophonic_B"), ",", "");
    const std::map<juce::String, Engine::Mode> modeOf{{"unconstrained", Engine::Mode::Unconstrained},
                                                      {"monophonic_A", Engine::Mode::LineLegato},
                                                      {"monophonic_B", Engine::Mode::LineRests}};
    auto dirs = juce::File(fixtures).findChildFiles(juce::File::findDirectories, false, "sample_*");
    dirs.sort();
    for (const auto &modeName: modes) {
        auto *all = new juce::DynamicObject();
        juce::Array<juce::var> perFixture;
        Engine::Stats total;
        double lateSum = 0;
        int lateN = 0;
        double lateAllSum = 0;
        int lateAllN = 0;
        int lateMax = 0;
        for (int fi = 0; fi < dirs.size(); ++fi) {
            if (only >= 0 && fi != only)
                continue;
            const auto wall0 = std::chrono::steady_clock::now();
            const long calls0 = backend.calls;
            const long batched0 = backend.batchedCalls;
            const auto piano = loadPiano(dirs[fi].getChildFile("input.mid"), endCs);
            if (piano.empty())
                continue;
            Engine engine;
            engine.setBackend(&backend);
            Engine::Settings s;
            s.mode = modeOf.at(modeName);
            s.seed = static_cast<uint64_t>(fi);
            s.precompute = precompute;
            engine.setSettings(s);
            std::vector<int32_t> fireLate;      // first stacked note
            std::vector<int32_t> fireLateAll;   // last stacked note
            int32_t lastFireOnset = INT32_MIN;
            int32_t openFire = INT32_MIN;       // onset of the fire whose stack is still coming
            int32_t openLeft = 0;
            bool firstSeen = false;
            const bool trace = envOr("IW_TRACE", "0") == "1";
            int32_t traceNow = 0;
            engine.onEvent = [&](const Engine::Event &e) {
                if (e.what == "fire") {
                    lastFireOnset = e.atCs;
                    openFire = e.atCs;
                    openLeft = e.a; // k
                    firstSeen = false;
                }
                if (trace)
                    std::cout << "  t=" << traceNow << " " << e.what << " @" << e.atCs << " " << e.a << " " << e.b << " "
                              << e.text << std::endl;
            };
            engine.reset(0);
            std::vector<std::pair<Engine::OutNote, int32_t>> orch;
            std::vector<std::pair<int32_t, size_t>> offs; // (time, piano index)
            size_t next = 0;
            int32_t now = 0;
            const auto stopAt = endCs + 300;
            while (now < stopAt) {
                while (next < piano.size() && piano[next].onset <= now) {
                    engine.noteOn(piano[next].onset, piano[next].pitch, piano[next].velocity);
                    offs.emplace_back(piano[next].onset + piano[next].duration, next);
                    ++next;
                }
                for (auto it = offs.begin(); it != offs.end();) {
                    if (it->first <= now) {
                        const auto &n = piano[it->second];
                        engine.noteOff(n.onset, n.pitch, n.duration);
                        it = offs.erase(it);
                    } else {
                        ++it;
                    }
                }
                lastFireOnset = INT32_MIN;
                backend.virtualMs = 0;
                traceNow = now;
                const long c0 = backend.calls;
                const long b0 = backend.batchedCalls;
                const auto out = engine.step(now);
                const double ms = backend.virtualMs;
                if (trace && ms > 0)
                    std::cout << "    step t=" << now << " cost " << ms << " ms (" << backend.calls - c0 << " calls, "
                              << backend.batchedCalls - b0 << " batched) out " << out.size() << std::endl;
                const auto cost = std::max<int32_t>(1, static_cast<int32_t>(std::ceil(ms / 10.0)));
                const int32_t producedAt = now + cost - 1; // the notes exist once the step has run
                const auto cancel = engine.takeCancelFrom();
                if (cancel != INT32_MAX)
                    orch.erase(std::remove_if(orch.begin(), orch.end(), [&](const auto &p) { return p.first.onsetCs >= cancel; }),
                               orch.end());
                for (const auto &n: out)
                    orch.emplace_back(n, producedAt);
                for (const auto &n: out)
                    if (openFire != INT32_MIN && n.onsetCs == openFire && openLeft > 0) {
                        if (! firstSeen) {
                            fireLate.push_back(producedAt - openFire);
                            firstSeen = true;
                        }
                        if (--openLeft == 0) {
                            fireLateAll.push_back(producedAt - openFire);
                            openFire = INT32_MIN;
                        }
                    }
                now += cost;
            }
            const auto dir = juce::File(outRoot).getChildFile(modeName).getChildFile(dirs[fi].getFileName());
            dir.createDirectory();
            writeMidi(dir.getChildFile("generated.mid"), piano, orch);
            juce::String toks;
            for (const auto t: engine.tokens())
                toks << t << " ";
            dir.getChildFile("tokens.txt").replaceWithText(toks.trimEnd() + "\n");

            const auto &stt = engine.stats();
            auto *o = new juce::DynamicObject();
            o->setProperty("fixture", dirs[fi].getFileName());
            o->setProperty("piano_notes", static_cast<int>(piano.size()));
            o->setProperty("orchestra_notes", static_cast<int>(orch.size()));
            o->setProperty("waits_drawn", stt.waitsDrawn);
            o->setProperty("waits_ge_10cs", stt.waitsGe10);
            o->setProperty("waits_provisional", stt.waitsProvisional);
            o->setProperty("fires", stt.fires);
            o->setProperty("expired", stt.expired);
            o->setProperty("removed", stt.removed);
            o->setProperty("early_piano", stt.earlyPiano);
            o->setProperty("cut_notes", stt.cutNotes);
            o->setProperty("anticipated", stt.anticipated);
            o->setProperty("antic_top1", stt.antTop1);
            o->setProperty("antic_top5", stt.antTop5);
            o->setProperty("precompute_started", stt.precomputeStarted);
            o->setProperty("precompute_completed", stt.precomputeCompleted);
            o->setProperty("precompute_hits", stt.precomputeHits);
            o->setProperty("fires_decoded", stt.fireDecoded);
            o->setProperty("mean_precompute_ms", stt.precomputeCompleted > 0 ? stt.precomputeMsTotal / stt.precomputeCompleted : 0.0);
            o->setProperty("mean_fire_decode_ms", stt.fireDecoded > 0 ? stt.fireDecodeMsTotal / stt.fireDecoded : 0.0);
            juce::Array<juce::var> lates;
            for (const auto l: fireLate) {
                lates.add(l);
                lateSum += l;
                ++lateN;
                lateMax = std::max(lateMax, l);
            }
            o->setProperty("fire_latency_cs", lates);
            juce::Array<juce::var> latesAll;
            for (const auto l: fireLateAll) {
                latesAll.add(l);
                lateAllSum += l;
                ++lateAllN;
            }
            o->setProperty("fire_latency_all_notes_cs", latesAll);
            o->setProperty("context_rebuilds", stt.contextRebuilds);
            o->setProperty("lag_skips", stt.lagSkips);
            o->setProperty("model_calls", static_cast<int>(backend.calls - calls0));
            o->setProperty("batched_calls", static_cast<int>(backend.batchedCalls - batched0));
            o->setProperty("wall_seconds_this_machine",
                           std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count());
            perFixture.add(juce::var(o));
            dir.getChildFile("summary.json").replaceWithText(juce::JSON::toString(juce::var(o)));
            std::cout << "[iw-sim] " << modeName << " " << dirs[fi].getFileName() << ": " << orch.size()
                      << " notes, fires " << stt.fires << ", hits " << stt.precomputeHits << ", top5 " << stt.antTop5
                      << "/" << stt.anticipated << ", expired " << stt.expired << ", removed " << stt.removed << std::endl;
            total.fires += stt.fires;
            total.precomputeHits += stt.precomputeHits;
            total.anticipated += stt.anticipated;
            total.antTop1 += stt.antTop1;
            total.antTop5 += stt.antTop5;
            total.waitsDrawn += stt.waitsDrawn;
            total.waitsGe10 += stt.waitsGe10;
            total.expired += stt.expired;
            total.removed += stt.removed;
            total.orchestraNotes += static_cast<int>(orch.size());
            total.fireDecoded += stt.fireDecoded;
            total.precomputePartial += stt.precomputePartial;
            total.precomputeCompleted += stt.precomputeCompleted;
            total.precomputeMsTotal += stt.precomputeMsTotal;
            total.fireDecodeMsTotal += stt.fireDecodeMsTotal;
        }
        all->setProperty("mode", modeName);
        all->setProperty("model", model);
        all->setProperty("precompute", precompute);
        all->setProperty("ms_per_call", backend.msPerCall);
        all->setProperty("ms_per_batched_call", backend.msPerBatched);
        all->setProperty("fixtures", perFixture);
        all->setProperty("fires", total.fires);
        all->setProperty("precompute_hit_rate", total.fires > 0 ? static_cast<double>(total.precomputeHits) / total.fires : 0.0);
        all->setProperty("antic_top1", total.anticipated > 0 ? static_cast<double>(total.antTop1) / total.anticipated : 0.0);
        all->setProperty("antic_top5", total.anticipated > 0 ? static_cast<double>(total.antTop5) / total.anticipated : 0.0);
        all->setProperty("waits_drawn", total.waitsDrawn);
        all->setProperty("waits_ge_10cs_share", total.waitsDrawn > 0 ? static_cast<double>(total.waitsGe10) / total.waitsDrawn : 0.0);
        all->setProperty("mean_fire_latency_cs", lateN > 0 ? lateSum / lateN : 0.0);
        all->setProperty("max_fire_latency_cs", lateMax);
        all->setProperty("mean_fire_latency_all_notes_cs", lateAllN > 0 ? lateAllSum / lateAllN : 0.0);
        all->setProperty("precompute_partial", total.precomputePartial);
        all->setProperty("mean_precompute_ms", total.precomputeCompleted > 0 ? total.precomputeMsTotal / total.precomputeCompleted : 0.0);
        all->setProperty("mean_fire_decode_ms", total.fireDecoded > 0 ? total.fireDecodeMsTotal / total.fireDecoded : 0.0);
        juce::File(outRoot).getChildFile(modeName).getChildFile("summary.json").replaceWithText(juce::JSON::toString(juce::var(all)));
        std::cout << "[iw-sim] " << modeName << " TOTAL fires " << total.fires << ", precompute hits "
                  << total.precomputeHits << ", top1 " << total.antTop1 << ", top5 " << total.antTop5 << "/"
                  << total.anticipated << ", mean fire latency " << (lateN > 0 ? lateSum / lateN : 0.0) << " cs" << std::endl;
    }
}

} // namespace

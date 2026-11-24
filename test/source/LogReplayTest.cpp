#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <string>

#include <VirtualOrch/sim/LogReplay.h>

namespace virtual_orch_test {

namespace {

auto fixture(const char *name) -> juce::File {
    return juce::File(VIRTUAL_ORCH_TEST_RESOURCES).getChildFile("log_replay").getChildFile(name);
}

auto runReplay(const char *name) -> LogReplay::Report {
    const auto log = LogReplay::readLog(fixture(name));
    EXPECT_FALSE(log.empty()) << name;
    LogReplay::Options options;
    options.preset = LogReplay::hybridV1W3Preset();
    // LOGREPLAY_KEEP=n: list up to n differences per tag (default 12).
    if (const auto keep = juce::SystemStats::getEnvironmentVariable("LOGREPLAY_KEEP", ""); keep.isNotEmpty())
        options.keepDifferences = keep.getIntValue();
    auto report = LogReplay::replay(log, options);
    std::cout << "---- " << name << "\n" << report.summary();
    // LOGREPLAY_DUMP="from,to" (ms): both logs side by side in that window, for debugging.
    const auto dump = juce::SystemStats::getEnvironmentVariable("LOGREPLAY_DUMP", "");
    if (dump.isNotEmpty()) {
        const double from = dump.upToFirstOccurrenceOf(",", false, false).getDoubleValue();
        const double to = dump.fromFirstOccurrenceOf(",", false, false).getDoubleValue();
        std::vector<std::pair<double, juce::String>> rows;
        for (const auto &l: log)
            if (l.ms >= from && l.ms <= to && l.tag != "MIDI_IN")
                rows.push_back({l.ms, "REC " + juce::String(l.ms, 1) + " c" + juce::String(l.clock) + " " + l.tag + " " + l.text.substring(0, juce::SystemStats::getEnvironmentVariable("LOGREPLAY_FULL", "").isNotEmpty() ? 100000 : 150)});
        for (const auto &l: report.replayed)
            if (l.ms >= from && l.ms <= to && l.tag != "MIDI_IN")
                rows.push_back({l.ms, "        SIM " + juce::String(l.ms, 1) + " c" + juce::String(l.clock) + " " + l.tag + " " + l.text.substring(0, juce::SystemStats::getEnvironmentVariable("LOGREPLAY_FULL", "").isNotEmpty() ? 100000 : 150)});
        for (const auto &l: log)
            if (l.ms >= from && l.ms <= to && l.tag == "MIDI_IN")
                rows.push_back({l.ms, "IN  " + juce::String(l.ms, 1) + " " + l.text.fromFirstOccurrenceOf("(", false, false).substring(0, 60)});
        std::stable_sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (const auto &r: rows)
            std::cout << r.second << "\n";
    }
    for (const auto &[tag, r]: report.tags) {
        if (r.exact())
            continue;
        for (const auto &m: r.missing)
            std::cout << "  " << tag << " missing: " << m << "\n";
        for (const auto &x: r.extra)
            std::cout << "  " << tag << " extra:   " << x << "\n";
    }
    return report;
}

} // namespace

TEST(LogReplay, parsesTheSessionLogFormat) {
    const auto lines = LogReplay::parseLog(
        "0.5 | clock - | SESSION | start 22 Oct 2026 1:20:27\n"
        "4011.8 | clock 0 | MIDI_IN | Keystation 88 MK3(USB MIDI) [piano2,pedal] | B0 40 00 (Controller Hold Pedal (on/off): 0 Channel 1)\n"
        "41280.6 | clock 12 | RT | Reduction Transformer Hybrid | (0, Clear Queue) voice -1 noteId -1 | TD 4\n"
        "not a log line\n");
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0].clock, -1);
    EXPECT_EQ(lines[1].tag, "MIDI_IN");
    EXPECT_NEAR(lines[1].ms, 4011.8, 1e-9);
    EXPECT_EQ(lines[2].clock, 12);
    EXPECT_TRUE(lines[2].text.startsWith("Reduction Transformer Hybrid | (0, Clear Queue)"));
}

TEST(LogReplay, presetOfTakesTheLoggedPresetElseTheFallbackWithTheLoggedStrength) {
    const auto rt = juce::String("200.0 | clock 1 | RT | Reduction Transformer Hybrid | (1, 10, 0 - 60, vel 64) voice 0 "
                                 "noteId 0 | TD 4 LogitTD on s 6.0\n");
    // A logged preset wins, its own strength too.
    const auto logged = LogReplay::parseLog(
        "100.0 | clock 0 | UI | Preset applied: {\"directInputWindowLength\": 10, \"logitAdjustments\": "
        "{\"logitTdStrength\": 3.0}}\n" + rt
        + "300.0 | clock 2 | UI | Preset applied: {\"directInputWindowLength\": 20}\n");
    const auto a = LogReplay::presetOf(logged);
    EXPECT_EQ(static_cast<int>(a["directInputWindowLength"]), 10);
    EXPECT_DOUBLE_EQ(static_cast<double>(a["logitAdjustments"]["logitTdStrength"]), 3.0);
    // No preset line: the fallback, with the strength the RT lines show; the fallback is not changed.
    auto fallback = juce::JSON::parse("{\"directInputWindowLength\": 10}");
    const auto b = LogReplay::presetOf(LogReplay::parseLog(rt), fallback);
    EXPECT_EQ(static_cast<int>(b["directInputWindowLength"]), 10);
    EXPECT_DOUBLE_EQ(static_cast<double>(b["logitAdjustments"]["logitTdStrength"]), 6.0);
    EXPECT_FALSE(fallback.hasProperty("logitAdjustments"));
    // ... also when the fallback has a strength of its own (presets of Nov 2026 did not save it).
    const auto d = LogReplay::presetOf(LogReplay::parseLog(rt), LogReplay::hybridV1W3Preset());
    EXPECT_DOUBLE_EQ(static_cast<double>(d["logitAdjustments"]["logitTdStrength"]), 6.0);
    // Nothing at all: the Hybrid default.
    const auto c = LogReplay::presetOf({});
    EXPECT_EQ(c["model"].toString(), "hybrid_v1_w3");
}

/**
 * One recorded session and what its replay must reach. A tag's match rate is matched lines /
 * max(recorded, replayed) (see LogReplay::TagResult). Tags not listed must match exactly.
 *
 * The minimums sit just under what the replay reached when the test was written (21 Nov 2026);
 * the gap to 100% is real-world timing the log cannot pin down (docs/LOG_REPLAY.md, "Accepted
 * differences"). A change that makes the replay worse fails here; one that makes it better can
 * raise them.
 */
struct ReplayCase {
    const char *file;
    std::map<std::string, double> minimum;
};

auto operator<<(std::ostream &os, const ReplayCase &c) -> std::ostream & { return os << c.file; }

auto matchRate(const LogReplay::TagResult &r) -> double {
    const int denominator = std::max(r.recorded, r.replayed);
    return denominator == 0 ? 1.0 : static_cast<double>(r.matched) / static_cast<double>(denominator);
}

// Every Oct 22 afternoon / evening session (main @ 9563268 was committed at 14:54).
class LogReplaySessions : public ::testing::TestWithParam<ReplayCase> {};

TEST_P(LogReplaySessions, reproducesTheRecordedRun) {
    auto gui = juce::ScopedJuceInitialiser_GUI{};
    const auto &c = GetParam();
    const auto report = runReplay(c.file);
    EXPECT_GT(report.generationRuns, 0);
    // The replay never lost track of the recorded run: every forced draw had a logged value and
    // simulated time always moved on.
    for (const auto &p: report.problems) {
        EXPECT_FALSE(p.contains("drew a note")) << p;
        EXPECT_FALSE(p.contains("never used")) << p;
        EXPECT_FALSE(p.contains("stalled")) << p;
    }
    ASSERT_FALSE(report.tags.empty());
    for (const auto &[tag, r]: report.tags) {
        const auto it = c.minimum.find(tag.toStdString());
        if (it == c.minimum.end()) {
            EXPECT_TRUE(r.exact()) << tag << ": recorded " << r.recorded << ", replayed " << r.replayed
                                   << ", matched " << r.matched;
        } else {
            EXPECT_GE(matchRate(r), it->second) << tag << ": recorded " << r.recorded << ", replayed "
                                                << r.replayed << ", matched " << r.matched;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Oct22, LogReplaySessions,
    ::testing::Values(
        // 3 min: every tag exact.
        ReplayCase{"session_2026-10-22_14-51-28.log.gz", {}},
        // 26 min, Edit / Jam / Accompaniment, dense playing.
        ReplayCase{"session_2026-10-22_14-25-31.log.gz",
                   {{"MIDI_OUT", 0.985}, {"OT", 0.94}, {"RT", 0.998}, {"RT_IN", 0.91}, {"WAIT", 0.985}}},
        // 3.6 h, mostly idle.
        ReplayCase{"session_2026-10-22_14-54-26.log.gz",
                   {{"MIDI_OUT", 0.999}, {"OT", 0.97}, {"RT", 0.995}, {"RT_IN", 0.85}, {"WAIT", 0.94},
                    {"HYBRID", 0.91}}},
        // 57 min.
        ReplayCase{"session_2026-10-22_18-30-05.log.gz",
                   {{"MIDI_OUT", 0.99}, {"OT", 0.95}, {"RT", 0.998}, {"RT_IN", 0.92}, {"CUT", 0.98},
                    {"HYBRID", 0.92}}}));

} // namespace virtual_orch_test

#include <gtest/gtest.h>

#include <iostream>

#include <VirtualOrch/ProjectPaths.h>
#include <VirtualOrch/sim/LogReplay.h>
#include <VirtualOrch/sim/OpLogReader.h>
#include <VirtualOrch/sim/Rerun.h>

namespace virtual_orch_test {

namespace {

auto fixture(const char *name) -> juce::File {
    return juce::File(VIRTUAL_ORCH_TEST_RESOURCES).getChildFile("log_replay").getChildFile(name);
}

} // namespace

TEST(Rerun, reproducesTheLoggedProbabilitiesWithTheSameModels) {
    const auto v1 = Rerun::defaultModelFile("v1");
    const auto w = Rerun::defaultModelFile("W");
    const auto ot = Rerun::defaultModelFile("OT");
    if (! v1.existsAsFile() || ! w.existsAsFile() || ! ot.existsAsFile())
        GTEST_SKIP() << "the models are not in " << modelsDir().getFullPathName();
    const juce::ScopedJuceInitialiser_GUI gui;

    // An oplog with the real models' probabilities: replay a recorded session with them.
    const auto file = juce::File::createTempFile(".jsonl.gz");
    {
        const auto lines = LogReplay::readLog(fixture("session_2026-10-22_14-51-28.log.gz"));
        ASSERT_FALSE(lines.empty());
        LogReplay::Options options;
        options.preset = LogReplay::hybridV1W3Preset();
        options.rtSilentModel = v1;
        options.rtWaitModel = w;
        options.otModel = ot;
        options.oplog = file;
        options.untilMs = lines.front().ms + 60000.0; // the first minute is plenty
        (void) LogReplay::replay(lines, options);
    }
    const auto log = OpLogReader::Log::load(file);
    file.deleteFile();
    ASSERT_TRUE(log->error.isEmpty()) << log->error;

    Rerun::Engine engine;
    int rtTried = 0, rtMatched = 0, otTried = 0, otMatched = 0;
    for (size_t i = 0; i < log->records().size() && (rtTried < 12 || otTried < 12); ++i) {
        const auto &op = log->records()[i].op;
        const bool isRt = op == "RT_DRAW" && rtTried < 12;
        const bool isOt = op == "OT_DRAW" && otTried < 12;
        if (! isRt && ! isOt)
            continue;
        const auto r = engine.rerun(*log, i);
        ASSERT_TRUE(r.error.empty()) << r.what << ": " << r.error;
        const bool ok = r.matchesLog();
        if (! ok) {
            std::cout << r.what << " context " << r.contextTokens << " / logged " << r.loggedContextTokens << "\n";
            for (const auto &f: r.fields)
                std::cout << "   " << f.name << " chosen " << f.chosen << " logged " << f.loggedPModel << " rerun "
                          << f.rerunPModel << "\n";
        }
        (isRt ? rtTried : otTried)++;
        (isRt ? rtMatched : otMatched) += ok ? 1 : 0;
    }
    std::cout << "RT reruns matching the log: " << rtMatched << " / " << rtTried << ", OT: " << otMatched << " / "
              << otTried << "\n";
    EXPECT_GT(rtTried, 0);
    EXPECT_GT(otTried, 0);
    EXPECT_EQ(rtMatched, rtTried);
    EXPECT_EQ(otMatched, otTried);

    // The reruns file: one JSON line per rerun, next to the oplog.
    const auto reruns = Rerun::rerunsFileFor(juce::File("/tmp/x/oplog_2026-11-22_10-00-00.jsonl.gz"));
    EXPECT_EQ(reruns.getFileName(), "oplog_2026-11-22_10-00-00.reruns.jsonl");
}

} // namespace virtual_orch_test

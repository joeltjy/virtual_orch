#include <gtest/gtest.h>

#include <iostream>

#include <VirtualOrch/OpLog.h>
#include <VirtualOrch/sim/LogReplay.h>
#include <VirtualOrch/sim/OpLogReader.h>

namespace virtual_orch_test {

namespace {

auto fixture(const char *name) -> juce::File {
    return juce::File(VIRTUAL_ORCH_TEST_RESOURCES).getChildFile("log_replay").getChildFile(name);
}

/** The oplog of a recorded session, written by replaying it (zero-logit models, choices forced). */
auto replayOplog(const char *name, const juce::File &out) -> void {
    const auto log = LogReplay::readLog(fixture(name));
    ASSERT_FALSE(log.empty());
    LogReplay::Options options;
    options.preset = LogReplay::hybridV1W3Preset();
    options.oplog = out;
    options.sourceName = name;
    (void) LogReplay::replay(log, options);
}

} // namespace

TEST(OpLogReader, readsHeadersAndSkipsACutOffLine) {
    const auto log = OpLogReader::Log::fromText(
        "{\"seq\":0,\"t\":0,\"clock\":-1,\"src\":\"ui\",\"op\":\"SESSION_START\",\"origin\":\"app\"}\n"
        "{\"seq\":1,\"t\":12.5,\"clock\":0,\"src\":\"ui\",\"op\":\"RUN_START\",\"run\":\"run0\"}\n"
        "{\"seq\":2,\"t\":40,\"clock\":28,\"src\":\"rt\",\"op\":\"Q_PUSH\",\"queue\":\"rt→play\",\"item\":\"r0\"}\n"
        "{\"seq\":3,\"t\":90,\"clock\":78,\"src\":\"ui\",\"op\":\"RUN_STOP\",\"run\":\"run0\",\"reason\":\"stop\"}\n"
        "{\"seq\":4,\"t\":95,\"clock\":78,\"src\":\"ui\",\"op\":\"RUN_START\",\"run\":\"run1\"}\n"
        "{\"seq\":5,\"t\":99,\"clock\":0,\"src\":\"midi\",\"op\":\"KEY_D");
    ASSERT_EQ(log->records().size(), 5u);
    EXPECT_EQ(log->skippedLines(), 1);
    EXPECT_EQ(log->records()[2].op, "Q_PUSH");
    EXPECT_DOUBLE_EQ(log->records()[1].t, 12.5);
    ASSERT_EQ(log->sessions().size(), 2u);
    EXPECT_EQ(log->sessions()[0].run, "run0");
    EXPECT_EQ(log->sessions()[0].stopReason, "stop");
    EXPECT_EQ(log->sessions()[1].run, "run1");
    EXPECT_EQ(log->sessions()[1].stopReason, "") << "no RUN_STOP: runs to the end";
    EXPECT_EQ(log->indexAt(41.0), 3u);
    const auto st = OpLogReader::stateAt(*log, 2);
    ASSERT_EQ(st.queues.at("rt→play").size(), 1u);
    EXPECT_EQ(st.queues.at("rt→play")[0].item, "r0");
    EXPECT_EQ(log->fields(2)["item"].toString(), "r0");
}

TEST(OpLogReader, rebuildsEveryCheckpointsContextFromTheOneBefore) {
    const juce::ScopedJuceInitialiser_GUI gui;
    const auto file = juce::File::createTempFile(".jsonl.gz");
    replayOplog("session_2026-10-22_14-51-28.log.gz", file);
    const auto log = OpLogReader::Log::load(file);
    file.deleteFile();
    ASSERT_TRUE(log->error.isEmpty()) << log->error;
    ASSERT_GT(log->records().size(), 100u);
    EXPECT_FALSE(log->sessions().empty());

    // At every rt checkpoint, the context rebuilt from the previous checkpoint and the changes
    // since must be exactly the checkpoint's tokens (unless a rebase came between).
    int compared = 0, mismatched = 0;
    const auto &recs = log->records();
    for (size_t i = 0; i < recs.size(); ++i) {
        if (recs[i].op != "CHECKPOINT" || log->line(i).find("\"part\":\"rt\"") == std::string::npos || i == 0)
            continue;
        const auto before = OpLogReader::stateAt(*log, i - 1);
        if (! before.rtCheckpointState.has_value() || OpLogReader::sessionOf(*log, i) != OpLogReader::sessionOf(*log, i - 1))
            continue;
        const auto f = log->fields(i);
        if (auto *models = f["rt"].getDynamicObject())
            for (const auto &prop: models->getProperties()) {
                const auto it = before.rt.find(prop.name.toString().toStdString());
                if (it == before.rt.end() || ! it->second.exact)
                    continue;
                std::vector<int> expected;
                if (const auto *a = prop.value["tokens"].getArray())
                    for (const auto &x: *a)
                        expected.push_back(static_cast<int>(x));
                ++compared;
                if (it->second.tokens() != expected)
                    ++mismatched;
            }
    }
    EXPECT_GT(compared, 5);
    EXPECT_EQ(mismatched, 0) << "of " << compared;
}


// OPLOG_FILE=<path to an oplog_*.jsonl.gz>: load a real recording and check its contexts against
// its own checkpoints (skipped otherwise).
TEST(OpLogReader, aRealRecordingIfGiven) {
    const auto path = juce::SystemStats::getEnvironmentVariable("OPLOG_FILE", "");
    if (path.isEmpty())
        GTEST_SKIP() << "set OPLOG_FILE";
    const auto started = juce::Time::getMillisecondCounterHiRes();
    const auto log = OpLogReader::Log::load(juce::File(path));
    const auto loadMs = juce::Time::getMillisecondCounterHiRes() - started;
    ASSERT_TRUE(log->error.isEmpty()) << log->error;
    std::cout << "records " << log->records().size() << ", sessions " << log->sessions().size() << ", skipped "
              << log->skippedLines() << ", loaded in " << loadMs << " ms\n";
    int compared = 0, mismatched = 0;
    const auto &recs = log->records();
    for (size_t i = 1; i < recs.size(); ++i) {
        if (recs[i].op != "CHECKPOINT" || log->line(i).find("\"part\":\"rt\"") == std::string::npos)
            continue;
        const auto before = OpLogReader::stateAt(*log, i - 1);
        if (! before.rtCheckpointState.has_value() || OpLogReader::sessionOf(*log, i) != OpLogReader::sessionOf(*log, i - 1))
            continue;
        const auto f = log->fields(i);
        if (auto *models = f["rt"].getDynamicObject())
            for (const auto &prop: models->getProperties()) {
                const auto it = before.rt.find(prop.name.toString().toStdString());
                if (it == before.rt.end() || ! it->second.exact)
                    continue;
                std::vector<int> expected;
                if (const auto *a = prop.value["tokens"].getArray())
                    for (const auto &x: *a)
                        expected.push_back(static_cast<int>(x));
                ++compared;
                if (it->second.tokens() != expected)
                    ++mismatched;
            }
    }
    std::cout << "checkpoints compared " << compared << ", mismatched " << mismatched << "\n";
    EXPECT_EQ(mismatched, 0);
}


TEST(OpLogReader, aLiveOplogKeepsEveryRecordAcrossWriterFlushes) {
    // The writer thread writes every 100 ms; all of it must be in the file (JUCE's gzip flush()
    // finishes the stream, which once dropped everything after the first batch).
    const auto file = juce::File::createTempFile(".jsonl.gz");
    OpLog::start(file);
    int emitted = 0;
    for (int batch = 0; batch < 4; ++batch) {
        for (int i = 0; i < 25; ++i, ++emitted)
            OpLog::emit(OpLog::Src::Ui, "PAD_ACTION", {{"action", "test " + std::to_string(emitted)}});
        juce::Thread::sleep(250);
    }
    OpLog::stop();
    const auto log = OpLogReader::Log::load(file);
    file.deleteFile();
    ASSERT_TRUE(log->error.isEmpty()) << log->error;
    EXPECT_EQ(static_cast<int>(log->records().size()), emitted + 1) << "every record and SESSION_END";
    EXPECT_EQ(log->records().back().op, "SESSION_END");
}

} // namespace virtual_orch_test

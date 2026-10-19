#include <gtest/gtest.h>

#include <set>
#include <vector>

#include <VirtualOrch/Clock.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/PedalClear.h>
#include <VirtualOrch/reduction/ReductionTransformer.h>

namespace {

auto onsetLess(const OrchestrationNote &a, const OrchestrationNote &b) -> bool {
    if (a.token.time != b.token.time)
        return a.token.time < b.token.time;
    return a.token.note < b.token.note;
}

auto endLess(const OrchestrationNote &a, const OrchestrationNote &b) -> bool {
    const auto aEnd = a.token.time + a.token.getRealDuration();
    const auto bEnd = b.token.time + b.token.getRealDuration();
    if (aEnd != bEnd)
        return aEnd < bEnd;
    return onsetLess(a, b);
}

using QueuedSet = std::multiset<OrchestrationNote, decltype(&onsetLess)>;
using PlayingSet = std::multiset<OrchestrationNote, decltype(&endLess)>;

auto note(int32_t onset, int32_t durationCs, int32_t pitch = 60) -> OrchestrationNote {
    OrchestrationNote n;
    n.token = Token{onset, static_cast<int32_t>(Vocab::DurOffset) + durationCs,
                    static_cast<int32_t>(Vocab::NoteOffset) + pitch};
    n.velocity = 90;
    n.localInstrumentId = 0;
    return n;
}

auto endOf(const OrchestrationNote &n) -> int32_t {
    return n.token.time + n.token.getRealDuration();
}

/** Reduction transformer stub that exposes the soft-stop emitter. */
class StubReduction : public ReductionTransformer {
public:
    explicit StubReduction(ModelConfig &config) : ReductionTransformer("stub", config) {}
    [[nodiscard]] auto isModelLoaded() const -> bool override { return true; }
    void run() override {}
    using ReductionTransformer::maybeEmitGenerationPauseSoftStopClear;
};

} // namespace

namespace virtual_orch_test {

TEST(PedalClear, queuedNotesNeverSound) {
    QueuedSet queued(onsetLess);
    PlayingSet playing(endLess);
    queued.insert(note(1020, 100));
    queued.insert(note(1030, 100)); // would start exactly at pedal + 0.3 s: still dropped
    queued.insert(note(1500, 100));

    int noteOffs = 0;
    const bool changed =
        PedalClear::apply(queued, playing, 1000, 1000, [&](const OrchestrationNote &) { ++noteOffs; });

    EXPECT_TRUE(changed);
    EXPECT_TRUE(queued.empty());
    EXPECT_EQ(noteOffs, 0);
}

TEST(PedalClear, soundingNoteIsShortenedToPedalPlusGrace) {
    QueuedSet queued(onsetLess);
    PlayingSet playing(endLess);
    playing.insert(note(900, 300)); // sounding, would end at 1200

    int noteOffs = 0;
    PedalClear::apply(queued, playing, 1000, 1000, [&](const OrchestrationNote &) { ++noteOffs; });

    ASSERT_EQ(playing.size(), 1u);
    EXPECT_EQ(endOf(*playing.begin()), 1000 + PedalClear::kGraceCs);
    EXPECT_EQ(noteOffs, 0) << "the note keeps sounding until pedal + 0.3 s";
}

TEST(PedalClear, soundingNoteThatEndsEarlierIsLeftAlone) {
    QueuedSet queued(onsetLess);
    PlayingSet playing(endLess);
    playing.insert(note(900, 110)); // ends at 1010 < 1030

    PedalClear::apply(queued, playing, 1000, 1000, [](const OrchestrationNote &) {});

    ASSERT_EQ(playing.size(), 1u);
    EXPECT_EQ(endOf(*playing.begin()), 1010);
}

TEST(PedalClear, graceAlreadyPassedMeansImmediateNoteOff) {
    QueuedSet queued(onsetLess);
    PlayingSet playing(endLess);
    playing.insert(note(900, 500));

    std::vector<int32_t> offPitches;
    // The playback thread only sees the request 50 cs after the press: 1000 + 30 < 1050.
    PedalClear::apply(queued, playing, 1000, 1050,
                      [&](const OrchestrationNote &n) { offPitches.push_back(n.token.getPitch()); });

    EXPECT_TRUE(playing.empty());
    ASSERT_EQ(offPitches.size(), 1u);
    EXPECT_EQ(offPitches[0], 60);
}

TEST(PedalClear, nothingToDoReportsNoChange) {
    QueuedSet queued(onsetLess);
    PlayingSet playing(endLess);
    EXPECT_FALSE(PedalClear::apply(queued, playing, 1000, 1000, [](const OrchestrationNote &) {}));
}

TEST(PedalClear, dropWindowDropsLateNotesUntilClearArrivesOrExpires) {
    PedalClear::DropWindow drop;
    EXPECT_FALSE(drop.drops(5000, 0)) << "inactive window drops nothing";

    drop.start(1000, 1005);
    EXPECT_TRUE(drop.drops(1001, 1010)) << "starts after the press";
    EXPECT_FALSE(drop.drops(1000, 1010)) << "started at or before the press: already sounded";
    EXPECT_FALSE(drop.drops(900, 1010));

    drop.finish();
    EXPECT_FALSE(drop.drops(1500, 1020)) << "upstream clear arrived";

    drop.start(1000, 1005);
    EXPECT_FALSE(drop.drops(1500, 1005 + PedalClear::kDropWindowCs + 1)) << "expired";
    EXPECT_FALSE(drop.drops(1500, 1010)) << "stays off after expiry";
}

TEST(PedalClear, clearTokensGoLastInOutputOrder) {
    std::vector<OrchestrationNote> notes;
    notes.push_back(note(1200, 50, 62));
    OrchestrationNote clear;
    clear.token = PedalClear::makeClearToken(1000);
    notes.push_back(clear);
    notes.push_back(note(1100, 50, 61));
    OrchestrationNote ordinaryClear;
    ordinaryClear.token = Token{0, static_cast<int32_t>(Vocab::DurOffset),
                                static_cast<int32_t>(Vocab::ClearQueue)};
    notes.push_back(ordinaryClear);

    PedalClear::sortForOutput(notes);

    ASSERT_EQ(notes.size(), 4u);
    EXPECT_FALSE(PedalClear::isPedalClear(notes[0].token));
    EXPECT_EQ(notes[0].token.time, 0) << "ordinary clears keep their time order, in front";
    EXPECT_EQ(notes[1].token.time, 1100);
    EXPECT_EQ(notes[2].token.time, 1200);
    EXPECT_TRUE(PedalClear::isPedalClear(notes[3].token)) << "pedal clear is behind the notes";
}

TEST(PedalClear, reductionEmitsStampedClearOnEveryPress) {
    ModelConfig config;
    Metrics metrics;
    Clock clock(metrics);
    StubReduction rt(config);
    rt.clock = &clock;

    // Press 1: pause starts.
    rt.requestPedalClear(500);
    rt.generationPause.set(true);
    rt.maybeEmitGenerationPauseSoftStopClear();

    Token token{};
    ASSERT_TRUE(rt.outputTokenQueue.pull(token));
    EXPECT_TRUE(PedalClear::isPedalClear(token));
    EXPECT_EQ(token.time, 500) << "stamped with the press time, not clock + 1 s";
    EXPECT_FALSE(rt.outputTokenQueue.pull(token)) << "no second clear for the same press";

    // Press 2 while generation is still paused: the old edge trigger would send nothing.
    rt.requestPedalClear(900);
    rt.maybeEmitGenerationPauseSoftStopClear();
    ASSERT_TRUE(rt.outputTokenQueue.pull(token));
    EXPECT_TRUE(PedalClear::isPedalClear(token));
    EXPECT_EQ(token.time, 900);
}

TEST(PedalClear, manualPauseStillUsesOneSecondSoftStop) {
    ModelConfig config;
    Metrics metrics;
    Clock clock(metrics);
    StubReduction rt(config);
    rt.clock = &clock;

    rt.generationPause.set(true); // e.g. the Launchpad pause pad: no pedal request
    rt.maybeEmitGenerationPauseSoftStopClear();

    Token token{};
    ASSERT_TRUE(rt.outputTokenQueue.pull(token));
    EXPECT_EQ(token.note, static_cast<int32_t>(Vocab::ClearQueue));
    EXPECT_FALSE(PedalClear::isPedalClear(token));
    EXPECT_EQ(token.time, static_cast<int32_t>(clock.getTime()) + Config::TimeResolution);
}

} // namespace virtual_orch_test

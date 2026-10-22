#include <gtest/gtest.h>

#include <VirtualOrch/InputFilter.h>
#include <VirtualOrch/InstrumentConstants.h>
#include <VirtualOrch/PerformanceActivity.h>
#include <VirtualOrch/ReductionInputGate.h>

namespace virtual_orch_test {

namespace {

struct GateOff {
    GateOff() { ReductionInputGate::setBlocked(false); }
    ~GateOff() { ReductionInputGate::setBlocked(false); }
};

auto key(int32_t onset, int32_t pitch) -> Token {
    return Token{onset, static_cast<int32_t>(Vocab::DurOffset) + 40,
                 static_cast<int32_t>(Vocab::NoteOffset) + pitch, 80};
}

} // namespace

TEST(ReductionInputGate, blockedInputSkipsTheReductionButStillReachesTheOrchestration) {
    const GateOff guard;
    CircularFifo<Token> reduction;
    CircularFifo<Token> conditioning;
    CircularFifo<TokenUpdate> updates;
    CircularFifo<Token> orchestration;
    PassthroughInputFilter filter(reduction, conditioning, updates);
    filter.orchestrationMidiOutgoing = &orchestration;

    filter.filter(key(100, 60));
    ReductionInputGate::setBlocked(true);
    filter.filter(key(200, 64));
    ReductionInputGate::setBlocked(false);
    filter.filter(key(300, 67));

    std::vector<int32_t> heard;
    Token t{};
    while (reduction.pull(t))
        heard.push_back(t.time);
    EXPECT_EQ(heard, (std::vector<int32_t>{100, 300})) << "the reduction never hears the blocked note";

    std::vector<int32_t> played;
    while (orchestration.pull(t))
        played.push_back(t.time);
    EXPECT_EQ(played, (std::vector<int32_t>{100, 200, 300})) << "input instruments still play it";
}

TEST(ReductionInputGate, aBlockedNotesNoteOffStillFlowsSoNothingHangs) {
    const GateOff guard;
    CircularFifo<Token> reduction;
    CircularFifo<Token> conditioning;
    CircularFifo<TokenUpdate> updates;
    PassthroughInputFilter filter(reduction, conditioning, updates);

    ReductionInputGate::setBlocked(true);
    auto held = key(200, 64);
    filter.filter(held);
    auto released = held;
    released.duration = static_cast<int32_t>(Vocab::DurOffset) + 90;
    filter.updatesFromMain.push({held, released});
    EXPECT_EQ(filter.processUpdates(), 1) << "the update is still forwarded (OT needs it)";
    TokenUpdate u{};
    EXPECT_TRUE(updates.pull(u));
}

TEST(ReductionInputGate, theConsoleDescribesTheButton) {
    const auto name = [](int32_t) { return juce::String(); };
    using A = PerformanceActivity::LaunchpadAction;
    EXPECT_EQ(PerformanceActivity::describe(A{A::Kind::InputBlock, 0, 0, true, false}, name),
              juce::String("Input to the reduction blocked"));
    EXPECT_EQ(PerformanceActivity::describe(A{A::Kind::InputBlock, 0, 0, false, false}, name),
              juce::String("Input to the reduction open"));
}

} // namespace virtual_orch_test

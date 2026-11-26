// OutputPlaybackTest.cpp developed by Bob Tianqi Wei
#include <gtest/gtest.h>

#include <VirtualOrch/AppTime.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/OutputPlayback.h>

namespace {

class RecordingOutput : public OutputProcessor {
public:
    void send(NoteOnEvent) override { ++noteOns; }
    void send(NoteOffEvent) override { ++noteOffs; }

    int noteOns = 0;
    int noteOffs = 0;
};

TEST(OutputPlayback, ExpiredNoteCanPlayForItsOriginalDuration) {
    struct ResetTime {
        ~ResetTime() { AppTime::setSimulatedMs(-1); }
    } resetTime;
    AppTime::setSimulatedMs(1000);
    Metrics metrics;
    Clock clock(metrics);
    OrchestrationTransformer orchestra;
    std::unique_ptr<OutputProcessor> output = std::make_unique<RecordingOutput>();
    auto &recording = static_cast<RecordingOutput &>(*output);
    std::unique_ptr<OSCBufferOutputProcessor> buffer;
    int32_t bufferSize = 0;
    std::atomic source{PlaybackSource::Orchestration};
    OutputPlayback playback(clock, orchestra, output, buffer, bufferSize, source);
    playback.beginSteps();

    const OrchestrationNote note{
        .token = Token{10, static_cast<int32_t>(Vocab::DurOffset) + 20,
                       static_cast<int32_t>(Vocab::NoteOffset) + 60},
        .velocity = 100,
        .localInstrumentId = 6,
    };

    clock.startAtTime(40);
    orchestra.outputTokenQueue.push(note);
    playback.stepOnce();
    EXPECT_EQ(recording.noteOns, 1);
    EXPECT_EQ(recording.noteOffs, 1);

    playback.playLateNotes = true;
    orchestra.outputTokenQueue.push(note);
    playback.stepOnce();
    EXPECT_EQ(recording.noteOns, 2);
    EXPECT_EQ(recording.noteOffs, 1);

    clock.setTime(59);
    playback.stepOnce();
    EXPECT_EQ(recording.noteOffs, 1);

    clock.setTime(60);
    playback.stepOnce();
    EXPECT_EQ(recording.noteOffs, 2);
}

} // namespace

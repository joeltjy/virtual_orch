#include "VirtualOrch/TestOrchestrationTransformerThread.h"

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/OrchestrationTransformer.h"

TestOrchestrationTransformerThread::TestOrchestrationTransformerThread()
    : juce::Thread("TestOrchestrationTransformerThread") {
}

TestOrchestrationTransformerThread::~TestOrchestrationTransformerThread() {
    stop();
}

auto TestOrchestrationTransformerThread::start(Clock &clockRef,
                                               OrchestrationTransformer &orchestrationTransformerRef)
    -> void {
    stop();
    clock = &clockRef;
    orchestrationTransformer = &orchestrationTransformerRef;
    lastFiredBoundary = -1;
    startThread();
}

auto TestOrchestrationTransformerThread::stop() -> void {
    signalThreadShouldExit();
    stopThread(2000);
    clock = nullptr;
    orchestrationTransformer = nullptr;
}

void TestOrchestrationTransformerThread::run() {
    while (! threadShouldExit()) {
        if (clock != nullptr && orchestrationTransformer != nullptr && clock->isRunning()) {
            const auto time = static_cast<int32_t>(clock->getTime());
            const int32_t boundary = (time / 50) * 50;

            int32_t next = lastFiredBoundary < 0 ? 0 : lastFiredBoundary + 50;
            while (next <= boundary) {
                const int32_t n = next / 50;
                const int32_t encodedDuration = static_cast<int32_t>(Vocab::DurOffset + 50);
                const Token midiToken{.time = next,
                                      .duration = encodedDuration,
                                      .note = static_cast<int32_t>(Vocab::NoteOffset + (n % 8) + 48)};
                const Token reductionToken{.time = next,
                                           .duration = encodedDuration,
                                           .note = static_cast<int32_t>(Vocab::NoteOffset + (n % 8) + 60)};
                orchestrationTransformer->midiInputIncoming.push(midiToken);
                orchestrationTransformer->reductionIncoming.push(reductionToken);
                lastFiredBoundary = next;
                next += 50;
            }
        }

        wait(5);
    }
}

#include "VirtualOrch/SessionLog.h"

#include "VirtualOrch/Clock.h"

class SessionLog::Writer : public juce::Thread {
public:
    explicit Writer(const juce::File &fileIn) : juce::Thread("SessionLog"), file(fileIn) {
        file.getParentDirectory().createDirectory();
        stream = std::make_unique<juce::FileOutputStream>(file);
        if (stream->failedToOpen())
            stream.reset();
        startMs = juce::Time::getMillisecondCounterHiRes();
    }

    ~Writer() override {
        stopThread(2000);
        flush();
    }

    auto append(const char *kind, const juce::String &text) -> void {
        const double ms = juce::Time::getMillisecondCounterHiRes() - startMs;
        const auto *clock = SessionLog::clock();
        const juce::String clockText =
            clock != nullptr ? juce::String(static_cast<int64_t>(clock->getTime())) : juce::String("-");
        juce::String line = juce::String(ms, 1) + " | clock " + clockText + " | " + kind + " | "
                            + text + "\n";
        const std::lock_guard<std::mutex> lock(pendingMutex);
        pending.push_back(std::move(line));
    }

    auto run() -> void override {
        while (! threadShouldExit()) {
            wait(100);
            flush();
        }
    }

    auto flush() -> void {
        std::vector<juce::String> lines;
        {
            const std::lock_guard<std::mutex> lock(pendingMutex);
            lines.swap(pending);
        }
        if (stream == nullptr || lines.empty())
            return;
        for (const auto &line: lines)
            stream->writeText(line, false, false, nullptr);
        stream->flush();
    }

    const juce::File file;

private:
    std::unique_ptr<juce::FileOutputStream> stream;
    double startMs = 0.0;
    std::mutex pendingMutex;
    std::vector<juce::String> pending;
};

auto SessionLog::writer() -> std::unique_ptr<Writer> & {
    static std::unique_ptr<Writer> w;
    return w;
}

auto SessionLog::writerMutex() -> std::mutex & {
    static std::mutex m;
    return m;
}

auto SessionLog::start() -> void {
    const std::lock_guard<std::mutex> lock(writerMutex());
    if (writer() != nullptr)
        return;
    const auto stamp = juce::Time::getCurrentTime().formatted("%Y-%m-%d_%H-%M-%S");
    const auto file = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                          .getChildFile("virtual-orch")
                          .getChildFile("Logs")
                          .getChildFile("sessions")
                          .getChildFile("session_" + stamp + ".log")
                          .getNonexistentSibling();
    writer() = std::make_unique<Writer>(file);
    writer()->startThread();
    // Not add(): it takes writerMutex, which this function already holds.
    writer()->append("SESSION", "start " + juce::Time::getCurrentTime().toString(true, true, true, true));
    running.store(true);
}

auto SessionLog::stop() -> void {
    const std::lock_guard<std::mutex> lock(writerMutex());
    if (writer() == nullptr)
        return;
    writer()->append("SESSION", "stop");
    running.store(false);
    writer().reset(); // stops the thread and flushes
}

auto SessionLog::isRunning() -> bool {
    return running.load();
}

auto SessionLog::currentFile() -> juce::File {
    const std::lock_guard<std::mutex> lock(writerMutex());
    return writer() != nullptr ? writer()->file : juce::File();
}

auto SessionLog::setClock(const Clock *clock) -> void {
    clockPtr.store(clock);
}

auto SessionLog::clock() -> const Clock * {
    return clockPtr.load();
}

auto SessionLog::add(const char *kind, const juce::String &text) -> void {
    if (! running.load(std::memory_order_relaxed))
        return;
    const std::lock_guard<std::mutex> lock(writerMutex());
    if (writer() != nullptr)
        writer()->append(kind, text);
}

auto SessionLog::describeMidi(const juce::MidiMessage &message) -> juce::String {
    return juce::String::toHexString(message.getRawData(), message.getRawDataSize()).toUpperCase()
           + " (" + message.getDescription() + ")";
}

auto SessionLog::isSkippedMidi(const juce::MidiMessage &message) -> bool {
    return message.isMidiClock() || message.isActiveSense();
}

#include "VirtualOrch/ScriptedInputPlayer.h"

#include <algorithm>
#include <array>

ScriptedInputPlayer::ScriptedInputPlayer(Sink sinkIn)
    : juce::Thread("Scripted input"), sink(std::move(sinkIn)) {}

ScriptedInputPlayer::~ScriptedInputPlayer() {
    stop();
}

auto ScriptedInputPlayer::eventsFrom(const juce::MidiFile &midiIn, int trackIndex) -> std::vector<Event> {
    juce::MidiFile midi(midiIn);
    midi.convertTimestampTicksToSeconds();
    std::vector<Event> out;
    for (int t = 0; t < midi.getNumTracks(); ++t) {
        if (trackIndex >= 0 && t != trackIndex)
            continue;
        const auto *seq = midi.getTrack(t);
        for (const auto *holder: *seq) {
            const auto &m = holder->message;
            const bool note = m.isNoteOnOrOff();
            if (! note && ! m.isController())
                continue;
            if (note && m.getChannel() == 10)
                continue; // drums
            out.push_back({m.getTimeStamp(), m});
        }
    }
    // Time order; at equal times note-offs first so a repeated pitch is released then struck.
    std::stable_sort(out.begin(), out.end(), [](const Event &a, const Event &b) {
        if (a.timeSeconds != b.timeSeconds)
            return a.timeSeconds < b.timeSeconds;
        const auto rank = [](const juce::MidiMessage &m) { return m.isNoteOff(true) ? 0 : m.isController() ? 1 : 2; };
        return rank(a.message) < rank(b.message);
    });
    return out;
}

auto ScriptedInputPlayer::load(const juce::File &newFile) -> juce::String {
    stop();
    juce::FileInputStream in(newFile);
    if (! in.openedOk())
        return "Could not open " + newFile.getFullPathName();
    juce::MidiFile parsed;
    int format = 0;
    if (! parsed.readFrom(in, true, &format))
        return "Not a MIDI file: " + newFile.getFileName();
    const juce::ScopedLock sl(lock);
    midiFile = parsed;
    file = newFile;
    tracks.clear();
    for (int t = 0; t < midiFile.getNumTracks(); ++t) {
        TrackInfo info{t, {}, 0};
        for (const auto *holder: *midiFile.getTrack(t)) {
            const auto &m = holder->message;
            if (m.isTrackNameEvent() && info.name.isEmpty())
                info.name = m.getTextFromTextMetaEvent();
            if (m.isNoteOn() && m.getChannel() != 10)
                ++info.noteCount;
        }
        if (info.noteCount > 0)
            tracks.push_back(info);
    }
    selectedTrack = -1;
    rebuild();
    if (events.empty())
        return "No notes in " + newFile.getFileName();
    return {};
}

auto ScriptedInputPlayer::setTrack(int trackIndex) -> void {
    stop();
    const juce::ScopedLock sl(lock);
    selectedTrack = trackIndex;
    rebuild();
}

auto ScriptedInputPlayer::rebuild() -> void {
    events = eventsFrom(midiFile, selectedTrack);
    lengthSeconds = events.empty() ? 0.0 : events.back().timeSeconds;
}

auto ScriptedInputPlayer::play() -> void {
    stop();
    if (events.empty())
        return;
    startMs.store(juce::Time::getMillisecondCounterHiRes());
    startThread(juce::Thread::Priority::high);
}

auto ScriptedInputPlayer::stop() -> void {
    if (isThreadRunning()) {
        signalThreadShouldExit();
        notify();
        stopThread(2000);
    }
}

auto ScriptedInputPlayer::getPositionSeconds() const -> double {
    if (! isThreadRunning())
        return 0.0;
    return (juce::Time::getMillisecondCounterHiRes() - startMs.load()) / 1000.0;
}

auto ScriptedInputPlayer::run() -> void {
    std::vector<Event> evs;
    {
        const juce::ScopedLock sl(lock);
        evs = events;
    }
    const juce::String id(kScriptedInputIdentifier);
    const juce::String name = juce::String(kScriptedInputName).replace("MIDI file", file.getFileName());
    std::array<std::array<bool, 128>, 17> held{};
    std::array<std::array<bool, 128>, 17> ccDown{}; // controllers 64-69 (pedals) still down

    const auto send = [&](const juce::MidiMessage &m) {
        if (m.isNoteOn())
            held[static_cast<size_t>(m.getChannel())][static_cast<size_t>(m.getNoteNumber())] = true;
        else if (m.isNoteOff())
            held[static_cast<size_t>(m.getChannel())][static_cast<size_t>(m.getNoteNumber())] = false;
        else if (m.isController() && m.getControllerNumber() >= 64 && m.getControllerNumber() <= 69)
            ccDown[static_cast<size_t>(m.getChannel())][static_cast<size_t>(m.getControllerNumber())] =
                m.getControllerValue() >= 64;
        if (sink)
            sink(id, name, m);
    };

    bool finished = false;
    while (! threadShouldExit()) {
        double origin = startMs.load();
        for (const auto &e: evs) {
            const double due = origin + e.timeSeconds * 1000.0;
            for (;;) {
                if (threadShouldExit())
                    break;
                const double now = juce::Time::getMillisecondCounterHiRes();
                if (now >= due)
                    break;
                const double waitMs = due - now;
                if (waitMs > 2.0)
                    wait(static_cast<int>(std::min(waitMs - 1.0, 50.0)));
                else
                    juce::Thread::yield();
            }
            if (threadShouldExit())
                break;
            send(e.message);
        }
        if (threadShouldExit())
            break;
        if (! loop.load()) {
            finished = true;
            break;
        }
        startMs.store(juce::Time::getMillisecondCounterHiRes() + 500.0); // half a second between loops
        // release anything still sounding before the next pass
        for (int ch = 1; ch <= 16; ++ch)
            for (int p = 0; p < 128; ++p)
                if (held[static_cast<size_t>(ch)][static_cast<size_t>(p)])
                    send(juce::MidiMessage::noteOff(ch, p));
    }

    // Like lifting the hands and the pedal: nothing stays held when the script stops.
    for (int ch = 1; ch <= 16; ++ch) {
        for (int p = 0; p < 128; ++p)
            if (held[static_cast<size_t>(ch)][static_cast<size_t>(p)])
                send(juce::MidiMessage::noteOff(ch, p));
        for (int cc = 64; cc <= 69; ++cc)
            if (ccDown[static_cast<size_t>(ch)][static_cast<size_t>(cc)])
                send(juce::MidiMessage::controllerEvent(ch, cc, 0));
    }

    if (finished && onFinished)
        juce::MessageManager::callAsync([cb = onFinished] { cb(); });
}

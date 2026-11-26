// sample_render: play a piano MIDI file into the real app pipeline (reduction RT + orchestration OT + playback),
// headless and in real time, with a version's preset and the viewer's settings, and write the operation log.
// The sample viewer (tools/sample_viewer) runs it on every Generate and reads the RT notes, the OT notes, the
// MIDI out and every draw's top 10 from the oplog.
//
//   sample_render --app APP_DIR --system v1.1|v1.2|v1.2_r4|v1.3 --input piano.mid --oplog out.jsonl.gz
//                 [--models DIR] [--mode simultaneous|continuation] [--prompt-s 5] [--seconds 15]
//                 [--tail-s 3] [--settings settings.json]
//
// APP_DIR: the repo's app/ folder (presets/version*.json, the model configs hybrid_v*.json).
// --models: the ONNX folder (default ~/Documents/virtual_orch_models, as the app).
// simultaneous: the whole input is played; the run ends --tail-s after its last note.
// continuation: only the notes that start in the first --prompt-s are played (cut at --prompt-s), then the piano is
//   silent for --seconds: the hybrid's model 1 (piano-silent side) continues. Model 2 is kept off (its start delay
//   is set past the end), so everything the orchestra plays is model 1's.
// --settings (JSON, all optional): {"logitAdjustments": {...preset format...}, "td": 1..64 | "auto",
//   "outputTemperatures": [a, b, c], "startDelayMs": n, "orchestrationMode": "jam" | "accompaniment",
//   "outerVoicesOutward": bool, "oneInstrumentPerOctave": bool, "sustainPedal": bool (send the file's CC64:
//   the app reads the pianos' pedal as its pause pedal; default false)}
//
// Progress goes to stdout (unbuffered). Exit 0 when the oplog was written.

#include <JuceHeader.h>

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/UserSettings.h"
#include "VirtualOrch/reduction/InformativeWaitReductionTransformer.h"

#include <chrono>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <set>
#include <thread>

namespace {

class NullOutput : public OutputProcessor {
public:
    void send(NoteOnEvent) override {}
    void send(NoteOffEvent) override {}
};

struct InputEvent {
    double ms;
    juce::MidiMessage message;
};

auto fail(const juce::String &why) -> int {
    std::printf("error: %s\n", why.toRawUTF8());
    std::fflush(stdout);
    std::_Exit(2);
}

auto parseFile(const juce::File &f) -> juce::var {
    if (! f.existsAsFile())
        fail("missing " + f.getFullPathName());
    auto v = juce::JSON::parse(f);
    if (! v.isObject())
        fail("cannot parse " + f.getFullPathName());
    return v;
}

/** Overlay b's properties on a (one level deep for objects, as the preset's logitAdjustments.model). */
auto merge(juce::var a, const juce::var &b) -> juce::var {
    if (! b.isObject())
        return a;
    if (! a.isObject())
        a = juce::var(new juce::DynamicObject());
    for (const auto &p: b.getDynamicObject()->getProperties()) {
        if (p.value.isObject() && a[p.name].isObject())
            a.getDynamicObject()->setProperty(p.name, merge(a[p.name], p.value));
        else
            a.getDynamicObject()->setProperty(p.name, p.value);
    }
    return a;
}

/** Note on / off (and CC64 when asked) of every track, in ms from the file's start. */
auto readInput(const juce::File &file, bool pedal, double cutMs) -> std::vector<InputEvent> {
    juce::FileInputStream in(file);
    juce::MidiFile midi;
    if (! in.openedOk() || ! midi.readFrom(in))
        fail("cannot read " + file.getFullPathName());
    midi.convertTimestampTicksToSeconds();
    std::vector<InputEvent> out;
    for (int t = 0; t < midi.getNumTracks(); ++t) {
        const auto *track = midi.getTrack(t);
        for (int i = 0; i < track->getNumEvents(); ++i) {
            auto m = track->getEventPointer(i)->message;
            const bool note = m.isNoteOnOrOff();
            const bool cc = pedal && m.isController() && m.getControllerNumber() == 64;
            if (! note && ! cc)
                continue;
            m.setChannel(1);
            out.push_back({m.getTimeStamp() * 1000.0, m});
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const InputEvent &a, const InputEvent &b) {
        if (a.ms != b.ms)
            return a.ms < b.ms;
        return a.message.isNoteOff() && ! b.message.isNoteOff(); // offs first at one time
    });
    if (cutMs < 0)
        return out;
    // continuation: keep the notes that start before the cut; release every key (and the pedal) at the cut
    std::vector<InputEvent> kept;
    std::set<int> down;
    for (const auto &e: out) {
        if (e.ms >= cutMs)
            break;
        if (e.message.isNoteOnOrOff()) {
            if (e.message.isNoteOn())
                down.insert(e.message.getNoteNumber());
            else
                down.erase(e.message.getNoteNumber());
        }
        kept.push_back(e);
    }
    for (const int p: down)
        kept.push_back({cutMs, juce::MidiMessage::noteOff(1, p)});
    if (pedal)
        kept.push_back({cutMs, juce::MidiMessage::controllerEvent(1, 64, 0)});
    return kept;
}

auto silentFileOf(const juce::var &config, const juce::File &models) -> juce::File {
    auto stem = config.getProperty("silentOnnx", {}).toString().trim();
    if (stem.endsWithIgnoreCase(".onnx"))
        stem = stem.dropLastCharacters(5);
    return models.getChildFile(stem + ".onnx");
}

auto bankFileOf(const juce::var &config, const juce::File &models, const juce::File &configFile) -> juce::File {
    const auto path = config.getProperty("voicingBank", {}).toString().trim();
    if (path.isEmpty())
        return {};
    juce::File f = juce::File::isAbsolutePath(path) ? juce::File(path) : models.getChildFile(path);
    if (! f.existsAsFile())
        f = models.getChildFile(juce::File(path).getFileName());
    if (! f.existsAsFile())
        f = configFile.getSiblingFile(juce::File(path).getFileName());
    return f;
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    juce::ScopedJuceInitialiser_GUI juce;
    juce::File appDir, input, oplog, settingsFile;
    juce::File models = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("virtual_orch_models");
    juce::String system = "v1.2", mode = "simultaneous";
    double promptS = 5.0, seconds = 15.0, tailS = 3.0;
    for (int i = 1; i < argc; ++i) {
        const juce::String a(argv[i]);
        const auto next = [&]() -> juce::String { return i + 1 < argc ? juce::String(argv[++i]) : juce::String(); };
        if (a == "--app") appDir = juce::File(next());
        else if (a == "--models") models = juce::File(next());
        else if (a == "--system") system = next();
        else if (a == "--mode") mode = next();
        else if (a == "--input") input = juce::File(next());
        else if (a == "--oplog") oplog = juce::File(next());
        else if (a == "--settings") settingsFile = juce::File(next());
        else if (a == "--prompt-s") promptS = next().getDoubleValue();
        else if (a == "--seconds") seconds = next().getDoubleValue();
        else if (a == "--tail-s") tailS = next().getDoubleValue();
        else fail("unknown argument " + a);
    }
    if (appDir == juce::File() || input == juce::File() || oplog == juce::File())
        fail("usage: sample_render --app APP_DIR --system v1.2 --input piano.mid --oplog out.jsonl.gz [...]");
    const bool continuation = mode == "continuation";
    const auto settings = settingsFile == juce::File() ? juce::var(new juce::DynamicObject()) : parseFile(settingsFile);

    // ---- the version's preset and model config (as MainComponent::updateModel / applyPreset)
    auto preset = parseFile(appDir.getChildFile("presets/version" + system.trimCharactersAtStart("v") + ".json"));
    const auto configFile = appDir.getChildFile(preset["model"].toString() + ".json");
    const auto config = parseFile(configFile);
    const int type = static_cast<int>(config.getProperty("reductionType", 0));
    if (settings.hasProperty("logitAdjustments"))
        preset.getDynamicObject()->setProperty("logitAdjustments", merge(preset["logitAdjustments"], settings["logitAdjustments"]));
    if (settings.hasProperty("outputTemperatures"))
        preset.getDynamicObject()->setProperty("outputTemperatures", settings["outputTemperatures"]);
    if (settings.hasProperty("orchestrationMode"))
        preset.getDynamicObject()->setProperty("orchestrationMode", settings["orchestrationMode"]);

    auto session = std::make_unique<AppSession>();
    AppSession &s = *session;
    // No message loop here: what the session posts to the message thread runs in the main loop below.
    std::mutex postedLock;
    std::deque<std::function<void()>> posted;
    s.postToMessageThread = [&](std::function<void()> f) {
        const std::lock_guard<std::mutex> g(postedLock);
        posted.push_back(std::move(f));
    };
    const auto pump = [&] {
        for (;;) {
            std::function<void()> f;
            {
                const std::lock_guard<std::mutex> g(postedLock);
                if (posted.empty())
                    return;
                f = std::move(posted.front());
                posted.pop_front();
            }
            f();
        }
    };

    s.modelConfig.outputInstruments.clear();
    if (auto *ins = config.getProperty("outputInstruments", juce::var()).getArray())
        for (int32_t id: *ins)
            s.modelConfig.outputInstruments.emplace(id, OutputInstrumentConfig{false, false, 36, 120});
    s.modelConfig.reductionContextNotes = juce::jmax(1, static_cast<int32_t>(config.getProperty("contextNotes", 160)));
    const auto modelFile = models.getChildFile(config["onnx"].toString() + ".onnx");
    const auto silentFile = silentFileOf(config, models);
    const auto bank = bankFileOf(config, models, configFile);
    // continuation: model 2 never starts within the run
    const int32_t runMs = static_cast<int32_t>((continuation ? promptS + seconds : 0.0) * 1000.0);
    const int32_t startDelay = continuation ? runMs + 60000
                                            : static_cast<int32_t>(settings.getProperty("startDelayMs", config.getProperty("startDelayMs", 3000)));
    std::printf("system %s (type %d): %s + %s\n", system.toRawUTF8(), type, silentFile.getFileName().toRawUTF8(),
                modelFile.getFileName().toRawUTF8());
    if (! modelFile.existsAsFile() || ! silentFile.existsAsFile())
        fail("model files missing in " + models.getFullPathName());
    if (type == 12) {
        s.setMusicModelArch(MusicModelArch::DenseHybridChordToken);
        auto &hy = s.reductionTransformerHybridChord;
        hy.setWaitMode(static_cast<ChordTokenEngine::Mode>(
            InformativeWaitReductionTransformer::modeFromName(config.getProperty("iwMode", "unconstrained").toString())));
        hy.setPrecompute(static_cast<bool>(config.getProperty("precompute", true)));
        hy.setMinWaitCs(static_cast<int32_t>(config.getProperty("minWaitCs", 0)));
        hy.setShortWaitRedraw(static_cast<bool>(config.getProperty("shortWaitRedraw", false)));
        hy.setStartDelayMs(startDelay);
        hy.setVoicingBank(nullptr);
        if (bank != juce::File())
            hy.loadVoicingBank(bank.getFullPathName().toStdString());
        if (! hy.init(silentFile.getFullPathName().toRawUTF8(), modelFile.getFullPathName().toRawUTF8()))
            fail("v1.2 init: " + juce::String(hy.lastInitError));
    } else if (type == 11) {
        s.setMusicModelArch(MusicModelArch::DenseHybridEvent);
        auto &hy = s.reductionTransformerHybridEvent;
        hy.setStartDelayMs(startDelay);
        hy.setForceAfterMs(static_cast<int32_t>(config.getProperty("forceAfterMs", 2000)));
        hy.setVoicingBank(nullptr);
        if (bank != juce::File())
            hy.loadVoicingBank(bank.getFullPathName().toStdString());
        if (! hy.init(silentFile.getFullPathName().toRawUTF8(), modelFile.getFullPathName().toRawUTF8()))
            fail("v1.3 init: " + juce::String(hy.lastInitError));
    } else if (type == 10) {
        s.setMusicModelArch(MusicModelArch::DenseHybridInformativeWait);
        auto &hy = s.reductionTransformerHybridIW;
        hy.setWaitMode(InformativeWaitReductionTransformer::modeFromName(config.getProperty("iwMode", "unconstrained").toString()));
        hy.setPrecompute(static_cast<bool>(config.getProperty("precompute", true)));
        hy.setMinWaitCs(static_cast<int32_t>(config.getProperty("minWaitCs", 0)));
        hy.setShortWaitRedraw(static_cast<bool>(config.getProperty("shortWaitRedraw", false)));
        if (continuation)
            std::printf("note: v1.1 has no start delay; in continuation mode model 2 may play during the prompt\n");
        if (! hy.init(silentFile.getFullPathName().toRawUTF8(), modelFile.getFullPathName().toRawUTF8()))
            fail("v1.1 init: " + juce::String(hy.lastInitError));
    } else {
        fail("reduction type " + juce::String(type) + " is not supported (v1.1 = 10, v1.2 = 12, v1.3 = 11)");
    }
    s.applyLogitAdjustments(preset.getProperty("logitAdjustments", {}));
    if (! s.setOrchestrationModel(preset.getProperty("orchestrationModel", "IodPretrained").toString()))
        fail("cannot load the orchestration model");
    const auto orchMode = preset.getProperty("orchestrationMode", "jam").toString();
    s.setOrchestrationMode(orchMode.equalsIgnoreCase("accompaniment") ? OrchestrationMode::Accompaniment
                           : orchMode.equalsIgnoreCase("edit")        ? OrchestrationMode::Edit
                                                                      : OrchestrationMode::Jam);
    s.presetStore.applyPreset(preset);
    s.rebuildInputFilter();
    if (settings.hasProperty("outerVoicesOutward"))
        UserSettings::setOuterVoicesOutward(static_cast<bool>(settings["outerVoicesOutward"]));
    if (settings.hasProperty("oneInstrumentPerOctave"))
        UserSettings::setOneInstrumentPerOctave(static_cast<bool>(settings["oneInstrumentPerOctave"]));
    s.outputProcessor = std::make_unique<NullOutput>();

    // TD: the Launchpad's TD row (cols 0..6 = 1, 2, 4, ..., 64; the side button = AUTO), callbacks on as live
    {
        auto &g = s.midiInputProcess.getLaunchpadGrid();
        const auto td = settings.getProperty("td", 4);
        if (td.toString().equalsIgnoreCase("auto")) {
            if (TemporalDensity::get() != TemporalDensity::kAuto)
                g.handleInput(LaunchpadGrid::kTdRow, LaunchpadGrid::kSideCol, true, 1u);
        } else {
            int col = 0;
            for (int v = static_cast<int>(td); v > 1 && col < 6; v /= 2)
                ++col;
            g.handleInput(LaunchpadGrid::kTdRow, col, true, 1u);
        }
    }

    const bool pedal = static_cast<bool>(settings.getProperty("sustainPedal", false));
    const auto events = readInput(input, pedal, continuation ? promptS * 1000.0 : -1.0);
    const double lastMs = events.empty() ? 0.0 : events.back().ms;
    const double endMs = continuation ? (promptS + seconds) * 1000.0 : lastMs + tailS * 1000.0;
    const juce::String device = "sample_render";
    s.selectedMidiInputIdentifier = device;

    oplog.getParentDirectory().createDirectory();
    oplog.deleteFile();
    OpLog::start(oplog);
    s.logSessionStart("sample_render");
    OpLog::emit(OpLog::Src::Ui, "SAMPLE_RENDER",
                {{"system", system.toStdString()}, {"mode", mode.toStdString()}, {"input", input.getFullPathName().toStdString()},
                 {"prompt_s", promptS}, {"seconds", seconds}, {"settings", OpLog::Raw{juce::JSON::toString(settings, true).toStdString()}},
                 {"start_delay_ms", startDelay}});
    std::printf("input: %zu events, %.1f s; running %.1f s ...\n", events.size(), lastMs / 1000.0, endMs / 1000.0);

    s.startGeneration();
    pump();
    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsedMs = [&] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    OpLog::emit(OpLog::Src::Ui, "SAMPLE_T0", {{"note", "the input file's time 0"}});
    size_t ei = 0;
    double nextTimer = 0.0, nextReport = 1000.0;
    while (elapsedMs() < endMs) {
        const double now = elapsedMs();
        while (ei < events.size() && events[ei].ms <= now) {
            s.midiInputProcess.handleMidiFrom(device, device, events[ei].message);
            ++ei;
        }
        if (now >= nextTimer) {
            if (s.midiInputProcess.isTimerRunning())
                s.midiInputProcess.timerCallback(); // the input window (a juce::Timer: no message loop here)
            nextTimer = now + 10.0;
        }
        pump();
        if (now >= nextReport) {
            std::printf("  %.0f / %.0f s\n", now / 1000.0, endMs / 1000.0);
            nextReport += 1000.0;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    s.stopGeneration();
    pump();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    pump();
    OpLog::stop();
    std::printf("done: %s\n", oplog.getFullPathName().toRawUTF8());
    std::fflush(stdout);
    // As oplog_replay: skip static destruction (ORT's CUDA provider teardown at exit)
    std::_Exit(0);
}

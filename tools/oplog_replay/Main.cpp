// oplog_replay: replay recorded session logs (Documents/virtual-orch/Logs/sessions/session_*.log)
// with the real models, every choice forced to the logged one, and write each session's operation
// log (format: orchestration/performance_streams/oplogs/README.md).
//
//   oplog_replay --models DIR --out DIR [--rt-v1 continuation_model.onnx] [--rt-wait simultaneous_model_v2.onnx]
//                [--ot instrument_octave_predictions.onnx] [--zero] session_*.log[.gz] ...
//
// Resumable: a session whose oplog_<stamp>.jsonl.gz exists is skipped; each file is written to
// .part and renamed when complete. Progress and a per-session summary go to stdout (unbuffered).
// --zero runs without models (zero logits), to check the pipeline quickly.

#include <JuceHeader.h>

#include "VirtualOrch/sim/LogReplay.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

auto presetFor(const std::vector<LogReplay::Line> &log) -> juce::var {
    // The last non-empty preset the session applied before its first Start; else the Hybrid default.
    juce::var preset;
    for (const auto &l: log) {
        if (l.tag == "UI" && l.text.startsWith("Start generation"))
            break;
        if (l.tag == "UI" && l.text.startsWith("Preset applied: ")) {
            const auto v = juce::JSON::parse(l.text.fromFirstOccurrenceOf("Preset applied: ", false, false));
            if (v.isObject() && v.getDynamicObject()->getProperties().size() > 0)
                preset = v;
        }
    }
    return preset.isObject() ? preset : LogReplay::hybridV1W3Preset();
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    juce::ScopedJuceInitialiser_GUI juce;
    juce::File models, out;
    juce::String rtV1 = "continuation_model.onnx", rtWait = "simultaneous_model_v2.onnx",
                 ot = "instrument_octave_predictions.onnx";
    bool zero = false, noOt = false, noRt = false;
    double untilMs = -1.0;
    std::vector<juce::File> logs;
    for (int i = 1; i < argc; ++i) {
        const juce::String a(argv[i]);
        const auto next = [&]() -> juce::String { return i + 1 < argc ? juce::String(argv[++i]) : juce::String(); };
        if (a == "--models")
            models = juce::File(next());
        else if (a == "--out")
            out = juce::File(next());
        else if (a == "--rt-v1")
            rtV1 = next();
        else if (a == "--rt-wait")
            rtWait = next();
        else if (a == "--ot")
            ot = next();
        else if (a == "--zero")
            zero = true;
        else if (a == "--no-ot") // profiling: the OT without its model
            noOt = true;
        else if (a == "--no-rt") // profiling: the RT on zero logits
            noRt = true;
        else if (a == "--until-ms")
            untilMs = next().getDoubleValue();
        else
            logs.emplace_back(juce::File(a));
    }
    if (out == juce::File() || logs.empty() || (! zero && models == juce::File())) {
        std::printf("usage: oplog_replay --models DIR --out DIR [--zero] session_*.log[.gz] ...\n");
        return 2;
    }
    out.createDirectory();
    int failed = 0;
    for (const auto &file: logs) {
        auto stamp = file.getFileName().upToFirstOccurrenceOf(".log", false, false);
        stamp = stamp.fromFirstOccurrenceOf("session_", false, false);
        const auto target = out.getChildFile("oplog_" + stamp + ".jsonl.gz");
        if (target.existsAsFile()) {
            std::printf("%s: done already, skipped\n", stamp.toRawUTF8());
            continue;
        }
        const auto part = out.getChildFile("oplog_" + stamp + ".jsonl.gz.part");
        const auto log = LogReplay::readLog(file);
        LogReplay::Options o;
        o.preset = presetFor(log);
        o.oplog = part;
        o.sourceName = file.getFileName();
        o.untilMs = untilMs;
        if (! zero && ! noRt) {
            o.rtSilentModel = models.getChildFile(rtV1);
            o.rtWaitModel = models.getChildFile(rtWait);
        }
        if (! zero && ! noOt)
            o.otModel = models.getChildFile(ot);
        const auto t0 = std::chrono::steady_clock::now();
        std::printf("%s: replaying %.1f min of log ...\n", stamp.toRawUTF8(),
                    log.empty() ? 0.0 : (log.back().ms - log.front().ms) / 60000.0);
        const auto report = LogReplay::replay(log, o);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        bool fatal = false;
        for (const auto &p: report.problems)
            if (p.startsWith("could not load") || p.startsWith("cannot write"))
                fatal = true;
        std::printf("%s: %.1f s\n%s", stamp.toRawUTF8(), secs, report.summary().toRawUTF8());
        // the summary next to the oplog, for checking later
        out.getChildFile("oplog_" + stamp + ".replay.txt")
            .replaceWithText("source " + file.getFullPathName() + "\nreplay " + juce::String(secs, 1) + " s\n"
                             + report.summary());
        // The replay's own session-log lines (old format), for checking the oplog against the same run.
        {
            std::string text;
            text.reserve(report.replayed.size() * 120);
            char ms[32];
            for (const auto &l: report.replayed) {
                std::snprintf(ms, sizeof(ms), "%.1f", l.ms);
                text += ms;
                text += " | clock ";
                text += l.clock >= 0 ? std::to_string(l.clock) : std::string("-");
                text += " | ";
                text += l.tag.toStdString();
                text += " | ";
                text += l.text.toStdString();
                text += '\n';
            }
            juce::FileOutputStream fos(out.getChildFile("oplog_" + stamp + ".replay_session.log.gz"));
            if (fos.openedOk()) {
                fos.setPosition(0);
                fos.truncate();
                juce::GZIPCompressorOutputStream gz(fos, 6, juce::GZIPCompressorOutputStream::windowBitsGZIP);
                gz.write(text.data(), text.size());
            }
        }
        if (fatal || ! part.existsAsFile()) {
            ++failed;
            part.deleteFile();
            continue;
        }
        part.moveFileTo(target);
    }
    // Skip static destruction: tearing down ORT's CUDA provider at exit corrupts the heap
    // ("corrupted double-linked list") after every file is written and closed.
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(failed > 0 ? 1 : 0);
}

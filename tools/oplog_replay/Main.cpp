// oplog_replay: replay recorded session logs (Documents/virtual-orch/Logs/sessions/session_*.log)
// with the real models, every choice forced to the logged one, and write each session's operation
// log (format: orchestration/performance_streams/oplogs/README.md).
//
//   oplog_replay --models DIR --out DIR [--rt-v1 rt_continuation_legacy.onnx] [--rt-wait rt_simultaneous_legacy.onnx]
//                [--ot ot_legacy.onnx] [--preset FILE] [--layout 1|2] [--zero] session_*.log[.gz] ...
//
// --preset: for logs without a "UI | Preset applied" line (before 22 Nov 2026), the preset the app
// was using, on top of the Hybrid default (LogReplay::presetOf; the LogitTD strength comes from the
// RT lines).
// --layout: the Launchpad layout for logs without a "launchpad layout" line (default: from the log).
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

// The fallback preset for a log that applied none: the Hybrid default with `file`'s settings on top.
auto fallbackPreset(const juce::File &file) -> juce::var {
    auto preset = LogReplay::hybridV1W3Preset();
    if (file == juce::File())
        return preset;
    const auto v = juce::JSON::parse(file);
    if (! v.isObject()) {
        std::printf("cannot read preset %s\n", file.getFullPathName().toRawUTF8());
        std::_Exit(2);
    }
    for (const auto &p: v.getDynamicObject()->getProperties())
        preset.getDynamicObject()->setProperty(p.name, p.value);
    return preset;
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    juce::ScopedJuceInitialiser_GUI juce;
    juce::File models, out, presetFile;
    juce::String rtV1 = "rt_continuation_legacy.onnx", rtWait = "rt_simultaneous_legacy.onnx",
                 ot = "ot_legacy.onnx";
    bool zero = false, noOt = false, noRt = false;
    double untilMs = -1.0;
    int layout = 0; // 0: from the log
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
        else if (a == "--preset") // for logs that applied no preset: the one the app was using
            presetFile = juce::File(next());
        else if (a == "--layout") // the Launchpad layout, for logs that do not say (1 or 2)
            layout = next().getIntValue();
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
        std::printf("usage: oplog_replay --models DIR --out DIR [--preset FILE] [--zero] session_*.log[.gz] ...\n");
        return 2;
    }
    out.createDirectory();
    const auto fallback = fallbackPreset(presetFile);
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
        o.preset = LogReplay::presetOf(log, fallback);
        o.oplog = part;
        o.sourceName = file.getFileName();
        o.untilMs = untilMs;
        if (layout > 0)
            o.layout1 = layout < 2;
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

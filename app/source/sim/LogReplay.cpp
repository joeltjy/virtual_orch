#include "VirtualOrch/sim/LogReplay.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/AppTime.h"
#include "VirtualOrch/LaunchpadProgrammerMap.h"
#include "VirtualOrch/ReductionInputGate.h"
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/orchestration-models/IodPretrained.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>

namespace LogReplay {

namespace {

// ---------------------------------------------------------------- parsing

auto parseLine(const juce::String &raw, Line &out) -> bool {
    // "<ms> | clock <cs or -> | <TAG> | <text>"
    const int a = raw.indexOf(" | ");
    if (a <= 0)
        return false;
    const int b = raw.indexOf(a + 3, " | ");
    if (b < 0)
        return false;
    const int c = raw.indexOf(b + 3, " | ");
    const auto msText = raw.substring(0, a).trim();
    const auto clockField = raw.substring(a + 3, b).trim();
    if (! msText.containsOnly("0123456789.") || ! clockField.startsWith("clock "))
        return false;
    out.ms = msText.getDoubleValue();
    const auto clockText = clockField.fromFirstOccurrenceOf("clock ", false, false).trim();
    out.clock = clockText.containsOnly("0123456789") && clockText.isNotEmpty() ? clockText.getIntValue() : -1;
    if (c < 0) {
        out.tag = raw.substring(b + 3).trim();
        out.text = {};
    } else {
        out.tag = raw.substring(b + 3, c).trim();
        out.text = raw.substring(c + 3);
    }
    return out.tag.isNotEmpty() && out.tag.containsOnly("ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789");
}

/** Hex bytes at the start of "90 3C 41 (Note on ...)". */
auto midiFromHex(const juce::String &text) -> std::optional<juce::MidiMessage> {
    juce::MemoryBlock bytes;
    juce::StringArray parts;
    parts.addTokens(text.upToFirstOccurrenceOf("(", false, false), " ", "");
    parts.removeEmptyStrings();
    std::vector<juce::uint8> data;
    for (const auto &p: parts) {
        if (p.length() != 2 || ! p.containsOnly("0123456789ABCDEFabcdef"))
            break;
        data.push_back(static_cast<juce::uint8>(p.getHexValue32()));
    }
    if (data.empty())
        return std::nullopt;
    if (data[0] == 0xF0)
        return juce::MidiMessage(data.data(), static_cast<int>(data.size()));
    const int len = juce::MidiMessage::getMessageLengthFromFirstByte(data[0]);
    if (static_cast<int>(data.size()) < len)
        return std::nullopt;
    return juce::MidiMessage(data.data(), len);
}

/** "(onset, dur, instr - pitch, vel v)" */
struct NoteFields {
    int32_t onset = 0, dur = 0, instr = 0, pitch = 0, vel = 0;
};
auto parseNoteFields(const juce::String &text) -> std::optional<NoteFields> {
    const int open = text.indexOfChar('(');
    const int close = text.indexOfChar(open, ')');
    if (open < 0 || close < 0)
        return std::nullopt;
    const auto inner = text.substring(open + 1, close);
    if (inner.contains("Clear Queue") || inner.contains("Bar Separator"))
        return std::nullopt;
    juce::StringArray parts;
    parts.addTokens(inner, ",", "");
    if (parts.size() < 3)
        return std::nullopt;
    NoteFields f;
    f.onset = parts[0].trim().getIntValue();
    f.dur = parts[1].trim().getIntValue();
    const auto ip = parts[2].trim();
    f.instr = ip.upToFirstOccurrenceOf("-", false, false).trim().getIntValue();
    f.pitch = ip.fromFirstOccurrenceOf("-", false, false).trim().getIntValue();
    if (parts.size() >= 4)
        f.vel = parts[3].fromFirstOccurrenceOf("vel", false, false).trim().getIntValue();
    return f;
}

auto clearQueueTime(const juce::String &text) -> std::optional<int32_t> {
    const int open = text.indexOfChar('(');
    if (open < 0 || ! text.contains("Clear Queue)"))
        return std::nullopt;
    return text.substring(open + 1).upToFirstOccurrenceOf(",", false, false).trim().getIntValue();
}

auto intAfter(const juce::String &text, const juce::String &key) -> std::optional<int32_t> {
    const int i = text.indexOf(key);
    if (i < 0)
        return std::nullopt;
    auto rest = text.substring(i + key.length()).trimStart();
    if (rest.isEmpty() || ! (juce::CharacterFunctions::isDigit(rest[0]) || rest[0] == '-'))
        return std::nullopt;
    return rest.getIntValue();
}

// ---------------------------------------------------------------- stand-ins

/** The KV model the RT runs on: every logit 0 (all draws are forced from the log). */
class ZeroBackend : public StreamingKv::Backend {
public:
    explicit ZeroBackend(size_t vocabIn) : vocab(vocabIn) {
        // The real T3'' / W2/W3 exports: 4 sink rows + 508 recent (eviction and rebasing depend on it).
        d.layers = 1;
        d.heads = 1;
        d.headDim = 1;
        d.maxRows = 512;
        d.sinkRows = 4;
    }
    [[nodiscard]] auto dims() const -> StreamingKv::Dims override { return d; }
    auto run(const StreamingKv::State &past, const std::vector<int64_t> &ids, StreamingKv::State &present,
             std::vector<float> &last) -> bool override {
        present = past;
        for (size_t i = 0; i < ids.size(); ++i) {
            present.key.push_back(0.0f);
            present.value.push_back(0.0f);
            ++present.rows;
        }
        last.assign(vocab, 0.0f);
        return true;
    }

private:
    size_t vocab;
    StreamingKv::Dims d;
};

/** MIDI out as MidiOutputProcessor logs it, without a device. */
class LoggingOutput : public OutputProcessor {
public:
    void send(NoteOnEvent event) override {
        const auto channel = channelForInstrument(event.instrument, event.note);
        if (! channel.has_value())
            return;
        const auto message = juce::MidiMessage::noteOn(*channel, event.note, event.velocity);
        SessionLog::add("MIDI_OUT", "instr " + juce::String(event.instrument) + " | " + SessionLog::describeMidi(message));
    }
    void send(NoteOffEvent event) override {
        const auto channel = channelForInstrument(event.instrument, event.note);
        if (! channel.has_value())
            return;
        const auto message = juce::MidiMessage::noteOff(*channel, event.note);
        SessionLog::add("MIDI_OUT", "instr " + juce::String(event.instrument) + " | " + SessionLog::describeMidi(message));
    }
    void relayMidi(const juce::MidiMessage &midiMessage) override {
        juce::MidiMessage m = midiMessage;
        m.setChannel(InstrumentConstants::kReductionMonitorMidiChannel);
        SessionLog::add("MIDI_OUT", "thru | " + SessionLog::describeMidi(m));
    }
    void clear() override {
        OutputProcessor::clear();
        SessionLog::add("MIDI_OUT", "clear: note off on every channel and pitch");
    }
};

/** The orchestration "model": returns the OT notes the recorded run produced for these inputs. */
class ReplayModel : public OrchestrationModel {
public:
    struct Logged {
        double ms = 0.0;
        OrchestrationNote note;
        bool used = false;
    };
    std::vector<Logged> logged;
    std::function<double()> now;
    /** A logged note is used when the call is at most this much before / after its logged time. */
    double lookaheadMs = 400.0;
    double lookbackMs = 60.0;
    int callsWithoutMatch = 0;
    /** Called with the recorded time of every logged note handed out. */
    std::function<void(double)> onMatched;
    /** The real model, run with its choices forced to the logged ones (oplog draws); may be null. */
    IodPretrained *real = nullptr;

    [[nodiscard]] auto getName() const -> std::string override { return real != nullptr ? real->getName() : "LogReplay"; }
    auto resetState() -> void override {
        OrchestrationModel::resetState();
        if (real != nullptr)
            real->resetState();
    }
    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override { return history; }
    auto getOutput(const std::vector<Token> &incoming, const std::vector<int32_t> &instruments,
                   const ConditioningSignal &signal, OrchestrationBalanceTracker *balance,
                   const OrchestrationBalanceTracker::BiasView *bias) -> std::vector<OrchestrationNote> override {
        std::multiset<int32_t> onsets;
        std::multimap<int32_t, Token> byOnset;
        for (const auto &t: incoming)
            if (t.note >= static_cast<int32_t>(Vocab::NoteOffset) && t.note < static_cast<int32_t>(Vocab::Rest)) {
                onsets.insert(t.time);
                byOnset.emplace(t.time, t);
            }
        std::vector<OrchestrationNote> out;
        if (onsets.empty())
            return out;
        const double t = now ? now() : 0.0;
        // Each input note gets its own outputs: same onset, duration and pitch class (the OT keeps
        // the timing and may only move a note by octaves); for the user's own notes, whose length
        // can change at the key release, same onset and pitch class.
        const auto take = [&](const Token &src, bool needDuration) {
            bool any = false;
            for (auto &l: logged) {
                if (l.used || l.ms < t - lookbackMs)
                    continue;
                if (l.ms > t + lookaheadMs)
                    break;
                const auto &o = l.note.token;
                if (o.time != src.time || o.getPitch() % 12 != src.getPitch() % 12)
                    continue;
                if (needDuration && o.duration != src.duration)
                    continue;
                l.used = true;
                any = true;
                if (onMatched)
                    onMatched(l.ms);
                OrchestrationNote n = l.note;
                n.token.keyId = src.keyId;
                n.sourceNoteId = src.noteId;
                out.push_back(n);
            }
            return any;
        };
        for (const auto &[onset, src]: byOnset)
            if (! take(src, true))
                take(src, false);
        if (out.empty())
            ++callsWithoutMatch;
        if (real != nullptr) {
            // The model sees the same inputs; for each input note it is forced to the instruments and
            // octaves the recorded run played for it (as (GM program, octave) pairs).
            real->forcedCombo = [&out](const Token &in) -> std::optional<std::array<int32_t, IodPretrainedTypes::comboLen>> {
                std::vector<int32_t> pairs;
                for (const auto &n: out) {
                    const bool same = in.keyId >= 0 ? n.token.keyId == in.keyId
                                                    : (n.sourceNoteId == in.noteId && n.token.time == in.time);
                    if (! same)
                        continue;
                    int32_t gm = -1;
                    for (const auto &[local, program]: InstrumentConstants::kInstrumentMappings)
                        if (local == n.localInstrumentId)
                            gm = program;
                    const int32_t delta = static_cast<int32_t>(std::lround((n.token.getPitch() - in.getPitch()) / 12.0));
                    if (gm >= 0)
                        pairs.push_back(IodPretrainedTypes::packProgramDelta(gm, delta));
                }
                return IodPretrainedTypes::comboFromPairs(pairs);
            };
            real->currentStream = currentStream;
            real->instrumentLogitSink = instrumentLogitSink;
            real->octaveDeltaSink = octaveDeltaSink;
            real->getOutputTimings = getOutputTimings;
            (void) real->getOutput(incoming, instruments, signal, balance, bias);
            real->forcedCombo = nullptr;
        }
        return out;
    }

private:
    std::vector<OrchestrationNote> history;
};

// ---------------------------------------------------------------- comparison

auto compareTag(const std::vector<const Line *> &recorded, const std::vector<const Line *> &replayed,
                double tolMs, int keep) -> TagResult {
    TagResult r;
    r.recorded = static_cast<int>(recorded.size());
    r.replayed = static_cast<int>(replayed.size());
    // text -> indices of replayed lines (in time order), greedy nearest-in-time match
    std::map<juce::String, std::deque<size_t>> byText;
    for (size_t i = 0; i < replayed.size(); ++i)
        byText[replayed[i]->text].push_back(i);
    std::vector<bool> used(replayed.size(), false);
    for (const auto *rec: recorded) {
        auto it = byText.find(rec->text);
        bool found = false;
        if (it != byText.end()) {
            auto &q = it->second;
            while (! q.empty() && (used[q.front()] || replayed[q.front()]->ms < rec->ms - tolMs))
                q.pop_front();
            for (auto qi = q.begin(); qi != q.end(); ++qi) {
                if (replayed[*qi]->ms > rec->ms + tolMs)
                    break;
                if (! used[*qi]) {
                    used[*qi] = true;
                    found = true;
                    break;
                }
            }
        }
        if (found)
            ++r.matched;
        else if (static_cast<int>(r.missing.size()) < keep)
            r.missing.push_back(juce::String(rec->ms, 1) + " | " + rec->text.substring(0, 240));
    }
    for (size_t i = 0; i < replayed.size(); ++i)
        if (! used[i] && static_cast<int>(r.extra.size()) < keep)
            r.extra.push_back(juce::String(replayed[i]->ms, 1) + " | " + replayed[i]->text.substring(0, 240));
    return r;
}

/**
 * The Launchpad layout a session log was recorded with: its "SESSION | launchpad layout N" line,
 * or 1 for logs without one (recorded before the two-sided layout, Nov 2026).
 */
auto launchpadLayoutOf(const std::vector<Line> &log) -> int {
    for (const auto &l: log)
        if (l.tag == "SESSION" && l.text.startsWith("launchpad layout "))
            return l.text.fromFirstOccurrenceOf("launchpad layout ", false, false).getIntValue();
    return 1;
}

/** A replayed line in the words a layout-1 session logged it with (Edit, the old flush and pedal). */
auto withLayout1Text(Line l) -> Line {
    if (l.tag == "MODE" && l.text.trim() == "Step-in")
        l.text = "Edit";
    else if (l.tag == "LAUNCHPAD" && l.text == "Mode: Step-in")
        l.text = "Mode: Edit";
    else if (l.tag == "LAUNCHPAD" && l.text == "Flush: your instruments added to the model")
        l.text = "Flush: input instruments moved to the model";
    else if (l.tag == "CUT" && l.text.startsWith("pedal up: future removed from"))
        l.text = "pedal down" + l.text.fromFirstOccurrenceOf("pedal up", false, false);
    return l;
}

/** Tags whose recorded and replayed lines are compared. */
const juce::StringArray kComparedTags{"MIDI_OUT", "RT_IN", "RT", "OT", "CUT", "DROP", "HYBRID",
                                      "WAIT", "MODE", "LAUNCHPAD", "MIDI_IN"};

} // namespace

// ---------------------------------------------------------------- public

auto parseLog(const juce::String &content) -> std::vector<Line> {
    std::vector<Line> out;
    juce::StringArray lines;
    lines.addLines(content);
    out.reserve(static_cast<size_t>(lines.size()));
    for (const auto &raw: lines) {
        Line l;
        if (parseLine(raw, l))
            out.push_back(std::move(l));
    }
    return out;
}

auto readLog(const juce::File &file) -> std::vector<Line> {
    if (file.getFileExtension().equalsIgnoreCase(".gz")) {
        juce::FileInputStream in(file);
        if (! in.openedOk())
            return {};
        juce::GZIPDecompressorInputStream gz(&in, false, juce::GZIPDecompressorInputStream::gzipFormat);
        return parseLog(gz.readEntireStreamAsString());
    }
    return parseLog(file.loadFileAsString());
}

auto hybridV1W3Preset() -> juce::var {
    return juce::JSON::parse(R"({"model": "hybrid_v1_w3", "reduction": "hybrid", "orchestrationModel": "IodPretrained", "orchestrationMode": "jam", "inputMode": "direct", "inputFilterType": "passthrough", "directInputWindowLength": 1, "directInputSendNoteOffs": true, "directInputStartOnInput": false, "directInputStartDelay": 0, "directInputHoldBass": false, "directInputBassLow": 24, "directInputBassHigh": 36, "directInputInitialBass": -1, "inputInstrument": 0, "inputLow": 36, "inputHigh": 120, "inputDuration": 400, "inputInitialData": "", "contextNotes": 40, "outputMinimumDuration": 10, "outputMaximumDuration": 999, "outputTemperatures": [0.5, 0.5, 0.5], "outputStartTime": 0, "outputForceStartTime": false, "outputMaxAheadSeconds": 5, "outputAheadThrottleSeconds": 2, "outputInstruments": [{"id": 0, "active": false, "monophony": false, "low": 36, "high": 120}, {"id": 1, "active": false, "monophony": false, "low": 36, "high": 120}, {"id": 2, "active": false, "monophony": false, "low": 36, "high": 120}, {"id": 3, "active": false, "monophony": false, "low": 36, "high": 120}, {"id": 4, "active": false, "monophony": false, "low": 36, "high": 120}], "logitAdjustments": {"biasesOn": true, "biasesOnlyWhenPianoSilent": false, "logitTdStrength": 5.5, "model": {"pitchBias": false, "durationBias": false, "onsetGapBias": false, "shortGapPenalty": false, "logitTd": false, "pitchTau": 0.2, "pitchTauPrime": 0.2, "durationTau": 0.5, "durationTauPrime": 0.0, "onsetGapTau": 0.5, "waitMinPlay": 0.3}, "hybridV1": {"pitchBias": true, "durationBias": true, "onsetGapBias": true, "shortGapPenalty": true, "logitTd": true}}})");
}

auto Report::summary() const -> juce::String {
    juce::String s;
    s << "generation runs: " << generationRuns << ", clock phase " << juce::String(clockPhaseMs, 1) << " ms\n";
    for (const auto &[tag, r]: tags)
        s << tag.paddedRight(' ', 10) << " recorded " << juce::String(r.recorded).paddedLeft(' ', 6) << "  replayed "
          << juce::String(r.replayed).paddedLeft(' ', 6) << "  matched " << juce::String(r.matched).paddedLeft(' ', 6)
          << (r.exact() ? "  exact" : "") << "\n";
    for (const auto &p: problems)
        s << "problem: " << p << "\n";
    return s;
}

auto replay(const std::vector<Line> &log, const Options &options) -> Report {
    Report report;
    if (log.empty())
        return report;

    double simMs = std::floor(log.front().ms);
    AppTime::setSimulatedMs(simMs);
    SamplingRng::seed(42); // nothing should draw at random; if something does, keep it repeatable

    struct Restore {
        ~Restore() {
            SessionLog::setCapture(nullptr);
            AppTime::setSimulatedMs(-1.0);
            SamplingRng::unseed();
        }
    } restore;

    auto session = std::make_unique<AppSession>();
    AppSession &s = *session;
    // Sessions recorded before the two-sided layout: read their Launchpad presses and pedal as
    // the layout-1 grid and pedal polarity did (NEW_LAUNCHPAD_LAYOUT.txt, part D).
    const bool layout1 = options.layout1.value_or(launchpadLayoutOf(log) < 2);
    s.midiInputProcess.getLaunchpadGrid().layout1 = layout1;
    s.midiInputProcess.layout1PedalPolarity = layout1;
    if (! options.preset.isVoid())
        s.presetStore.applyPreset(options.preset);
    s.setMusicModelArch(MusicModelArch::DenseHybrid);
    auto &rt = s.reductionTransformerHybrid;
    const bool realModels = options.rtSilentModel != juce::File() && options.rtWaitModel != juce::File();
    if (realModels) {
        if (! rt.initHybrid(options.rtSilentModel.getFullPathName().toRawUTF8(),
                            options.rtWaitModel.getFullPathName().toRawUTF8())) {
            report.problems.push_back("could not load the RT models: " + juce::String(rt.lastInitError));
            return report;
        }
    } else {
        rt.initHybridWithBackends(std::make_unique<ZeroBackend>(11776), 11776, std::make_unique<ZeroBackend>(11777), 11777);
    }
    std::unique_ptr<IodPretrained> realOt;
    if (options.otModel != juce::File()) {
        realOt = std::make_unique<IodPretrained>();
        realOt->init(options.otModel.getFullPathName().toRawUTF8());
        if (! realOt->isModelLoaded()) {
            report.problems.push_back("could not load the OT model " + options.otModel.getFullPathName());
            return report;
        }
        if (s.orchestrationModel != nullptr)
            realOt->setVocsepEnsembleEnabled(s.orchestrationModel->getVocsepEnsembleEnabled() && options.voiceSeparation);
    }

    // ---- the operation log (written in order, on simulated time)
    std::unique_ptr<juce::OutputStream> oplogOut;
    struct StopOplog {
        std::unique_ptr<juce::OutputStream> &out;
        ~StopOplog() {
            if (out == nullptr)
                return;
            OpLog::emit(OpLog::Src::Ui, "SESSION_END", {});
            OpLog::setCapture(nullptr);
            OpLog::setTime(nullptr);
            out.reset();
        }
    } stopOplog{oplogOut};
    if (options.oplog != juce::File()) {
        options.oplog.getParentDirectory().createDirectory();
        auto fos = std::make_unique<juce::FileOutputStream>(options.oplog);
        if (! fos->openedOk()) {
            report.problems.push_back("cannot write " + options.oplog.getFullPathName());
            return report;
        }
        fos->setPosition(0);
        fos->truncate();
        oplogOut = std::make_unique<juce::GZIPCompressorOutputStream>(fos.release(), 6, true, juce::GZIPCompressorOutputStream::windowBitsGZIP);
        OpLog::setTime([&simMs] { return simMs; });
        OpLog::setCapture([&oplogOut](const std::string &line) {
            oplogOut->write(line.data(), line.size());
            oplogOut->writeByte('\n');
        });
    }
    if (options.preset.isObject() && options.preset.hasProperty("logitAdjustments"))
        s.applyLogitAdjustments(options.preset["logitAdjustments"]);
    // The hybrid's two model names as the recorded run logged them ("<name> from <clock> (why)").
    for (const auto &l: log) {
        if (l.tag != "HYBRID")
            continue;
        const auto name = l.text.upToFirstOccurrenceOf(" from ", false, false);
        if (l.text.contains("(playing 1 s)"))
            rt.hybridWaitName = name.toStdString();
        else
            rt.hybridV1Name = name.toStdString();
    }
    s.outputProcessor = std::make_unique<LoggingOutput>();
    if (! options.voiceSeparation) {
        rt.voiceSeparation = nullptr;
        s.orchestrationTransformer.inputVoiceSeparation = nullptr;
    }

    // Process-wide state the app starts with: TD 4, register AUTO, input not blocked. An earlier
    // replay in the same process may have changed it; reset it through the fresh grid's own buttons
    // (TemporalDensity / RegisterRange are only settable by LaunchpadGrid), with its callbacks off.
    {
        auto &g = s.midiInputProcess.getLaunchpadGrid();
        const auto td = g.td_callback;
        const auto reg = g.register_callback;
        const auto act = g.action_callback;
        g.td_callback = nullptr;
        g.register_callback = nullptr;
        g.action_callback = nullptr;
        if (TemporalDensity::get() != TemporalDensity::kDefault)
            g.handleInput(LaunchpadGrid::kTdRow, 2 /* TD 4 */, true, 1u);
        if (! RegisterRange::isAuto())
            g.handleInput(LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol, true, 1u);
        g.td_callback = td;
        g.register_callback = reg;
        g.action_callback = act;
        ReductionInputGate::setBlocked(false);
    }

    // ---- what the recorded run decided: RT draws, WAIT decisions, OT outputs
    struct Action {
        double ms = 0.0;
        bool wait = false; // a WAIT decision (true) or a drawn note
        NoteFields note;
        float waitProbability = 0.0f; // the logged P(wait) of a "wait" decision
    };
    std::vector<Action> actions;
    auto replayModel = std::make_unique<ReplayModel>();
    replayModel->now = [&simMs] { return simMs; };
    for (const auto &l: log) {
        if (options.untilMs >= 0.0 && l.ms > options.untilMs)
            break;
        if (l.tag == "RT" && l.text.startsWith(rt.getThreadName())) {
            if (const auto f = parseNoteFields(l.text))
                actions.push_back({l.ms, false, *f});
        } else if (l.tag == "WAIT" && l.text.startsWith(rt.getThreadName())) {
            const auto what = l.text.fromFirstOccurrenceOf("| ", false, false).trim();
            if (what.startsWith("wait ") || what.startsWith("wait-given")) {
                Action a{l.ms, true, {}};
                if (what.contains("P(wait)"))
                    a.waitProbability = what.fromFirstOccurrenceOf("P(wait)", false, false).trim().getFloatValue();
                actions.push_back(a);
            }
        } else if (l.tag == "OT") {
            if (const auto f = parseNoteFields(l.text)) {
                ReplayModel::Logged lo;
                lo.ms = l.ms;
                lo.note.token = Token{f->onset, static_cast<int32_t>(Vocab::DurOffset) + f->dur,
                                      static_cast<int32_t>(Vocab::NoteOffset) + Config::MaxPitch * f->instr + f->pitch,
                                      f->vel};
                lo.note.localInstrumentId = intAfter(l.text, ") instr").value_or(0);
                lo.note.velocity = intAfter(l.text.fromFirstOccurrenceOf(") instr", false, false), "vel").value_or(0);
                lo.note.token.voiceId = intAfter(l.text, "voice").value_or(-1);
                replayModel->logged.push_back(lo);
            }
        }
    }
    s.orchestrationTransformer.orchestrationModel = replayModel.get();
    replayModel->getOutputTimings = &s.orchestrationTransformer.getOutputAccum;
    replayModel->real = realOt.get();
    if (OpLog::on()) {
        s.logSessionStart("replay");
        OpLog::emit(OpLog::Src::Ui, "REPLAY",
                    {{"source", options.sourceName},
                     {"rt_v1", options.rtSilentModel.getFileName()}, {"rt_wait", options.rtWaitModel.getFileName()},
                     {"ot", options.otModel.getFileName()}, {"real_models", realModels},
                     {"note", "every model choice is the one the live run made (forced); p / p_model / top are the models' probabilities for it"}});
    }

    // How the recorded threads spent their time. A draw starts at the first pass that wants one
    // (the live thread started computing then) and its output is delivered at the recorded time
    // (when the live model finished); the thread is busy in between. The same for a WAIT.
    size_t nextAction = 0;
    double rtBusyUntil = -1.0;    // no RT pass before this
    double restampMs = -1.0;      // RT_IN / WAIT lines logged during a deferred draw get this time
    std::vector<Token> rtHeld;    // drawn notes waiting for their recorded time
    double rtDeliverMs = -1.0;
    const double maxDrawMs = 400.0; // the recorded draw started at most this long before its note
    rt.holdOutput = [&](const Token &t) {
        if (rtDeliverMs <= simMs + 1e-9)
            return false;
        if (t.note < static_cast<int32_t>(Vocab::NoteOffset) || t.note >= static_cast<int32_t>(Vocab::Rest))
            return false; // clears from cuts go at once, as live
        rtHeld.push_back(t);
        return true;
    };
    for (auto *sampler: rt.samplersForReplay()) {
        sampler->drawHeld = [&] {
            return nextAction >= actions.size() || simMs + 1e-9 < actions[nextAction].ms - maxDrawMs;
        };
        sampler->forcedWait = [&, sampler](const AnticipationSampler::Context &) -> std::optional<bool> {
            if (nextAction < actions.size() && actions[nextAction].wait) {
                sampler->lastWaitProbability = actions[nextAction].waitProbability; // the WAIT line prints it
                rtBusyUntil = std::max(rtBusyUntil, actions[nextAction].ms);
                restampMs = std::max(simMs, actions[nextAction].ms);
                ++nextAction;
                return true;
            }
            return false;
        };
        sampler->forcedSample = [&](int field, const AnticipationSampler::Context &ctx) -> std::optional<int32_t> {
            if (nextAction >= actions.size() || actions[nextAction].wait) {
                if (report.problems.size() < 50)
                    report.problems.push_back("t=" + juce::String(simMs, 0) + ": the RT drew a note where the log has "
                                              + (nextAction >= actions.size() ? juce::String("none") : juce::String("a WAIT")));
                return std::nullopt;
            }
            const auto &n = actions[nextAction].note;
            switch (field) {
                case 0:
                    rtDeliverMs = actions[nextAction].ms;
                    restampMs = std::max(simMs, actions[nextAction].ms);
                    return n.onset - ctx.base;
                case 1: return static_cast<int32_t>(DenseVocab::DurOffset) + n.dur;
                case 2:
                    return static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * AnticipatedStream::kEventInstrument
                           + n.pitch;
                default:
                    ++nextAction;
                    return AnticipationSampler::kDenseVelocityBase + n.vel;
            }
        };
    }

    // The OT the same way: a step whose model call the recorded run answered later is held until then.
    std::vector<OrchestrationNote> otHeld;
    double otDeliverMs = -1.0;
    double otStepRelease = -1.0;
    bool otStepping = false;
    replayModel->onMatched = [&](double loggedMs) { otStepRelease = std::max(otStepRelease, loggedMs); };
    s.orchestrationTransformer.holdOutput = [&](const OrchestrationNote &n) {
        if (! otStepping && otDeliverMs <= simMs + 1e-9)
            return false;
        otHeld.push_back(n);
        return true;
    };

    // ---- devices and their roles, from the recorded MIDI_IN lines
    std::map<juce::String, juce::StringArray> rolesOf;
    for (const auto &l: log) {
        if (l.tag != "MIDI_IN")
            continue;
        const int open = l.text.indexOf(" [");
        const int close = l.text.indexOf(open, "] | ");
        if (open < 0 || close < 0)
            continue;
        juce::StringArray roles;
        roles.addTokens(l.text.substring(open + 2, close), ",", "");
        rolesOf[l.text.substring(0, open)] = roles;
    }
    for (const auto &[name, roles]: rolesOf) {
        if (roles.contains("piano"))
            s.selectedMidiInputIdentifier = name;
        if (roles.contains("piano2"))
            s.selectedAdditionalMidiInputIdentifier = name;
        if (roles.contains("pedal"))
            s.selectedPedalMidiInputIdentifier = name;
        if (roles.contains("launchpad"))
            s.selectedLaunchpadMidiIdentifier = name;
    }

    // Force restart (top button 0) restarts threads in the app; here the restart comes from the
    // log's own stream reset, so the button only logs.
    auto &grid = s.midiInputProcess.getLaunchpadGrid();
    auto appScene = grid.scene_callback;
    grid.scene_callback = [&, appScene](int idx) {
        if (idx == 0) {
            SessionLog::add("LAUNCHPAD", "Force restart");
            if (OpLog::on())
                OpLog::emit(OpLog::Src::Ui, "PAD_ACTION", {{"p", OpLog::id('p', OpLog::currentPad)}, {"action", "Force restart"}});
            return;
        }
        if (appScene)
            appScene(idx);
    };

    // ---- the event schedule
    enum class Kind { Input, Start, Stop, Mode };
    struct Event {
        double ms;
        Kind kind;
        const Line *line;
        int32_t clock;
        int order;
        size_t lineIndex = 0;
    };
    std::vector<Event> events;
    int order = 0;
    // Generation runs: every reduction stream reset starts one; it ends when the clock stops
    // (the first MIDI_OUT clear after the clock's last move) or at the next start.
    std::vector<size_t> starts;
    for (size_t i = 0; i < log.size(); ++i)
        if (log[i].tag == "RT_IN" && log[i].text.startsWith(rt.getThreadName()) && log[i].text.contains("stream reset at clock"))
            starts.push_back(i);
    for (size_t k = 0; k < starts.size(); ++k) {
        const size_t i0 = starts[k];
        const size_t i1 = k + 1 < starts.size() ? starts[k + 1] : log.size();
        const int32_t startClock = intAfter(log[i0].text, "stream reset at clock").value_or(0);
        events.push_back({log[i0].ms - 0.01, Kind::Start, &log[i0], startClock, order++, i0});
        double lastMove = log[i0].ms;
        int32_t prev = startClock;
        size_t lastMoveIdx = i0;
        for (size_t i = i0 + 1; i < i1; ++i) {
            if (log[i].clock >= 0 && log[i].clock > prev) {
                prev = log[i].clock;
                lastMove = log[i].ms;
                lastMoveIdx = i;
            }
        }
        bool frozen = false; // lines after the last move still show the same clock: generation stopped
        for (size_t i = lastMoveIdx + 1; i < i1; ++i)
            if (log[i].clock >= 0 && log[i].ms - lastMove > 200.0) {
                frozen = true;
                break;
            }
        bool cleared = false;
        for (size_t i = i0; i < i1; ++i)
            if (log[i].tag == "MIDI_OUT" && log[i].text.startsWith("clear"))
                cleared = true;
        if (frozen && ! cleared) // no Stop recorded (no clear line): stop when the clock did
            events.push_back({lastMove + 10.0, Kind::Stop, nullptr, 0, order++});
    }
    // Every Stop (and every Force restart's stop) logs a MIDI_OUT clear: replay it as a stop.
    for (const auto &l: log)
        if (l.tag == "MIDI_OUT" && l.text.startsWith("clear"))
            events.push_back({l.ms - 0.01, Kind::Stop, nullptr, 0, order++});
    // Inputs, and mode changes that did not come from a Launchpad mode pad.
    double lastModePadMs = -1e9;
    for (const auto &l: log) {
        if (l.tag == "MIDI_IN") {
            events.push_back({l.ms, Kind::Input, &l, 0, order++});
            const int open = l.text.indexOf(" [");
            if (open > 0 && rolesOf[l.text.substring(0, open)].contains("launchpad"))
                if (const auto m = midiFromHex(l.text.fromFirstOccurrenceOf("] | ", false, false)))
                    if (m->isNoteOn() && m->getNoteNumber() >= 81 && m->getNoteNumber() <= 83)
                        lastModePadMs = l.ms;
        } else if (l.tag == "MODE" && l.ms - lastModePadMs > 50.0) {
            events.push_back({l.ms, Kind::Mode, &l, 0, order++});
        }
    }
    std::stable_sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
        return a.ms != b.ms ? a.ms < b.ms : a.order < b.order;
    });

    // When the recorded RT took in a new key: every Clear Queue it pushed on input (a cut that no
    // CUT / HYBRID line explains). The app's input window flushes on the message thread, which
    // can run late; the replay holds a flush until the recorded RT had the note.
    std::vector<double> drainCuts;
    {
        double lastExplained = -1e9;
        for (const auto &l: log) {
            if (l.tag == "CUT" || l.tag == "HYBRID")
                lastExplained = l.ms;
            else if (l.tag == "RT" && l.text.startsWith(rt.getThreadName()) && l.text.contains("Clear Queue)")
                     && l.ms - lastExplained > 5.0) {
                const auto v = clearQueueTime(l.text);
                if (v.has_value() && l.clock >= 0 && *v >= l.clock)
                    drainCuts.push_back(l.ms);
            }
        }
    }
    // Moments a recorded RT pass began, from the lines a pass logs at its start: a hybrid switch
    // (or the Clear Queue its cut logged just before), a WAIT resume / timeout / reorder, and a
    // Clear Queue cut on input. While idle the live thread checks every 5 ms, at moments that depend
    // on when it woke; the replay runs its pass at each recorded moment instead, and skips a routine
    // check in the 6 ms before it.
    std::vector<double> passAnchors;
    {
        double lastClearMs = -1e9;
        for (const auto &l: log) {
            const bool rtLine = (l.tag == "RT" || l.tag == "WAIT") && l.text.startsWith(rt.getThreadName());
            if (l.tag == "RT" && rtLine && l.text.contains("Clear Queue)"))
                lastClearMs = l.ms;
            if (l.tag == "HYBRID") {
                passAnchors.push_back(l.ms - lastClearMs <= 2.0 ? lastClearMs : l.ms);
            } else if (l.tag == "WAIT" && rtLine) {
                const auto what = l.text.fromFirstOccurrenceOf("| ", false, false).trim();
                if (what.startsWith("resume") || what.startsWith("timeout") || what.startsWith("reorder"))
                    passAnchors.push_back(l.ms);
            }
        }
    }
    passAnchors.insert(passAnchors.end(), drainCuts.begin(), drainCuts.end());
    std::sort(passAnchors.begin(), passAnchors.end());
    size_t nextPassAnchor = 0;
    size_t nextDrainCut = 0;
    double flushHeldUntil = -1.0;
    const double maxFlushDelayMs = 250.0;

    // The app clock ticks when the machine's ms counter crosses a multiple of 10 ms, at a phase the
    // log's own ms do not show: clock = floor((ms + phase) / 10) - k, k fixed per run. The phase
    // is the one that explains the most recorded (ms, clock) pairs; the replay's simulated ms are
    // the log's ms + phase, so its clock ticks on the same boundaries.
    double phaseMs = 0.0;
    {
        int bestScore = -1;
        for (int tenth = 0; tenth < 100; ++tenth) {
            const double phase = tenth * 0.1;
            int score = 0;
            for (size_t k = 0; k < starts.size(); ++k) {
                const size_t i0 = starts[k];
                const size_t i1 = k + 1 < starts.size() ? starts[k + 1] : log.size();
                std::map<int64_t, int> offsets;
                for (size_t i = i0; i < i1 && i < i0 + 4000; ++i)
                    if (log[i].clock > 0)
                        ++offsets[static_cast<int64_t>(std::floor((log[i].ms + phase) / 10.0)) - log[i].clock];
                int best = 0;
                for (const auto &[off, n]: offsets)
                    best = std::max(best, n);
                score += best;
            }
            if (score > bestScore) {
                bestScore = score;
                phaseMs = phase;
            }
        }
    }
    report.clockPhaseMs = phaseMs;
    const auto applySimTime = [&] {
        AppTime::setSimulatedMs(simMs + phaseMs);
    };
    // k per run: the clock value the run's first lines show for their ms.
    std::map<size_t, int64_t> runK;
    for (size_t k = 0; k < starts.size(); ++k) {
        const size_t i0 = starts[k];
        const size_t i1 = k + 1 < starts.size() ? starts[k + 1] : log.size();
        std::map<int64_t, int> offsets;
        for (size_t i = i0; i < i1 && i < i0 + 4000; ++i)
            if (log[i].clock >= 0)
                ++offsets[static_cast<int64_t>(std::floor((log[i].ms + phaseMs) / 10.0)) - log[i].clock];
        int best = -1;
        for (const auto &[off, n]: offsets)
            if (n > best) {
                best = n;
                runK[i0] = off;
            }
    }

    // ---- capture what the replay logs
    SessionLog::setCapture([&](const char *kind, const juce::String &text) {
        const bool restamp = restampMs > simMs && (juce::String(kind) == "RT_IN" || juce::String(kind) == "WAIT");
        report.replayed.push_back({restamp ? restampMs : simMs, static_cast<int32_t>(s.clock.getTime()), kind, text});
    });

    // ---- run
    bool running = false;
    int64_t otLastClock = -1;       // OT / playback passes only when their input or the clock changed
    int64_t playbackLastClock = -1;
    double rtNextPassMs = 0.0;
    double timerNextMs = -1.0;
    const double endMs = options.untilMs >= 0.0 ? options.untilMs : log.back().ms;
    size_t ei = 0;

    const auto stopRun = [&] {
        rt.steppedBySimulator = false;
        s.stopGenerationForReplay(); // also when stopped: the app's Stop clears MIDI out either way
        running = false;
        // A draw in flight still finishes: Stop only asks the threads to exit, so the live RT
        // logged that note after the Stop (rtHeld is delivered at its time; nothing plays it).
        otHeld.clear();
        rtBusyUntil = otDeliverMs = -1.0;
    };
    const auto startRun = [&](int32_t clockValue, size_t lineIndex) {
        if (running)
            stopRun();
        rtHeld.clear();
        rtDeliverMs = -1.0;
        // The recorded clock at this run's boundaries: floor((ms + phase) / 10) - k.
        if (const auto it = runK.find(lineIndex); it != runK.end())
            clockValue = static_cast<int32_t>(std::floor((simMs + phaseMs) / 10.0) - static_cast<double>(it->second));
        s.startGenerationForReplay(static_cast<uint32_t>(std::max(0, clockValue)));
        rt.steppedBySimulator = true;
        rt.beginLive();
        s.orchestrationTransformer.beginSteps();
        s.outputPlayback.beginSteps();
        running = true;
        rtNextPassMs = simMs;
        otLastClock = playbackLastClock = -1;
        ++report.generationRuns;
    };

    // Discrete events at their exact times (sub-ms): inputs, the input window's timer, RT passes and
    // deliveries, OT deliveries. The OT and playback threads spin live, so they step after every
    // event and on every 1 ms tick.
    const auto otStep = [&] {
        if (! running)
            return;
        if (! otHeld.empty() && simMs + 1e-9 >= otDeliverMs) {
            auto held = std::move(otHeld);
            otHeld.clear();
            otDeliverMs = -1.0;
            for (const auto &n: held)
                s.orchestrationTransformer.deliverOutputNote(n);
        }
        if (! otHeld.empty())
            return; // busy: the recorded model had not answered yet
        // The live thread spins, but a pass can only do something new when one of its queues has
        // something or the clock has moved (hold-back release and the paused drop read the clock).
        auto &ot = s.orchestrationTransformer;
        const auto clockNow = static_cast<int64_t>(s.clock.getTime());
        const bool queued = ot.midiInputIncoming.getNumAvailableForReading() > 0
                            || ot.conditioningIncoming.getNumAvailableForReading() > 0
                            || ot.reductionIncoming.getNumAvailableForReading() > 0
                            || ot.instrumentUpdates.getNumAvailableForReading() > 0
                            || ot.updatesIncoming.getNumAvailableForReading() > 0
                            || ot.restampsIncoming.getNumAvailableForReading() > 0;
        if (! queued && clockNow == otLastClock)
            return;
        otLastClock = clockNow;
        otStepRelease = -1.0;
        otStepping = true;
        s.orchestrationTransformer.stepOnce();
        otStepping = false;
        if (! otHeld.empty()) {
            if (otStepRelease > simMs + 1e-9) {
                otDeliverMs = otStepRelease;
            } else {
                auto held = std::move(otHeld);
                otHeld.clear();
                for (const auto &n: held)
                    s.orchestrationTransformer.deliverOutputNote(n);
            }
        }
    };
    const auto rtFreeAt = [&]() -> double {
        return ! rtHeld.empty() ? rtDeliverMs : std::max({rtNextPassMs, rtBusyUntil, simMs});
    };
    // The recorded RT had a new key at time T: a Clear Queue cut on input, or a WAIT resume (the
    // soloist's group had arrived). The flush came before T, at whatever moment the message thread
    // ran the input window's timer. If the replay's own 10 ms ticks would not flush before T, one
    // extra tick runs just before it (it can only make a note available earlier, never later).
    std::vector<double> drainEvidence(drainCuts.begin(), drainCuts.end());
    for (const auto &l: log)
        if (l.tag == "WAIT" && l.text.startsWith(rt.getThreadName())
            && l.text.fromFirstOccurrenceOf("| ", false, false).trim().startsWith("resume"))
            drainEvidence.push_back(l.ms);
    std::sort(drainEvidence.begin(), drainEvidence.end());
    size_t nextEvidence = 0;
    size_t nextLateEvidence = 0;
    double extraFlushDoneFor = -1.0; // at most one extra flush per piece of evidence
    const auto earlyFlushBefore = [&]() -> double {
        if (timerNextMs < 0.0)
            return -1.0;
        while (nextEvidence < drainEvidence.size() && drainEvidence[nextEvidence] < simMs + 0.5)
            ++nextEvidence;
        size_t k = nextEvidence;
        while (k < drainEvidence.size() && drainEvidence[k] <= extraFlushDoneFor)
            ++k;
        if (k >= drainEvidence.size())
            return -1.0;
        const double t = drainEvidence[k];
        if (t - simMs > maxFlushDelayMs || timerNextMs <= t - 0.5)
            return -1.0;
        return t - 0.5;
    };
    double extraFlushMs = -1.0;
    const auto spin = [&] {
        otStep();
        if (running) {
            // Playback reads the clock and its queues; a pass with neither changed does nothing.
            const auto clockNow = static_cast<int64_t>(s.clock.getTime());
            const bool queued = s.orchestrationTransformer.outputTokenQueue.getNumAvailableForReading() > 0
                                || s.orchestrationTransformer.outputDurationUpdates.getNumAvailableForReading() > 0
                                || s.outputPlayback.pedalClearRequests.getNumAvailableForReading() > 0
                                || s.outputPlayback.modeCutRequests.getNumAvailableForReading() > 0
                                || (s.outputPlayback.reductionOutputQueue != nullptr
                                    && s.outputPlayback.reductionOutputQueue->getNumAvailableForReading() > 0);
            if (queued || clockNow != playbackLastClock) {
                playbackLastClock = clockNow;
                s.outputPlayback.stepOnce();
            }
        }
        if (s.midiInputProcess.isTimerRunning()) {
            if (timerNextMs < 0.0)
                timerNextMs = simMs + 10.0; // started by what just happened
        } else {
            timerNextMs = -1.0;
        }
        if (extraFlushMs < simMs - 1e-9)
            extraFlushMs = earlyFlushBefore();
    };
    const auto rtPassDue = [&]() -> double {
        if (! running || ! rtHeld.empty())
            return 1e300;
        double due = std::max(rtNextPassMs, rtBusyUntil);
        while (nextPassAnchor < passAnchors.size() && passAnchors[nextPassAnchor] < simMs - 1e-6)
            ++nextPassAnchor;
        if (nextPassAnchor < passAnchors.size()) {
            const double a = passAnchors[nextPassAnchor];
            if (a > due && a <= due + 6.0)
                due = a; // the recorded pass came then (see passAnchors)
        }
        return due;
    };

    double tick = std::floor(log.front().ms);
    simMs = tick;
    while (tick <= endMs + 1e-9) {
        int actionsAtThisTick = 0;
        for (;;) {
            enum class Next { None, Event, Timer, ExtraFlush, RtDeliver, OtDeliver, RtPass } which = Next::None;
            double next = 1e300;
            const auto consider = [&](double t, Next w) {
                if (t < next) {
                    next = t;
                    which = w;
                }
            };
            if (ei < events.size())
                consider(events[ei].ms, Next::Event);
            if (timerNextMs >= 0.0)
                consider(timerNextMs, Next::Timer);
            if (extraFlushMs >= 0.0 && timerNextMs >= 0.0)
                consider(extraFlushMs, Next::ExtraFlush);
            if (! rtHeld.empty())
                consider(rtDeliverMs, Next::RtDeliver);
            if (running && ! otHeld.empty())
                consider(otDeliverMs, Next::OtDeliver);
            consider(rtPassDue(), Next::RtPass);
            if (which == Next::None || next > tick + 1e-9)
                break;
            if (++actionsAtThisTick > 200000) {
                report.problems.push_back("replay stalled at " + juce::String(simMs, 1) + " ms");
                break;
            }
            simMs = std::max(simMs, next);
            applySimTime();
            switch (which) {
                case Next::Event: {
                    const auto &e = events[ei++];
                    switch (e.kind) {
                        case Kind::Start: startRun(e.clock, e.lineIndex); break;
                        case Kind::Stop: stopRun(); break;
                        case Kind::Mode: {
                            const auto m = e.line->text.trim();
                            s.setOrchestrationMode(m == "Edit" || m == "Step-in" ? OrchestrationMode::Edit
                                                   : m == "Accompaniment" ? OrchestrationMode::Accompaniment
                                                                          : OrchestrationMode::Jam);
                            break;
                        }
                        case Kind::Input: {
                            const int open = e.line->text.indexOf(" [");
                            if (open <= 0)
                                break;
                            const auto name = e.line->text.substring(0, open);
                            if (const auto m = midiFromHex(e.line->text.fromFirstOccurrenceOf("] | ", false, false)))
                                s.midiInputProcess.handleMidiFrom(name, name, *m);
                            break;
                        }
                    }
                    break;
                }
                case Next::Timer: {
                    if (! s.midiInputProcess.isTimerRunning()) {
                        timerNextMs = -1.0;
                        break;
                    }
                    // Evidence of when the recorded RT took a new key in: input cuts and WAIT resumes.
                    while (nextLateEvidence < drainEvidence.size() && drainEvidence[nextLateEvidence] < simMs - 1.0)
                        ++nextLateEvidence;
                    // Hold the flush only when the RT would have taken the note in before the recorded
                    // moment; when it is busy until then, the recorded delay is the RT's, not the timer's.
                    // ... and only when no other key went down before the cut: then the cut is that
                    // key's, and this flush kept its normal time.
                    const auto keyDownBefore = [&](double until) {
                        for (size_t k = ei; k < events.size() && events[k].ms < until; ++k)
                            if (events[k].kind == Kind::Input && events[k].line->text.contains("(Note on"))
                                return true;
                        return false;
                    };
                    const double evidence =
                        nextLateEvidence < drainEvidence.size() ? drainEvidence[nextLateEvidence] : 1e300;
                    const bool late = flushHeldUntil < 0.0 && evidence < 1e299
                                      && evidence - 1.0 > simMs + 1e-9
                                      && evidence - simMs <= maxFlushDelayMs
                                      && rtFreeAt() < evidence - 1.5
                                      && ! keyDownBefore(evidence);
                    if (late) {
                        flushHeldUntil = evidence;
                        timerNextMs = flushHeldUntil - 1.0; // the recorded flush came then
                        break;
                    }
                    s.midiInputProcess.timerCallback();
                    if (flushHeldUntil >= 0.0) {
                        rtNextPassMs = std::min(std::max(rtNextPassMs, simMs), flushHeldUntil); // the RT took it in then
                        flushHeldUntil = -1.0;
                    }
                    timerNextMs = s.midiInputProcess.isTimerRunning() ? timerNextMs + 10.0 : -1.0;
                    break;
                }
                case Next::ExtraFlush:
                    extraFlushDoneFor = extraFlushMs + 0.5;
                    extraFlushMs = -1.0;
                    if (s.midiInputProcess.isTimerRunning())
                        s.midiInputProcess.timerCallback(); // the regular ticks keep their schedule
                    break;
                case Next::RtDeliver: {
                    auto held = std::move(rtHeld);
                    rtHeld.clear();
                    rtDeliverMs = -1.0;
                    for (const auto &t: held) {
                        rt.deliverOutputToken(t);
                        rt.afterOutputDelivered(t);
                    }
                    rtNextPassMs = std::max(rtNextPassMs, simMs); // the next pass follows at once
                    break;
                }
                case Next::OtDeliver:
                    break; // otStep (in spin) delivers it
                case Next::RtPass: {
                    const auto result = rt.liveStep();
                    restampMs = -1.0;
                    const bool idle = result == TwoInstrumentAnticipatedReductionTransformer::LiveStep::Idle
                                      || result == TwoInstrumentAnticipatedReductionTransformer::LiveStep::NothingGenerated;
                    if (! rtHeld.empty() || rtBusyUntil > simMs)
                        rtNextPassMs = simMs; // busy: the delivery / busy time gates the next pass
                    else
                        rtNextPassMs = idle ? simMs + 5.0 : simMs + 0.01;
                    break;
                }
                case Next::None: break;
            }
            spin();
        }
        simMs = std::max(simMs, tick);
        applySimTime();
        spin();
        // Between events nothing changes but the app clock, so the next tick is the next clock tick
        // (a multiple of 10 ms on the shifted ms); with generation stopped, the next event.
        const double nextClockTick = (std::floor((tick + phaseMs) / 10.0) + 1.0) * 10.0 - phaseMs;
        double nextTick = running ? std::min(tick + options.stepMs * 10.0, nextClockTick) : 1e300;
        if (ei < events.size())
            nextTick = std::min(nextTick, std::ceil(events[ei].ms));
        if (timerNextMs >= 0.0)
            nextTick = std::min(nextTick, std::ceil(timerNextMs));
        if (! running && nextTick >= 1e299)
            nextTick = endMs + 1.0;
        tick = std::max(tick + 1e-3, nextTick);
    }
    // The log ends here; the recorded session did not necessarily stop (no closing clear).
    rt.steppedBySimulator = false;
    SessionLog::setCapture(nullptr);

    if (nextAction < actions.size())
        report.problems.push_back(juce::String(static_cast<int>(actions.size() - nextAction))
                                  + " recorded RT draws / WAITs were never used (first at "
                                  + juce::String(actions[nextAction].ms, 1) + " ms)");
    int unusedOt = 0;
    for (const auto &l: replayModel->logged)
        unusedOt += l.used ? 0 : 1;
    if (unusedOt > 0)
        report.problems.push_back(juce::String(unusedOt) + " recorded OT notes were never asked for");

    // ---- compare
    std::map<juce::String, std::vector<const Line *>> rec, rep;
    const double firstMs = log.front().ms;
    for (const auto &l: log)
        if (kComparedTags.contains(l.tag) && (options.untilMs < 0.0 || l.ms <= options.untilMs) && l.ms >= firstMs)
            rec[l.tag].push_back(&l);
    // A layout-1 session was logged with the texts of its time: compare the replay's in those words.
    std::vector<Line> replayedAsRecorded;
    replayedAsRecorded.reserve(report.replayed.size());
    for (const auto &l: report.replayed)
        replayedAsRecorded.push_back(layout1 ? withLayout1Text(l) : l);
    for (const auto &l: replayedAsRecorded)
        if (kComparedTags.contains(l.tag))
            rep[l.tag].push_back(&l);
    // RT lines are compared without their voice id: voices come from the vocsep model, which may not
    // be the one the recorded run used. "RT voice" compares them separately.
    std::vector<Line> recRt, repRt;
    const auto stripVoice = [](const Line *l) {
        Line c = *l;
        const int v = c.text.indexOf(") voice ");
        if (v >= 0)
            c.text = c.text.substring(0, v + 1) + c.text.fromFirstOccurrenceOf(" noteId", true, false);
        return c;
    };
    for (const auto *l: rec["RT"])
        recRt.push_back(stripVoice(l));
    for (const auto *l: rep["RT"])
        repRt.push_back(stripVoice(l));
    std::vector<const Line *> recRtPtr, repRtPtr;
    for (const auto &l: recRt)
        recRtPtr.push_back(&l);
    for (const auto &l: repRt)
        repRtPtr.push_back(&l);
    for (const auto &tag: kComparedTags)
        report.tags[tag] = tag == "RT" ? compareTag(recRtPtr, repRtPtr, options.matchToleranceMs, options.keepDifferences)
                                       : compareTag(rec[tag], rep[tag], options.matchToleranceMs, options.keepDifferences);
    if (options.voiceSeparation)
        report.tags["RT voice"] = compareTag(rec["RT"], rep["RT"], options.matchToleranceMs, 0);

    s.orchestrationTransformer.orchestrationModel = s.orchestrationModel.get();
    replayModel->real = nullptr;
    return report;
}

} // namespace LogReplay

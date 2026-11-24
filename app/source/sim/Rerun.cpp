#include "VirtualOrch/sim/Rerun.h"

#include <algorithm>
#include <cmath>

#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/OpLogDraw.h"
#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/orchestration-models/IodPretrained.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

namespace Rerun {

namespace {

auto approxEqual(double a, double b) -> bool { return std::abs(a - b) <= 1e-3 + 0.02 * std::max(a, b); }

/** The field's most probable tokens by `p` (absolute ids). */
auto topOf(const std::vector<float> &p, int lo, int hi, int count, int offset = 0) -> std::vector<std::pair<int, double>> {
    std::vector<std::pair<int, double>> all;
    for (int t = lo; t < hi && t - offset < static_cast<int>(p.size()); ++t)
        all.push_back({t, p[static_cast<size_t>(t - offset)]});
    std::partial_sort(all.begin(), all.begin() + std::min<size_t>(all.size(), static_cast<size_t>(count)), all.end(),
                      [](const auto &a, const auto &b) { return a.second > b.second; });
    all.resize(std::min<size_t>(all.size(), static_cast<size_t>(count)));
    return all;
}

/** RT field token ranges in the dense vocab. */
auto rtRange(const std::string &field) -> std::pair<int, int> {
    if (field == "onset")
        return {static_cast<int>(DenseVocab::TimeOffset), static_cast<int>(DenseVocab::TimeOffset) + DenseConfig::MaxTime};
    if (field == "dur")
        return {static_cast<int>(DenseVocab::DurOffset), static_cast<int>(DenseVocab::DurOffset) + DenseConfig::MaxDur};
    if (field == "pitch")
        return {static_cast<int>(DenseVocab::NoteOffset), static_cast<int>(DenseVocab::NoteOffset) + DenseConfig::MaxNote};
    return {static_cast<int>(DenseVocab::VelocityOffset), static_cast<int>(DenseVocab::VelocityOffset) + DenseConfig::MaxVelocity};
}

/** OT step token ranges (absolute ids). */
auto otRange(const std::string &field) -> std::pair<int, int> {
    using namespace IodPretrainedTypes;
    static constexpr const char *kFamilies[] = {"strings", "woodwinds", "brass", "other"};
    for (int f = 0; f < numFamilies; ++f)
        if (field == kFamilies[f]) {
            const auto &spec = familySpecs[static_cast<size_t>(f)];
            return {spec.tokenOffset, spec.tokenOffset + spec.tokenSize};
        }
    return {groupsOffset, groupsHi};
}

auto otLogitsIndex(int noteIdx, int step) -> size_t {
    using namespace IodPretrainedTypes;
    return (static_cast<size_t>(noteIdx) * static_cast<size_t>(comboLen) + static_cast<size_t>(step))
           * static_cast<size_t>(comboVocabSize);
}

} // namespace

auto Result::matchesLog() const -> bool {
    if (! error.empty() || fields.empty())
        return false;
    for (const auto &f: fields)
        if (f.chosen >= 0 && ! approxEqual(f.loggedPModel, f.rerunPModel))
            return false;
    return true;
}

auto defaultModelFile(const std::string &tag) -> juce::File {
    if (tag == "v1" || tag == "T3")
        return modelsDir().getChildFile("rt_continuation_legacy.onnx");
    if (tag == "W")
        return modelsDir().getChildFile("rt_simultaneous_legacy.onnx"); // W3 (the retired hybrid_v1_w3)
    return modelsDir().getChildFile("ot_legacy.onnx");
}

auto rerunsFileFor(const juce::File &oplog) -> juce::File {
    auto name = oplog.getFileName();
    if (name.endsWith(".jsonl.gz"))
        name = name.dropLastCharacters(9);
    return oplog.getSiblingFile(name + ".reruns.jsonl");
}

auto toJson(const Result &r, const juce::String &source) -> std::string {
    std::string fields = "[";
    for (size_t i = 0; i < r.fields.size(); ++i) {
        const auto &f = r.fields[i];
        std::string top = "[";
        for (size_t k = 0; k < f.top.size(); ++k)
            top += (k ? "," : "") + std::string("[") + std::to_string(f.top[k].first) + ","
                   + OpLog::valueJson(f.top[k].second) + "]";
        top += "]";
        OpLog::Obj o;
        o.add("field", f.name).add("lo", static_cast<int64_t>(f.lo)).add("hi", static_cast<int64_t>(f.hi))
            .add("chosen", static_cast<int64_t>(f.chosen)).add("p_model_logged", f.loggedPModel)
            .add("p_model_rerun", f.rerunPModel).add("top", OpLog::Raw{top});
        fields += (i ? "," : "") + o.str();
    }
    fields += "]";
    OpLog::Obj o;
    o.add("source", source.toStdString()).add("seq", r.seq).add("t", r.t).add("what", r.what).add("model", r.model)
        .add("model_file", r.modelFile).add("context_tokens", static_cast<int64_t>(r.contextTokens))
        .add("logged_context_tokens", static_cast<int64_t>(r.loggedContextTokens)).add("exact_context", r.exactContext)
        .add("matches_log", r.matchesLog()).add("fields", OpLog::Raw{fields});
    if (! r.error.empty())
        o.add("error", r.error);
    return o.str();
}

auto append(const juce::File &file, const Result &r, const juce::String &source) -> bool {
    juce::FileOutputStream out(file); // appends
    if (! out.openedOk())
        return false;
    const auto line = toJson(r, source) + "\n";
    return out.write(line.data(), line.size());
}

Engine::Engine() = default;
Engine::~Engine() = default;

auto Engine::rtBackend(const std::string &tag, Result &r) -> StreamingKv::OnnxBackend * {
    const auto file = modelFiles.count(tag) != 0 ? modelFiles[tag] : defaultModelFile(tag);
    r.modelFile = file.getFullPathName().toStdString();
    auto &slot = rt[tag];
    if (slot != nullptr && rtFile[tag] == r.modelFile)
        return slot.get();
    if (! file.existsAsFile()) {
        r.error = "model file not found: " + r.modelFile;
        return nullptr;
    }
    slot = std::make_unique<StreamingKv::OnnxBackend>();
    std::string error;
    if (! slot->init(r.modelFile.c_str(), error)) {
        r.error = "cannot load " + r.modelFile + ": " + error;
        slot.reset();
        return nullptr;
    }
    rtFile[tag] = r.modelFile;
    return slot.get();
}

auto Engine::otModel(Result &r) -> IodPretrained * {
    const auto file = modelFiles.count("OT") != 0 ? modelFiles["OT"] : defaultModelFile("OT");
    r.modelFile = file.getFullPathName().toStdString();
    if (ot != nullptr && otFile == r.modelFile)
        return ot.get();
    if (! file.existsAsFile()) {
        r.error = "model file not found: " + r.modelFile;
        return nullptr;
    }
    ot = std::make_unique<IodPretrained>();
    ot->init(r.modelFile.c_str());
    otFile = r.modelFile;
    return ot.get();
}

auto Engine::rerun(const OpLogReader::Log &log, size_t index) -> Result {
    Result r;
    if (index >= log.records().size()) {
        r.error = "no such record";
        return r;
    }
    r.t = log.records()[index].t;
    r.seq = log.records()[index].seq;
    const auto &op = log.records()[index].op;
    if (op == "RT_DRAW")
        rerunRt(log, index, r);
    else if (op == "OT_DRAW")
        rerunOt(log, index, r);
    else
        r.error = "only RT_DRAW and OT_DRAW records can be rerun";
    return r;
}

auto Engine::rerunRt(const OpLogReader::Log &log, size_t index, Result &r) -> void {
    const auto f = log.fields(index);
    r.model = f["model"].toString().toStdString();
    r.what = "RT_DRAW " + f["r"].toString().toStdString() + " (" + r.model + ")";
    r.loggedContextTokens = static_cast<int>(f["context_len"]);
    // The context when it drew. The pass's context changes are logged after the draw, in stream
    // order: everything just before the drawn note's own RT_ADD (given notes the same pass took
    // in come first), else this record's state.
    size_t at = index;
    const auto drawnId = f["r"].toString().toStdString();
    for (size_t i = index + 1; i < log.records().size() && i < index + 400; ++i) {
        const auto &op = log.records()[i].op;
        if (op == "RT_DRAW" || op == "RUN_STOP")
            break;
        if (op == "RT_ADD") {
            const auto line = log.line(i);
            if (line.find("\"id\":\"" + drawnId + "\"") != std::string::npos
                && line.find("\"model\":\"" + r.model + "\"") != std::string::npos) {
                at = i - 1;
                break;
            }
        }
    }
    const auto st = OpLogReader::stateAt(log, at);
    const auto it = st.rt.find(r.model);
    if (it == st.rt.end()) {
        r.error = "no context for model " + r.model + " (no checkpoint before this draw in the session)";
        return;
    }
    auto ids = it->second.tokens();
    r.contextTokens = static_cast<int>(ids.size());
    r.exactContext = it->second.exact && r.contextTokens == r.loggedContextTokens;
    auto *backend = rtBackend(r.model, r);
    if (backend == nullptr)
        return;
    const auto rows = backend->dims().maxRows;
    if (static_cast<int>(ids.size()) > rows) // the cache holds the sink rows and the newest ones
        ids.erase(ids.begin() + backend->dims().sinkRows, ids.end() - (rows - backend->dims().sinkRows));
    StreamingKv::State past, present;
    std::vector<float> logits;
    std::vector<int64_t> feed(ids.begin(), ids.end());
    if (! backend->run(past, feed, present, logits)) {
        r.error = "the model failed: " + backend->lastError;
        return;
    }
    if (const auto *drawn = f["fields"].getArray())
        for (int i = 0; i < drawn->size(); ++i) {
            const auto &d = (*drawn)[i];
            Field field;
            field.name = d["field"].toString().toStdString();
            const auto [lo, hi] = rtRange(field.name);
            field.lo = lo;
            field.hi = hi;
            field.chosen = static_cast<int>(d["chosen"]);
            field.loggedPModel = static_cast<double>(d["p_model"]);
            const auto p = OpLogDraw::softmax(logits.data(), logits.size());
            if (field.chosen >= 0 && field.chosen < static_cast<int>(p.size()))
                field.rerunPModel = p[static_cast<size_t>(field.chosen)];
            field.top = topOf(p, lo, hi, topCount);
            r.fields.push_back(std::move(field));
            if (i + 1 < drawn->size()) { // feed the logged choice, as the draw did
                past = std::move(present);
                present = {};
                if (! backend->run(past, {static_cast<int64_t>(r.fields.back().chosen)}, present, logits)) {
                    r.error = "the model failed: " + backend->lastError;
                    return;
                }
            }
        }
}

auto Engine::rerunOt(const OpLogReader::Log &log, size_t index, Result &r) -> void {
    using namespace IodPretrainedTypes;
    const auto f = log.fields(index);
    r.model = "OT";
    const auto forId = f["for"].toString();
    r.what = "OT_DRAW for " + forId.toStdString();
    const auto st = OpLogReader::stateAt(log, index);
    if (! st.otInput.has_value()) {
        r.error = "no OT_INPUT before this draw (recorded before OT inputs were logged)";
        return;
    }
    const auto in = log.fields(*st.otInput);
    const auto *notes = in["notes"].getArray();
    const auto *inputIds = in["input_ids"].getArray();
    const auto *combos = in["combo_in"].getArray();
    if (notes == nullptr || inputIds == nullptr || combos == nullptr) {
        r.error = "OT_INPUT incomplete";
        return;
    }
    EncoderWindow window;
    for (auto &row: window.comboIn)
        row.fill(bos);
    window.nNotes = std::min(static_cast<int>(notes->size()), nNotesMax);
    int noteIdx = -1;
    for (int i = 0; i < window.nNotes; ++i) {
        for (int k = 0; k < noteTokensPerNote; ++k) {
            const auto at = static_cast<size_t>(i * noteTokensPerNote + k);
            window.inputIds[at] = static_cast<int64_t>(static_cast<juce::int64>((*inputIds)[static_cast<int>(at)]));
            window.attentionMask[at] = 1;
        }
        window.noteValid[static_cast<size_t>(i)] = 1;
        for (int k = 0; k < comboLen; ++k)
            window.comboIn[static_cast<size_t>(i)][static_cast<size_t>(k)] =
                static_cast<int64_t>(static_cast<juce::int64>((*combos)[i * comboLen + k]));
        if ((*notes)[i].toString() == forId)
            noteIdx = i;
    }
    if (noteIdx < 0) {
        r.error = "the draw's note is not in the encoder window";
        return;
    }
    // Combos drawn earlier in this window are teacher-forced in, as the draw saw them.
    for (const auto d: st.otDrawsSinceInput) {
        if (d >= index)
            break;
        const auto df = log.fields(d);
        const auto id = df["for"].toString();
        for (int i = 0; i < window.nNotes; ++i)
            if ((*notes)[i].toString() == id)
                if (const auto *c = df["combo"].getArray()) {
                    auto &row = window.comboIn[static_cast<size_t>(i)];
                    row.fill(bos);
                    for (int s = 0; s + 1 < comboLen && s < c->size(); ++s)
                        row[static_cast<size_t>(s + 1)] = static_cast<int64_t>(static_cast<juce::int64>((*c)[s]));
                }
    }
    auto *model = otModel(r);
    if (model == nullptr)
        return;
    auto &target = window.comboIn[static_cast<size_t>(noteIdx)];
    target.fill(bos);
    if (const auto *steps = f["steps"].getArray())
        for (int step = 0; step < steps->size() && step < comboLen; ++step) {
            const auto &d = (*steps)[step];
            Field field;
            field.name = d["field"].toString().toStdString();
            const auto [lo, hi] = otRange(field.name);
            field.lo = lo;
            field.hi = hi;
            const int chosenRel = static_cast<int>(d["chosen"]);
            field.chosen = chosenRel >= 0 ? lo + chosenRel : -1;
            field.loggedPModel = static_cast<double>(d["p_model"]);
            const auto logits = model->runLogits(window);
            const auto at = otLogitsIndex(noteIdx, step);
            if (logits.size() < at + static_cast<size_t>(hi)) {
                r.error = "the OT model returned too few logits";
                return;
            }
            // p_model over the field's slice, as logged (OpLogDraw::describe)
            const auto p = OpLogDraw::softmax(logits.data() + at + lo, static_cast<size_t>(hi - lo));
            if (chosenRel >= 0 && chosenRel < static_cast<int>(p.size()))
                field.rerunPModel = p[static_cast<size_t>(chosenRel)];
            field.top = topOf(p, lo, hi, topCount, lo);
            r.fields.push_back(std::move(field));
            if (step + 1 < comboLen && chosenRel >= 0)
                target[static_cast<size_t>(step + 1)] = lo + chosenRel; // teacher-force the logged choice
        }
}

} // namespace Rerun

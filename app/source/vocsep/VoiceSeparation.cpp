#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/vocsep/VocsepGraph.h"

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace {

auto nowMs() -> double {
    return juce::Time::getMillisecondCounterHiRes();
}

constexpr int64_t kAssignLogMaxBytes = 2 * 1024 * 1024;
/** Onsets within this are treated as concurrent for conflict diagnostics (matches NearConsecutiveTolCs). */
constexpr int kConcurrentOnsetTolCs = Vocsep::NearConsecutiveTolCs;

auto vocsepAssignLogFile() -> juce::File {
    const auto dir = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                         .getChildFile("virtual-orch")
                         .getChildFile("Logs");
    dir.createDirectory();
    return dir.getChildFile("vocsep_assign.log");
}

auto appendAssignLogAsync(juce::String body) -> void {
    juce::Thread::launch([body = std::move(body)] {
        auto file = vocsepAssignLogFile();
        if (file.getSize() > kAssignLogMaxBytes) {
            const auto rotated = file.getSiblingFile("vocsep_assign.prev.log");
            rotated.deleteFile();
            file.moveFileTo(rotated);
        }
        file.appendText(body);
        if (! body.endsWithChar('\n'))
            file.appendText("\n");
    });
}

auto formatNoteBrief(const Token &t) -> juce::String {
    return juce::String::formatted("onset=%d dur=%d pitch=%d voice=%d",
                                   static_cast<int>(t.time),
                                   static_cast<int>(t.getRealDuration()),
                                   static_cast<int>(t.getPitch()),
                                   static_cast<int>(t.voiceId));
}

/**
 * Log one voice assignment. `parentScore` is NaN when unused.
 * Lists concurrent window notes (onset within tol) and whether they share the assigned voice.
 */
auto logVoiceAssignment(const Token &token,
                        const char *reason,
                        const std::vector<Token> &window,
                        int parentIdx,
                        float parentScore,
                        const Vocsep::GraphTensors *graph,
                        const std::vector<float> *edgeScores) -> void {
    juce::String line;
    line << juce::Time::getCurrentTime().toString(true, true, true, true)
         << " | " << reason
         << " | new{" << formatNoteBrief(token) << "}";

    if (parentIdx >= 0 && parentIdx < static_cast<int>(window.size())) {
        line << " | parentIdx=" << parentIdx
             << " parent{" << formatNoteBrief(window[static_cast<size_t>(parentIdx)]) << "}";
        if (std::isfinite(parentScore))
            line << " score=" << juce::String(parentScore, 4);
    } else {
        line << " | parentIdx=-1";
    }

    // Pot edges into the newest node (candidate parents).
    const int newest = static_cast<int>(window.size()) - 1;
    if (graph != nullptr && edgeScores != nullptr && newest >= 0) {
        line << " | potsIntoNew=";
        bool any = false;
        for (int64_t e = 0; e < graph->targetEdgeCount; ++e) {
            const int d = static_cast<int>(
                graph->targetEdgeIndex[static_cast<size_t>(graph->targetEdgeCount + e)]);
            if (d != newest)
                continue;
            const int s =
                static_cast<int>(graph->targetEdgeIndex[static_cast<size_t>(e)]);
            const float sc =
                e < static_cast<int64_t>(edgeScores->size())
                    ? (*edgeScores)[static_cast<size_t>(e)]
                    : 0.0f;
            if (any)
                line << ",";
            line << s << "->" << d << "@" << juce::String(sc, 3);
            any = true;
        }
        if (! any)
            line << "(none)";
    }

    // Concurrent notes in window (excluding newest).
    line << " | concurrent=";
    bool anyConc = false;
    bool sameVoiceConflict = false;
    for (int i = 0; i < newest; ++i) {
        const auto &o = window[static_cast<size_t>(i)];
        if (std::abs(o.time - token.time) > kConcurrentOnsetTolCs)
            continue;
        if (anyConc)
            line << ";";
        line << "[" << i << " " << formatNoteBrief(o) << "]";
        anyConc = true;
        if (o.voiceId >= 0 && o.voiceId == token.voiceId)
            sameVoiceConflict = true;
    }
    if (! anyConc)
        line << "(none)";
    if (sameVoiceConflict)
        line << " | CONFLICT_SAME_VOICE_CONCURRENT";

    appendAssignLogAsync(std::move(line));
}

auto potEdgeScore(const Vocsep::GraphTensors &graph,
                  const std::vector<float> &edgeScores,
                  int src,
                  int dst) -> float {
    for (int64_t e = 0; e < graph.targetEdgeCount; ++e) {
        const int s = static_cast<int>(graph.targetEdgeIndex[static_cast<size_t>(e)]);
        const int d = static_cast<int>(
            graph.targetEdgeIndex[static_cast<size_t>(graph.targetEdgeCount + e)]);
        if (s == src && d == dst && e < static_cast<int64_t>(edgeScores.size()))
            return edgeScores[static_cast<size_t>(e)];
    }
    return -std::numeric_limits<float>::infinity();
}

} // namespace

auto VoiceSeparation::init(const char *modelPath) -> void {
    Ort::SessionOptions sessionOptions;
#ifdef __linux__
    OrtCUDAProviderOptions cudaOptions{};
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
#endif
    session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, sessionOptions);

    allocatedInputNames.clear();
    allocatedOutputNames.clear();
    const Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i)
        allocatedInputNames.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session->GetOutputCount(); ++i)
        allocatedOutputNames.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
}

auto VoiceSeparation::reset() -> void {
    nextVoiceId = 0;
    const juce::ScopedLock lock(profileLock);
    profile = {};
}

auto VoiceSeparation::isSoundingNote(const Token &token) -> bool {
    if (token.note == static_cast<int32_t>(Vocab::ClearQueue)
        || token.note == static_cast<int32_t>(Vocab::BarSeparator)
        || token.note == static_cast<int32_t>(Vocab::Rest))
        return false;
    return token.note >= static_cast<int32_t>(Vocab::NoteOffset)
           && token.note < static_cast<int32_t>(Vocab::Rest);
}

auto VoiceSeparation::getLastProfile() const -> VocsepLoopProfile {
    const juce::ScopedLock lock(profileLock);
    return profile;
}

auto VoiceSeparation::stampVoiceId(Token &token, const std::vector<Token> &outputHistory)
    -> std::vector<HistoryRestamp> {
    std::vector<HistoryRestamp> restamps;
    if (session == nullptr || ! isSoundingNote(token))
        return restamps;

    const double t0 = nowMs();

    // Sounding history indices aligned with the pre-newest window prefix.
    std::vector<size_t> soundingHistoryIdx;
    soundingHistoryIdx.reserve(outputHistory.size());
    std::vector<Token> window;
    window.reserve(static_cast<size_t>(WindowNotes));
    for (size_t hi = 0; hi < outputHistory.size(); ++hi) {
        if (! isSoundingNote(outputHistory[hi]))
            continue;
        soundingHistoryIdx.push_back(hi);
        window.push_back(outputHistory[hi]);
    }
    if (static_cast<int>(window.size()) >= WindowNotes) {
        const auto drop = static_cast<size_t>(window.size() - (WindowNotes - 1));
        window.erase(window.begin(), window.begin() + static_cast<std::ptrdiff_t>(drop));
        soundingHistoryIdx.erase(soundingHistoryIdx.begin(),
                                 soundingHistoryIdx.begin() + static_cast<std::ptrdiff_t>(drop));
    }
    window.push_back(token);

    VocsepLoopStepMs step;
    step.nodes = static_cast<int>(window.size());

    std::vector<Vocsep::Note> notes;
    notes.reserve(window.size());
    for (const auto &t: window)
        notes.push_back(Vocsep::noteFromToken(t));

    const double tGraph0 = nowMs();
    auto graph = Vocsep::buildGraph(notes);
    step.graph = static_cast<float>(nowMs() - tGraph0);
    step.edges = static_cast<int>(graph.targetEdgeCount);

    auto finishEarly = [&](const char *reason) {
        token.voiceId = nextVoiceId++;
        window.back().voiceId = token.voiceId;
        logVoiceAssignment(token, reason, window, -1, NAN, &graph, nullptr);
        step.total = static_cast<float>(nowMs() - t0);
        const juce::ScopedLock lock(profileLock);
        profile.publishBusy(step);
    };

    if (graph.numNodes <= 0) {
        finishEarly("new_empty_graph");
        return restamps;
    }

    if (graph.targetEdgeCount <= 0 || graph.numNodes == 1) {
        finishEarly(graph.numNodes == 1 ? "new_singleton" : "new_no_pots");
        return restamps;
    }

    const double tOnnx0 = nowMs();
    std::vector<float> edgeScores;
    try {
        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const std::array<int64_t, 2> tgtShape{2, graph.targetEdgeCount};
        const std::array<int64_t, 2> xShape{graph.numNodes, Vocsep::InFeats};
        const std::array<int64_t, 2> embShape{2, graph.embedEdgeCount};
        const std::array<int64_t, 1> etShape{graph.embedEdgeCount};
        const std::array<int64_t, 2> psShape{graph.targetEdgeCount, 1};
        const std::array<int64_t, 2> osShape{graph.targetEdgeCount, 2};

        auto tTarget = Ort::Value::CreateTensor<int64_t>(
            mem, graph.targetEdgeIndex.data(), graph.targetEdgeIndex.size(), tgtShape.data(), 2);
        auto tX = Ort::Value::CreateTensor<float>(mem, graph.x.data(), graph.x.size(),
                                                   xShape.data(), 2);
        auto tEmbed = Ort::Value::CreateTensor<int64_t>(
            mem, graph.embedEdgeIndex.data(), graph.embedEdgeIndex.size(), embShape.data(), 2);
        auto tType = Ort::Value::CreateTensor<int64_t>(mem, graph.edgeType.data(),
                                                       graph.edgeType.size(), etShape.data(), 1);
        auto tPitch = Ort::Value::CreateTensor<float>(
            mem, graph.pitchScore.data(), graph.pitchScore.size(), psShape.data(), 2);
        auto tOnset = Ort::Value::CreateTensor<float>(
            mem, graph.onsetScore.data(), graph.onsetScore.size(), osShape.data(), 2);

        std::array<Ort::Value, 6> inputs{std::move(tTarget), std::move(tX), std::move(tEmbed),
                                         std::move(tType), std::move(tPitch), std::move(tOnset)};

        std::vector<const char *> inNames;
        inNames.reserve(allocatedInputNames.size());
        for (const auto &n: allocatedInputNames)
            inNames.push_back(n.c_str());
        std::vector<const char *> outNames;
        outNames.reserve(allocatedOutputNames.size());
        for (const auto &n: allocatedOutputNames)
            outNames.push_back(n.c_str());

        auto outputs = session->Run(Ort::RunOptions{nullptr}, inNames.data(), inputs.data(),
                                    inputs.size(), outNames.data(), outNames.size());
        if (! outputs.empty() && outputs[0].IsTensor()) {
            const float *data = outputs[0].GetTensorData<float>();
            const auto info = outputs[0].GetTensorTypeAndShapeInfo();
            const auto count = info.GetElementCount();
            edgeScores.assign(data, data + count);
        }
    } catch (const Ort::Exception &) {
        step.onnx = static_cast<float>(nowMs() - tOnnx0);
        finishEarly("new_onnx_error");
        return restamps;
    }
    step.onnx = static_cast<float>(nowMs() - tOnnx0);

    const double tHung0 = nowMs();
    const auto parents = Vocsep::hungarianParents(edgeScores, graph.targetEdgeIndex,
                                                  graph.targetEdgeCount,
                                                  static_cast<int>(graph.numNodes),
                                                  Vocsep::Threshold);
    step.hungarian = static_cast<float>(nowMs() - tHung0);

    const int newest = static_cast<int>(window.size()) - 1;
    const int32_t parentIdx = (newest >= 0 && newest < static_cast<int>(parents.size()))
                                  ? parents[static_cast<size_t>(newest)]
                                  : -1;

    const float parentScore =
        parentIdx >= 0 ? potEdgeScore(graph, edgeScores, parentIdx, newest) : NAN;

    const char *reason = "new_no_parent";
    if (parentIdx >= 0 && parentIdx < static_cast<int>(window.size())
        && window[static_cast<size_t>(parentIdx)].voiceId >= 0) {
        const int32_t candidateVoice = window[static_cast<size_t>(parentIdx)].voiceId;
        const float sNew = std::isfinite(parentScore)
                               ? parentScore
                               : -std::numeric_limits<float>::infinity();

        // Concurrent claimants already holding candidateVoice (±tol onset).
        std::vector<int> concurrentClaimants;
        float maxOld = -std::numeric_limits<float>::infinity();
        bool anyClaimant = false;
        for (int i = 0; i < newest; ++i) {
            const auto &o = window[static_cast<size_t>(i)];
            if (o.voiceId != candidateVoice)
                continue;
            if (std::abs(o.time - token.time) > kConcurrentOnsetTolCs)
                continue;
            anyClaimant = true;
            concurrentClaimants.push_back(i);
            maxOld = std::max(maxOld, potEdgeScore(graph, edgeScores, parentIdx, i));
        }

        if (! anyClaimant) {
            token.voiceId = candidateVoice;
            reason = "inherit";
        } else if (sNew > maxOld) {
            // Newest wins on score: take voice, restamp losers to fresh ids.
            token.voiceId = candidateVoice;
            reason = "inherit_win_score";
            for (int i: concurrentClaimants) {
                const int32_t stolen = nextVoiceId++;
                window[static_cast<size_t>(i)].voiceId = stolen;
                if (i >= 0 && static_cast<size_t>(i) < soundingHistoryIdx.size()) {
                    restamps.push_back(
                        HistoryRestamp{soundingHistoryIdx[static_cast<size_t>(i)], stolen});
                }
            }
        } else {
            // Tie or lower: newest does not inherit (no first-wins).
            token.voiceId = nextVoiceId++;
            reason = "inherit_lose_score";
        }
    } else {
        token.voiceId = nextVoiceId++;
        reason = parentIdx >= 0 ? "new_parent_no_voice" : "new_no_parent";
    }
    window.back().voiceId = token.voiceId;
    logVoiceAssignment(token, reason, window, parentIdx, parentScore, &graph, &edgeScores);

    step.total = static_cast<float>(nowMs() - t0);
    const juce::ScopedLock lock(profileLock);
    profile.publishBusy(step);
    return restamps;
}

#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include "VirtualOrch/OrtEnv.h"

#include <array>

namespace {

auto nowMs() -> double {
    return juce::Time::getMillisecondCounterHiRes();
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

auto VoiceSeparation::stampVoiceId(Token &token, const std::vector<Token> &outputHistory) -> void {
    if (session == nullptr || ! isSoundingNote(token))
        return;

    const double t0 = nowMs();

    std::vector<Token> window;
    window.reserve(static_cast<size_t>(WindowNotes));
    for (const auto &t: outputHistory) {
        if (isSoundingNote(t))
            window.push_back(t);
    }
    if (static_cast<int>(window.size()) >= WindowNotes)
        window.erase(window.begin(),
                     window.end() - static_cast<std::ptrdiff_t>(WindowNotes - 1));
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

    if (graph.numNodes <= 0) {
        token.voiceId = nextVoiceId++;
        step.total = static_cast<float>(nowMs() - t0);
        const juce::ScopedLock lock(profileLock);
        profile.publishBusy(step);
        return;
    }

    if (graph.targetEdgeCount <= 0 || graph.numNodes == 1) {
        token.voiceId = nextVoiceId++;
        step.total = static_cast<float>(nowMs() - t0);
        const juce::ScopedLock lock(profileLock);
        profile.publishBusy(step);
        return;
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
        token.voiceId = nextVoiceId++;
        step.onnx = static_cast<float>(nowMs() - tOnnx0);
        step.total = static_cast<float>(nowMs() - t0);
        const juce::ScopedLock lock(profileLock);
        profile.publishBusy(step);
        return;
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
    if (parentIdx >= 0 && parentIdx < static_cast<int>(window.size())
        && window[static_cast<size_t>(parentIdx)].voiceId >= 0) {
        token.voiceId = window[static_cast<size_t>(parentIdx)].voiceId;
    } else {
        token.voiceId = nextVoiceId++;
    }

    step.total = static_cast<float>(nowMs() - t0);
    const juce::ScopedLock lock(profileLock);
    profile.publishBusy(step);
}

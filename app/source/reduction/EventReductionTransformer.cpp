#include "VirtualOrch/reduction/EventReductionTransformer.h"

#include "VirtualOrch/reduction/DenseTypes.h"

#include <algorithm>

namespace {

auto denseQuad(int32_t time, int32_t durCs, int32_t instrument, int32_t pitch, int32_t velocity) -> std::array<int32_t, 4> {
    return {time, static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * instrument + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity};
}

constexpr size_t kWidth = 4;

} // namespace

EventReductionTransformer::EventReductionTransformer(const juce::String &threadName, ModelConfig &modelConfigIn)
    : ReductionTransformer(threadName, modelConfigIn) {}

auto EventReductionTransformer::nowCs() const -> int32_t {
    if (nowCsProvider)
        return nowCsProvider();
    return clock != nullptr ? static_cast<int32_t>(clock->getTime()) : 0;
}

auto EventReductionTransformer::clearIo() -> void {
    scheduledOffs.clear();
    open.clear();
}

auto EventReductionTransformer::drainPianoEvents(int32_t now) -> std::vector<PianoEvent> {
    std::vector<PianoEvent> out;
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
    }
    Token in{-1, -1, -1};
    if (keepsQueueOnInput()) // Accompaniment: the pianist is ignored (releases of earlier keys still apply)
        while (inputTokenQueue.pull(in)) {
        }
    bool changed = false;
    while (inputTokenQueue.pull(in)) {
        const int32_t pitch = DenseQuantize::snapPianoPitch(in.getPitch());
        const int32_t velocity = std::clamp(in.velocity > 0 ? in.velocity : DenseConfig::DefaultVelocity, 1, 127);
        out.push_back({true, in.time, pitch, velocity});
        const bool held = DenseSampling::isProvisionalDurationToken(in.duration, modelConfig.inputDuration);
        if (! held)
            scheduledOffs.push_back({false, in.time + std::max(1, in.getRealDuration()), pitch, 0});
        const auto quad = denseQuad(in.time, DenseQuantize::snapDurationCs(in.getRealDuration()),
                                    DenseConfig::PianoReductionInputInstrument, pitch, DenseQuantize::snapVelocity(velocity));
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        changed = true;
    }
    TokenUpdate update{};
    while (updatesFromFilter.pull(update)) {
        if (orchestrationUpdatesIncoming != nullptr)
            orchestrationUpdatesIncoming->push(update);
        const int32_t pitch = DenseQuantize::snapPianoPitch(update.oldNote.getPitch());
        const int32_t dur = std::max(1, update.newNote.getRealDuration());
        scheduledOffs.push_back({false, update.oldNote.time + dur, pitch, 0});
        for (size_t i = 0; i + 3 < inputData.size(); i += kWidth)
            if (inputData[i] == update.oldNote.time
                && DenseQuantize::instrumentOfNoteToken(inputData[i + 2]) == DenseConfig::PianoReductionInputInstrument
                && inputData[i + 2] - static_cast<int32_t>(DenseVocab::NoteOffset) == pitch) {
                inputData[i + 1] = static_cast<int32_t>(DenseVocab::DurOffset) + DenseQuantize::snapDurationCs(dur);
                changed = true;
            }
    }
    // key-ups when the clock reaches them, in time order, after the key-downs that came with them
    std::stable_sort(scheduledOffs.begin(), scheduledOffs.end(), [](const auto &a, const auto &b) { return a.timeCs < b.timeCs; });
    while (! scheduledOffs.empty() && scheduledOffs.front().timeCs <= now) {
        out.push_back(scheduledOffs.front());
        scheduledOffs.erase(scheduledOffs.begin());
    }
    std::stable_sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.timeCs < b.timeCs; });
    if (changed)
        notifyInputDataChanged();
    return out;
}

auto EventReductionTransformer::emitOrchestra(const EventEngine::Out &o, int32_t velocityBoost, int32_t noteId) -> void {
    if (o.on) {
        const int32_t dur = std::min<int32_t>(kMaxNoteDurationCs, 995);
        Token token{o.timeCs, static_cast<int32_t>(DenseVocab::DurOffset) + dur,
                    static_cast<int32_t>(DenseVocab::NoteOffset) + o.pitch, std::clamp(o.velocity + velocityBoost, 1, 127)};
        token.noteId = noteId;
        pushOutputToken(token);
        afterOutputDelivered(token);
        open.push_back({token});
        const auto quad = denseQuad(o.timeCs, dur, 1, o.pitch, std::clamp(o.velocity, 1, 127));
        inputData.insert(inputData.end(), quad.begin(), quad.end());
        currentTime = std::max(currentTime, o.timeCs);
        logGeneratedTokenIfResumed(o.timeCs);
        notifyInputDataChanged();
        return;
    }
    const auto it = std::find_if(open.rbegin(), open.rend(), [&](const Open &n) { return n.token.getPitch() == o.pitch; });
    if (it == open.rend())
        return;
    const Token old = it->token;
    open.erase(std::next(it).base());
    const int32_t dur = std::clamp(o.timeCs - old.time, 1, 995);
    Token now = old;
    now.duration = static_cast<int32_t>(DenseVocab::DurOffset) + dur;
    if (orchestrationUpdatesIncoming != nullptr)
        orchestrationUpdatesIncoming->push(TokenUpdate{old, now});
    {
        const juce::ScopedLock lock(outputHistoryLock);
        for (auto &t: outputHistory)
            if (t.noteId == old.noteId && old.noteId >= 0)
                t.duration = now.duration;
    }
    for (size_t i = 0; i + 3 < inputData.size(); i += kWidth)
        if (inputData[i] == old.time
            && inputData[i + 2] == static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch + o.pitch)
            inputData[i + 1] = static_cast<int32_t>(DenseVocab::DurOffset) + DenseQuantize::snapDurationCs(dur);
    notifyInputDataChanged();
}

auto EventReductionTransformer::forgetOpenFrom(int32_t cutCs) -> void {
    open.erase(std::remove_if(open.begin(), open.end(), [&](const Open &n) { return n.token.time >= cutCs; }), open.end());
}

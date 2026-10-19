#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

#include "VirtualOrch/orchestration-models/VocsepEnsemble.h"

namespace {

[[nodiscard]] auto isSoundingNote(const Token &token) -> bool {
    return token.note >= static_cast<int32_t>(Vocab::NoteOffset)
           && token.note < static_cast<int32_t>(Vocab::Rest);
}

} // namespace

auto OrchestrationModel::soundingNotes(const std::vector<Token> &tokens) -> std::vector<Token> {
    std::vector<Token> out;
    out.reserve(tokens.size());
    for (const auto &token: tokens) {
        if (isSoundingNote(token))
            out.push_back(token);
    }
    return out;
}

auto OrchestrationModel::recordIncoming(const std::vector<Token> &incoming,
                                        int32_t gapClearCs,
                                        bool *cleared) -> size_t {
    size_t newBegin = inputNotes.size();
    for (const auto &token: incoming) {
        if (! isSoundingNote(token))
            continue;
        if (gapClearCs > 0 && ! inputNotes.empty()
            && token.time - inputNotes.back().token.time > gapClearCs) {
            inputNotes.clear();
            if (cleared != nullptr)
                *cleared = true;
        }
        if (inputNotes.empty())
            newBegin = 0;
        InputNote record;
        record.token = token;
        record.stream = currentStream;
        inputNotes.push_back(std::move(record));
    }

    if (inputNotes.size() > kInputNoteCap) {
        const size_t drop = inputNotes.size() - kInputNoteCap;
        inputNotes.erase(inputNotes.begin(), inputNotes.begin() + static_cast<std::ptrdiff_t>(drop));
        newBegin = newBegin >= drop ? newBegin - drop : 0;
    }
    return newBegin;
}

auto OrchestrationModel::ensembleKeysB(size_t idx) const -> std::vector<int32_t> {
    return VocsepEnsemble::computeKeysB(inputNotes, idx, ensembleAddP.load(), ensembleRemoveP.load());
}

auto OrchestrationModel::applyVoiceRestamp(int32_t noteId, int32_t newVoiceId) -> void {
    if (noteId < 0)
        return;
    for (auto it = inputNotes.rbegin(); it != inputNotes.rend(); ++it) {
        if (it->token.noteId == noteId) {
            it->token.voiceId = newVoiceId;
            return;
        }
    }
}

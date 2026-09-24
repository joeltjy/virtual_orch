#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/vocsep/VocsepGraph.h"
#include "onnxruntime_cxx_api.h"

/**
 * Online voice separation over the last reduction notes.
 * Stamps Token::voiceId from Hungarian parent links before OT push.
 */
class VoiceSeparation {
public:
    static constexpr int WindowNotes = 40;

    /** Rewrite an already-stamped history note's voiceId (score steal from concurrent). */
    struct HistoryRestamp {
        size_t historyIndex = 0;
        int32_t newVoiceId = -1;
    };

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto isLoaded() const -> bool { return session != nullptr; }

    /** Clear voice counter + profile max (call when reduction thread starts). */
    auto reset() -> void;

    /**
     * If loaded and `token` is a sounding note: build last-40 graph with history,
     * run ONNX + Hungarian, set token.voiceId (inherit parent or allocate).
     * Non-notes / unloaded leave voiceId unchanged (-1) and return {}.
     *
     * Concurrent notes (± NearConsecutiveTolCs onset) that already hold the parent's
     * voice are resolved by pot-edge score to that parent: newest inherits only if
     * strictly greater than every concurrent claimant's score; then losers are
     * restamped (returned for the caller to write into outputHistory). On tie or
     * lower score, newest allocates a new voiceId.
     *
     * Logs to Documents/virtual-orch/Logs/vocsep_assign.log.
     */
    auto stampVoiceId(Token &token, const std::vector<Token> &outputHistory)
        -> std::vector<HistoryRestamp>;

    [[nodiscard]] auto getLastProfile() const -> VocsepLoopProfile;

private:
    [[nodiscard]] static auto isSoundingNote(const Token &token) -> bool;

    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    int32_t nextVoiceId = 0;

    mutable juce::CriticalSection profileLock;
    VocsepLoopProfile profile;
};

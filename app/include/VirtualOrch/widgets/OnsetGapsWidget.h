#pragma once

#include <JuceHeader.h>

#include <array>
#include <vector>

#include "VirtualOrch/reduction/TdOnsetBias.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Onset-gap distributions of the active reduction model, in 5 cs bins over 0..4 s:
 *   - top: what the model samples from at its latest onset draw (TdOnsetBias::Snapshot, softmax at
 *     the onset temperature), before (grey) and after (colour) the LogitTD shift;
 *   - bottom: the gaps between the onsets of the last generated notes (0 = a chord).
 * A dashed line marks the TD level's mean gap 200 / k cs.
 */
class OnsetGapsView : public juce::Component {
public:
    static constexpr int kBinCs = 5;
    static constexpr int kBins = TdOnsetBias::kSnapshotGaps / kBinCs;

    auto setData(const TdOnsetBias::Snapshot &snapshot, std::vector<int32_t> generatedGaps) -> void;
    auto paint(juce::Graphics &g) -> void override;

    /** Gaps (cs) between consecutive onsets of the last `maxNotes` notes of `onsets` (any order). */
    [[nodiscard]] static auto gapsBetweenOnsets(std::vector<int32_t> onsets, size_t maxNotes)
        -> std::vector<int32_t>;

private:
    auto paintPredicted(juce::Graphics &g, juce::Rectangle<float> area) -> void;
    auto paintGenerated(juce::Graphics &g, juce::Rectangle<float> area) -> void;
    auto paintAxis(juce::Graphics &g, juce::Rectangle<float> plot) -> void;
    auto paintTdMarker(juce::Graphics &g, juce::Rectangle<float> plot) -> void;

    TdOnsetBias::Snapshot snap;
    std::vector<int32_t> gaps;
};

class OnsetGapsWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit OnsetGapsWidget(AppSession &session);
    ~OnsetGapsWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    AppSession &session;
    OnsetGapsView view;
    uint64_t lastSequence = 0;
    size_t lastHistorySize = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OnsetGapsWidget)
};

#include "VirtualOrch/widgets/OnsetGapsWidget.h"

#include <algorithm>

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

constexpr size_t kGeneratedNotes = 128;
const juce::Colour kBeforeColour{juce::Colours::grey.withAlpha(0.8f)};
const juce::Colour kAfterColour{0xffe67e22};
const juce::Colour kGeneratedColour{0xff3498db};
const juce::Colour kTdMarkerColour{juce::Colours::white.withAlpha(0.5f)};

auto binned(const std::array<float, TdOnsetBias::kSnapshotGaps> &perCs)
    -> std::array<float, OnsetGapsView::kBins> {
    std::array<float, OnsetGapsView::kBins> bins{};
    for (int c = 0; c < TdOnsetBias::kSnapshotGaps; ++c)
        bins[static_cast<size_t>(c / OnsetGapsView::kBinCs)] += perCs[static_cast<size_t>(c)];
    return bins;
}

/** Mean gap (cs) over the shown range, renormalised without the tail. */
auto meanGap(const std::array<float, TdOnsetBias::kSnapshotGaps> &perCs) -> float {
    double total = 0.0;
    double weighted = 0.0;
    for (int c = 0; c < TdOnsetBias::kSnapshotGaps; ++c) {
        total += perCs[static_cast<size_t>(c)];
        weighted += static_cast<double>(c) * perCs[static_cast<size_t>(c)];
    }
    return total > 0.0 ? static_cast<float>(weighted / total) : 0.0f;
}

auto xFor(juce::Rectangle<float> plot, float gapCs) -> float {
    return plot.getX() + plot.getWidth() * gapCs / static_cast<float>(TdOnsetBias::kSnapshotGaps);
}

auto drawCurve(juce::Graphics &g, juce::Rectangle<float> plot,
               const std::array<float, OnsetGapsView::kBins> &bins, float peak, juce::Colour colour,
               bool filled) -> void {
    juce::Path path;
    const float binW = plot.getWidth() / static_cast<float>(OnsetGapsView::kBins);
    path.startNewSubPath(plot.getX(), plot.getBottom());
    for (int i = 0; i < OnsetGapsView::kBins; ++i) {
        const float y = plot.getBottom() - plot.getHeight() * bins[static_cast<size_t>(i)] / peak;
        path.lineTo(plot.getX() + binW * static_cast<float>(i), y);
        path.lineTo(plot.getX() + binW * static_cast<float>(i + 1), y);
    }
    path.lineTo(plot.getRight(), plot.getBottom());
    if (filled) {
        g.setColour(colour.withAlpha(0.35f));
        g.fillPath(path);
    }
    g.setColour(colour);
    g.strokePath(path, juce::PathStrokeType(1.5f));
}

} // namespace

auto OnsetGapsView::gapsBetweenOnsets(std::vector<int32_t> onsets, size_t maxNotes)
    -> std::vector<int32_t> {
    std::sort(onsets.begin(), onsets.end());
    if (onsets.size() > maxNotes)
        onsets.erase(onsets.begin(), onsets.end() - static_cast<std::ptrdiff_t>(maxNotes));
    std::vector<int32_t> out;
    for (size_t i = 1; i < onsets.size(); ++i)
        out.push_back(onsets[i] - onsets[i - 1]);
    return out;
}

auto OnsetGapsView::setData(const TdOnsetBias::Snapshot &snapshot, std::vector<int32_t> generatedGaps)
    -> void {
    snap = snapshot;
    gaps = std::move(generatedGaps);
    repaint();
}

auto OnsetGapsView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    auto area = getLocalBounds().toFloat().reduced(8.0f);
    const auto top = area.removeFromTop(area.getHeight() * 0.55f);
    area.removeFromTop(6.0f);
    paintPredicted(g, top);
    paintGenerated(g, area);
}

auto OnsetGapsView::paintAxis(juce::Graphics &g, juce::Rectangle<float> plot) -> void {
    g.setColour(juce::Colours::white.withAlpha(0.25f));
    g.drawHorizontalLine(static_cast<int>(plot.getBottom()), plot.getX(), plot.getRight());
    g.setFont(juce::Font(11.0f));
    for (int cs = 0; cs <= TdOnsetBias::kSnapshotGaps; cs += 50) {
        const float x = xFor(plot, static_cast<float>(cs));
        g.setColour(juce::Colours::white.withAlpha(0.12f));
        g.drawVerticalLine(static_cast<int>(x), plot.getY(), plot.getBottom());
        g.setColour(UiConstants::workspaceWidgetTitleTextColour.withAlpha(0.8f));
        const juce::String label = cs % 100 == 0 ? juce::String(cs / 100) + " s" : juce::String();
        g.drawText(label, juce::Rectangle<float>(x - 20.0f, plot.getBottom() + 1.0f, 40.0f, 14.0f),
                   juce::Justification::centred);
    }
}

auto OnsetGapsView::paintTdMarker(juce::Graphics &g, juce::Rectangle<float> plot) -> void {
    if (snap.tdLevel == TemporalDensity::kAuto || snap.tdLevel <= 0)
        return;
    const float x = xFor(plot, TdOnsetBias::kWindowCs / static_cast<float>(snap.tdLevel));
    const float dashes[] = {4.0f, 3.0f};
    g.setColour(kTdMarkerColour);
    g.drawDashedLine(juce::Line<float>(x, plot.getY(), x, plot.getBottom()), dashes, 2, 1.0f);
}

auto OnsetGapsView::paintPredicted(juce::Graphics &g, juce::Rectangle<float> area) -> void {
    const auto header = area.removeFromTop(18.0f);
    const auto plot = area.withTrimmedBottom(16.0f);
    g.setFont(juce::Font(12.0f, juce::Font::bold));
    if (snap.sequence == 0) {
        g.setColour(UiConstants::workspaceWidgetTitleTextColour);
        g.drawText("Predicted gap: no onset drawn yet", header, juce::Justification::centredLeft);
        paintAxis(g, plot);
        return;
    }

    const auto before = binned(snap.before);
    const auto after = binned(snap.after);
    float peak = 1e-6f;
    for (int i = 0; i < kBins; ++i)
        peak = std::max({peak, before[static_cast<size_t>(i)], after[static_cast<size_t>(i)]});

    const juce::String td = snap.tdLevel == TemporalDensity::kAuto ? juce::String("AUTO")
                                                                   : juce::String(snap.tdLevel);
    juce::String text = "Predicted gap  TD " + td + "   before: mean "
                        + juce::String(juce::roundToInt(meanGap(snap.before))) + " cs, >4 s "
                        + juce::String(juce::roundToInt(snap.beforeTail * 100.0f)) + "%";
    if (snap.shifted && snap.chordNotes > 0)
        text += "   chord n " + juce::String(snap.chordNotes) + " h "
                + juce::String(TdOnsetBias::chordShift(snap.tdLevel, snap.chordNotes), 1);
    if (snap.shifted)
        text += "   after: mean " + juce::String(juce::roundToInt(meanGap(snap.after))) + " cs, >4 s "
                + juce::String(juce::roundToInt(snap.afterTail * 100.0f)) + "%";
    else
        text += "   (LogitTD idle)";
    g.setColour(UiConstants::workspaceWidgetTitleTextColour);
    g.drawFittedText(text, header.toNearestInt(), juce::Justification::centredLeft, 1, 0.7f);

    paintAxis(g, plot);
    drawCurve(g, plot, before, peak, kBeforeColour, ! snap.shifted);
    if (snap.shifted)
        drawCurve(g, plot, after, peak, kAfterColour, true);
    paintTdMarker(g, plot);
}

auto OnsetGapsView::paintGenerated(juce::Graphics &g, juce::Rectangle<float> area) -> void {
    const auto header = area.removeFromTop(18.0f);
    const auto plot = area.withTrimmedBottom(16.0f);

    std::array<float, kBins> bins{};
    int beyond = 0;
    double sum = 0.0;
    for (const auto gap: gaps) {
        sum += gap;
        if (gap >= 0 && gap < TdOnsetBias::kSnapshotGaps)
            bins[static_cast<size_t>(gap / kBinCs)] += 1.0f;
        else
            ++beyond;
    }
    float peak = 1.0f;
    for (const auto b: bins)
        peak = std::max(peak, b);

    const auto chords = std::count(gaps.begin(), gaps.end(), 0);
    juce::String text = "Generated gaps (last " + juce::String(static_cast<int>(gaps.size())) + ")";
    if (! gaps.empty())
        text += "   mean " + juce::String(juce::roundToInt(sum / static_cast<double>(gaps.size())))
                + " cs, 0 cs " + juce::String(juce::roundToInt(100.0 * static_cast<double>(chords)
                                                              / static_cast<double>(gaps.size())))
                + "%, >4 s " + juce::String(beyond);
    g.setFont(juce::Font(12.0f, juce::Font::bold));
    g.setColour(UiConstants::workspaceWidgetTitleTextColour);
    g.drawFittedText(text, header.toNearestInt(), juce::Justification::centredLeft, 1, 0.7f);

    paintAxis(g, plot);
    drawCurve(g, plot, bins, peak, kGeneratedColour, true);
    paintTdMarker(g, plot);
}

OnsetGapsWidget::OnsetGapsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceOnsetGapsTitle, UiConstants::workspaceWidgetTypeOnsetGaps),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(10);
    timerCallback();
}

OnsetGapsWidget::~OnsetGapsWidget() {
    stopTimer();
}

auto OnsetGapsWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto OnsetGapsWidget::timerCallback() -> void {
    const auto snapshot = TdOnsetBias::latestSnapshot();
    const auto history = session.activeReduction().getOutputHistory();
    if (snapshot.sequence == lastSequence && history.size() == lastHistorySize)
        return;
    lastSequence = snapshot.sequence;
    lastHistorySize = history.size();

    std::vector<int32_t> onsets;
    for (const auto &token: history)
        if (token.note >= static_cast<int32_t>(Vocab::NoteOffset)
            && token.note < static_cast<int32_t>(Vocab::Rest))
            onsets.push_back(token.time);
    view.setData(snapshot, OnsetGapsView::gapsBetweenOnsets(std::move(onsets), kGeneratedNotes));
}

#include "VirtualOrch/widgets/PianoRollView.h"

#include "VirtualOrch/ui/UiConstants.h"

PianoRollView::PianoRollView(Clock &clockIn)
    : clock(clockIn),
      historyNoteColour(UiConstants::pianoRollNoteColour),
      pendingNoteColour(UiConstants::pianoRollPendingNoteColour) {
    setOpaque(true);
    startTimer(UiConstants::pianoRollTimerIntervalMs);
}

PianoRollView::~PianoRollView() {
    stopTimer();
}

auto PianoRollView::setNotes(std::vector<Token> notesIn) -> void {
    notes = std::move(notesIn);
    repaint();
}

auto PianoRollView::setPendingNotes(std::vector<Token> notesIn) -> void {
    pendingNotes = std::move(notesIn);
    repaint();
}

auto PianoRollView::setNoteColours(juce::Colour history, juce::Colour pending) -> void {
    historyNoteColour = history;
    pendingNoteColour = pending;
    repaint();
}

auto PianoRollView::setPitchRange(int32_t low, int32_t high) -> void {
    pitchLow = low;
    pitchHigh = std::max(high, low);
    repaint();
}

auto PianoRollView::nowBarX(float width) const -> float {
    const auto total = static_cast<float>(UiConstants::pianoRollPastWindow + UiConstants::pianoRollLookaheadWindow);
    if (total <= 0.0f || width <= 0.0f)
        return width * UiConstants::pianoRollNowbarFraction;
    return width * (static_cast<float>(UiConstants::pianoRollPastWindow) / total);
}

auto PianoRollView::timeToX(int32_t time, int32_t nowTime, float width) const -> float {
    const auto leftTime = nowTime - UiConstants::pianoRollPastWindow;
    const auto total = static_cast<float>(UiConstants::pianoRollPastWindow + UiConstants::pianoRollLookaheadWindow);
    if (total <= 0.0f)
        return 0.0f;
    return (static_cast<float>(time - leftTime) / total) * width;
}

auto PianoRollView::pitchToY(int32_t pitch, float height) const -> float {
    const auto span = std::max(1, pitchHigh - pitchLow + 1);
    // High pitches at the top.
    return ((static_cast<float>(pitchHigh - pitch) + 0.0f) / static_cast<float>(span)) * height;
}

auto PianoRollView::paintNotes(juce::Graphics &g,
                               const std::vector<Token> &tokens,
                               juce::Colour colour,
                               int32_t nowTime,
                               float width,
                               float height,
                               float noteHeight) const -> void {
    g.setColour(colour);
    for (const auto &token: tokens) {
        if (token.note == Vocab::BarSeparator || token.note == Vocab::ClearQueue || token.note == Vocab::Rest)
            continue;

        const auto pitch = token.getPitch();
        if (pitch < pitchLow || pitch > pitchHigh)
            continue;

        const auto x = timeToX(token.time, nowTime, width);
        const auto duration = std::max(1, token.getRealDuration());
        const auto w = std::max(2.0f, timeToX(token.time + duration, nowTime, width) - x);
        const auto y = pitchToY(pitch, height);

        if (x + w < 0.0f || x > width)
            continue;

        g.fillRoundedRectangle(x, y, w, noteHeight * 0.9f, UiConstants::pianoRollNoteCornerRadius);
    }
}

auto PianoRollView::paint(juce::Graphics &g) -> void {
    const auto bounds = getLocalBounds().toFloat();
    g.fillAll(UiConstants::pianoRollBackground);

    const auto width = bounds.getWidth();
    const auto height = bounds.getHeight();
    if (width <= 1.0f || height <= 1.0f)
        return;

    const auto nowTime = static_cast<int32_t>(clock.getTime());
    const auto span = std::max(1, pitchHigh - pitchLow + 1);
    const auto noteHeight = std::max(1.0f, height / static_cast<float>(span));

    // Horizontal pitch grid (octaves).
    g.setColour(UiConstants::pianoRollGridColour);
    for (int32_t pitch = pitchLow; pitch <= pitchHigh; ++pitch) {
        if (pitch % 12 != 0)
            continue;
        const auto y = pitchToY(pitch, height);
        g.drawHorizontalLine(juce::roundToInt(y), 0.0f, width);
    }

    paintNotes(g, notes, historyNoteColour, nowTime, width, height, noteHeight);
    paintNotes(g, pendingNotes, pendingNoteColour, nowTime, width, height, noteHeight);

    // Fixed nowbar.
    const auto barX = nowBarX(width);
    g.setColour(UiConstants::pianoRollNowbarColour);
    g.drawLine(barX, 0.0f, barX, height, 2.0f);
}

auto PianoRollView::timerCallback() -> void {
    repaint();
}

#include "VirtualOrch/widgets/PianoRollView.h"

#include "VirtualOrch/ui/UiConstants.h"

namespace {

/** Cap how far we zoom out when fitting table history into the roll. */
constexpr int32_t pianoRollMaxVisibleSpanCs = 6000; // 60s

} // namespace

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

auto PianoRollView::isDrawableNote(const Token &token) const -> bool {
    if (token.note == static_cast<int32_t>(Vocab::BarSeparator)
        || token.note == static_cast<int32_t>(Vocab::ClearQueue)
        || token.note == static_cast<int32_t>(Vocab::Rest))
        return false;

    const auto pitch = token.getPitch();
    return pitch >= pitchLow && pitch <= pitchHigh;
}

auto PianoRollView::computeTimeWindow() const -> TimeWindow {
    TimeWindow window;
    window.nowTime = static_cast<int32_t>(clock.getTime());
    const int32_t defaultLeft = window.nowTime - UiConstants::pianoRollPastWindow;
    const int32_t defaultRight = window.nowTime + UiConstants::pianoRollLookaheadWindow;
    window.leftTime = defaultLeft;
    window.rightTime = defaultRight;

    auto expandWith = [&](const std::vector<Token> &tokens) {
        for (const auto &token: tokens) {
            if (! isDrawableNote(token))
                continue;
            const auto duration = std::max(1, token.getRealDuration());
            window.leftTime = std::min(window.leftTime, token.time);
            window.rightTime = std::max(window.rightTime, token.time + duration);
        }
    };
    expandWith(notes);
    expandWith(pendingNotes);

    // Always keep the default around-now window (and thus the nowbar) in range.
    window.leftTime = std::min(window.leftTime, defaultLeft);
    window.rightTime = std::max(window.rightTime, defaultRight);

    if (window.rightTime <= window.leftTime)
        window.rightTime = window.leftTime + 1;

    const auto span = window.rightTime - window.leftTime;
    if (span > pianoRollMaxVisibleSpanCs) {
        // Cap zoom while pinning now at the default past/lookahead fraction.
        const int32_t defaultSpan =
            UiConstants::pianoRollPastWindow + UiConstants::pianoRollLookaheadWindow;
        const int32_t pastPart =
            (pianoRollMaxVisibleSpanCs * UiConstants::pianoRollPastWindow) / defaultSpan;
        window.leftTime = window.nowTime - pastPart;
        window.rightTime = window.leftTime + pianoRollMaxVisibleSpanCs;
    }

    return window;
}

auto PianoRollView::nowBarX(const TimeWindow &window, float width) const -> float {
    const auto total = static_cast<float>(window.rightTime - window.leftTime);
    if (total <= 0.0f || width <= 0.0f)
        return width * UiConstants::pianoRollNowbarFraction;
    return ((static_cast<float>(window.nowTime - window.leftTime)) / total) * width;
}

auto PianoRollView::timeToX(int32_t time, const TimeWindow &window, float width) const -> float {
    const auto total = static_cast<float>(window.rightTime - window.leftTime);
    if (total <= 0.0f)
        return 0.0f;
    return (static_cast<float>(time - window.leftTime) / total) * width;
}

auto PianoRollView::pitchToY(int32_t pitch, float height) const -> float {
    const auto span = std::max(1, pitchHigh - pitchLow + 1);
    return ((static_cast<float>(pitchHigh - pitch) + 0.0f) / static_cast<float>(span)) * height;
}

auto PianoRollView::paintNotes(juce::Graphics &g,
                               const std::vector<Token> &tokens,
                               juce::Colour colour,
                               const TimeWindow &window,
                               float width,
                               float height,
                               float noteHeight) const -> void {
    g.setColour(colour);
    for (const auto &token: tokens) {
        if (! isDrawableNote(token))
            continue;

        const auto pitch = token.getPitch();
        const auto x = timeToX(token.time, window, width);
        const auto duration = std::max(1, token.getRealDuration());
        const auto w = std::max(2.0f, timeToX(token.time + duration, window, width) - x);
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

    const auto window = computeTimeWindow();
    const auto span = std::max(1, pitchHigh - pitchLow + 1);
    const auto noteHeight = std::max(1.0f, height / static_cast<float>(span));

    g.setColour(UiConstants::pianoRollGridColour);
    for (int32_t pitch = pitchLow; pitch <= pitchHigh; ++pitch) {
        if (pitch % 12 != 0)
            continue;
        const auto y = pitchToY(pitch, height);
        g.drawHorizontalLine(juce::roundToInt(y), 0.0f, width);
    }

    paintNotes(g, notes, historyNoteColour, window, width, height, noteHeight);
    paintNotes(g, pendingNotes, pendingNoteColour, window, width, height, noteHeight);

    const auto barX = nowBarX(window, width);
    g.setColour(UiConstants::pianoRollNowbarColour);
    g.drawLine(barX, 0.0f, barX, height, 2.0f);
}

auto PianoRollView::timerCallback() -> void {
    repaint();
}

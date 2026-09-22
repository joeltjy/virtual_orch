#include "VirtualOrch/widgets/PianoRollView.h"

#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>
#include <unordered_map>

namespace {

/** Cap how far we zoom out when fitting table history into the roll. */
constexpr int32_t pianoRollMaxVisibleSpanCs = NoteWindow::visibleSpanCs;

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

auto PianoRollView::setNotes(std::vector<Token> notesIn, std::vector<juce::Colour> coloursIn) -> void {
    notes = std::move(notesIn);
    noteColours = std::move(coloursIn);
    repaint();
}

auto PianoRollView::setPendingNotes(std::vector<Token> notesIn,
                                    std::vector<juce::Colour> coloursIn) -> void {
    pendingNotes = std::move(notesIn);
    pendingNoteColours = std::move(coloursIn);
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
                               const std::vector<juce::Colour> &colours,
                               juce::Colour fallbackColour,
                               const TimeWindow &window,
                               float width,
                               float height,
                               float noteHeight) const -> void {
    struct NoteKey {
        int32_t time = 0;
        int32_t duration = 0;
        int32_t pitch = 0;

        auto operator==(const NoteKey &other) const -> bool {
            return time == other.time && duration == other.duration && pitch == other.pitch;
        }
    };

    struct NoteKeyHash {
        auto operator()(const NoteKey &key) const -> size_t {
            size_t h = static_cast<size_t>(key.time);
            h ^= static_cast<size_t>(key.duration) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.pitch) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    const bool usePerNote = colours.size() == tokens.size();
    std::unordered_map<NoteKey, std::vector<juce::Colour>, NoteKeyHash> grouped;
    grouped.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto &token = tokens[i];
        if (! isDrawableNote(token))
            continue;

        const NoteKey key{.time = token.time,
                          .duration = std::max(1, token.getRealDuration()),
                          .pitch = token.getPitch()};
        const auto colour = usePerNote ? colours[i] : fallbackColour;
        auto &bucket = grouped[key];
        if (std::find(bucket.begin(), bucket.end(), colour) == bucket.end())
            bucket.push_back(colour);
    }

    const float bodyHeight = noteHeight * 0.9f;
    for (const auto &[key, noteColoursForKey]: grouped) {
        const auto x = timeToX(key.time, window, width);
        const auto w = std::max(2.0f, timeToX(key.time + key.duration, window, width) - x);
        const auto y = pitchToY(key.pitch, height);

        if (x + w < 0.0f || x > width)
            continue;

        const auto bounds = juce::Rectangle<float>(x, y, w, bodyHeight);
        const float radius = UiConstants::pianoRollNoteCornerRadius;

        if (noteColoursForKey.size() <= 1) {
            g.setColour(noteColoursForKey.empty() ? fallbackColour : noteColoursForKey.front());
            g.fillRoundedRectangle(bounds, radius);
            continue;
        }

        // Time stripes: cycle instrument colours every pianoRollMultiColourStripeCs.
        juce::Path clip;
        clip.addRoundedRectangle(bounds, radius);
        const juce::Graphics::ScopedSaveState save(g);
        g.reduceClipRegion(clip);

        const auto stripeCs = UiConstants::pianoRollMultiColourStripeCs;
        const auto colourCount = noteColoursForKey.size();
        const int32_t endTime = key.time + key.duration;
        for (int32_t t = key.time; t < endTime; t += stripeCs) {
            const auto stripeEnd = std::min(t + stripeCs, endTime);
            const auto sx = timeToX(t, window, width);
            const auto ex = timeToX(stripeEnd, window, width);
            const auto colourIndex =
                static_cast<size_t>((t - key.time) / stripeCs) % colourCount;
            g.setColour(noteColoursForKey[colourIndex]);
            g.fillRect(sx, bounds.getY(), std::max(1.0f, ex - sx + 0.5f), bounds.getHeight());
        }
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

    paintNotes(g, notes, noteColours, historyNoteColour, window, width, height, noteHeight);
    paintNotes(g,
               pendingNotes,
               pendingNoteColours,
               pendingNoteColour,
               window,
               width,
               height,
               noteHeight);

    const auto barX = nowBarX(window, width);
    g.setColour(UiConstants::pianoRollNowbarColour);
    g.drawLine(barX, 0.0f, barX, height, 2.0f);
}

auto PianoRollView::timerCallback() -> void {
    repaint();
}

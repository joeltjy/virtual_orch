#include "VirtualOrch/widgets/WaitVisualizerWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/reduction/HybridInformativeWaitReductionTransformer.h"
#include "VirtualOrch/reduction/HybridChordTokenReductionTransformer.h"
#include "VirtualOrch/reduction/HybridGroupReductionTransformer.h"
#include "VirtualOrch/reduction/InformativeWaitReductionTransformer.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>
#include <vector>

namespace {

const juce::Colour kInk{0xffd6d6d6};
const juce::Colour kDim{0xff8a8a8a};
const juce::Colour kLine{0xff6a6a6a};
const juce::Colour kDivider{0xff3a3a3a};
const juce::Colour kPredicted{0xff4fc3f7};
const juce::Colour kCount{0xffffb347};
const juce::Colour kHit{0xff81c784};
const juce::Colour kOutside{0xffe6c36a};
const juce::Colour kExpired{0xffef8a8a};
const juce::String kDot{juce::CharPointer_UTF8(" \xc2\xb7 ")}; // " · " (a plain literal is read as Latin-1)

auto resultColour(const WaitTracker::Result &r) -> juce::Colour {
    if (r.kind == WaitTracker::Result::Kind::Expired)
        return kExpired;
    return r.rank > 5 ? kOutside : kHit;
}

/** A placement time in ms: "<1 ms" below a millisecond, else "12.3 ms" (one decimal under 100 ms). */
auto msText(int64_t us) -> juce::String {
    if (us < 1000)
        return "<1 ms";
    const double ms = static_cast<double>(us) / 1000.0;
    return juce::String(ms, ms < 100.0 ? 1 : 0) + " ms";
}

auto seconds(int32_t cs) -> juce::String {
    return juce::String(static_cast<double>(cs) / 100.0, 2) + " s";
}

} // namespace

auto WaitVisualizerView::resultText(const WaitTracker::Result &r) -> juce::String {
    if (r.kind == WaitTracker::Result::Kind::Expired)
        return "EXPIRED";
    if (r.rank <= 0)
        return "HIT";
    return r.rank > 5 ? juce::String(">5") : "#" + juce::String(r.rank);
}

auto WaitVisualizerView::setState(WaitTracker::Snapshot next, int32_t nowCs, std::string source) -> void {
    state = std::move(next);
    now = nowCs;
    sourceName = std::move(source);
    repaint();
}

namespace {

/**
 * One row of a panel: its height, and how dispensable it is (0 = always shown; higher numbers
 * are dropped first when the panel is too short). `flex`: takes the space left over.
 */
struct Row {
    int id;
    float height;
    int drop;
    bool flex = false;
};

/** Drops rows (highest `drop` first) until they fit `available`; returns the kept ones, in order. */
auto fitRows(std::vector<Row> rows, float available) -> std::vector<Row> {
    const auto total = [&] {
        float t = 0.0f;
        for (const auto &r: rows)
            t += r.height;
        return t;
    };
    while (total() > available) {
        auto worst = std::max_element(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.drop < b.drop; });
        if (worst == rows.end() || worst->drop == 0)
            break;
        rows.erase(worst);
    }
    float spare = available - total();
    for (auto &r: rows)
        if (r.flex && spare > 0.0f) {
            r.height += spare;
            spare = 0.0f;
        }
    return rows;
}

/** One line of text in `box`: squeezed horizontally (down to 70%) or ellipsised, never spilling out. */
auto textLine(juce::Graphics &g, const juce::String &text, juce::Rectangle<float> box, juce::Justification j) -> void {
    g.drawFittedText(text, box.toNearestInt(), j, 1, 0.7f);
}

} // namespace

auto WaitVisualizerView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    const auto bounds = getLocalBounds().toFloat().reduced(6.0f);
    if (bounds.getWidth() < 60.0f || bounds.getHeight() < 40.0f)
        return;
    // The text grows with the widget's text scale (PerformanceView: 2.2) only as far as the widget
    // is tall enough for it: about 150 px of height per unit of scale.
    const float ts = std::clamp(std::min(textScale, bounds.getHeight() / 150.0f), 0.75f, std::max(0.75f, textScale));
    const float small = 11.0f * ts;
    const float body = 12.5f * ts;

    if (sourceName.empty()) {
        g.setColour(kDim);
        g.setFont(juce::Font(body));
        g.drawFittedText("No WAIT model: choose Hybrid v1.1 or Informative WAIT", bounds.toNearestInt(),
                         juce::Justification::centred, 3);
        return;
    }

    auto left = bounds.withWidth(bounds.getWidth() * 0.56f);
    auto right = bounds.withLeft(left.getRight() + 8.0f);
    g.setColour(kDivider);
    g.drawVerticalLine(juce::roundToInt(left.getRight() + 4.0f), bounds.getY(), bounds.getBottom());

    // ---- left: the pending WAIT, NOW -> PREDICTED
    {
        enum { Header, Countdown, Line, Labels, Info, Times, Spacer };
        const float radius = std::clamp(9.0f * ts, 7.0f, std::max(7.0f, left.getHeight() * 0.18f));
        const bool pending = state.pending.has_value();
        const auto rows = fitRows({{Header, small + 4.0f, 5},
                                   {Spacer, 0.0f, 0, true}, // spare room above the countdown: the labels stay under the line
                                   {Countdown, pending ? small + 4.0f : 0.0f, 3},
                                   {Line, radius * 2.0f + 6.0f, 0},
                                   {Labels, small + 4.0f, 2},
                                   {Info, body + 4.0f, 4},
                                   {Times, small + 4.0f, 6}},
                                  left.getHeight());
        auto area = left;
        const auto has = [&](int id) {
            return std::any_of(rows.begin(), rows.end(), [id](const Row &r) { return r.id == id; });
        };
        juce::Rectangle<float> rowBox[7];
        for (const auto &r: rows)
            rowBox[r.id] = area.removeFromTop(r.height);

        if (has(Header)) {
            g.setColour(kDim);
            g.setFont(juce::Font(small));
            textLine(g, "Pending WAIT  (" + juce::String(sourceName) + ")", rowBox[Header], juce::Justification::centredLeft);
        }
        const auto lineBox = rowBox[Line];
        const float x0 = lineBox.getX() + radius + 2.0f;
        const float x1 = lineBox.getRight() - radius - 2.0f;
        const float y = lineBox.getCentreY();
        const float windowW = std::max(6.0f, (x1 - x0) * 0.06f);
        g.setColour(kPredicted.withAlpha(0.15f));
        g.fillRect(x1 - windowW * 0.5f, y - radius * 1.3f, windowW, radius * 2.6f);
        g.setColour(kLine);
        g.drawLine(x0, y, x1, y, 2.0f);
        g.setColour(kInk);
        g.drawLine(x0, y - radius, x0, y + radius, 2.0f);
        g.setColour(kPredicted);
        g.drawLine(x1, y - radius * 1.2f, x1, y + radius * 1.2f, 2.0f);
        if (has(Labels)) {
            // NOW at the left end, PREDICTED (or PRED) ending at the right end: never on top of each other
            const auto font = juce::Font(small);
            g.setFont(font);
            const auto box = rowBox[Labels];
            const juce::String predicted =
                font.getStringWidthFloat("NOW  PREDICTED") <= box.getWidth() ? "PREDICTED" : "PRED";
            g.setColour(kInk);
            textLine(g, "NOW", box.withWidth(box.getWidth() * 0.4f), juce::Justification::centredLeft);
            g.setColour(kPredicted);
            textLine(g, predicted, box.withLeft(box.getX() + box.getWidth() * 0.4f), juce::Justification::centredRight);
        }

        if (pending) {
            const auto &p = *state.pending;
            const int32_t span = std::max(1, p.predictedCs - p.drawnAtCs);
            const float f = std::clamp(static_cast<float>(now - p.drawnAtCs) / static_cast<float>(span), 0.0f, 1.0f);
            const float cx = x0 + (x1 - x0) * f;
            g.setColour(kCount.withAlpha(0.5f));
            g.drawLine(x0, y, cx, y, 3.0f);
            g.setColour(kCount);
            g.fillEllipse(cx - radius, y - radius, radius * 2.0f, radius * 2.0f);
            g.setColour(juce::Colour(0xff2a1a00));
            g.setFont(juce::Font(radius * 1.2f, juce::Font::bold));
            g.drawText(juce::String(p.k), juce::Rectangle<float>(cx - radius, y - radius, radius * 2.0f, radius * 2.0f),
                       juce::Justification::centred, false);
            if (has(Countdown)) {
                const int32_t toGo = p.predictedCs - now;
                g.setColour(kCount);
                g.setFont(juce::Font(small));
                const auto box = rowBox[Countdown];
                const float w = std::min(box.getWidth(), juce::Font(small).getStringWidthFloat("0.00 s to go") + 8.0f);
                textLine(g, toGo > 0 ? seconds(toGo) + " to go" : juce::String("in the window"),
                         juce::Rectangle<float>(std::clamp(cx - w * 0.5f, box.getX(), box.getRight() - w), box.getY(), w,
                                                box.getHeight()),
                         juce::Justification::centred);
            }
            if (has(Info)) {
                g.setColour(juce::Colour(0xffbdbdbd));
                g.setFont(juce::Font(body));
                textLine(g, (p.provisional ? "provisional WAIT " : "WAIT ") + juce::String(p.d) + " cs   COUNT " + juce::String(p.k),
                         rowBox[Info], juce::Justification::centredLeft);
            }
            if (has(Times)) {
                g.setColour(kDim);
                g.setFont(juce::Font(small));
                textLine(g, "drawn at " + seconds(p.drawnAtCs) + kDot + "predicted " + seconds(p.predictedCs),
                         rowBox[Times], juce::Justification::centredLeft);
            }
        } else if (has(Info)) {
            g.setColour(kDim);
            g.setFont(juce::Font(small));
            textLine(g, "no WAIT pending", rowBox[Info], juce::Justification::centredLeft);
        }
    }

    // ---- right: the last window's result, then the recent ones
    {
        enum { Header, Big, Placed, Detail, CountAt, Chips };
        if (state.recent.empty()) {
            g.setColour(kDim);
            g.setFont(juce::Font(small));
            g.drawFittedText("no window closed yet", right.toNearestInt(), juce::Justification::centred, 2);
            return;
        }
        const auto &last = state.recent.back();
        const bool hit = last.kind == WaitTracker::Result::Kind::Hit;
        const float chipH = std::max(14.0f, small + 8.0f);
        const auto rows = fitRows({{Header, small + 4.0f, 5},
                                   {Big, body * 1.8f, 0, true},
                                   {Placed, hit ? body + 4.0f : 0.0f, 1},
                                   {Detail, body + 4.0f, 4},
                                   {CountAt, small + 4.0f, 6},
                                   {Chips, chipH + 4.0f, 3}},
                                  right.getHeight());
        auto area = right;
        const auto has = [&](int id) {
            return std::any_of(rows.begin(), rows.end(), [id](const Row &r) { return r.id == id; });
        };
        juce::Rectangle<float> rowBox[6];
        for (const auto &r: rows)
            rowBox[r.id] = area.removeFromTop(r.height);

        if (has(Header)) {
            g.setColour(kDim);
            g.setFont(juce::Font(small));
            textLine(g, "Last window", rowBox[Header], juce::Justification::centredLeft);
        }
        {
            const juce::String text = resultText(last);
            const auto big = rowBox[Big];
            float size = std::min(big.getHeight() * 0.9f, 64.0f * ts);
            const juce::Font probe(size, juce::Font::bold);
            const float w = probe.getStringWidthFloat(text);
            if (w > big.getWidth() * 0.95f)
                size *= big.getWidth() * 0.95f / w; // EXPIRED in a narrow panel
            g.setColour(resultColour(last));
            g.setFont(juce::Font(size, juce::Font::bold));
            g.drawText(text, big, juce::Justification::centred, false);
        }
        if (has(Placed)) {
            // the time to place the COUNT orchestra notes on the onset, from the pianist's key
            juce::String placed;
            if (last.placeUs < 0)
                placed = "placing " + juce::String(last.k) + " notes...";
            else
                placed = juce::String(last.placed) + "/" + juce::String(last.k) + " notes in " + msText(last.placeUs)
                         + (last.fromKey ? "" : " (from the fire)") + kDot + (last.precomputed ? "precomputed" : "decoded");
            g.setColour(last.precomputed ? kHit : kCount);
            g.setFont(juce::Font(body, juce::Font::bold));
            textLine(g, placed, rowBox[Placed], juce::Justification::centred);
        }
        if (has(Detail)) {
            juce::String detail;
            if (! hit)
                detail = "nothing played in the window";
            else if (last.rank <= 0)
                detail = "no anticipation was ready";
            else if (last.rank > 5)
                detail = "your note was not in the top 5";
            else
                detail = "your note ranked " + juce::String(last.rank) + " in the top 5";
            g.setColour(juce::Colour(0xffbdbdbd));
            g.setFont(juce::Font(body));
            textLine(g, detail, rowBox[Detail], juce::Justification::centred);
        }
        if (has(CountAt)) {
            g.setColour(kDim);
            g.setFont(juce::Font(small));
            textLine(g, "COUNT " + juce::String(last.k) + kDot + "at " + seconds(last.atCs), rowBox[CountAt],
                     juce::Justification::centred);
        }
        if (has(Chips)) {
            const auto box = rowBox[Chips].withTrimmedTop(4.0f);
            float cxs = box.getX();
            const juce::Font chipFont(small);
            for (auto it = state.recent.rbegin(); it != state.recent.rend(); ++it) {
                const auto t = resultText(*it);
                const float w = chipFont.getStringWidthFloat(t) + 10.0f;
                if (cxs + w > box.getRight())
                    break;
                const auto c = resultColour(*it);
                g.setColour(c.withAlpha(0.18f));
                g.fillRoundedRectangle(cxs, box.getY(), w, box.getHeight(), 4.0f);
                g.setColour(c);
                g.setFont(chipFont);
                g.drawText(t, juce::Rectangle<float>(cxs, box.getY(), w, box.getHeight()), juce::Justification::centred, false);
                cxs += w + 4.0f;
            }
        }
    }
}

WaitVisualizerWidget::WaitVisualizerWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceWaitVisualizerTitle, UiConstants::workspaceWidgetTypeWaitVisualizer),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(30); // the circle moves smoothly
}

WaitVisualizerWidget::~WaitVisualizerWidget() {
    stopTimer();
}

auto WaitVisualizerWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto WaitVisualizerWidget::timerCallback() -> void {
    const auto now = static_cast<int32_t>(session.clock.getTime());
    auto &rt = session.activeReduction();
    if (auto *hy = dynamic_cast<HybridInformativeWaitReductionTransformer *>(&rt))
        view.setState(hy->waitState(), now, "Hybrid v1.1, playing side");
    else if (auto *h12 = dynamic_cast<HybridChordTokenReductionTransformer *>(&rt))
        view.setState(h12->waitState(), now, "Hybrid v1.2, playing side");
    else if (auto *h14 = dynamic_cast<HybridGroupReductionTransformer *>(&rt))
        view.setState(h14->waitState(), now, "Hybrid v1.4, playing side");
    else if (auto *iw = dynamic_cast<InformativeWaitReductionTransformer *>(&rt))
        view.setState(iw->waitState(), now, "Informative WAIT");
    else
        view.setState({}, now, "");
}

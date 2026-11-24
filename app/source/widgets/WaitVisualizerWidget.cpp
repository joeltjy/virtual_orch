#include "VirtualOrch/widgets/WaitVisualizerWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/reduction/HybridInformativeWaitReductionTransformer.h"
#include "VirtualOrch/reduction/InformativeWaitReductionTransformer.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>

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

auto WaitVisualizerView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    const auto bounds = getLocalBounds().toFloat().reduced(8.0f);
    if (bounds.getWidth() < 80.0f || bounds.getHeight() < 60.0f)
        return;
    const float ts = textScale;
    const float small = 11.0f * ts;
    const float body = 12.5f * ts;

    if (sourceName.empty()) {
        g.setColour(kDim);
        g.setFont(juce::Font(body));
        g.drawText("No WAIT model: choose Hybrid v1.1 or Informative WAIT", bounds,
                   juce::Justification::centred, true);
        return;
    }

    auto left = bounds.withWidth(bounds.getWidth() * 0.58f);
    auto right = bounds.withLeft(left.getRight() + 10.0f);
    g.setColour(kDivider);
    g.drawVerticalLine(juce::roundToInt(left.getRight() + 5.0f), bounds.getY(), bounds.getBottom());

    // ---- left: the pending WAIT, NOW -> PREDICTED
    g.setColour(kDim);
    g.setFont(juce::Font(small));
    g.drawText("Pending WAIT  (" + juce::String(sourceName) + ")", left.removeFromTop(small + 6.0f),
               juce::Justification::centredLeft, true);

    const float radius = std::clamp(left.getHeight() * 0.11f, 9.0f, 22.0f) * std::sqrt(ts);
    const float x0 = left.getX() + radius + 4.0f;
    const float x1 = left.getRight() - std::max(radius, 40.0f * ts);
    const float y = left.getY() + left.getHeight() * 0.42f;

    g.setColour(kPredicted.withAlpha(0.15f));
    const float windowW = std::max(8.0f, (x1 - x0) * 0.06f);
    g.fillRect(x1 - windowW * 0.5f, y - radius * 1.5f, windowW, radius * 3.0f);
    g.setColour(kLine);
    g.drawLine(x0, y, x1, y, 2.0f);
    g.setColour(kInk);
    g.drawLine(x0, y - radius, x0, y + radius, 2.0f);
    g.setColour(kPredicted);
    g.drawLine(x1, y - radius * 1.3f, x1, y + radius * 1.3f, 2.0f);
    g.setFont(juce::Font(body));
    g.setColour(kInk);
    g.drawText("NOW", juce::Rectangle<float>(x0 - 40.0f, y + radius * 1.4f, 80.0f, body + 4.0f),
               juce::Justification::centred, false);
    g.setColour(kPredicted);
    g.drawText("PREDICTED", juce::Rectangle<float>(x1 - 60.0f, y + radius * 1.4f, 120.0f, body + 4.0f),
               juce::Justification::centred, false);

    const float infoY = y + radius * 1.4f + body + 14.0f;
    if (state.pending.has_value()) {
        const auto &p = *state.pending;
        const int32_t span = std::max(1, p.predictedCs - p.drawnAtCs);
        const float f = std::clamp(static_cast<float>(now - p.drawnAtCs) / static_cast<float>(span), 0.0f, 1.0f);
        const float cx = x0 + (x1 - x0) * f;
        g.setColour(kCount.withAlpha(0.5f));
        g.drawLine(x0, y, cx, y, 3.0f);
        g.setColour(kCount);
        g.fillEllipse(cx - radius, y - radius, radius * 2.0f, radius * 2.0f);
        g.setColour(juce::Colour(0xff2a1a00));
        g.setFont(juce::Font(radius * 1.15f, juce::Font::bold));
        g.drawText(juce::String(p.k), juce::Rectangle<float>(cx - radius, y - radius, radius * 2.0f, radius * 2.0f),
                   juce::Justification::centred, false);
        const int32_t toGo = p.predictedCs - now;
        g.setColour(kCount);
        g.setFont(juce::Font(small));
        g.drawText(toGo > 0 ? seconds(toGo) + " to go" : juce::String("in the window"),
                   juce::Rectangle<float>(cx - 60.0f, y - radius - small - 8.0f, 120.0f, small + 4.0f),
                   juce::Justification::centred, false);

        g.setColour(juce::Colour(0xffbdbdbd));
        g.setFont(juce::Font(body));
        g.drawText((p.provisional ? "provisional WAIT " : "WAIT ") + juce::String(p.d) + " cs    COUNT "
                       + juce::String(p.k),
                   juce::Rectangle<float>(left.getX(), infoY, left.getWidth(), body + 4.0f),
                   juce::Justification::centredLeft, true);
        g.setColour(kDim);
        g.setFont(juce::Font(small));
        g.drawText("drawn at " + seconds(p.drawnAtCs) + " " + kDot + " predicted " + seconds(p.predictedCs),
                   juce::Rectangle<float>(left.getX(), infoY + body + 6.0f, left.getWidth(), small + 4.0f),
                   juce::Justification::centredLeft, true);
    } else {
        g.setColour(kDim);
        g.setFont(juce::Font(small));
        g.drawText("no WAIT pending", juce::Rectangle<float>(left.getX(), infoY, left.getWidth(), small + 4.0f),
                   juce::Justification::centredLeft, true);
    }

    // ---- right: the last window's result, then the recent ones
    g.setColour(kDim);
    g.setFont(juce::Font(small));
    g.drawText("Last window", right.removeFromTop(small + 6.0f), juce::Justification::centredLeft, true);
    const float chipH = std::max(18.0f, 22.0f * ts);
    auto chips = right.removeFromBottom(chipH);
    right.removeFromBottom(small + 10.0f);
    if (state.recent.empty()) {
        g.drawText("no window closed yet", right, juce::Justification::centred, true);
        return;
    }
    const auto &last = state.recent.back();
    const juce::String text = resultText(last);
    auto big = right.removeFromTop(right.getHeight() * 0.62f);
    float bigSize = std::min(big.getHeight() * 0.95f, 64.0f * ts);
    if (text.length() > 3)
        bigSize = std::min(bigSize, big.getWidth() / (0.62f * static_cast<float>(text.length())));
    g.setColour(resultColour(last));
    g.setFont(juce::Font(bigSize, juce::Font::bold));
    g.drawText(text, big, juce::Justification::centredBottom, false);
    juce::String detail;
    if (last.kind == WaitTracker::Result::Kind::Expired)
        detail = "nothing played in the window";
    else if (last.rank <= 0)
        detail = "hit" + kDot + "no anticipation was ready";
    else if (last.rank > 5)
        detail = "hit" + kDot + "your note was not in the top 5";
    else
        detail = "hit" + kDot + "your note ranked " + juce::String(last.rank) + " in the top 5";
    g.setColour(juce::Colour(0xffbdbdbd));
    g.setFont(juce::Font(body));
    g.drawText(detail, right.removeFromTop(body + 6.0f), juce::Justification::centred, true);
    if (last.kind == WaitTracker::Result::Kind::Hit) {
        // the time to place the COUNT orchestra notes on the onset, from the fire
        juce::String placed;
        if (last.placeUs < 0)
            placed = "placing " + juce::String(last.k) + " notes...";
        else
            placed = juce::String(last.placed) + " of " + juce::String(last.k) + " notes in " + msText(last.placeUs)
                     + (last.fromKey ? "" : " (from the fire)") + (last.precomputed ? " " + kDot + " precomputed" : " " + kDot + " decoded");
        g.setColour(last.precomputed ? kHit : kCount);
        g.setFont(juce::Font(body, juce::Font::bold));
        g.drawText(placed, right.removeFromTop(body + 6.0f), juce::Justification::centred, true);
    }
    g.setColour(kDim);
    g.setFont(juce::Font(small));
    g.drawText("COUNT " + juce::String(last.k) + kDot + "at " + seconds(last.atCs), right.removeFromTop(small + 6.0f),
               juce::Justification::centred, true);

    g.drawText("Recent", chips.translated(0.0f, -(small + 6.0f)).withHeight(small + 4.0f),
               juce::Justification::centredLeft, false);
    float cxs = chips.getX();
    const juce::Font chipFont(small * 1.05f);
    for (auto it = state.recent.rbegin(); it != state.recent.rend(); ++it) {
        const auto t = resultText(*it);
        const float w = chipFont.getStringWidthFloat(t) + 14.0f;
        if (cxs + w > chips.getRight())
            break;
        const auto c = resultColour(*it);
        g.setColour(c.withAlpha(0.18f));
        g.fillRoundedRectangle(cxs, chips.getY(), w, chips.getHeight(), 4.0f);
        g.setColour(c);
        g.setFont(chipFont);
        g.drawText(t, juce::Rectangle<float>(cxs, chips.getY(), w, chips.getHeight()), juce::Justification::centred,
                   false);
        cxs += w + 6.0f;
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
    else if (auto *iw = dynamic_cast<InformativeWaitReductionTransformer *>(&rt))
        view.setState(iw->waitState(), now, "Informative WAIT");
    else
        view.setState({}, now, "");
}

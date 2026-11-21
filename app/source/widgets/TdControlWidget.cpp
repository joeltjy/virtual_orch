#include "VirtualOrch/widgets/TdControlWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/AppTime.h"
#include "VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

const juce::Colour kInk{juce::Colours::lightgrey};
const juce::Colour kTarget{0xff4fc3f7};   // target TD
const juce::Colour kMeasured{0xffffb347}; // measured TD
const juce::Colour kGamma{0xff81c784};    // gamma

auto tdText(int32_t td) -> juce::String {
    return td == TemporalDensity::kAuto ? juce::String("AUTO") : juce::String(td);
}

} // namespace

auto TdControlView::setRows(std::vector<Row> nextRows, double nowSeconds) -> void {
    rows = std::move(nextRows);
    now = nowSeconds;
    for (const auto &r: rows) {
        auto it = std::find_if(history.begin(), history.end(), [&](const auto &h) { return h.first == r.model; });
        if (it == history.end()) {
            history.push_back({r.model, {}});
            it = history.end() - 1;
        }
        it->second.push_back({now, r.state.target, r.state.measured, r.state.gamma, r.state.frozen});
        while (! it->second.empty() && it->second.front().seconds < now - kHistorySeconds)
            it->second.pop_front();
    }
    repaint();
}

auto TdControlView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    auto area = getLocalBounds().toFloat().reduced(6.0f);
    const float rowH = 20.0f * textScale;
    const auto font = juce::Font(13.0f * textScale);
    const auto bold = juce::Font(13.0f * textScale, juce::Font::bold);

    if (rows.empty()) {
        g.setColour(kInk.withAlpha(0.6f));
        g.setFont(font);
        g.drawText("Dynamic LogitTD runs with the W2/W3, T3 and Hybrid reductions", area, juce::Justification::centred);
        return;
    }

    // Table: model | target | measured | K | gamma | S | state
    const std::array<const char *, 7> heads{"model", "target", "actual", "K", "gamma", "S", ""};
    const std::array<float, 7> widths{0.10f, 0.12f, 0.12f, 0.12f, 0.12f, 0.12f, 0.30f};
    const auto cell = [&](float y, int col, const juce::String &text, juce::Colour colour, const juce::Font &f) {
        float x = area.getX();
        for (int i = 0; i < col; ++i)
            x += widths[static_cast<size_t>(i)] * area.getWidth();
        g.setColour(colour);
        g.setFont(f);
        g.drawText(text, juce::Rectangle<float>(x, y, widths[static_cast<size_t>(col)] * area.getWidth(), rowH),
                   juce::Justification::centredLeft, false);
    };
    float y = area.getY();
    for (int c = 0; c < 7; ++c)
        cell(y, c, heads[static_cast<size_t>(c)], kInk.withAlpha(0.55f), font);
    y += rowH;
    for (const auto &r: rows) {
        const auto &s = r.state;
        const auto ink = kInk.withAlpha(s.frozen ? 0.45f : 0.95f);
        cell(y, 0, r.model, ink, bold);
        cell(y, 1, tdText(s.target), kTarget.withMultipliedAlpha(s.frozen ? 0.6f : 1.0f), bold);
        cell(y, 2, juce::String(s.measured, 1), kMeasured.withMultipliedAlpha(s.frozen ? 0.6f : 1.0f), bold);
        cell(y, 3, juce::String(s.k, 2), ink, font);
        cell(y, 4, juce::String(s.gamma, 2), kGamma.withMultipliedAlpha(s.frozen ? 0.6f : 1.0f), font);
        cell(y, 5, juce::String(s.sensitivity, 2), ink, font);
        cell(y, 6, s.frozen ? "frozen: " + juce::String(s.why) : juce::String(s.why), ink, font);
        y += rowH;
    }

    // Graph: TD on a log scale (1 .. 64) for each model, gamma in a strip below.
    auto graph = area.withTrimmedTop(y - area.getY() + 6.0f);
    if (graph.getHeight() < 40.0f)
        return;
    const float legendH = rowH;
    auto legend = graph.removeFromBottom(legendH);
    const float panelGap = 6.0f;
    const auto panels = static_cast<float>(history.size());
    const float panelH = (graph.getHeight() - panelGap * (panels - 1.0f)) / panels;
    float top = graph.getY();
    for (const auto &[model, samples]: history) {
        const juce::Rectangle<float> panel(graph.getX(), top, graph.getWidth(), panelH);
        top += panelH + panelGap;
        auto tdArea = panel.withTrimmedBottom(panelH * 0.3f);
        auto gammaArea = panel.withTrimmedTop(panelH * 0.72f);
        g.setColour(kInk.withAlpha(0.15f));
        g.drawRect(panel, 1.0f);
        // gridlines at TD 1 4 16 64
        const auto yTd = [&](double td) {
            // TD 0.5 .. 90 (log), so 64 and its label sit inside the panel
            const double v = std::clamp(std::log2(std::max(td, 0.5)), -1.0, 6.5);
            return tdArea.getBottom() - static_cast<float>((v + 1.0) / 7.5) * tdArea.getHeight();
        };
        g.setFont(juce::Font(10.0f * textScale));
        for (const double td: {1.0, 4.0, 16.0, 64.0}) {
            g.setColour(kInk.withAlpha(0.12f));
            g.drawHorizontalLine(juce::roundToInt(yTd(td)), tdArea.getX(), tdArea.getRight());
            g.setColour(kInk.withAlpha(0.4f));
            g.drawText(juce::String(static_cast<int>(td)), juce::Rectangle<float>(tdArea.getX() + 2.0f, yTd(td) - 7.0f, 24.0f, 12.0f),
                       juce::Justification::centredLeft, false);
        }
        g.setColour(kInk.withAlpha(0.6f));
        g.drawText(model, panel.withHeight(14.0f * textScale).reduced(4.0f, 0.0f), juce::Justification::topRight, false);
        const auto xOf = [&](double seconds) {
            return panel.getRight() - static_cast<float>((now - seconds) / kHistorySeconds) * panel.getWidth();
        };
        juce::Path target, measured, gamma;
        bool targetOn = false;
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto &s = samples[i];
            const float x = xOf(s.seconds);
            if (s.target != TemporalDensity::kAuto) {
                const float ty = yTd(s.target);
                if (! targetOn)
                    target.startNewSubPath(x, ty);
                else
                    target.lineTo(x, ty);
                targetOn = true;
            } else {
                targetOn = false;
            }
            const float my = yTd(s.measured);
            const float gy = gammaArea.getCentreY()
                             - static_cast<float>(s.gamma / TdController::kMaxGamma) * gammaArea.getHeight() * 0.48f;
            if (i == 0) {
                measured.startNewSubPath(x, my);
                gamma.startNewSubPath(x, gy);
            } else {
                measured.lineTo(x, my);
                gamma.lineTo(x, gy);
            }
        }
        g.setColour(kInk.withAlpha(0.15f));
        g.drawHorizontalLine(juce::roundToInt(gammaArea.getCentreY()), gammaArea.getX(), gammaArea.getRight());
        juce::Path dashed;
        const float dashes[] = {5.0f, 4.0f};
        juce::PathStrokeType(1.6f).createDashedStroke(dashed, target, dashes, 2);
        g.setColour(kTarget);
        g.fillPath(dashed);
        g.setColour(kMeasured);
        g.strokePath(measured, juce::PathStrokeType(1.8f));
        g.setColour(kGamma);
        g.strokePath(gamma, juce::PathStrokeType(1.4f));
    }
    // legend
    g.setFont(font);
    float lx = legend.getX();
    for (const auto &[text, colour]: std::vector<std::pair<juce::String, juce::Colour>>{
             {"target TD", kTarget}, {"actual TD", kMeasured}, {"gamma (0 = midline)", kGamma}}) {
        g.setColour(colour);
        g.fillRect(lx, legend.getCentreY() - 2.0f, 14.0f, 4.0f);
        g.setColour(kInk.withAlpha(0.7f));
        const float w = font.getStringWidthFloat(text) + 12.0f;
        g.drawText(text, juce::Rectangle<float>(lx + 18.0f, legend.getY(), w, legend.getHeight()),
                   juce::Justification::centredLeft, false);
        lx += 18.0f + w + 10.0f;
    }
}

TdControlWidget::TdControlWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceTdControlTitle, UiConstants::workspaceWidgetTypeTdControl),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(8);
}

TdControlWidget::~TdControlWidget() {
    stopTimer();
}

auto TdControlWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto TdControlWidget::timerCallback() -> void {
    std::vector<TdControlView::Row> rows;
    if (auto *rt = dynamic_cast<TwoInstrumentAnticipatedReductionTransformer *>(&session.activeReduction()))
        for (auto &v: rt->getTdControlStates())
            rows.push_back({v.model, v.state});
    view.setRows(std::move(rows), AppTime::hiResMs() / 1000.0);
}

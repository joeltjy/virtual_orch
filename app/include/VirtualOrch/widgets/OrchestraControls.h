// OrchestraControls.h developed by Bob Tianqi Wei
#pragma once

#include <JuceHeader.h>

class AppSession;

class OrchestraControls final : public juce::TabbedComponent {
public:
    OrchestraControls(AppSession &session, juce::Component &seatingPlan);
    ~OrchestraControls() override;

    void updateVisibility();

private:
    class ConductingPage;
    class TangiblePage;

    void currentTabChanged(int index, const juce::String &) override;

    std::unique_ptr<ConductingPage> conducting;
    std::unique_ptr<TangiblePage> tangible;
};

#include <JuceHeader.h>

#include <cstdio>
#include <cstdlib>
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/LaunchpadGrid.h"
#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/ui/AppRootComponent.h"

namespace {

class DesktopLookAndFeel final : public juce::LookAndFeel_V4 {
public:
    auto getTextButtonFont(juce::TextButton &, int height) -> juce::Font override {
        return controlFont(height);
    }

    auto getComboBoxFont(juce::ComboBox &box) -> juce::Font override {
        return controlFont(box.getHeight());
    }

    void drawComboBox(juce::Graphics &g, int width, int height, bool,
                      int, int, int, int, juce::ComboBox &box) override {
        const auto bounds = juce::Rectangle<float>(0, 0, float(width), float(height)).reduced(0.5f);
        const auto opacity = box.isEnabled() ? 1.0f : 0.45f;
        g.setColour(box.findColour(juce::ComboBox::backgroundColourId).withMultipliedAlpha(opacity));
        g.fillRoundedRectangle(bounds, radius);
        g.setColour(box.findColour(juce::ComboBox::outlineColourId).withMultipliedAlpha(opacity));
        g.drawRoundedRectangle(bounds, radius, 1.0f);

        const float x = float(width) - 16.0f;
        const float y = float(height) / 2.0f;
        juce::Path arrow;
        arrow.startNewSubPath(x - 4.0f, y - 2.0f);
        arrow.lineTo(x, y + 2.0f);
        arrow.lineTo(x + 4.0f, y - 2.0f);
        g.setColour(box.findColour(juce::ComboBox::arrowColourId).withMultipliedAlpha(opacity));
        g.strokePath(arrow, juce::PathStrokeType(1.5f));
    }

    void fillTextEditorBackground(juce::Graphics &g, int width, int height, juce::TextEditor &editor) override {
        g.setColour(editor.findColour(juce::TextEditor::backgroundColourId));
        g.fillRoundedRectangle(juce::Rectangle<float>(0, 0, float(width), float(height)), radius);
    }

    void drawTextEditorOutline(juce::Graphics &g, int width, int height, juce::TextEditor &editor) override {
        const bool focused = editor.isEnabled() && ! editor.isReadOnly() && editor.hasKeyboardFocus(true);
        g.setColour(editor.findColour(focused ? juce::TextEditor::focusedOutlineColourId
                                             : juce::TextEditor::outlineColourId));
        g.drawRoundedRectangle(juce::Rectangle<float>(0, 0, float(width), float(height)).reduced(0.5f), radius, 1.0f);
    }

private:
    static constexpr float radius = UiConstants::chromeCornerRadius;

    static auto controlFont(int height) -> juce::Font {
        return juce::Font(juce::jmin(UiConstants::chromeControlFontSize, height * 0.75f));
    }
};

} // namespace

//==============================================================================
class VirtualOrchApplication : public juce::JUCEApplication {
public:
    //==============================================================================
    VirtualOrchApplication() {
    }

    const juce::String getApplicationName() override { return "Virtual Orchestra"; }
    const juce::String getApplicationVersion() override { return ProjectInfo::versionString; }
    bool moreThanOneInstanceAllowed() override { return true; }

    //==============================================================================
    void initialise(const juce::String &commandLine) override {
        // This method is where you should put your application's initialisation code..
        // One pair of logs per launch with the same stamp (the Replay page pairs them by it):
        // Logs/sessions/session_<stamp>.log and Logs/oplogs/oplog_<stamp>.jsonl.gz. The oplog's
        // format: docs/OPLOGS.txt.
        const auto stamp = juce::Time::getCurrentTime().formatted("%Y-%m-%d_%H-%M-%S");
        auto sessionFile = sessionLogsDir().getChildFile("session_" + stamp + ".log");
        auto oplogFile = oplogsDir().getChildFile("oplog_" + stamp + ".jsonl.gz");
        for (int n = 2; sessionFile.exists() || oplogFile.exists(); ++n) { // two launches in one second
            sessionFile = sessionLogsDir().getChildFile("session_" + stamp + "_" + juce::String(n) + ".log");
            oplogFile = oplogsDir().getChildFile("oplog_" + stamp + "_" + juce::String(n) + ".jsonl.gz");
        }
        sessionLogsDir().createDirectory();
        oplogsDir().createDirectory();
        SessionLog::start(sessionFile);
        // Which Launchpad layout this session uses: LogReplay replays logs without this line
        // (recorded before Nov 2026) with the layout-1 grid and pedal.
        SessionLog::add("SESSION", "launchpad layout " + juce::String(LaunchpadGrid::kLayoutVersion));
        OpLog::start(oplogFile);

        const auto text = UiConstants::workspaceWidgetTitleTextColour;
        const auto control = UiConstants::controlBackground;
        const auto accent = UiConstants::controlAccent;
        const auto outline = UiConstants::workspaceWidgetBorderColour;
        lookAndFeel.setColour(juce::ResizableWindow::backgroundColourId, UiConstants::workspacePageBackground);
        lookAndFeel.setColour(juce::Label::textColourId, text);
        lookAndFeel.setColour(juce::TextButton::buttonColourId, control);
        lookAndFeel.setColour(juce::TextButton::buttonOnColourId, accent);
        lookAndFeel.setColour(juce::TextButton::textColourOffId, text);
        lookAndFeel.setColour(juce::TextButton::textColourOnId, UiConstants::workspacePageBackground);
        lookAndFeel.setColour(juce::ToggleButton::textColourId, text);
        lookAndFeel.setColour(juce::ToggleButton::tickColourId, accent);
        lookAndFeel.setColour(juce::ToggleButton::tickDisabledColourId, outline);
        lookAndFeel.setColour(juce::ComboBox::backgroundColourId, control);
        lookAndFeel.setColour(juce::ComboBox::textColourId, text);
        lookAndFeel.setColour(juce::ComboBox::arrowColourId, text);
        lookAndFeel.setColour(juce::ComboBox::outlineColourId, outline);
        lookAndFeel.setColour(juce::TextEditor::backgroundColourId, UiConstants::workspacePromptEditorBackground);
        lookAndFeel.setColour(juce::TextEditor::textColourId, text);
        lookAndFeel.setColour(juce::TextEditor::outlineColourId, outline);
        lookAndFeel.setColour(juce::TextEditor::focusedOutlineColourId, accent);
        lookAndFeel.setColour(juce::PopupMenu::backgroundColourId, UiConstants::workspaceToolbarBackground);
        lookAndFeel.setColour(juce::PopupMenu::textColourId, text);
        lookAndFeel.setColour(juce::PopupMenu::highlightedBackgroundColourId, control);
        lookAndFeel.setColour(juce::PopupMenu::highlightedTextColourId, text);
        lookAndFeel.setColour(juce::TableHeaderComponent::backgroundColourId, UiConstants::workspaceWidgetTitleBarBackground);
        lookAndFeel.setColour(juce::TableHeaderComponent::textColourId, text);
        lookAndFeel.setColour(juce::TableHeaderComponent::outlineColourId, outline);
        lookAndFeel.setColour(juce::TableHeaderComponent::highlightColourId, control);
        lookAndFeel.setColour(juce::TabbedButtonBar::tabOutlineColourId, outline);
        lookAndFeel.setColour(juce::TabbedButtonBar::tabTextColourId, text);
        lookAndFeel.setColour(juce::TabbedButtonBar::frontTextColourId, accent);
        juce::LookAndFeel::setDefaultLookAndFeel(&lookAndFeel);

        mainWindow.reset(new MainWindow(getApplicationName()));
    }

    void shutdown() override {
        // Add your application's shutdown code here..

        mainWindow = nullptr; // (deletes our window)
        juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
        OpLog::stop();
        SessionLog::stop();
        // Everything is saved and every worker joined: end here, without the exit handlers. Run,
        // they aborted about every other quit with "corrupted double-linked list", detected in
        // libcuda's own teardown after the CUDA sessions (AddressSanitizer finds nothing in our
        // code; with CUDA hidden it never happens).
        std::fflush(nullptr);
        std::_Exit(0);
    }

    //==============================================================================
    void systemRequestedQuit() override {
        // This is called when the app is being asked to quit: you can ignore this
        // request and let the app carry on running, or call quit() to allow the app to close.
        quit();
    }

    void anotherInstanceStarted(const juce::String &commandLine) override {
        // When another instance of the app is launched while this one is running,
        // this method is invoked, and the commandLine parameter tells you what
        // the other instance's command-line arguments were.
    }

    //==============================================================================
    /*
        This class implements the desktop window that contains an instance of
        our MainComponent class.
    */
    class MainWindow : public juce::DocumentWindow {
    public:
        MainWindow(juce::String name)
            : DocumentWindow(name,
                             juce::Desktop::getInstance().getDefaultLookAndFeel()
                             .findColour(juce::ResizableWindow::backgroundColourId),
                             DocumentWindow::allButtons) {
            setUsingNativeTitleBar(true);
            // false: don't shrink the window to AppRoot/Settings preferred size (~700x800).
            setContentOwned(new AppRootComponent(), false);
            setResizable(true, true);
            setResizeLimits(640, 480, 10000, 10000);

            if (auto *display = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()) {
                auto area = display->userArea.reduced(40);
                setBounds(area);
            } else {
                centreWithSize(1280, 800);
            }

            setVisible(true);
        }

        void closeButtonPressed() override {
            // This is called when the user tries to close this window. Here, we'll just
            // ask the app to quit when this happens, but you can change this to do
            // whatever you need.
            JUCEApplication::getInstance()->systemRequestedQuit();
        }

        /* Note: Be careful if you override any DocumentWindow methods - the base
           class uses a lot of them, so by overriding you might break its functionality.
           It's best to do all your work in your content component instead, but if
           you really have to override any DocumentWindow methods, make sure your
           subclass also calls the superclass's method.
        */

    private:
        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainWindow)
    };

private:
    DesktopLookAndFeel lookAndFeel;
    std::unique_ptr<MainWindow> mainWindow;
};

//==============================================================================
// This macro generates the main() routine that launches the app.
START_JUCE_APPLICATION(VirtualOrchApplication)

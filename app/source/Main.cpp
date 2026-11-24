#include <JuceHeader.h>

#include <cstdio>
#include <cstdlib>
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/LaunchpadGrid.h"
#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/ui/AppRootComponent.h"

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

        mainWindow.reset(new MainWindow(getApplicationName()));
    }

    void shutdown() override {
        // Add your application's shutdown code here..

        mainWindow = nullptr; // (deletes our window)
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
    std::unique_ptr<MainWindow> mainWindow;
};

//==============================================================================
// This macro generates the main() routine that launches the app.
START_JUCE_APPLICATION(VirtualOrchApplication)

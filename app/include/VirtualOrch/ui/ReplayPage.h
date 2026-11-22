#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include "VirtualOrch/sim/OpLogReader.h"
#include "VirtualOrch/sim/ReplayDriver.h"
#include "VirtualOrch/sim/Rerun.h"

class AppSession;
class WorkspacePage;

/**
 * Replay page: load a past launch from Documents/virtual-orch/Logs, pick one of its sessions
 * (a Start -> Stop), and step through its operation log.
 *
 *  - Left: the session's records (filterable). Clicking one moves the replay there.
 *  - Right, "Workspace": the usual workspace, view-only, showing the replay's AppSession at that
 *    moment (ReplayDriver runs the session log through the app's pipeline up to it).
 *  - Right, "Model inputs": what every model had as input at that record (OpLogReader::stateAt):
 *    each RT model's token context, the OT's encoder input, vocsep's window, and the queues
 *    between the stages.
 *  - Right, "Draw": a selected draw's chosen token and its top 10 (RT_DRAW, OT_DRAW), a WAIT
 *    decision, or a vocsep assignment.
 *
 * While a replay is loaded, live generation cannot start (AppSession::replayActive).
 */
class ReplayPage : public juce::Component, private juce::ListBoxModel, private juce::Timer {
public:
    explicit ReplayPage(AppSession &liveSession);
    ~ReplayPage() override;

    auto resized() -> void override;
    auto paint(juce::Graphics &g) -> void override;

private:
    // ListBoxModel
    auto getNumRows() -> int override;
    auto paintListBoxItem(int row, juce::Graphics &g, int width, int height, bool selected) -> void override;
    auto selectedRowsChanged(int lastRowSelected) -> void override;

    auto timerCallback() -> void override;

    auto refreshLaunches() -> void;
    auto loadSelectedLaunch() -> void;
    auto closeReplay() -> void;
    auto selectSession(int sessionIndex) -> void;
    auto applyFilter() -> void;
    auto showRecord(size_t recordIndex) -> void;
    /** Drop the workspace (before anything that ends the replay's session). */
    auto detachWorkspace() -> void;
    auto attachWorkspace(AppSession *replaySession) -> void;
    [[nodiscard]] auto inputsText(size_t recordIndex) const -> juce::String;
    [[nodiscard]] auto drawText(size_t recordIndex) const -> juce::String;
    /** Rerun the selected draw through the model (a worker thread), show and save the result. */
    auto rerunSelected() -> void;
    [[nodiscard]] static auto rerunText(const Rerun::Result &r) -> juce::String;

    AppSession &live;
    std::vector<OpLogReader::Launch> launches;
    std::shared_ptr<OpLogReader::Log> log; // shared: a running rerun keeps it alive
    std::unique_ptr<ReplayDriver> driver;
    std::vector<size_t> rows; // record indices shown in the list
    int currentSession = -1;

    juce::Component toolbar;
    juce::ComboBox launchCombo, sessionCombo;
    juce::TextButton refreshButton{"Refresh"}, loadButton{"Load"}, closeButton{"Close replay"};
    juce::TextEditor filterEditor;
    juce::Label statusLabel;
    juce::ListBox recordList{"records", this};
    juce::TabbedComponent rightTabs{juce::TabbedButtonBar::TabsAtTop};
    juce::Component workspaceHolder;
    juce::TextEditor inputsView, drawView;
    juce::Component drawPanel;
    juce::TextButton rerunButton{"Run through the model again"};
    juce::Label rerunNote;
    std::shared_ptr<Rerun::Engine> rerunEngine = std::make_shared<Rerun::Engine>();
    std::atomic<bool> rerunning{false};
    std::optional<size_t> selectedRecord;
    juce::File loadedOplog;
    std::unique_ptr<WorkspacePage> workspace;
    std::atomic<int> epoch{0}; // bumped whenever the workspace is dropped (stale hand-outs ignored)

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReplayPage)
};

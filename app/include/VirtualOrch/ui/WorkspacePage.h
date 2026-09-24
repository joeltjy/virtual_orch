#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/ViewStore.h"
#include "VirtualOrch/ui/WorkspaceCanvas.h"

class AppSession;

/**
 * Workspace tab content: toolbar strip + canvas region.
 */
class WorkspacePage : public juce::Component {
public:
    explicit WorkspacePage(AppSession &session);

    ~WorkspacePage() override = default;

    auto paint(juce::Graphics &g) -> void override;

    auto resized() -> void override;

private:
    auto refreshViewCombo(const juce::String &selectName) -> void;

    auto loadSelectedView() -> void;

    auto saveCurrentView() -> void;

    auto performSave(const juce::String &viewName) -> void;

    auto promptSaveAs() -> void;

    auto applySaveAsName(const juce::String &rawName) -> void;

    auto promptNewView() -> void;

    auto applyNewViewName(const juce::String &rawName) -> void;

    auto confirmOverwriteThenSave(const juce::String &viewName) -> void;

    /** Persist built-in layouts once if missing; never overwrite a saved view. */
    auto ensureBuiltinViews() -> void;

    auto setCurrentViewAsDefault() -> void;

    [[nodiscard]] auto getStartupViewName() const -> juce::String;

    auto markDirty() -> void;

    auto clearDirty() -> void;

    auto updateDirtyUi() -> void;

    [[nodiscard]] auto getSelectedViewName() const -> juce::String;

    juce::Component toolbar;
    juce::Label toolbarLabel;
    juce::TextButton addButton{"Add"};
    juce::ComboBox viewCombo;
    juce::TextButton newViewButton{"New View"};
    juce::TextButton saveButton{"Save"};
    juce::TextButton saveAsButton{"Save As"};
    juce::TextButton setDefaultButton{"Set Default"};

    ViewStore viewStore;
    WorkspaceCanvas canvas;
    AppSession &session;
    juce::String currentViewName;
    bool ignoreViewComboCallbacks = false;
    bool isDirty = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WorkspacePage)
};

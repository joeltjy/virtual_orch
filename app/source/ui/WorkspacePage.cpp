#include "VirtualOrch/ui/WorkspacePage.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>
#include <array>

WorkspacePage::WorkspacePage(AppSession &sessionIn) : session(sessionIn) {
    setOpaque(true);

    canvas.setSession(&session);
    canvas.onLayoutChanged = [this] { markDirty(); };

    addAndMakeVisible(toolbar);

    toolbarLabel.setText("Workspace", juce::dontSendNotification);
    toolbarLabel.setJustificationType(juce::Justification::centredLeft);
    toolbarLabel.setFont(juce::Font(UiConstants::chromeTitleFontSize, juce::Font::bold));
    toolbarLabel.setColour(juce::Label::textColourId, UiConstants::workspaceToolbarTitleColour);
    toolbar.addAndMakeVisible(toolbarLabel);

    viewCombo.setTextWhenNothingSelected(UiConstants::workspaceViewsComboPlaceholder);
    viewCombo.setTextWhenNoChoicesAvailable(UiConstants::workspaceViewsComboPlaceholder);
    viewCombo.onChange = [this] {
        if (ignoreViewComboCallbacks)
            return;
        loadSelectedView();
    };

    addButton.onClick = [this] {
        juce::PopupMenu menu;
        menu.addItem(UiConstants::workspaceAddMenuPromptItemId, UiConstants::workspaceAddMenuPromptItem);
        menu.addItem(UiConstants::workspaceAddMenuGenerationItemId,
                     UiConstants::workspaceAddMenuGenerationItem);
        menu.addItem(UiConstants::workspaceAddMenuTransportItemId,
                     UiConstants::workspaceAddMenuTransportItem);
        menu.addItem(UiConstants::workspaceAddMenuTokenPianoRollItemId,
                     UiConstants::workspaceAddMenuTokenPianoRollItem);
        menu.addItem(UiConstants::workspaceAddMenuConditioningPianoRollItemId,
                     UiConstants::workspaceAddMenuConditioningPianoRollItem);
        menu.addItem(UiConstants::workspaceAddMenuOrchestrationMidiPianoRollItemId,
                     UiConstants::workspaceAddMenuOrchestrationMidiPianoRollItem);
        menu.addItem(UiConstants::workspaceAddMenuOrchestrationConditioningPianoRollItemId,
                     UiConstants::workspaceAddMenuOrchestrationConditioningPianoRollItem);
        menu.addItem(UiConstants::workspaceAddMenuOrchestrationReductionPianoRollItemId,
                     UiConstants::workspaceAddMenuOrchestrationReductionPianoRollItem);
        menu.addItem(UiConstants::workspaceAddMenuActiveInstrumentsItemId,
                     UiConstants::workspaceAddMenuActiveInstrumentsItem);
        menu.addItem(UiConstants::workspaceAddMenuOrchestrationOutputVisualizerItemId,
                     UiConstants::workspaceAddMenuOrchestrationOutputVisualizerItem);
        menu.addItem(UiConstants::workspaceAddMenuPlaybackOutputItemId,
                     UiConstants::workspaceAddMenuPlaybackOutputItem);
        menu.addItem(UiConstants::workspaceAddMenuReductionTransformerOutputItemId,
                     UiConstants::workspaceAddMenuReductionTransformerOutputItem);
        menu.addItem(UiConstants::workspaceAddMenuVocsepOutputItemId,
                     UiConstants::workspaceAddMenuVocsepOutputItem);
        menu.addItem(UiConstants::workspaceAddMenuReductionModelInputItemId,
                     UiConstants::workspaceAddMenuReductionModelInputItem);
        menu.addItem(UiConstants::workspaceAddMenuModeItemId, UiConstants::workspaceAddMenuModeItem);
        menu.addItem(UiConstants::workspaceAddMenuInstrumentLogitsItemId,
                     UiConstants::workspaceAddMenuInstrumentLogitsItem);
        menu.addItem(UiConstants::workspaceAddMenuDurationLogitsItemId,
                     UiConstants::workspaceAddMenuDurationLogitsItem);
        menu.addItem(UiConstants::workspaceAddMenuReductionTemperatureItemId,
                     UiConstants::workspaceAddMenuReductionTemperatureItem);
        menu.addItem(UiConstants::workspaceAddMenuReductionTauItemId,
                     UiConstants::workspaceAddMenuReductionTauItem);
        menu.addItem(UiConstants::workspaceAddMenuOctaveDeltaItemId,
                     UiConstants::workspaceAddMenuOctaveDeltaItem);
        menu.addItem(UiConstants::workspaceAddMenuLogitAdjustmentsItemId,
                     UiConstants::workspaceAddMenuLogitAdjustmentsItem);
        menu.addItem(UiConstants::workspaceAddMenuInstrumentActiveLightItemId,
                     UiConstants::workspaceAddMenuInstrumentActiveLightItem);
        menu.addItem(UiConstants::workspaceAddMenuInstrumentGroupActivityItemId,
                     UiConstants::workspaceAddMenuInstrumentGroupActivityItem);
        menu.addItem(UiConstants::workspaceAddMenuTdViewerItemId,
                     UiConstants::workspaceAddMenuTdViewerItem);
        menu.addItem(UiConstants::workspaceAddMenuOnsetGapsItemId,
                     UiConstants::workspaceAddMenuOnsetGapsItem);
        menu.addItem(UiConstants::workspaceAddMenuActiveModelItemId,
                     UiConstants::workspaceAddMenuActiveModelItem);
        menu.addItem(UiConstants::workspaceAddMenuUserSelectionItemId,
                     UiConstants::workspaceAddMenuUserSelectionItem);
        menu.addItem(UiConstants::workspaceAddMenuScoreItemId,
                     UiConstants::workspaceAddMenuScoreItem);
        menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&addButton),
                           [this](int result) {
                               if (result == UiConstants::workspaceAddMenuPromptItemId)
                                   canvas.addPromptWidget();
                               else if (result == UiConstants::workspaceAddMenuGenerationItemId)
                                   canvas.addGenerationWidget();
                               else if (result == UiConstants::workspaceAddMenuTransportItemId)
                                   canvas.addTransportWidget();
                               else if (result == UiConstants::workspaceAddMenuTokenPianoRollItemId)
                                   canvas.addTokenPianoRollWidget();
                               else if (result == UiConstants::workspaceAddMenuConditioningPianoRollItemId)
                                   canvas.addConditioningPianoRollWidget();
                               else if (result
                                        == UiConstants::workspaceAddMenuOrchestrationMidiPianoRollItemId)
                                   canvas.addOrchestrationMidiPianoRollWidget();
                               else if (result
                                        == UiConstants::
                                            workspaceAddMenuOrchestrationConditioningPianoRollItemId)
                                   canvas.addOrchestrationConditioningPianoRollWidget();
                               else if (result
                                        == UiConstants::
                                            workspaceAddMenuOrchestrationReductionPianoRollItemId)
                                   canvas.addOrchestrationReductionPianoRollWidget();
                               else if (result == UiConstants::workspaceAddMenuActiveInstrumentsItemId)
                                   canvas.addActiveInstrumentsWidget();
                               else if (result
                                        == UiConstants::
                                            workspaceAddMenuOrchestrationOutputVisualizerItemId)
                                   canvas.addOrchestrationOutputVisualizerWidget();
                               else if (result == UiConstants::workspaceAddMenuPlaybackOutputItemId)
                                   canvas.addPlaybackOutputWidget();
                               else if (result
                                        == UiConstants::workspaceAddMenuReductionTransformerOutputItemId)
                                   canvas.addReductionTransformerOutputWidget();
                               else if (result == UiConstants::workspaceAddMenuVocsepOutputItemId)
                                   canvas.addVocsepOutputWidget();
                               else if (result
                                        == UiConstants::workspaceAddMenuReductionModelInputItemId)
                                   canvas.addReductionModelInputWidget();
                               else if (result == UiConstants::workspaceAddMenuModeItemId)
                                   canvas.addModeWidget();
                               else if (result == UiConstants::workspaceAddMenuInstrumentLogitsItemId)
                                   canvas.addInstrumentLogitsWidget();
                               else if (result == UiConstants::workspaceAddMenuDurationLogitsItemId)
                                   canvas.addDurationLogitsWidget();
                               else if (result
                                        == UiConstants::workspaceAddMenuReductionTemperatureItemId)
                                   canvas.addReductionTemperatureWidget();
                               else if (result == UiConstants::workspaceAddMenuReductionTauItemId)
                                   canvas.addReductionTauWidget();
                               else if (result == UiConstants::workspaceAddMenuOctaveDeltaItemId)
                                   canvas.addOctaveDeltaWidget();
                               else if (result == UiConstants::workspaceAddMenuLogitAdjustmentsItemId)
                                   canvas.addLogitAdjustmentsWidget();
                               else if (result
                                        == UiConstants::workspaceAddMenuInstrumentActiveLightItemId)
                                   canvas.addWidgetByType(
                                       UiConstants::workspaceWidgetTypeInstrumentActiveLight);
                               else if (result
                                        == UiConstants::workspaceAddMenuInstrumentGroupActivityItemId)
                                   canvas.addWidgetByType(
                                       UiConstants::workspaceWidgetTypeInstrumentGroupActivity);
                               else if (result == UiConstants::workspaceAddMenuTdViewerItemId)
                                   canvas.addWidgetByType(UiConstants::workspaceWidgetTypeTdViewer);
                               else if (result == UiConstants::workspaceAddMenuOnsetGapsItemId)
                                   canvas.addWidgetByType(UiConstants::workspaceWidgetTypeOnsetGaps);
                               else if (result == UiConstants::workspaceAddMenuActiveModelItemId)
                                   canvas.addWidgetByType(UiConstants::workspaceWidgetTypeActiveModel);
                               else if (result == UiConstants::workspaceAddMenuUserSelectionItemId)
                                   canvas.addWidgetByType(UiConstants::workspaceWidgetTypeUserSelection);
                               else if (result == UiConstants::workspaceAddMenuScoreItemId)
                                   canvas.addWidgetByType(UiConstants::workspaceWidgetTypeScore);
                           });
    };
    newViewButton.onClick = [this] { promptNewView(); };
    saveButton.onClick = [this] { saveCurrentView(); };
    saveAsButton.onClick = [this] { promptSaveAs(); };
    setDefaultButton.onClick = [this] { setCurrentViewAsDefault(); };

    toolbar.addAndMakeVisible(addButton);
    toolbar.addAndMakeVisible(viewCombo);
    toolbar.addAndMakeVisible(newViewButton);
    toolbar.addAndMakeVisible(saveButton);
    toolbar.addAndMakeVisible(saveAsButton);
    toolbar.addAndMakeVisible(setDefaultButton);

    addAndMakeVisible(canvas);

    ensureBuiltinViews();

    auto names = viewStore.listViewNames();
    juce::String initialName = getStartupViewName();
    if (initialName.isEmpty() || ! names.contains(initialName)) {
        if (names.contains(UiConstants::workspaceOrchestrationDebugViewName))
            initialName = UiConstants::workspaceOrchestrationDebugViewName;
        else if (names.contains(UiConstants::workspaceDefaultViewName))
            initialName = UiConstants::workspaceDefaultViewName;
        else if (names.size() > 0)
            initialName = names[0];
        else
            initialName = {};
    }

    refreshViewCombo(initialName);

    if (initialName.isNotEmpty())
        loadSelectedView();
}

auto WorkspacePage::getStartupViewName() const -> juce::String {
    const auto workspace = session.presetStore.settings.getChildWithName(
        UiConstants::workspaceSettingsWorkspaceChild);
    if (! workspace.isValid())
        return UiConstants::workspaceOrchestrationDebugViewName;

    const auto name =
        workspace.getProperty(UiConstants::workspaceSettingsDefaultViewProperty, {}).toString().trim();
    return name.isNotEmpty() ? name
                             : juce::String(UiConstants::workspaceOrchestrationDebugViewName);
}

auto WorkspacePage::setCurrentViewAsDefault() -> void {
    const auto name = currentViewName.isNotEmpty() ? currentViewName : getSelectedViewName();
    if (name.isEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(
            juce::AlertWindow::WarningIcon,
            UiConstants::workspaceSetDefaultViewTitle,
            UiConstants::workspaceSetDefaultViewEmptyMessage);
        return;
    }

    auto workspace = session.presetStore.settings.getOrCreateChildWithName(
        UiConstants::workspaceSettingsWorkspaceChild, nullptr);
    workspace.setProperty(UiConstants::workspaceSettingsDefaultViewProperty, name, nullptr);
    session.presetStore.saveSettings();

    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::InfoIcon,
        UiConstants::workspaceSetDefaultViewTitle,
        juce::String(UiConstants::workspaceSetDefaultViewMessagePrefix) + name
            + UiConstants::workspaceSetDefaultViewMessageSuffix);
}

auto WorkspacePage::ensureBuiltinViews() -> void {
    // Seed each built-in layout once. After that, Save owns the file — never re-seed.
    const auto seedIfMissing = [this](const juce::String &name, auto addWidgets) {
        if (viewStore.viewExists(name))
            return;

        const auto snapshot = canvas.toVar();
        const bool restore = canvas.getWidgetCount() > 0;

        addWidgets();
        viewStore.saveView(name, canvas.toVar());

        if (restore)
            canvas.fromVar(snapshot);
    };

    seedIfMissing(UiConstants::workspaceDefaultViewName,
                  [this] { canvas.addDefaultWidgets(); });
    seedIfMissing(UiConstants::workspaceInputViewName,
                  [this] { canvas.addInputViewWidgets(); });
    seedIfMissing(UiConstants::workspaceOrchestrationDebugViewName,
                  [this] { canvas.addOrchestrationDebugViewWidgets(); });
    seedIfMissing(UiConstants::workspacePerformanceViewName,
                  [this] { canvas.addPerformanceViewWidgets(); });
    clearDirty();
}

auto WorkspacePage::getSelectedViewName() const -> juce::String {
    if (viewCombo.getSelectedId() <= 0)
        return {};
    return viewCombo.getText().trimCharactersAtEnd("* ").trimEnd();
}

auto WorkspacePage::updateDirtyUi() -> void {
    juce::String label = "Workspace";
    if (isDirty)
        label += UiConstants::workspaceDirtyTitleSuffix;
    toolbarLabel.setText(label, juce::dontSendNotification);
}

auto WorkspacePage::markDirty() -> void {
    if (isDirty)
        return;
    isDirty = true;
    updateDirtyUi();
}

auto WorkspacePage::clearDirty() -> void {
    isDirty = false;
    updateDirtyUi();
}

auto WorkspacePage::refreshViewCombo(const juce::String &selectName) -> void {
    const auto names = viewStore.listViewNames();

    ignoreViewComboCallbacks = true;
    viewCombo.clear(juce::dontSendNotification);

    for (int i = 0; i < names.size(); ++i)
        viewCombo.addItem(names[i], i + 1);

    if (selectName.isNotEmpty()) {
        const int index = names.indexOf(selectName);
        if (index >= 0)
            viewCombo.setSelectedId(index + 1, juce::dontSendNotification);
    }

    currentViewName = getSelectedViewName();
    saveButton.setEnabled(currentViewName.isNotEmpty());
    ignoreViewComboCallbacks = false;
}

auto WorkspacePage::loadSelectedView() -> void {
    const auto name = getSelectedViewName();
    if (name.isEmpty())
        return;

    const auto loaded = viewStore.loadView(name);
    if (name == UiConstants::workspacePerformanceViewName && loaded.isObject()) {
        const auto widgets = loaded.getProperty("widgets", {});
        if (const auto *items = widgets.getArray(); items != nullptr && items->size() == 6) {
            const std::array<juce::Rectangle<int>, 6> oldBounds{{
                {20, 20, 980, 980}, {20, 1020, 980, 300}, {20, 1340, 980, 460},
                {20, 1820, 480, 160}, {520, 1820, 480, 160}, {20, 2000, 980, 160}
            }};
            const std::array<juce::Rectangle<int>, 6> previousBounds{{
                {630, 20, 370, 705}, {20, 745, 980, 400}, {20, 20, 590, 705},
                {20, 1165, 310, 255}, {350, 1165, 310, 255}, {680, 1165, 320, 255}
            }};
            const std::array<const char *, 6> oldTypes{{
                UiConstants::workspaceWidgetTypeActiveInstruments, UiConstants::workspaceWidgetTypeScore,
                UiConstants::workspaceWidgetTypeInstrumentActiveLight,
                UiConstants::workspaceWidgetTypeInstrumentGroupActivity,
                UiConstants::workspaceWidgetTypeTdViewer, UiConstants::workspaceWidgetTypeUserSelection
            }};
            bool oldDefault = true;
            bool previousDefault = true;
            bool widerControlsDefault = true;
            for (int i = 0; i < items->size(); ++i) {
                const auto &item = items->getReference(i);
                const juce::Rectangle<int> bounds{int(item.getProperty("x", 0)), int(item.getProperty("y", 0)),
                                                  int(item.getProperty("w", 0)), int(item.getProperty("h", 0))};
                const auto typeMatches = item.getProperty("type", {}).toString() == oldTypes[i];
                oldDefault &= typeMatches && bounds == oldBounds[i];
                previousDefault &= typeMatches && bounds == previousBounds[i];
                const auto widerControlsBounds = i == 0 ? juce::Rectangle<int>(600, 20, 400, 705)
                    : (i == 2 ? juce::Rectangle<int>(20, 20, 560, 705) : previousBounds[i]);
                widerControlsDefault &= typeMatches && bounds == widerControlsBounds;
            }
            if (oldDefault || previousDefault || widerControlsDefault) {
                canvas.addPerformanceViewWidgets();
                currentViewName = name;
                canvas.setPerformanceControlsEnabled(true);
                canvas.setTextScale(UiConstants::performanceTextScale);
                saveButton.setEnabled(true);
                clearDirty();
                return;
            }
        }
    }
    bool useUpdatedDebugLayout = false;
    if (name == UiConstants::workspaceOrchestrationDebugViewName) {
        struct Placement {
            const char *type;
            juce::Rectangle<int> bounds;
        };
        using namespace UiConstants;
        const std::array<Placement, 14> legacyLayout{{
            {workspaceWidgetTypeOrchestrationMidiPianoRoll, {20, 40, 480, 200}},
            {workspaceWidgetTypeOrchestrationReductionPianoRoll, {20, 260, 480, 200}},
            {workspaceWidgetTypeGeneration, {520, 40, 230, 180}},
            {workspaceWidgetTypeReductionTemperature, {760, 40, 240, 280}},
            {workspaceWidgetTypeMode, {520, 220, 480, 490}},
            {workspaceWidgetTypeDurationLogits, {520, 720, 200, 200}},
            {workspaceWidgetTypeOctaveDelta, {730, 720, 160, 200}},
            {workspaceWidgetTypeReductionTau, {900, 720, 180, 200}},
            {workspaceWidgetTypeActiveInstruments, {520, 930, 480, 490}},
            {workspaceWidgetTypeOrchestrationOutputVisualizer, {20, 510, 480, 240}},
            {workspaceWidgetTypePlaybackOutput, {20, 770, 480, 240}},
            {workspaceWidgetTypeReductionTransformerOutput, {20, 1030, 480, 180}},
            {workspaceWidgetTypeReductionModelInput, {520, 1030, 480, 180}},
            {workspaceWidgetTypeVocsepOutput, {20, 1230, 480, 180}}
        }};
        const auto widgets = loaded.getProperty("widgets", {});
        if (const auto *items = widgets.getArray(); items != nullptr && items->size() == int(legacyLayout.size())) {
            useUpdatedDebugLayout = std::equal(legacyLayout.begin(), legacyLayout.end(), items->begin(),
                [](const auto &placement, const auto &item) {
                    const juce::Rectangle<int> bounds{int(item.getProperty("x", 0)), int(item.getProperty("y", 0)),
                                                      int(item.getProperty("w", 0)), int(item.getProperty("h", 0))};
                    return item.getProperty("type", {}).toString() == placement.type && bounds == placement.bounds;
                });
        }
    }
    if (useUpdatedDebugLayout) {
        canvas.addOrchestrationDebugViewWidgets();
    } else if (loaded.isVoid() || ! canvas.fromVar(loaded)) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Load View",
                                               "Could not load view \"" + name + "\".");
        return;
    }

    currentViewName = name;
    canvas.setPerformanceControlsEnabled(name == UiConstants::workspacePerformanceViewName);
    canvas.setTextScale(name == UiConstants::workspacePerformanceViewName ? UiConstants::performanceTextScale : 1.0f);
    saveButton.setEnabled(true);
    clearDirty();
}

auto WorkspacePage::performSave(const juce::String &viewName) -> void {
    if (! viewStore.saveView(viewName, canvas.toVar())) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Save View",
                                               "Could not save view \"" + viewName + "\".");
        return;
    }

    currentViewName = viewName;
    canvas.setPerformanceControlsEnabled(viewName == UiConstants::workspacePerformanceViewName);
    canvas.setTextScale(viewName == UiConstants::workspacePerformanceViewName ? UiConstants::performanceTextScale
                                                                              : 1.0f);
    refreshViewCombo(viewName);
    clearDirty();
}

auto WorkspacePage::confirmOverwriteThenSave(const juce::String &viewName) -> void {
    juce::Component::SafePointer<WorkspacePage> safeThis(this);
    juce::AlertWindow::showAsync(
        juce::MessageBoxOptions()
            .withIconType(juce::AlertWindow::QuestionIcon)
            .withTitle(UiConstants::workspaceOverwriteSaveTitle)
            .withMessage(juce::String(UiConstants::workspaceOverwriteSaveMessagePrefix) + viewName
                         + UiConstants::workspaceOverwriteSaveMessageSuffix)
            .withButton(UiConstants::workspaceOverwriteConfirmButton)
            .withButton(UiConstants::workspaceOverwriteCancelButton),
        [safeThis, viewName](int result) {
            if (safeThis == nullptr || result != UiConstants::workspaceOverwriteConfirmResult)
                return;
            safeThis->performSave(viewName);
        });
}

auto WorkspacePage::saveCurrentView() -> void {
    if (currentViewName.isEmpty()) {
        promptSaveAs();
        return;
    }

    if (viewStore.viewExists(currentViewName))
        confirmOverwriteThenSave(currentViewName);
    else
        performSave(currentViewName);
}

auto WorkspacePage::promptSaveAs() -> void {
    auto *dialog = new juce::AlertWindow(UiConstants::workspaceSaveAsDialogTitle,
                                         UiConstants::workspaceSaveAsDialogMessage,
                                         juce::AlertWindow::QuestionIcon);
    dialog->addTextEditor(UiConstants::workspaceSaveAsNameFieldId, currentViewName,
                          UiConstants::workspaceSaveAsNameFieldLabel);
    dialog->addButton(UiConstants::workspaceSaveAsConfirmButton,
                      UiConstants::workspaceSaveAsConfirmResult);
    dialog->addButton(UiConstants::workspaceSaveAsCancelButton,
                      UiConstants::workspaceSaveAsCancelResult);

    juce::Component::SafePointer<WorkspacePage> safeThis(this);
    dialog->enterModalState(
        true,
        juce::ModalCallbackFunction::create([safeThis, dialog](int result) {
            std::unique_ptr<juce::AlertWindow> cleanup(dialog);
            if (safeThis == nullptr || result != UiConstants::workspaceSaveAsConfirmResult)
                return;

            const auto rawName =
                dialog->getTextEditorContents(UiConstants::workspaceSaveAsNameFieldId);
            safeThis->applySaveAsName(rawName);
        }),
        true);
}

auto WorkspacePage::applySaveAsName(const juce::String &rawName) -> void {
    auto name = rawName.trim();
    name = name.replaceCharacter('/', '_').replaceCharacter('\\', '_');

    if (name.isEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Save View As",
                                               "View name cannot be empty.");
        return;
    }

    if (viewStore.viewExists(name))
        confirmOverwriteThenSave(name);
    else
        performSave(name);
}

auto WorkspacePage::promptNewView() -> void {
    auto *dialog = new juce::AlertWindow(UiConstants::workspaceNewViewDialogTitle,
                                         UiConstants::workspaceNewViewDialogMessage,
                                         juce::AlertWindow::QuestionIcon);
    dialog->addTextEditor(UiConstants::workspaceSaveAsNameFieldId, {},
                          UiConstants::workspaceSaveAsNameFieldLabel);
    dialog->addButton(UiConstants::workspaceNewViewConfirmButton,
                      UiConstants::workspaceSaveAsConfirmResult);
    dialog->addButton(UiConstants::workspaceNewViewCancelButton,
                      UiConstants::workspaceSaveAsCancelResult);

    juce::Component::SafePointer<WorkspacePage> safeThis(this);
    dialog->enterModalState(
        true,
        juce::ModalCallbackFunction::create([safeThis, dialog](int result) {
            std::unique_ptr<juce::AlertWindow> cleanup(dialog);
            if (safeThis == nullptr || result != UiConstants::workspaceSaveAsConfirmResult)
                return;

            const auto rawName =
                dialog->getTextEditorContents(UiConstants::workspaceSaveAsNameFieldId);
            safeThis->applyNewViewName(rawName);
        }),
        true);
}

auto WorkspacePage::applyNewViewName(const juce::String &rawName) -> void {
    auto name = rawName.trim();
    name = name.replaceCharacter('/', '_').replaceCharacter('\\', '_');

    if (name.isEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               UiConstants::workspaceNewViewDialogTitle,
                                               "View name cannot be empty.");
        return;
    }

    if (! viewStore.viewExists(name)) {
        canvas.beginViewLayout();
        canvas.endViewLayout();
        performSave(name);
        return;
    }

    juce::Component::SafePointer<WorkspacePage> safeThis(this);
    juce::AlertWindow::showAsync(
        juce::MessageBoxOptions()
            .withIconType(juce::AlertWindow::QuestionIcon)
            .withTitle(UiConstants::workspaceOverwriteSaveTitle)
            .withMessage(juce::String(UiConstants::workspaceOverwriteSaveMessagePrefix) + name
                         + UiConstants::workspaceOverwriteSaveMessageSuffix)
            .withButton(UiConstants::workspaceOverwriteConfirmButton)
            .withButton(UiConstants::workspaceOverwriteCancelButton),
        [safeThis, name](int result) {
            if (safeThis == nullptr || result != UiConstants::workspaceOverwriteConfirmResult)
                return;
            safeThis->canvas.beginViewLayout();
            safeThis->canvas.endViewLayout();
            safeThis->performSave(name);
        });
}

auto WorkspacePage::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::workspacePageBackground);

    const auto toolbarBounds = toolbar.getBounds();
    g.setColour(UiConstants::workspaceToolbarBackground);
    g.fillRect(toolbarBounds);
    g.setColour(UiConstants::workspaceToolbarDivider);
    g.drawLine(
        (float) toolbarBounds.getX(),
        (float) toolbarBounds.getBottom() - UiConstants::workspaceToolbarDividerPixelCenterOffset,
        (float) toolbarBounds.getRight(),
        (float) toolbarBounds.getBottom() - UiConstants::workspaceToolbarDividerPixelCenterOffset,
        UiConstants::workspaceToolbarDividerThickness);
}

auto WorkspacePage::resized() -> void {
    auto bounds = getLocalBounds();
    const int rowHeight = UiConstants::workspaceToolbarHeight;
    const int gap = UiConstants::workspaceToolbarControlGap;
    const int labelWidth = UiConstants::workspaceToolbarTitleWidth + 12;
    const int labelGap = UiConstants::workspaceToolbarTitleTrailingGap;
    const int buttonWidth = UiConstants::workspaceToolbarButtonWidth;
    const int viewWidth = UiConstants::workspaceToolbarViewComboWidth;
    const int newWidth = UiConstants::workspaceToolbarNewViewButtonWidth;
    const int saveAsWidth = UiConstants::workspaceToolbarSaveAsButtonWidth;
    const int defaultWidth = UiConstants::workspaceToolbarSetDefaultButtonWidth;
    const int singleRowWidth = labelWidth + labelGap + buttonWidth * 2 + viewWidth + newWidth
                             + saveAsWidth + defaultWidth + gap * 5
                             + UiConstants::workspaceToolbarPaddingX * 2;
    const bool wrap = bounds.getWidth() < singleRowWidth;
    toolbar.setBounds(bounds.removeFromTop(rowHeight * (wrap ? 2 : 1)));
    canvas.setBounds(bounds);

    auto rows = toolbar.getLocalBounds();
    const auto nextRow = [&rows, rowHeight] {
        return rows.removeFromTop(rowHeight).reduced(UiConstants::workspaceToolbarPaddingX,
                                                    UiConstants::workspaceToolbarPaddingY);
    };
    auto row = nextRow();
    toolbarLabel.setBounds(row.removeFromLeft(labelWidth));
    row.removeFromLeft(labelGap);
    const auto place = [&row, gap](auto &control, int width) {
        control.setBounds(row.removeFromLeft(width));
        row.removeFromLeft(gap);
    };
    place(addButton, buttonWidth);
    place(viewCombo, viewWidth);
    if (wrap)
        row = nextRow();
    place(newViewButton, newWidth);
    place(saveButton, buttonWidth);
    place(saveAsButton, saveAsWidth);
    setDefaultButton.setBounds(row.removeFromLeft(defaultWidth));
}

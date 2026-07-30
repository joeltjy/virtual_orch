#include "VirtualOrch/ui/WorkspacePage.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

WorkspacePage::WorkspacePage(AppSession &sessionIn) : session(sessionIn) {
    setOpaque(true);

    canvas.setSession(&session);
    canvas.onLayoutChanged = [this] { markDirty(); };

    addAndMakeVisible(toolbar);

    toolbarLabel.setText("Workspace", juce::dontSendNotification);
    toolbarLabel.setJustificationType(juce::Justification::centredLeft);
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
                           });
    };
    saveButton.onClick = [this] { saveCurrentView(); };
    saveAsButton.onClick = [this] { promptSaveAs(); };

    toolbar.addAndMakeVisible(addButton);
    toolbar.addAndMakeVisible(viewCombo);
    toolbar.addAndMakeVisible(saveButton);
    toolbar.addAndMakeVisible(saveAsButton);

    addAndMakeVisible(canvas);

    ensureDefaultView();
    ensureInputView();

    auto names = viewStore.listViewNames();
    juce::String initialName;
    if (names.contains(UiConstants::workspaceDefaultViewName))
        initialName = UiConstants::workspaceDefaultViewName;
    else if (names.size() > 0)
        initialName = names[0];

    refreshViewCombo(initialName);

    if (initialName.isNotEmpty())
        loadSelectedView();
}

auto WorkspacePage::ensureDefaultView() -> void {
    if (viewStore.listViewNames().size() > 0)
        return;

    canvas.addDefaultWidgets();
    viewStore.saveView(UiConstants::workspaceDefaultViewName, canvas.toVar());
    clearDirty();
}

auto WorkspacePage::ensureInputView() -> void {
    const auto viewHasGeneration = [](const juce::var &viewJson) {
        if (! viewJson.isObject())
            return false;
        const auto widgetsVar = viewJson.getProperty("widgets", juce::var());
        if (! widgetsVar.isArray())
            return false;
        for (const auto &entry : *widgetsVar.getArray()) {
            if (! entry.isObject())
                continue;
            if (entry.getProperty("type", {}).toString() == UiConstants::workspaceWidgetTypeGeneration)
                return true;
        }
        return false;
    };

    if (viewStore.viewExists(UiConstants::workspaceInputViewName)
        && viewHasGeneration(viewStore.loadView(UiConstants::workspaceInputViewName)))
        return;

    const auto snapshot = canvas.toVar();
    const bool restore = canvas.getWidgetCount() > 0;

    canvas.addInputViewWidgets();
    viewStore.saveView(UiConstants::workspaceInputViewName, canvas.toVar());

    if (restore)
        canvas.fromVar(snapshot);

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
    if (loaded.isVoid() || ! canvas.fromVar(loaded)) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Load View",
                                               "Could not load view \"" + name + "\".");
        return;
    }

    if (canvas.getWidgetCount() == 0)
        canvas.addPromptWidget();

    currentViewName = name;
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
    toolbar.setBounds(bounds.removeFromTop(UiConstants::workspaceToolbarHeight));
    canvas.setBounds(bounds);

    auto bar = toolbar.getLocalBounds().reduced(
        UiConstants::workspaceToolbarPaddingX,
        UiConstants::workspaceToolbarPaddingY);
    toolbarLabel.setBounds(bar.removeFromLeft(UiConstants::workspaceToolbarTitleWidth + 12));
    bar.removeFromLeft(UiConstants::workspaceToolbarTitleTrailingGap);

    addButton.setBounds(bar.removeFromLeft(UiConstants::workspaceToolbarButtonWidth));
    bar.removeFromLeft(UiConstants::workspaceToolbarControlGap);
    viewCombo.setBounds(bar.removeFromLeft(UiConstants::workspaceToolbarViewComboWidth));
    bar.removeFromLeft(UiConstants::workspaceToolbarControlGap);
    saveButton.setBounds(bar.removeFromLeft(UiConstants::workspaceToolbarButtonWidth));
    bar.removeFromLeft(UiConstants::workspaceToolbarControlGap);
    saveAsButton.setBounds(bar.removeFromLeft(UiConstants::workspaceToolbarSaveAsButtonWidth));
}

#include "VirtualOrch/ViewStore.h"

ViewStore::ViewStore()
    : viewsDir(juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
                   .getChildFile("virtual-orch")
                   .getChildFile("Views")) {
    viewsDir.createDirectory();
}

auto ViewStore::viewFile(const juce::String &viewName) const -> juce::File {
    return viewsDir.getChildFile(viewName + ".json");
}

auto ViewStore::listViewNames() const -> juce::StringArray {
    juce::StringArray names;
    juce::Array<juce::File> files;
    viewsDir.findChildFiles(files, juce::File::findFiles, false, "*.json");
    for (const auto &file : files)
        names.add(file.getFileNameWithoutExtension());
    names.sort(false);
    return names;
}

auto ViewStore::saveView(const juce::String &viewName, const juce::var &viewJson) const -> bool {
    if (viewName.isEmpty() || viewJson.isVoid())
        return false;

    const auto jsonText = juce::JSON::toString(viewJson, true);
    return viewFile(viewName).replaceWithText(jsonText);
}

auto ViewStore::loadView(const juce::String &viewName) const -> juce::var {
    const auto file = viewFile(viewName);
    if (! file.existsAsFile())
        return {};

    const auto parsed = juce::JSON::parse(file);
    if (parsed.isVoid() || ! parsed.isObject())
        return {};

    return parsed;
}

auto ViewStore::viewExists(const juce::String &viewName) const -> bool {
    return ! viewName.isEmpty() && viewFile(viewName).existsAsFile();
}

#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <VirtualOrch/PresetStore.h>

/** Checked-in presets follow the app's version unless edited here (PresetStore::installBundledPresets). */
namespace {

struct Dirs {
    juce::TemporaryFile root;
    juce::File bundled, presets;
    Dirs() {
        root.getFile().createDirectory();
        bundled = root.getFile().getChildFile("bundled");
        presets = root.getFile().getChildFile("Presets");
        bundled.createDirectory();
        presets.createDirectory();
    }
    ~Dirs() { root.getFile().deleteRecursively(); }
    auto install() const -> void { PresetStore::installBundledPresets(bundled, presets); }
    auto installed(const char *name) const -> juce::String { return presets.getChildFile(name).loadFileAsString(); }
    auto backups() const -> int {
        return presets.getChildFile(".bundled").getNumberOfChildFiles(juce::File::findFiles, "*.bak");
    }
};

} // namespace

TEST(PresetInstall, aMissingPresetIsInstalled) {
    Dirs d;
    d.bundled.getChildFile("v.json").replaceWithText("{\"a\": 1}");
    d.install();
    EXPECT_EQ(d.installed("v.json"), "{\"a\": 1}");
}

TEST(PresetInstall, anUneditedPresetFollowsTheAppsVersion) {
    Dirs d;
    d.bundled.getChildFile("v.json").replaceWithText("{\"a\": 1}");
    d.install();
    d.bundled.getChildFile("v.json").replaceWithText("{\"a\": 2}");
    d.install();
    EXPECT_EQ(d.installed("v.json"), "{\"a\": 2}");
    EXPECT_EQ(d.backups(), 0);
}

TEST(PresetInstall, anEditedPresetIsKept) {
    Dirs d;
    d.bundled.getChildFile("v.json").replaceWithText("{\"a\": 1}");
    d.install();
    d.presets.getChildFile("v.json").replaceWithText("{\"a\": 9}"); // saved over it in the app
    d.bundled.getChildFile("v.json").replaceWithText("{\"a\": 2}");
    d.install();
    EXPECT_EQ(d.installed("v.json"), "{\"a\": 9}");
}

TEST(PresetInstall, aCopyFromBeforeTheRecordsIsUpdatedWithABackup) {
    Dirs d;
    d.presets.getChildFile("v.json").replaceWithText("{\"ot\": \"IodPretrainedExpressive\"}");
    d.bundled.getChildFile("v.json").replaceWithText("{\"ot\": \"IodPretrained\"}");
    d.install();
    EXPECT_EQ(d.installed("v.json"), "{\"ot\": \"IodPretrained\"}");
    EXPECT_EQ(d.backups(), 1);
    d.install(); // recorded now: nothing more happens
    EXPECT_EQ(d.backups(), 1);
}

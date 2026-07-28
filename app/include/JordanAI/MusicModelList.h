#pragma once

#include <cstddef>
#include <map>
#include <JuceHeader.h>

#include "JordanAI/MLFramework/MLFramework.h"

struct MusicModel {
    juce::String friendlyName;
    juce::String name;
    juce::String path;
    juce::String jsonPath;
    MLFramework::ACCELERATOR accelerator;

    // Allows ordered set sorting MusicModel by friendlyName
    friend bool operator <(const MusicModel& lhs, const MusicModel& rhs) {
       return lhs.friendlyName.toLowerCase() < rhs.friendlyName.toLowerCase();
    }
    friend bool operator >(const MusicModel& lhs, const MusicModel& rhs) {
       return lhs.friendlyName.toLowerCase() > rhs.friendlyName.toLowerCase();
    }
    friend bool operator ==(const MusicModel& lhs, const MusicModel& rhs) {
       return (
            lhs.friendlyName == rhs.friendlyName &&
            lhs.name == rhs.name &&
            lhs.path == rhs.path &&
            lhs.jsonPath == rhs.jsonPath &&
            lhs.accelerator == rhs.accelerator
        );
    }
};

class MusicModelList {
    public:
        MusicModelList();

        int buildList();

        auto getList() -> std::map<std::size_t, MusicModel>;

        auto getMusicModelByIndex(std::size_t index) -> MusicModel;

        auto getIndexByMusicModel(const MusicModel& musicModel) -> std::size_t;

        auto getListComboBox() -> juce::StringArray;

        auto getMusicModelByNameAndAccelerator(const juce::String &name, const juce::String &acceleratorName) -> MusicModel;

    private:
        std::map<std::size_t, MusicModel> modelList;
};

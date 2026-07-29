#include <cstddef>
#include <map>
#include <set>

#include "VirtualOrch/MusicModelList.h"

#include <juce_core/juce_core.h>

MusicModelList::MusicModelList() {
    // pass
}

auto MusicModelList::buildList() -> int {
    // Get available accelerators
    std::unordered_set<MLFramework::ACCELERATOR> availableAccels = MLFramework::getAllAvailableAccelerators();

    juce::File modelsDir = juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
            .getChildFile("VirtualOrch")
            .getChildFile("Models");

    std::set<MusicModel> orderedModelList;

    for (const auto &accel : availableAccels) {
        std::string fileExtensionGlob = "*" + MLFramework::ACCELERATOR_FILE_EXTENSIONS.at(accel);

        juce::Array<juce::File> modelFiles;
        modelsDir.findChildFiles(modelFiles, juce::File::findFiles, false, fileExtensionGlob.c_str());

        for (const auto &file: modelFiles) {
            // Don't add if associated .json doesn't exist
            if (!modelsDir.getChildFile(file.getFileNameWithoutExtension() + ".json").exists()) {
                continue;
            }
            // TODO: Parse JSON here and omit if invalid

            juce::String friendlyName = file.getFileNameWithoutExtension() + " [" + juce::String(MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(accel).c_str()) + "]";
            orderedModelList.insert(MusicModel{
                .friendlyName = friendlyName,
                .name = file.getFileNameWithoutExtension(),
                .path = file.getFullPathName(),
                .jsonPath = modelsDir.getChildFile(file.getFileNameWithoutExtension() + ".json").getFullPathName(),
                .accelerator = accel
            });
            DBG("Model added: " + friendlyName);
        }
    }

    for (const auto& model : orderedModelList) {
        std::size_t index = modelList.size();
        modelList.insert({index, model});
    }

    return 0;
}

auto MusicModelList::getList() -> std::map<std::size_t, MusicModel> {
    return modelList;
}

auto MusicModelList::getMusicModelByIndex(std::size_t index) -> MusicModel {
    return modelList.at(index);
}

auto MusicModelList::getIndexByMusicModel(const MusicModel& musicModel) -> std::size_t {
    for (const auto& [index, model] : modelList) {
        if (model == musicModel) {
            return index;
        }
    }
    throw std::runtime_error("MusicModel not found in the list.");
}

auto MusicModelList::getListComboBox() -> juce::StringArray {
    juce::StringArray result;
    for (const auto& [index, model] : modelList) {
        result.add(model.friendlyName);
    }
    return result;
}

auto MusicModelList::getMusicModelByNameAndAccelerator(const juce::String &name, const juce::String &acceleratorName) -> MusicModel {
    for (const auto& [index, model] : modelList) {
        if (model.name == name &&
            juce::String(MLFramework::ACCELERATOR_NAMES.at(model.accelerator)) == acceleratorName) {
            return model;
        }
    }
    throw std::runtime_error("Model with name " + name.toStdString() +
                             " and accelerator " + acceleratorName.toStdString() + " not found.");
}

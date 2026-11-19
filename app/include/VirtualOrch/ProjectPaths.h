#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

/**
 * The one folder every model weight (.onnx) is read from: Documents/virtual_orch_models.
 * Mirrors /data/scratch-oc40/joel/virtual_orch_models on the cluster (same file names), e.g.
 * continuation_model.onnx (v1 / T3''), simultaneous_model_v1/v2.onnx (W2 / W3),
 * instrument_octave_predictions.onnx (IOD pretrained), vocsep.onnx.
 */
[[nodiscard]] inline auto modelsDir() -> juce::File {
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("virtual_orch_models");
}

/** Checked-in .json configs (and presets) under the app/ folder (VIRTUAL_ORCH_APP_DIR). */
[[nodiscard]] inline auto projectModelsDir() -> juce::File {
#if defined(VIRTUAL_ORCH_APP_DIR)
    return juce::File{VIRTUAL_ORCH_APP_DIR};
#else
    return modelsDir();
#endif
}

struct ModelCheckpoint {
    juce::String name;
    juce::File onnxFile;
    juce::File jsonFile;
    bool isDense = false;
    /** JSON "reductionType": the only reduction type that lists it (0 = every dense type). */
    int reductionType = 0;
};

/** True when JSON arch is "dense" (missing/other → AMT / MusicTransformer). */
[[nodiscard]] inline auto checkpointJsonIsDense(const juce::var &parsedJson) -> bool {
    return parsedJson.getProperty("arch", "amt").toString().equalsIgnoreCase("dense");
}

/**
 * Scan the project dir and modelsDir(). Same basename: project dir wins.
 * Only entries with both .onnx and .json are included.
 *
 * Also pairs `app/<name>.json` with ONNX from modelsDir() (Documents/virtual_orch_models),
 * so large weights need not live in the repo. Optional JSON `"onnx"` overrides
 * the stem (e.g. vocsep_reduction_best → vocsep.onnx).
 */
[[nodiscard]] inline auto listModelCheckpoints() -> std::vector<ModelCheckpoint> {
    std::vector<ModelCheckpoint> result;
    juce::StringArray seenNames;

    const auto scanDir = [&](const juce::File &dir) {
        if (! dir.isDirectory())
            return;
        juce::Array<juce::File> onnxFiles;
        dir.findChildFiles(onnxFiles, juce::File::findFiles, false, "*.onnx");
        for (const auto &onnx: onnxFiles) {
            const auto name = onnx.getFileNameWithoutExtension();
            if (seenNames.contains(name))
                continue;
            const auto json = dir.getChildFile(name + ".json");
            if (! json.existsAsFile())
                continue;
            const auto parsed = juce::JSON::parse(json.loadFileAsString());
            if (parsed.isVoid())
                continue;
            seenNames.add(name);
            result.push_back(ModelCheckpoint{
                .name = name,
                .onnxFile = onnx,
                .jsonFile = json,
                .isDense = checkpointJsonIsDense(parsed),
                .reductionType = static_cast<int>(parsed.getProperty("reductionType", 0)),
            });
        }
    };

    scanDir(projectModelsDir());
    scanDir(modelsDir());

    // Cross-dir: project JSON + modelsDir() ONNX.
    {
        const auto projectDir = projectModelsDir();
        if (projectDir.isDirectory()) {
            const std::vector<juce::File> onnxDirs{modelsDir()};
            juce::Array<juce::File> jsonFiles;
            projectDir.findChildFiles(jsonFiles, juce::File::findFiles, false, "*.json");
            for (const auto &json: jsonFiles) {
                const auto name = json.getFileNameWithoutExtension();
                if (seenNames.contains(name))
                    continue;
                const auto parsed = juce::JSON::parse(json.loadFileAsString());
                if (parsed.isVoid())
                    continue;

                // Optional JSON "onnx" overrides basename (e.g. vocsep_reduction_best → vocsep.onnx).
                juce::String onnxStem = name;
                const auto onnxProp = parsed.getProperty("onnx", {}).toString().trim();
                if (onnxProp.isNotEmpty())
                    onnxStem = onnxProp.endsWithIgnoreCase(".onnx")
                                   ? onnxProp.dropLastCharacters(5)
                                   : onnxProp;

                juce::File onnx;
                for (const auto &onnxDir: onnxDirs) {
                    if (! onnxDir.isDirectory())
                        continue;
                    const auto candidate = onnxDir.getChildFile(onnxStem + ".onnx");
                    if (candidate.existsAsFile()) {
                        onnx = candidate;
                        break;
                    }
                }
                if (! onnx.existsAsFile())
                    continue;
                seenNames.add(name);
                result.push_back(ModelCheckpoint{
                    .name = name,
                    .onnxFile = onnx,
                    .jsonFile = json,
                    .isDense = checkpointJsonIsDense(parsed),
                    .reductionType = static_cast<int>(parsed.getProperty("reductionType", 0)),
                });
            }
        }
    }

    return result;
}

[[nodiscard]] inline auto findModelCheckpoint(const juce::String &name) -> std::optional<ModelCheckpoint> {
    for (const auto &cp: listModelCheckpoints()) {
        if (cp.name == name)
            return cp;
    }
    return std::nullopt;
}

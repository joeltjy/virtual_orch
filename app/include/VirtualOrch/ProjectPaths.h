#pragma once

#include <JuceHeader.h>

#include <array>
#include <optional>
#include <vector>

/** Checked-in .onnx / .json under the app/ folder (VIRTUAL_ORCH_APP_DIR). */
[[nodiscard]] inline auto projectModelsDir() -> juce::File {
#if defined(VIRTUAL_ORCH_APP_DIR)
    return juce::File{VIRTUAL_ORCH_APP_DIR};
#else
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("virtual-orch")
        .getChildFile("Models");
#endif
}

/** User Documents models (Chick Corea, trading checkpoints, etc.). */
[[nodiscard]] inline auto documentsModelsDir() -> juce::File {
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("virtual-orch")
        .getChildFile("Models");
}

[[nodiscard]] inline auto onnxExportDir() -> juce::File {
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("onnx_export_092126");
}

/** Older export kept as fallback for names not yet present in onnxExportDir(). */
[[nodiscard]] inline auto onnxExportFallbackDir() -> juce::File {
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("onnx_export_091426");
}

struct ModelCheckpoint {
    juce::String name;
    juce::File onnxFile;
    juce::File jsonFile;
    bool isDense = false;
};

/** True when JSON arch is "dense" (missing/other → AMT / MusicTransformer). */
[[nodiscard]] inline auto checkpointJsonIsDense(const juce::var &parsedJson) -> bool {
    return parsedJson.getProperty("arch", "amt").toString().equalsIgnoreCase("dense");
}

/**
 * Scan project + Documents model dirs. Same basename: project dir wins.
 * Only entries with both .onnx and .json are included.
 *
 * Also pairs `app/<name>.json` with `Documents/onnx_export_092126/<name>.onnx`
 * (fallback: `onnx_export_091426`) so large export weights need not be copied
 * into the repo. Prefer the newer export when both exist (e.g. amt_causal).
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
            });
        }
    };

    scanDir(projectModelsDir());
    scanDir(documentsModelsDir());

    // Cross-dir: project JSON + onnx_export ONNX (e.g. amt_causal).
    {
        const auto projectDir = projectModelsDir();
        if (projectDir.isDirectory()) {
            const std::array exportDirs{onnxExportDir(), onnxExportFallbackDir()};
            juce::Array<juce::File> jsonFiles;
            projectDir.findChildFiles(jsonFiles, juce::File::findFiles, false, "*.json");
            for (const auto &json: jsonFiles) {
                const auto name = json.getFileNameWithoutExtension();
                if (seenNames.contains(name))
                    continue;
                juce::File onnx;
                for (const auto &exportDir: exportDirs) {
                    if (! exportDir.isDirectory())
                        continue;
                    const auto candidate = exportDir.getChildFile(name + ".onnx");
                    if (candidate.existsAsFile()) {
                        onnx = candidate;
                        break;
                    }
                }
                if (! onnx.existsAsFile())
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

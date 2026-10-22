#pragma once

#include <JuceHeader.h>

#include <algorithm>
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

/**
 * Every onnx_export_* folder, newest name first: Documents/ (where the app finds its models on
 * any laptop), then Desktop/ (the reference location of the exports while they are developed).
 * A model is taken from the first folder that has it.
 */
[[nodiscard]] inline auto onnxExportDirs() -> std::vector<juce::File> {
    std::vector<juce::File> dirs;
    for (const auto location: {juce::File::userDocumentsDirectory, juce::File::userDesktopDirectory}) {
        juce::Array<juce::File> found;
        juce::File::getSpecialLocation(location).findChildFiles(found, juce::File::findDirectories,
                                                                 false, "onnx_export_*");
        std::vector<juce::File> sorted(found.begin(), found.end());
        std::sort(sorted.begin(), sorted.end(), [](const juce::File &a, const juce::File &b) {
            return a.getFileName() > b.getFileName();
        });
        dirs.insert(dirs.end(), sorted.begin(), sorted.end());
    }
    return dirs;
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
 * Scan project + Documents model dirs. Same basename: project dir wins.
 * Only entries with both .onnx and .json are included.
 *
 * Also pairs `app/<name>.json` with ONNX from, in order:
 *   Documents/virtual-orch/Models, onnx_export_092126, onnx_export_091426
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
    scanDir(documentsModelsDir());

    // Cross-dir: project JSON + Documents Models / onnx_export ONNX.
    {
        const auto projectDir = projectModelsDir();
        if (projectDir.isDirectory()) {
            std::vector<juce::File> onnxDirs{documentsModelsDir(), onnxExportDir(),
                                             onnxExportFallbackDir()};
            for (const auto &dir: onnxExportDirs())
                onnxDirs.push_back(dir);
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

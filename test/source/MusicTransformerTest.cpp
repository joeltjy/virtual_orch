#include <gtest/gtest.h>
#include <VirtualOrch/MusicTransformer.h>
#include <chrono>
#include <thread>

ModelConfig modelConfig;

std::string modelPath = juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
        .getChildFile("virtual-orch")
        .getChildFile("Models").getChildFile("bassAndChords.onnx").getFullPathName().toStdString();
const ModelType modelType = ModelType::Small;
std::vector<int32_t> instruments{33};

namespace virtual_orch_test {
    TEST(MusicTransformer, CanInstantiate) {
        MusicTransformer musicTransformer{modelConfig};
        ASSERT_TRUE(true);
    }

    TEST(MusicTransformer, CanInitializeModel) {
        MusicTransformer musicTransformer{modelConfig};
        musicTransformer.init(modelPath.c_str(), modelType);
        ASSERT_TRUE(true);
    }

    TEST(MusicTransformer, CanGenerateTokens) {
        MusicTransformer musicTransformer{modelConfig};
        musicTransformer.init(modelPath.c_str(), modelType);
        musicTransformer.startThread();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Token token{-1, -1, -1};
        musicTransformer.outputTokenQueue.pull(token);
        ASSERT_NE(token.time, -1);
        ASSERT_NE(token.duration, -1);
        ASSERT_NE(token.note, -1);
        musicTransformer.stopThread(2000);
    }

    TEST(MusicTransformer, CanComplyWithMinimumDuration) {
        MusicTransformer musicTransformer{modelConfig};
        modelConfig.outputMinimumDuration = 999;
        musicTransformer.init(modelPath.c_str(), modelType);
        musicTransformer.startThread();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Token token{-1, -1, -1};
        musicTransformer.outputTokenQueue.pull(token);
        ASSERT_EQ(token.duration, Vocab::DurOffset + 999);
        musicTransformer.stopThread(2000);
    }

    TEST(MusicTransformer, DoesntClearPastEventsOnInput) {
        MusicTransformer musicTransformer{modelConfig};
        musicTransformer.init(modelPath.c_str(), modelType);
        musicTransformer.startThread();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        size_t sizeBefore = musicTransformer.getInputData().size();
        // Add a token (very far in time so it doesn't trigger a filtering) to the input queue
        Token inputToken = {55027, Vocab::DurOffset + 10, Vocab::NoteOffset + 5};
        musicTransformer.inputTokenQueue.push(inputToken);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        size_t sizeAfter = musicTransformer.getInputData().size();
        ASSERT_GT(sizeAfter, sizeBefore);
        musicTransformer.stopThread(2000);
    }

    TEST(MusicTransformer, ClearsFutureEventsOnInput) {
        MusicTransformer musicTransformer{modelConfig};
        musicTransformer.init(modelPath.c_str(), modelType);
        musicTransformer.startThread();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        size_t sizeBefore = musicTransformer.getInputData().size();
        Token inputToken = {Vocab::TimeOffset, Vocab::DurOffset + 10, Vocab::NoteOffset + 5};
        musicTransformer.inputTokenQueue.push(inputToken);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        size_t sizeAfter = musicTransformer.getInputData().size();
        ASSERT_LT(sizeAfter, sizeBefore);
        musicTransformer.stopThread(2000);
    }

    TEST(MusicTransformer, CanComplyWithOutputInstrumentAndPitch) {
        MusicTransformer musicTransformer{modelConfig};
        modelConfig.outputInstruments.clear();
        modelConfig.sortedActiveOutputInstruments.clear();
        constexpr OutputInstrumentConfig outputInstrument = {true, true, 30, 31};
        modelConfig.outputInstruments[1] = outputInstrument;
        modelConfig.sortedActiveOutputInstruments.emplace(1);
        musicTransformer.init(modelPath.c_str(), modelType);
        musicTransformer.startThread();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Token token{-1, -1, -1};
        musicTransformer.outputTokenQueue.pull(token);
        ASSERT_EQ(token.note, Vocab::NoteOffset + Config::MaxPitch * 1 + 30);
        musicTransformer.stopThread(2000);
    }

    TEST(MusicTransformer, CanGoOverMaxTime) {
        MusicTransformer musicTransformer{modelConfig};
        musicTransformer.init(modelPath.c_str(), modelType);
        musicTransformer.inputTokenQueue.push({Config::MaxTime + 100, Vocab::DurOffset + 10, Vocab::NoteOffset + 5});
        musicTransformer.startThread();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Token token{-1, -1, -1};
        musicTransformer.outputTokenQueue.pull(token);
        // May pull ClearQueue first if input was processed
        if (token.note == Vocab::ClearQueue) {
            musicTransformer.outputTokenQueue.pull(token);
        }
        ASSERT_GE(token.time, Config::MaxTime + 100);
        musicTransformer.stopThread(2000);
    }
} // namespace virtual_orch_test

#include <chrono>

#include "JordanAI/MusicTransformer.h"

#include "JordanAI/MetricsComponent.h"
#include "JordanAI/MusicModelList.h"

MusicTransformer::MusicTransformer(ModelConfig &modelConfig): Thread("Music Transformer"), modelConfig(modelConfig) {
}

void MusicTransformer::init(MusicModel newMusicModel, ModelType &modelType) {
    musicModel = newMusicModel;
    switch (musicModel.accelerator) {
        #ifdef ENABLE_COREML
        case MLFramework::ACCELERATOR::COREML_COREML:
            mlFramework = std::make_unique<MLFrameworkCoreML>();
            break;
        #endif
        #ifdef ENABLE_GGML
        case MLFramework::ACCELERATOR::GGML_CPU:
        case MLFramework::ACCELERATOR::GGML_CUDA:
        case MLFramework::ACCELERATOR::GGML_METAL:
        case MLFramework::ACCELERATOR::GGML_SYCL:
        case MLFramework::ACCELERATOR::GGML_VULKAN:
            mlFramework = std::make_unique<MLFrameworkGGML>();
            break;
        #endif
        #ifdef ENABLE_ONNXRUNTIME
        case MLFramework::ACCELERATOR::ONNXRUNTIME_CPU:
        case MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT:
        case MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA:
            mlFramework = std::make_unique<MLFrameworkONNXRuntime>();
            break;
        #endif
        #ifdef ENABLE_TORCH
        case MLFramework::ACCELERATOR::TORCH_CUDA:
        case MLFramework::ACCELERATOR::TORCH_MPS:
        case MLFramework::ACCELERATOR::TORCH_CPU:
            mlFramework = std::make_unique<MLFrameworkTorch>();
            break;
        #endif
        default:
            throw std::runtime_error("Unsupported accelerator for music transformer.");
            break;
    }

    // Load the model
    mlFramework->init(musicModel.path.toStdString().c_str(), modelType, musicModel.accelerator);
}

auto MusicTransformer::getCurrentMusicModel() const -> MusicModel {
    return musicModel;
}

void MusicTransformer::run() {
    // If no session has been started, stop
    if (!mlFramework) {
        return;
    }

    inputData.clear();
    clearInputTokenQueue();
    clearOutputTokenQueue();

    // Clear pause
    paused = false;

    currentTime = modelConfig.outputStartTime;
    currentBar = currentTime / modelConfig.outputBarLength;

    // Whether we want to generate a token at a precise time
    // (Used during Trading when the model needs to generate at the start of the bar)
    int32_t forceAtTime = -1;

    // If we force the start time, set the forceAtTime to the start time
    if (modelConfig.outputForceStartTime) {
        forceAtTime = modelConfig.outputStartTime;
    }

    std::multiset<Token> controlTokens;
    int32_t anticipatedTime = -1;

    // If in Playback mode, set the current time and control sequence
    if (modelConfig.inputMode == InputMode::Playback) {
        // Split the control sequence by space
        juce::StringArray controlSequence;
        controlSequence.addTokens(modelConfig.playbackInputControlSequence, " ", "");
        for (int i = 0; i < controlSequence.size(); i += 3) {
            controlTokens.insert({
                controlSequence[i].getIntValue(),
                controlSequence[i + 1].getIntValue(),
                controlSequence[i + 2].getIntValue()
            });
            DBG("Control token: " + controlSequence[i] + " " + controlSequence[i + 1] + " " + controlSequence[i + 2]);
        }
        anticipatedTime = controlTokens.begin()->time;
        DBG("Anticipated time: " + juce::String(anticipatedTime));

        // If anticipated time is before start time, set current time to start time
        // Otherwise, set current time to anticipated time
        if (anticipatedTime < modelConfig.playbackInputStartTime) {
            currentTime = modelConfig.playbackInputStartTime;
        } else {
            DBG("Anticipated time is after start time, using anticipated time instead.");
            currentTime = anticipatedTime;
        }
        currentBar = currentTime / modelConfig.outputBarLength;
    }

    // If in Trading mode, set initial offset
    if (modelConfig.inputMode == InputMode::Trading) {
        tradingOffset.set(modelConfig.tradingInputInitialOffset);
    }


    // If outputPauseAfterTradingSequence and we shouldn't trade to start, start paused but send the necessary BarSeparators to get to the start of the trading
    if (modelConfig.outputPauseAfterTradingSequence && modelConfig.inputMode == InputMode::Trading && !shouldTradeNow()) {
        paused = true;
        DBG("Calculated trading start bar: " + std::to_string(tradingStart() / modelConfig.outputBarLength));
        for (int32_t time = (currentBar + 1) * modelConfig.outputBarLength; time <= tradingStart(); time += modelConfig.outputBarLength) {
            outputTokenQueue.push({time, Vocab::DurOffset, Vocab::BarSeparator});
        }
    }

    // If we have initial input data, add it to the input data
    std::vector<int32_t> inputInitialData;
    int32_t lastInputInitialDataTime = 0;
    if (modelConfig.inputInitialData.length() > 0) {
        // Split the control sequence by space
        juce::StringArray inputInitialDataStringArray;
        inputInitialDataStringArray.addTokens(modelConfig.inputInitialData, " ", "");
        for (int i = 0; i < inputInitialDataStringArray.size(); i += 3) {
            inputInitialData.push_back(inputInitialDataStringArray[i].getIntValue());
            inputInitialData.push_back(inputInitialDataStringArray[i + 1].getIntValue());
            inputInitialData.push_back(inputInitialDataStringArray[i + 2].getIntValue());

            inputData.push_back(inputInitialDataStringArray[i].getIntValue());
            inputData.push_back(inputInitialDataStringArray[i + 1].getIntValue());
            inputData.push_back(inputInitialDataStringArray[i + 2].getIntValue());
        }
    }

    firstToken = true;

    // We trigger the start of the clock
    outputTokenQueue.push({Vocab::TimeOffset, Vocab::DurOffset, Vocab::BarSeparator});

    // We keep track of a flag to clear the queue
    bool clearFlag = false;

    // If input clicks, set up the first beat
    if (modelConfig.inputClicks) {
        auto beatLength = modelConfig.outputBarLength / 4;
        inputData.push_back(Vocab::TimeOffset + 0);
        inputData.push_back(Vocab::DurOffset + modelConfig.outputBarLength - 1);
        inputData.push_back(Vocab::BarClick);
        inputData.push_back(Vocab::TimeOffset + 0);
        inputData.push_back(Vocab::DurOffset + beatLength - 1);
        inputData.push_back(Vocab::BeatClick);
    }

    // Until thread is not stopped (or until we reach the end of the time in Playback mode)
    while (!threadShouldExit()
           && (modelConfig.inputMode != InputMode::Playback || currentTime < modelConfig.playbackInputEndTime)) {
        // DIRECT INPUT: WAIT FOR INPUT BLOCK
        if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput
            && directInputBlock.value) {
            continue;
        }

        // Watch for input tokens
        Token inputToken = {-1, -1, -1};
        int32_t lastTokenTime = -1;
        Token bassToken = {-1, -1, -1}; // TODO Cleanup

        /* COLLECT INPUT */
        while (inputTokenQueue.pull(inputToken) && modelConfig.inputMode != InputMode::Playback) {
            // CLEARING INPUT DATA IF FIRST TOKEN
            if (!clearFlag) {
                if (modelConfig.inputClearsPast) {
                    // If inputClearsPast, clear all past events
                    // Otherwise, clear events that are in the future
                    inputData.clear();
                } else {
                    for (size_t i = 0; i < inputData.size(); i += 3) {
                        if (inputData[i] > inputToken.time) {
                            inputData.erase(inputData.begin() + i, inputData.end());
                            break;
                        }
                    }
                }
            }

            // IF BAR SEPARATOR, THIS MEANS THE PLAY THREAD WANTS TO CLEAR THE QUEUE (CHANGE OF TRADING PATTERN)
            if (inputToken.note == Vocab::BarSeparator) {
                currentTime = inputToken.time;
                currentBar = currentTime / modelConfig.outputBarLength;

                clearFlag = true;
                continue;
            }

            // INPUT CLICKS: ADD KICK AND HIHATS
            if (modelConfig.inputClicks && lastTokenTime == -1) {
                // If first, add a bar click
                auto beatLength = modelConfig.outputBarLength / 4;
                auto lastBar = modelConfig.outputBarLength * (inputToken.time / modelConfig.outputBarLength);
                inputData.push_back(Vocab::TimeOffset + lastBar);
                inputData.push_back(Vocab::DurOffset + beatLength - 1);
                inputData.push_back(Vocab::NoteOffset + Vocab::BarClick);
            } else if (modelConfig.inputClicks) {
                // Otherwise, add hihats on beats between lastTokenTime and inputToken.time
                auto beatLength = modelConfig.outputBarLength / 4;
                auto lastBeat = beatLength * (lastTokenTime / beatLength); // TODO Fix this, it doesn't really work
                for (int32_t t = lastBeat; t < inputToken.time; t += beatLength) {
                    inputData.push_back(Vocab::TimeOffset + t);
                    inputData.push_back(Vocab::DurOffset + beatLength - 1);
                    inputData.push_back(Vocab::NoteOffset + Vocab::BeatClick);
                }
            }
            // Used for adding beats
            lastTokenTime = inputToken.time;

            // KEEP TRACK OF BASS NOTE (FIRST NOTE IF BUFFER)
            if (!clearFlag) {
                bassToken.time = inputToken.time;
                bassToken.duration = inputToken.duration;
                bassToken.note = inputToken.note;
            }

            // ADD TOKEN TO INPUT DATA (IF BUFFER DO SOME PROCESSING)
            if (modelConfig.inputMode != InputMode::Buffer) {
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(inputToken.note);
            } else {
                // NOTE: Should the output ranges be a parameter?
                // If bass, add an octave to range 36-48 and 48-60
                // Otherwise, put in range 60-72
                if (!clearFlag) {
                    inputData.push_back(inputToken.time);
                    inputData.push_back(inputToken.duration);
                    inputData.push_back(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                        + 36 + (inputToken.getPitch() % 12));
                    inputData.push_back(inputToken.time);
                    inputData.push_back(inputToken.duration);
                    inputData.push_back(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                        + 48 + (inputToken.getPitch() % 12));
                } else {
                    inputData.push_back(inputToken.time);
                    inputData.push_back(inputToken.duration);
                    inputData.push_back(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                        + 60 + (inputToken.getPitch() % 12));
                }
            }

            DBG("In prompt: " + inputToken.toUnderstandableString());

            // Set current time to last token time
            currentTime = inputToken.time;
            currentBar = currentTime / modelConfig.outputBarLength;

            // SNAP ON BAR: Force the generation at the given closest bar
            if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputSnapOnBar) {
                forceAtTime = currentTime;
            }

            // Remove the pause, and set the lastGeneratedTokenTime to -1
            paused = false;
            lastGeneratedTokenTime = -1;

            // Set Clear Flag
            clearFlag = true;
        }
        /* END COLLECT INPUT */

        // Do not generate if paused
        if (paused.get()) {
            continue;
        }

        // PLAYBACK INPUT: ANTICIPATE CONTROL SEQUENCE
        while (modelConfig.inputMode == InputMode::Playback
               && controlTokens.size() > 0
               && currentTime >= anticipatedTime + Config::AnticipationDelta) {
            DBG("Anticipating " + controlTokens.begin()->toUnderstandableString());
            inputData.push_back(controlTokens.begin()->time);
            inputData.push_back(controlTokens.begin()->duration);
            inputData.push_back(controlTokens.begin()->note);

            controlTokens.erase(controlTokens.begin());
            if (controlTokens.size() > 0) {
                anticipatedTime = controlTokens.begin()->time;
                DBG("NEW ANTICIPATED TIME: " + juce::String(anticipatedTime));
            } else {
                DBG("DONE ANTICIPATING");
            }
        }

        // Refresh Input Initial Data
        if (modelConfig.inputInitialDataRefreshRate > 0) {
            if (currentTime >= lastInputInitialDataTime + modelConfig.inputInitialDataRefreshRate) {
                DBG("Refreshing prompt at time " + juce::String(currentTime));
                inputData.clear();
                lastInputInitialDataTime = currentTime;
                for (int32_t i = 0; i < inputInitialData.size(); i += 3) {
                    inputData.push_back(inputInitialData[i]);
                    inputData.push_back(inputInitialData[i + 1]);
                    inputData.push_back(inputInitialData[i + 2]);
                }
            }
        }

        // BUFFER INPUT FORCE INSTRUMENT INPUT ON NEXT BAR
        if (modelConfig.inputMode == InputMode::Buffer && modelConfig.bufferInputForceOnNextBar && clearFlag) {
            // Generate token for all instrument
            for (const auto &[id, instrument]: modelConfig.outputInstruments) {
                auto beatLength = modelConfig.outputBarLength / 4; // TODO: Make beatsPerBar a parameter
                Token token{
                    static_cast<int32_t>(Vocab::TimeOffset + currentTime),
                    static_cast<int32_t>(Vocab::DurOffset + beatLength - 1),
                    static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * id
                                         + 12 * (instrument.low / 12) + bassToken.getPitch() % 12)
                };
                inputData.push_back(token.time);
                inputData.push_back(token.duration);
                inputData.push_back(token.note);
                outputTokenQueue.push(token);
            }
        }

        // INPUT CLICKS: CONTROL OF BASS NOTE AND DRUMS
        // if (clearFlag && modelConfig.inputClicks) {
        //     auto beatLength = modelConfig.outputBarLength / 4; // TODO: Make beatsPerBar a parameter
        //     auto nextBar = modelConfig.outputBarLength * ((currentTime / modelConfig.outputBarLength) + 1);
        //     if (modelConfig.inputMode == InputMode::Buffer) {
        //         inputData.push_back(Vocab::AtimeOffset + nextBar);
        //         inputData.push_back(Vocab::AdurOffset + beatLength / 2);
        //         inputData.push_back(Vocab::AnoteOffset + Config::MaxPitch * 33 + bassToken.getPitch());
        //         inputData.push_back(Vocab::AtimeOffset + nextBar);
        //         inputData.push_back(Vocab::AdurOffset + beatLength);
        //         inputData.push_back(Vocab::AnoteOffset + Config::MaxPitch * 49 + bassToken.getPitch() + 12 * 2);
        //         // TODO: !!!!!!! THIS IS WRONG !!!!!!!!
        //     }
        //     for (int32_t t = nextBar; t < currentTime + modelConfig.outputBarLength * 4; t += beatLength) {
        //         if (t % modelConfig.outputBarLength == 0) {
        //             inputData.push_back(Vocab::AtimeOffset + t);
        //             inputData.push_back(Vocab::AdurOffset + beatLength - 1);
        //             inputData.push_back(Vocab::ControlOffset + Vocab::BarClick);
        //         }
        //         inputData.push_back(Vocab::AtimeOffset + t);
        //         inputData.push_back(Vocab::AdurOffset + beatLength - 1);
        //         inputData.push_back(Vocab::ControlOffset + Vocab::BeatClick);
        //     }
        // }

        // GENERATE NEW TOKEN (OR REST)
        Token newToken = {-1, -1, -1};
        if (modelConfig.inputMode == InputMode::Trading && !shouldTradeNow()) {
            // If in Trading mode and it's not time to play, generate rests
            newToken.time = Vocab::TimeOffset + (currentBar + 1) * modelConfig.outputBarLength;
            newToken.duration = Vocab::DurOffset + modelConfig.outputBarLength - 1;
            newToken.note = Vocab::Rest;

            // We set forceAtTime to the next bar for the model to prepare for the first token of the trading sequence
            forceAtTime = (currentBar + 1) * modelConfig.outputBarLength;
        } else {
            // Otherwise, generate new token
            newToken = generateNewToken(forceAtTime);
            DBG("generating " + newToken.toUnderstandableString());

            // OUTPUT PAUSE:
            // PAUSE AFTER TIME INTERVAL: If more than time interval, ignore and pause
            if (lastGeneratedTokenTime != -1 && modelConfig.outputPauseAfterTimeInterval
                && (newToken.time - lastGeneratedTokenTime) > modelConfig.outputPauseAfterTimeIntervalValue) {
                DBG("PAUSING - TIME INTERVAL");
                paused = true;
                newToken.note = Vocab::Rest;
            }
            // PAUSE AFTER DURATION: If more than duration, pause but keep the token
            if (modelConfig.outputPauseAfterDuration
                && newToken.duration >= (Vocab::DurOffset + modelConfig.outputPauseAfterDurationValue)) {
                DBG("PAUSING - DURATION");
                paused = true;
            }
            // PAUSE AFTER TRADING SEQUENCE: If we finished a trading sequence, ignore and pause until next bar
            if (modelConfig.outputPauseAfterTradingSequence && newToken.time == nextTradingStop()) {
                DBG("PAUSING - TRADING SEQUENCE");
                paused = true;
                newToken.note = Vocab::Rest;
                if (modelConfig.inputMode == InputMode::Trading) {
                    // TODO (Lancelot): Now I'm thinking that the triggering of the trading sequence should not come
                    // TODO (Lancelot): from the MusicTransformer sending BarSeparators. We should change that.

                    // TODO (Lancelot): This also breaks the changing from 2 bars to 4 bars since the corresponding BarSeparator won't be there !!
                    // If in Trading mode, we want to be able to trigger the next trading sequence
                    // so we add the next bar separators to the queue
                    for (int i = 1; i <= 5; i++) {
                        const int32_t barTime = Vocab::TimeOffset + (currentBar + i) * modelConfig.outputBarLength;
                        outputTokenQueue.push({barTime, Vocab::DurOffset, Vocab::BarSeparator});
                    }
                }
            }

            // Set the new lastGeneratedTokenTime if it's not a rest
            if (newToken.note != Vocab::Rest) {
                lastGeneratedTokenTime = newToken.time;
            }

            // We set forceAtTime back to -1
            forceAtTime = -1;

            firstToken = false;
        }

        // INPUT CLICKS: ADD MISSING BEATS
        if (modelConfig.inputClicks) {
            auto beatLength = modelConfig.outputBarLength / 4; // TODO Make beatsPerBar a parameter
            int32_t previousNextBeat = (currentTime / beatLength) + 1;
            int32_t currentLastBeat = newToken.time / beatLength;
            // If we passed some beats, add them
            if (currentLastBeat >= previousNextBeat) {
                for (int32_t t = previousNextBeat; t <= currentLastBeat; t += 1) {
                    if (t % 4 == 0) {
                        // TODO Make beatsPerBar a parameter
                        inputData.push_back(Vocab::TimeOffset + t * beatLength);
                        inputData.push_back(Vocab::DurOffset + modelConfig.outputBarLength - 1);
                        inputData.push_back(Vocab::BarClick);
                    }
                    inputData.push_back(Vocab::TimeOffset + t * beatLength);
                    inputData.push_back(Vocab::DurOffset + beatLength - 1);
                    inputData.push_back(Vocab::BeatClick);
                }

                // TODO Then add next bar-worth of beats as anticipation (??)
            }
        }

        // Add the token to inputData
        inputData.push_back(newToken.time);
        inputData.push_back(newToken.duration);
        inputData.push_back(newToken.note);

        // Update current time
        currentTime = newToken.time;

        // If we need to clear the queue, send a clear queue token
        // We do it here to ensure we can push the new token right after, and not have a moment without any token
        if (clearFlag) {
            outputTokenQueue.push({Vocab::TimeOffset, Vocab::DurOffset, Vocab::ClearQueue});
        }

        // If "Send Bar Separators":
        // If we are at a new bar, set it and send a Bar Separator signal.
        if (modelConfig.outputSendBarSeparators && (currentTime / modelConfig.outputBarLength) > currentBar
            && !clearFlag) {
            currentBar = currentTime / modelConfig.outputBarLength;
            const int32_t barTime = Vocab::TimeOffset + modelConfig.outputBarLength * currentBar;
            outputTokenQueue.push({barTime, Vocab::DurOffset, Vocab::BarSeparator});
        } else {
            currentBar = currentTime / modelConfig.outputBarLength;
        }

        if (clearFlag) { clearFlag = false; }

        // Push new token to output queue
        outputTokenQueue.push(newToken);
    }
}

void MusicTransformer::instrLogits(std::vector<float> &logits) {
    auto it = modelConfig.sortedActiveOutputInstruments.begin();
    auto end = modelConfig.sortedActiveOutputInstruments.end();

    if (it == end) return; // Early exit if the set is empty

    // Hide all instruments up to the lowest note of the first
    std::fill(logits.begin() + Vocab::NoteOffset,
              logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*it)
              + modelConfig.outputInstruments[*it].low,
              -std::numeric_limits<float>::infinity());

    auto prev_it = it;
    ++it;

    // Hide all instruments between the active instruments
    for (; it != end; ++it, ++prev_it) {
        std::fill(
            logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*prev_it)
            + modelConfig.outputInstruments[*prev_it].high,
            logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*it)
            + modelConfig.outputInstruments[*it].low,
            -std::numeric_limits<float>::infinity());
    }

    // Hide all instruments after the highest note of the last
    std::fill(
        logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*prev_it)
        + modelConfig.outputInstruments[*prev_it].high,
        logits.begin() + Vocab::Rest,
        -std::numeric_limits<float>::infinity());
}

/**
 * - Don't sample events in the past
 * - Don't sample events too far in the future
*/
void MusicTransformer::futureLogits(std::vector<float> &logits, const int currentTime, int32_t forceAtTime) {
    // If we need to force generation at currentTime, leave only this slot valid
    if (forceAtTime != -1 && forceAtTime < Config::MaxTime) {
        std::fill_n(logits.begin() + Vocab::TimeOffset, forceAtTime, -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + Vocab::TimeOffset + forceAtTime + 1, logits.begin() + Vocab::DurOffset,
                  -std::numeric_limits<float>::infinity());
        return;
    }

    // Force generation in the future
    if (currentTime > 0) {
        std::fill_n(logits.begin() + Vocab::TimeOffset, currentTime,
                    -std::numeric_limits<float>::infinity());
    }

    // Follow minimum time interval (if not out of bounds and not first token)
    if ((currentTime + modelConfig.outputMinimumTimeInterval) < Config::MaxTime && !firstToken) {
        std::fill_n(logits.begin() + Vocab::TimeOffset + currentTime + 1, modelConfig.outputMinimumTimeInterval,
                    -std::numeric_limits<float>::infinity());
        // If chords are not allowed (or if it is already too large), also set current time to -inf
        if (modelConfig.outputMinimumTimeInterval > 0 && (
                !modelConfig.outputAllowChords || currentChordSize >= modelConfig.outputMaximumChordSize)) {
            std::fill_n(logits.begin() + Vocab::TimeOffset + currentTime, 1,
                        -std::numeric_limits<float>::infinity());
        }
    }

    // Do not generate too far in the future (TODO: manually set to 200, but could be part of config)
    if (currentTime < (Config::MaxTime - maximumFuture - modelConfig.outputMinimumTimeInterval)) {
        std::fill(
            logits.begin() + Vocab::TimeOffset + currentTime + maximumFuture + modelConfig.outputMinimumTimeInterval,
            logits.begin() + Vocab::DurOffset,
            -std::numeric_limits<float>::infinity());
    }
}

void MusicTransformer::durLogits(std::vector<float> &logits) {
    std::fill_n(logits.begin() + Vocab::DurOffset, modelConfig.outputMinimumDuration,
                -std::numeric_limits<float>::infinity());
    std::fill(logits.begin() + Vocab::DurOffset + modelConfig.outputMaximumDuration + 1,
              logits.begin() + Vocab::NoteOffset,
              -std::numeric_limits<float>::infinity());
}

void safeLogits(std::vector<float> &logits, const size_t idx) {
    std::fill(logits.begin() + Vocab::ControlOffset, logits.begin() + Vocab::SpecialOffset,
              -std::numeric_limits<float>::infinity());
    std::fill(logits.begin() + Vocab::SpecialOffset, logits.end(), -std::numeric_limits<float>::infinity());

    if (idx % 3 == 0) {
        std::fill(logits.begin() + Vocab::DurOffset, logits.begin() + Vocab::NoteOffset,
                  -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + Vocab::NoteOffset, logits.begin() + Vocab::ControlOffset,
                  -std::numeric_limits<float>::infinity());
    } else if (idx % 3 == 1) {
        std::fill(logits.begin() + Vocab::TimeOffset, logits.begin() + Vocab::DurOffset,
                  -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + Vocab::NoteOffset, logits.begin() + Vocab::ControlOffset,
                  -std::numeric_limits<float>::infinity());
    } else if (idx % 3 == 2) {
        std::fill(logits.begin() + Vocab::TimeOffset, logits.begin() + Vocab::DurOffset,
                  -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + Vocab::DurOffset, logits.begin() + Vocab::NoteOffset,
                  -std::numeric_limits<float>::infinity());
    }
}

Token MusicTransformer::generateNewToken(int32_t forceAtTime) {
    if (inputData.size() % 3 != 0) {
        throw std::runtime_error("inputData must be a multiple of 3");
    }

    const int lookback = std::max(static_cast<int>(inputData.size() - 120), 0);
    std::vector history(inputData.begin() + lookback, inputData.end());
    int32_t offset = 0;
    if (!history.empty()) {
        offset = minTime(history);
    }

    // Relativize time in the history buffer
    for (size_t i = 0; i < history.size(); i++) {
        if (i % 3 == 0) {
            history[i] -= offset;
        }
    }

    history.insert(history.begin(), Vocab::Anticipate);

    Token newToken{-1, -1, -1};

    for (int i = 0; i < 3; i++) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::vector<float> scores = mlFramework->runModelAndGetLogits(history, false);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (metricsWindow) {
            metricsWindow->tokenLatencyFifo.push(
                static_cast<int32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()));
        }
        safeLogits(scores, i % 3);
        if (i == 0) {
            // If forceAtTime is not -1, then pass it by removing the offset
            futureLogits(scores, currentTime - offset, forceAtTime != -1 ? forceAtTime - offset : -1);
        } else if (i == 1) {
            durLogits(scores);
        } else if (i == 2) {
            instrLogits(scores);
        }
        int32_t token = sampleTopP(scores, 0.9, modelConfig.outputTemperatures[i]);

        history.push_back(token);

        if (i == 0) {
            newToken.time = token + offset;

            // If we reached the end of the trading sequence, return a REST
            if (modelConfig.inputMode == InputMode::Trading && (token + offset >= nextTradingStop())) {
                newToken.time = nextTradingStop();
                newToken.duration = Vocab::DurOffset + modelConfig.outputBarLength - 1;
                newToken.note = Vocab::Rest;
                return newToken;
            }

            // If we are building a chord, increment the currentChordSize
            if ((token + offset) == currentTime) {
                currentChordSize++;
            } else {
                currentChordSize = 1;
            }
        } else if (i == 1) {
            newToken.duration = token;
        } else if (i == 2) {
            newToken.note = token;
        }
    }

    return newToken;
}

/**
 * @return whether the model should trade now based on the current bar and the setting of the model
 */
auto MusicTransformer::shouldTradeNow() const -> bool {
    auto _tradingOffset = tradingOffset.get();
    return ((modelConfig.tradingInputModelGoesFirst &&
             ((currentBar - _tradingOffset) % (modelConfig.tradingInputNumberOfBars * 2))
             < modelConfig.tradingInputNumberOfBars)
            || (!modelConfig.tradingInputModelGoesFirst &&
                ((currentBar - _tradingOffset) % (modelConfig.tradingInputNumberOfBars * 2))
                >= modelConfig.tradingInputNumberOfBars));
}

/**
 * Returns the time of the next time at which Trading should stop, or -1 if the model is not currently trading.
 * Two examples:
 *   - In a Trading sequence of 4 bars with Model Goes First:
 *      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 *      ------- ------- --------- -----------
 *        = 4     = -1    = 12       = -1           ( converted to time )
 *
*   - In a Trading sequence of 4 bars with Model Goes Second:
 *      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 *      ------- ------- --------- -----------
 *        = -1    = 8     = -1       = 16           ( converted to time )
 *
 */
auto MusicTransformer::nextTradingStop() const -> int32_t {
    if (!shouldTradeNow()) {
        return -1;
    }

    auto _tradingOffset = tradingOffset.get();
    // Return the next group of tradingInputNumberOfBars
    const auto nextGroup = modelConfig.tradingInputNumberOfBars
                           * ((currentBar - _tradingOffset) / modelConfig.tradingInputNumberOfBars)
                           + modelConfig.tradingInputNumberOfBars + _tradingOffset;
    return modelConfig.outputBarLength * nextGroup;
}

/** Calculates when the trading should start */
auto MusicTransformer::tradingStart() const -> int32_t {
    int32_t barToStart = 0;
    auto _tradingOffset = tradingOffset.get();
    barToStart += _tradingOffset;
    if (!modelConfig.tradingInputModelGoesFirst) {
        barToStart += modelConfig.tradingInputNumberOfBars;
    }
    return modelConfig.outputBarLength * barToStart;
}
#pragma once

#include <JuceHeader.h>

class OSCController : juce::OSCReceiver,
                      juce::OSCReceiver::ListenerWithOSCAddress<juce::OSCReceiver::MessageLoopCallback> {
public:
    OSCController(const uint32_t &port);
    explicit OSCController(juce::DatagramSocket &gestureSocket);

    ~OSCController() override;

    void oscMessageReceived(const OSCMessage &message) override;

    std::function<void(const juce::String &presetName)> onPresetChanged;

    std::function<void()> onStart;
    std::function<void()> onStop;
    std::function<void()> onOpenTransport;
    std::function<void(const juce::int32 &id, const juce::int32 &low, const juce::int32 &high)> onSetOutputRange;
    std::function<void(const juce::OSCMessage &message)> onSetModelConfig;

    /**
     * Sympathetic Orchestra gestures (/sym/...). The cue/select/tempo/expression state lives in
     * Sympathetic::state(); Stop/Play go through the pedal path: true = Stop (pedal down).
     */
    std::function<void(bool stop)> onSympatheticPedal;
    std::function<void()> onSympatheticGesture;

private:
    auto handleSympathetic(const juce::OSCMessage &message) -> void;
};

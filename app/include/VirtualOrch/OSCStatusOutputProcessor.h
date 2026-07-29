#pragma once

#include <JuceHeader.h>

class OSCStatusOutputProcessor {
public:
    OSCStatusOutputProcessor(const juce::String &host, const uint32_t &port);

    ~OSCStatusOutputProcessor();

    void modelLoading();

    void modelLoaded();

    void panic();

private:
    juce::OSCSender oscSender;
};

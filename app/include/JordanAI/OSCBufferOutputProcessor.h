#pragma once

#include <JuceHeader.h>

#include "MusicTransformer.h"

class OSCBufferOutputProcessor {
public:
    OSCBufferOutputProcessor(const juce::String &host, const uint32_t &port);

    ~OSCBufferOutputProcessor();

    void setTime(const uint32_t &time);

    void addToBuffer(const Token &token);

    void sendBuffer(const uint32_t &time);

private:
    juce::OSCSender oscSender;

    juce::OSCBundle bundle;
};

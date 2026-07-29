#pragma once

#include "VirtualOrch/OutputProcessor.h"

class OSCOutputProcessor : public OutputProcessor {
public:
    OSCOutputProcessor(const std::string &host, const uint32_t &port);

    ~OSCOutputProcessor() override;

    void send(NoteOnEvent event) override;

    void send(NoteOffEvent event) override;

private:
    juce::OSCSender oscSender;
};

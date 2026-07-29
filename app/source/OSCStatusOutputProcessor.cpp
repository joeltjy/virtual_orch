#include "VirtualOrch/OSCStatusOutputProcessor.h"

OSCStatusOutputProcessor::OSCStatusOutputProcessor(const juce::String &host, const uint32_t &port) {
    if (!oscSender.connect(host, port)) {
        DBG("OSC connection failed");
        // TODO handle error
    } else {
        oscSender.send(
            juce::OSCMessage("/workspace/E94F875A-457E-4849-B304-74A32472F50F/connect", juce::String("3940")));
    }
}

OSCStatusOutputProcessor::~OSCStatusOutputProcessor() {
    oscSender.disconnect();
}

void OSCStatusOutputProcessor::modelLoading() {
    DBG("modelLoading");
    oscSender.send("/workspace/E94F875A-457E-4849-B304-74A32472F50F/cue/0.99.1/go"); // SHOW RED MODEL LED
}

void OSCStatusOutputProcessor::modelLoaded() {
    oscSender.send("/workspace/E94F875A-457E-4849-B304-74A32472F50F/cue/0.99.2/go"); // SHOW GREEN MODEL LED
}

void OSCStatusOutputProcessor::panic() {
    oscSender.send("/workspace/E94F875A-457E-4849-B304-74A32472F50F/panic"); // PANIC
}

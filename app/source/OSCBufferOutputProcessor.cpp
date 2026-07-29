#include "VirtualOrch/OSCBufferOutputProcessor.h"

OSCBufferOutputProcessor::OSCBufferOutputProcessor(const juce::String &host, const uint32_t &port) {
    if (!oscSender.connect(host, port)) {
        DBG("OSC connection failed");
        // TODO handle error
    } else {
        oscSender.send("/connected");
    }
}

OSCBufferOutputProcessor::~OSCBufferOutputProcessor() {
    oscSender.disconnect();
}

// TODO Maybe not needed, but bundle timetag doesn't seem to work
void OSCBufferOutputProcessor::setTime(const uint32_t &time) {
    juce::MemoryOutputStream stream;
    stream.writeInt(time);
    bundle.addElement(juce::OSCMessage("/time", stream.getMemoryBlock()));
}

void OSCBufferOutputProcessor::addToBuffer(const Token &token) {
    bundle.addElement(juce::OSCMessage("/token", token.toMemoryBlock()));
}

void OSCBufferOutputProcessor::sendBuffer(const uint32_t &time) {
    bundle.setTimeTag(time);
    oscSender.send(bundle);
    bundle = juce::OSCBundle();
}

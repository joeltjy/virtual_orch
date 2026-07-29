#include "VirtualOrch/OSCOutputProcessor.h"

OSCOutputProcessor::OSCOutputProcessor(const std::string &host, const uint32_t &port) {
    if (!oscSender.connect(host, port)) {
        DBG("OSC connection failed");
        // TODO handle error
    } else {
        oscSender.send("/connected");
    }
}

OSCOutputProcessor::~OSCOutputProcessor() {
    oscSender.disconnect();
}

void OSCOutputProcessor::send(NoteOnEvent event) {
    oscSender.send("/noteon", event.instrument, event.note, event.velocity);
}

void OSCOutputProcessor::send(NoteOffEvent event) {
    oscSender.send("/noteoff", event.instrument, event.note);
}

#include "VirtualOrch/OSCController.h"

OSCController::OSCController(const uint32_t &port) {
    if (!connect(port)) {
        DBG("OSC connection failed");
        // TODO handle error
    }

    addListener(this, "/moveToPreset");
    addListener(this, "/start");
    addListener(this, "/stop");
    addListener(this, "/openTransport");
    addListener(this, "/setOutputRange");
    addListener(this, "/setModelConfig");
}

OSCController::~OSCController() {
    disconnect();
}

void OSCController::oscMessageReceived(const OSCMessage &message) {
    if (message.getAddressPattern().matches("/moveToPreset")) {
        const auto preset = message[0].getString();
        DBG("MOVING TO PRESET: " + preset);
        onPresetChanged(preset);
    } else if (message.getAddressPattern().matches("/start")) {
        onStart();
    } else if (message.getAddressPattern().matches("/stop")) {
        onStop();
    } else if (message.getAddressPattern().matches("/openTransport")) {
        onOpenTransport();
    } else if (message.getAddressPattern().matches("/setOutputRange")) {
        const auto ID = message[0].getInt32();
        const auto LOW = message[1].getInt32();
        const auto HIGH = message[2].getInt32();
        DBG("LOW SET TO: " + std::to_string(LOW));
        DBG("HIGH SET TO: " + std::to_string(HIGH));
        onSetOutputRange(ID, LOW, HIGH);
    } else if (message.getAddressPattern().matches("/setModelConfig")) {
        onSetModelConfig(message);
    }
}

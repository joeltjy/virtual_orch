#include "VirtualOrch/OSCController.h"
#include "VirtualOrch/SessionLog.h"
#include "VirtualOrch/Sympathetic.h"

namespace {
constexpr const char *kSymAddresses[] = {"/sym/point",  "/sym/release", "/sym/expression",
                                         "/sym/beat",   "/sym/stop",    "/sym/play",
                                         "/sym/reset",  "/sym/enable",  "/sym/warp"};

auto numberArg(const juce::OSCMessage &message, int index) -> std::optional<double> {
    if (message.size() <= index)
        return std::nullopt;
    const auto &arg = message[index];
    if (arg.isFloat32())
        return arg.getFloat32();
    if (arg.isInt32())
        return arg.getInt32();
    return std::nullopt;
}
} // namespace

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
    for (const auto *address: kSymAddresses)
        addListener(this, address);
}

OSCController::OSCController(juce::DatagramSocket &gestureSocket) {
    for (const auto *address: kSymAddresses)
        addListener(this, address);
    connectToSocket(gestureSocket);
}

OSCController::~OSCController() {
    disconnect();
}

void OSCController::oscMessageReceived(const OSCMessage &message) {
    if (message.getAddressPattern().toString().startsWith("/sym/")) {
        if (onSympatheticGesture)
            onSympatheticGesture();
        handleSympathetic(message);
        return;
    }
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

auto OSCController::handleSympathetic(const juce::OSCMessage &message) -> void {
    auto &sym = Sympathetic::state();
    const auto address = message.getAddressPattern().toString();

    if (address == "/sym/enable") {
        const bool on = numberArg(message, 0).value_or(1.0) != 0.0;
        sym.enabled.store(on);
        SessionLog::add("SYM", juce::String("enable ") + (on ? "1" : "0"));
        return;
    }
    // Any other gesture message turns the layer on (from then on only cued parts play).
    if (! sym.enabled.exchange(true))
        SessionLog::add("SYM", "enable 1 (first gesture)");

    if (address == "/sym/point") {
        if (message.size() < 1 || ! message[0].isString())
            return;
        const auto name = message[0].getString();
        const auto part = Sympathetic::partIndex(name.toStdString());
        if (! part.has_value()) {
            SessionLog::add("SYM", "point " + name + " (no counterpart)");
            return;
        }
        const bool cued = sym.point(*part);
        SessionLog::add("SYM", juce::String(cued ? "cue " : "select ") + name);
    } else if (address == "/sym/release") {
        sym.selectedPart.store(-1);
        SessionLog::add("SYM", "release");
    } else if (address == "/sym/expression") {
        if (const auto spread = numberArg(message, 0)) {
            sym.expressionScale.store(
                Sympathetic::expressionScaleForSpread(static_cast<float>(*spread)));
        }
    } else if (address == "/sym/beat") {
        // Beat time in ms on the tracker's clock (only differences matter).
        if (const auto timeMs = numberArg(message, 0)) {
            sym.beat(*timeMs);
            SessionLog::add("SYM", "beat " + juce::String(*timeMs, 0) + " warp "
                                       + juce::String(sym.warp.load(), 3));
        }
    } else if (address == "/sym/warp") {
        // Direct override (testing without a camera).
        if (const auto w = numberArg(message, 0)) {
            sym.warp.store(std::clamp(static_cast<float>(*w), Sympathetic::kMinWarp,
                                      Sympathetic::kMaxWarp));
            SessionLog::add("SYM", "warp " + juce::String(sym.warp.load(), 3));
        }
    } else if (address == "/sym/stop") {
        SessionLog::add("SYM", "stop");
        if (onSympatheticPedal)
            onSympatheticPedal(true);
    } else if (address == "/sym/play") {
        SessionLog::add("SYM", "play");
        if (onSympatheticPedal)
            onSympatheticPedal(false);
    } else if (address == "/sym/reset") {
        sym.reset();
        SessionLog::add("SYM", "reset");
    }
}

#pragma once

#include <JuceHeader.h>

#include <array>
#include <optional>
#include <set>

#include "VirtualOrch/LaunchpadGrid.h"

/**
 * Computer-keyboard stand-in for the Launchpad ("Keyboard as Launchpad" on the Settings page).
 * Every key press drives LaunchpadGrid exactly like the hardware button it maps to
 * (handleInput for pads / side buttons, handleSceneInput for the top buttons), so all the
 * actions, LEDs and logging are the same.
 *
 * Map, grouped by function (rows are LaunchpadGrid rows, side = LaunchpadGrid::kSideCol):
 *   1 2 3 4 5 6 7        TD 1 2 4 8 16 32 64          (row 1, cols 0-6)
 *   8                    TD AUTO                       (row 1, side)
 *   9 0 -                register LOW / MEDIUM / HIGH (row 6, cols 0-2)
 *   =                    register AUTO                 (row 6, side)
 *   Q W E                Str Ens Pizz Tremolo          (row 2)
 *   A S D F              Flute Oboe Clarinet Bassoon   (row 3)
 *   Z X C V              Horn Trumpet Trombone Tuba    (row 4)
 *   T                    Timpani                       (row 5)
 *                        Instrument keys press the side the pads control: yours (cols 0-3)
 *                        in Edit, the model's (cols 4-7) in Jam / Accompaniment.
 *   Return               flush your instruments to the model (row 2, side; Edit only)
 *   Backspace            reset the model's instruments (row 3, side; Edit only)
 *   Y U I                mode Step-in (Edit) / Jam / Accompaniment (row 0, cols 0-2)
 *   N M , .              max polyphony 1-4             (row 6, cols 4-7)
 *   F5 … F12             mood presets 1-8              (row 7, cols 0-7)
 *   F1 F2 F3 F4          top buttons: force restart, RT pause, OT pause, block input
 */
namespace KeyboardLaunchpad {

struct Target {
    int row = -1;      // LaunchpadGrid row; -1 for a top (scene) button
    int col = -1;      // 0-7, or LaunchpadGrid::kSideCol
    int scene = -1;    // 0-7 for the top buttons, else -1
    [[nodiscard]] auto isScene() const -> bool { return scene >= 0; }
    auto operator==(const Target &o) const -> bool { return row == o.row && col == o.col && scene == o.scene; }
};

struct Binding {
    int key;        // juce key code: an upper-case letter / digit / '-' / '=' / ',' / '.' or a juce::KeyPress code
    Target target;  // instrument keys name your side (cols 0-3); see resolve()
};

namespace detail {
inline constexpr int kSide = LaunchpadGrid::kSideCol;
inline auto pad(int row, int col) -> Target { return {row, col, -1}; }
inline auto scene(int idx) -> Target { return {-1, -1, idx}; }
} // namespace detail

/** Every binding (built once). */
inline auto bindings() -> const std::vector<Binding> & {
    static const std::vector<Binding> all = [] {
        using namespace detail;
        std::vector<Binding> b;
        const char td[] = {'1', '2', '3', '4', '5', '6', '7'};
        for (int i = 0; i < 7; ++i)
            b.push_back({td[i], pad(LaunchpadGrid::kTdRow, i)});
        b.push_back({'8', pad(LaunchpadGrid::kTdRow, kSide)});
        b.push_back({'9', pad(LaunchpadGrid::kRegisterRow, 0)});
        b.push_back({'0', pad(LaunchpadGrid::kRegisterRow, 1)});
        b.push_back({'-', pad(LaunchpadGrid::kRegisterRow, 2)});
        b.push_back({'=', pad(LaunchpadGrid::kRegisterRow, kSide)});
        // Instrument rows (your side; resolve() moves them to the model's side in Jam).
        const char *const familyKeys[LaunchpadGrid::kInstrumentRowCount] = {"QWE", "ASDF", "ZXCV", "T"};
        for (int row = 0; row < LaunchpadGrid::kInstrumentRowCount; ++row)
            for (int col = 0; familyKeys[row][col] != 0; ++col)
                b.push_back({familyKeys[row][col], pad(LaunchpadGrid::kInstrumentRowFirst + row,
                                                       LaunchpadGrid::kUserColFirst + col)});
        b.push_back({juce::KeyPress::returnKey, pad(LaunchpadGrid::kFlushRow, kSide)});
        b.push_back({juce::KeyPress::backspaceKey, pad(LaunchpadGrid::kResetRow, kSide)});
        const char modes[] = {'Y', 'U', 'I'};
        for (int i = 0; i < 3; ++i)
            b.push_back({modes[i], pad(LaunchpadGrid::kModeRow, i)});
        const char poly[] = {'N', 'M', ',', '.'};
        for (int i = 0; i < 4; ++i)
            b.push_back({poly[i], pad(LaunchpadGrid::kAccompRow, LaunchpadGrid::kPolyphonyColFirst + i)});
        const int presetKeys[] = {juce::KeyPress::F5Key,  juce::KeyPress::F6Key,  juce::KeyPress::F7Key,
                                  juce::KeyPress::F8Key,  juce::KeyPress::F9Key,  juce::KeyPress::F10Key,
                                  juce::KeyPress::F11Key, juce::KeyPress::F12Key};
        for (int i = 0; i < LaunchpadGrid::kPresetCount; ++i)
            b.push_back({presetKeys[i], pad(LaunchpadGrid::kPresetRow, i)});
        const int fkeys[] = {juce::KeyPress::F1Key, juce::KeyPress::F2Key, juce::KeyPress::F3Key,
                             juce::KeyPress::F4Key};
        for (int i = 0; i < 4; ++i)
            b.push_back({fkeys[i], scene(i)});
        return b;
    }();
    return all;
}

/** Normalise a juce key code: letters upper-case. */
[[nodiscard]] inline auto normaliseKey(int keyCode) -> int {
    if (keyCode >= 'a' && keyCode <= 'z')
        return keyCode - 'a' + 'A';
    return keyCode;
}

/** The target for a key (Shift makes no difference), or nullopt. */
[[nodiscard]] inline auto targetFor(int keyCode) -> std::optional<Target> {
    const int key = normaliseKey(keyCode);
    for (const auto &b: bindings())
        if (b.key == key)
            return b.target;
    return std::nullopt;
}

/** Human label for a key ("Q", "Ret", "Bksp", "F1"). */
[[nodiscard]] inline auto keyLabel(const Binding &b) -> juce::String {
    if (b.key == juce::KeyPress::returnKey)
        return "Ret";
    if (b.key == juce::KeyPress::backspaceKey)
        return "Bksp";
    const int fkeys[] = {juce::KeyPress::F1Key,  juce::KeyPress::F2Key,  juce::KeyPress::F3Key,
                         juce::KeyPress::F4Key,  juce::KeyPress::F5Key,  juce::KeyPress::F6Key,
                         juce::KeyPress::F7Key,  juce::KeyPress::F8Key,  juce::KeyPress::F9Key,
                         juce::KeyPress::F10Key, juce::KeyPress::F11Key, juce::KeyPress::F12Key};
    for (int i = 0; i < 12; ++i)
        if (b.key == fkeys[i])
            return "F" + juce::String(i + 1);
    return juce::String::charToString(static_cast<juce::juce_wchar>(b.key));
}

/**
 * The key label for a grid button (row, col incl. side) or top button, or empty. A model-side
 * instrument pad shows the key of the matching pad on your side (the same key presses it in Jam).
 */
[[nodiscard]] inline auto labelForTarget(const Target &t) -> juce::String {
    Target lookup = t;
    if (! t.isScene() && LaunchpadGrid::targetForPad(t.row, t.col) == InstrumentUpdateTarget::Model)
        lookup.col = t.col - LaunchpadGrid::kModelColFirst;
    for (const auto &b: bindings())
        if (b.target == lookup)
            return keyLabel(b);
    return {};
}

/** The button a key presses now: instrument keys go to the side the pads control. */
[[nodiscard]] inline auto resolve(const LaunchpadGrid &grid, const Target &t) -> Target {
    if (t.isScene() || LaunchpadGrid::targetForPad(t.row, t.col) != InstrumentUpdateTarget::User
        || grid.controlledTarget() != InstrumentUpdateTarget::Model)
        return t;
    return {t.row, t.col + LaunchpadGrid::kModelColFirst, -1};
}

/** Send a press to the grid, as the Launchpad would (press only; the grid ignores releases). */
inline auto press(LaunchpadGrid &grid, const Target &t) -> void {
    if (t.isScene()) {
        grid.handleSceneInput(t.scene);
        return;
    }
    const auto target = resolve(grid, t);
    grid.handleInput(target.row, target.col, true);
}

/**
 * Key-down tracking so a held key presses its button once (no auto-repeat), on the message
 * thread. `onPress` is called for each new press with the key's label and target.
 */
class Listener : public juce::KeyListener {
public:
    std::function<bool()> isEnabled;
    std::function<void(const Target &, const juce::String &label)> onPress;

    auto keyPressed(const juce::KeyPress &key, juce::Component *) -> bool override {
        if (! isEnabled || ! isEnabled())
            return false;
        // Cmd / Ctrl / Alt shortcuts stay with the app.
        const auto mods = key.getModifiers();
        if (mods.isCommandDown() || mods.isCtrlDown() || mods.isAltDown())
            return false;
        const int code = normaliseKey(key.getKeyCode());
        const auto target = targetFor(code);
        if (! target.has_value())
            return false;
        if (held.count(code) != 0)
            return true; // auto-repeat of a key that is still down
        held.insert(code);
        if (onPress)
            onPress(*target, keyLabel({code, *target}));
        return true;
    }

    auto keyStateChanged(bool, juce::Component *) -> bool override {
        for (auto it = held.begin(); it != held.end();) {
            if (! juce::KeyPress::isKeyCurrentlyDown(*it) && ! juce::KeyPress::isKeyCurrentlyDown(lowerOf(*it)))
                it = held.erase(it);
            else
                ++it;
        }
        return false;
    }

private:
    static auto lowerOf(int code) -> int { return code >= 'A' && code <= 'Z' ? code - 'A' + 'a' : code; }
    std::set<int> held;
};

} // namespace KeyboardLaunchpad

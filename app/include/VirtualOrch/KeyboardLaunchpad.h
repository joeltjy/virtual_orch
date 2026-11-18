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
 *   Q W E   T Y U I      model: Str Ens Pizz Tremolo  Flute Oboe Clarinet Bassoon (row 4)
 *   A S D F G            model: Horn Trumpet Trombone Tuba Timpani               (row 5)
 *   Shift + Q…I, A…G     the same instruments in the input (User) rows 2 / 3
 *   Return               flush input instruments to the model (row 2, side)
 *   Z X C                mode Edit / Jam / Accompaniment (row 0, cols 0-2)
 *   V B N M              max polyphony 1-4             (row 7, cols 0-3)
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
    int key;        // juce key code: an upper-case letter / digit / '-' / '=' or a juce::KeyPress code
    bool shift;     // Shift must be held (only for the input-instrument rows)
    Target target;
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
            b.push_back({td[i], false, pad(LaunchpadGrid::kTdRow, i)});
        b.push_back({'8', false, pad(LaunchpadGrid::kTdRow, kSide)});
        b.push_back({'9', false, pad(LaunchpadGrid::kRegisterRow, 0)});
        b.push_back({'0', false, pad(LaunchpadGrid::kRegisterRow, 1)});
        b.push_back({'-', false, pad(LaunchpadGrid::kRegisterRow, 2)});
        b.push_back({'=', false, pad(LaunchpadGrid::kRegisterRow, kSide)});
        // Instrument rows: first row of a set on Q W E _ T Y U I, second on A S D F G.
        const char firstRow[] = {'Q', 'W', 'E', 0, 'T', 'Y', 'U', 'I'};
        const char secondRow[] = {'A', 'S', 'D', 'F', 'G'};
        for (const bool shift: {false, true}) {
            const int base = shift ? LaunchpadGrid::kUserRowFirst : LaunchpadGrid::kModelRowFirst;
            for (int col = 0; col < 8; ++col)
                if (firstRow[col] != 0)
                    b.push_back({firstRow[col], shift, pad(base, col)});
            for (int col = 0; col < 5; ++col)
                b.push_back({secondRow[col], shift, pad(base + 1, col)});
        }
        b.push_back({juce::KeyPress::returnKey, false, pad(LaunchpadGrid::kFlushRow, kSide)});
        const char modes[] = {'Z', 'X', 'C'};
        for (int i = 0; i < 3; ++i)
            b.push_back({modes[i], false, pad(LaunchpadGrid::kModeRow, i)});
        const char poly[] = {'V', 'B', 'N', 'M'};
        for (int i = 0; i < 4; ++i)
            b.push_back({poly[i], false, pad(LaunchpadGrid::kAccompRow, i)});
        const int fkeys[] = {juce::KeyPress::F1Key, juce::KeyPress::F2Key, juce::KeyPress::F3Key,
                             juce::KeyPress::F4Key};
        for (int i = 0; i < 4; ++i)
            b.push_back({fkeys[i], false, scene(i)});
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

/**
 * The target for a key, or nullopt. `shift` only selects the input rows for letters: digits,
 * '-', '=', Return and F-keys ignore it (Shift+1 is still TD 1).
 */
[[nodiscard]] inline auto targetFor(int keyCode, bool shift) -> std::optional<Target> {
    const int key = normaliseKey(keyCode);
    const bool letter = key >= 'A' && key <= 'Z';
    for (const auto &b: bindings())
        if (b.key == key && (! letter || b.shift == shift))
            return b.target;
    return std::nullopt;
}

/** Human label for a key ("Q", "⇧Q", "Return", "F1"). */
[[nodiscard]] inline auto keyLabel(const Binding &b) -> juce::String {
    juce::String k;
    if (b.key == juce::KeyPress::returnKey)
        k = "Ret";
    else if (b.key >= juce::KeyPress::F1Key && b.key <= juce::KeyPress::F4Key)
        k = "F" + juce::String(b.key - juce::KeyPress::F1Key + 1);
    else
        k = juce::String::charToString(static_cast<juce::juce_wchar>(b.key));
    return (b.shift ? juce::String(juce::CharPointer_UTF8("\xe2\x87\xa7")) : juce::String()) + k;
}

/** The key label for a grid button (row, col incl. side) or top button, or empty. */
[[nodiscard]] inline auto labelForTarget(const Target &t) -> juce::String {
    for (const auto &b: bindings())
        if (b.target == t)
            return keyLabel(b);
    return {};
}

/** Send a press to the grid, as the Launchpad would (press only; the grid ignores releases). */
inline auto press(LaunchpadGrid &grid, const Target &t) -> void {
    if (t.isScene())
        grid.handleSceneInput(t.scene);
    else
        grid.handleInput(t.row, t.col, true);
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
        const auto target = targetFor(code, mods.isShiftDown());
        if (! target.has_value())
            return false;
        if (held.count(code) != 0)
            return true; // auto-repeat of a key that is still down
        held.insert(code);
        if (onPress)
            onPress(*target, keyLabel({code, mods.isShiftDown() && code >= 'A' && code <= 'Z', *target}));
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

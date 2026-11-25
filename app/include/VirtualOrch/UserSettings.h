#pragma once

#include <atomic>

/**
 * App-wide switches that used to be fixed in the code and are now user settings (Nov 25 2026, Joel). Shown in the
 * Settings widget, saved in settings.xml ("userSettings") by AppSession::saveUserSettings, loaded at startup.
 * JUCE-free so the OT's combo rules (IodPretrainedTypes) can read them.
 *
 * The reduction models' logit adjustments (pitch / duration / onset gap, LogitTD) are not here: they are the active
 * transformer's own switches (RtLogitAdjustments), saved in presets; the Settings widget edits them in place.
 */
struct UserSettings {
    /** OT: an outer voice may only shift outward (lowest note down, highest up); inner notes keep their octave. */
    static constexpr bool kOuterVoicesOutwardDefault = true;
    /** OT: at most one instrument per octave shift (-1 / 0 / +1) for a note, across all families. */
    static constexpr bool kOneInstrumentPerOctaveDefault = true;
    /** Score viewer: draw the key signature (accidentals only where a note departs from it). */
    static constexpr bool kShowKeySignatureDefault = false;

    [[nodiscard]] static auto outerVoicesOutward() -> bool { return outerVoicesOutwardFlag.load(std::memory_order_relaxed); }
    static auto setOuterVoicesOutward(bool on) -> void { outerVoicesOutwardFlag.store(on, std::memory_order_relaxed); }

    [[nodiscard]] static auto oneInstrumentPerOctave() -> bool {
        return oneInstrumentPerOctaveFlag.load(std::memory_order_relaxed);
    }
    static auto setOneInstrumentPerOctave(bool on) -> void { oneInstrumentPerOctaveFlag.store(on, std::memory_order_relaxed); }

    [[nodiscard]] static auto showKeySignature() -> bool { return showKeySignatureFlag.load(std::memory_order_relaxed); }
    static auto setShowKeySignature(bool on) -> void { showKeySignatureFlag.store(on, std::memory_order_relaxed); }

    /** Back to the defaults above. */
    static auto resetToDefaults() -> void {
        setOuterVoicesOutward(kOuterVoicesOutwardDefault);
        setOneInstrumentPerOctave(kOneInstrumentPerOctaveDefault);
        setShowKeySignature(kShowKeySignatureDefault);
    }

private:
    static inline std::atomic<bool> outerVoicesOutwardFlag{kOuterVoicesOutwardDefault};
    static inline std::atomic<bool> oneInstrumentPerOctaveFlag{kOneInstrumentPerOctaveDefault};
    static inline std::atomic<bool> showKeySignatureFlag{kShowKeySignatureDefault};
};

# Sampling notes: what works best so far

A running list of the sampling settings and rules that have sounded best, most recent first.
Each entry says where it lives and whether it is in the code yet.
Status: **done** = in the code; **chosen** = agreed, not built yet; **trial** = built, still judging.

## Model 1 (continuation) defaults for v1.2 — done (2026-11-26)

| Setting | Value |
|---|---|
| pitch τ / τ′ | 0.2 / 0.2 |
| duration τ / τ′ | 0.1 / 0 |
| onset-gap bias | off |

- Where: `app/presets/version1.2.json` and `version1.2_r4.json`, under `logitAdjustments.hybridV1`.
- Model 1 got its own τ settings for this (`RtLogitAdjustments::hybridV1*Tau*`). Before, it always used the base
  values (duration τ 0.5, τ′ 0), whatever the τ fields said: those are model 2's.
- Branch: `sample_viewer` (not committed yet).

## TD controller: γ drops faster when far over the target — trial (2026-11-26)

- Problem: Solemn (TD 2) fell into a repeated 13 cs chord (measured 16); γ could only drop 0.25 per 2 s and
  reached −1.3 after 10 s, so it never came back (run 20261126_152926, maestro_sample_07).
- Rule: when measured ≥ 2× target (error ≤ −1), γ drops by at least 0.5 × |error| and up to 1.0 per update.
  Near the target nothing changes (≤ 0.25 down, ≤ 1.0 up). The floor stays −2.
- Where: `TdController::kFarError / kFarGain / kMaxStepFar`. Branch: `sample_viewer`.
- Effect (two re-runs of the same request): one stayed at 2–3 onsets/s with γ at −2 by 10 s; one had a burst
  of 6–8 onsets/s at 6–10 s that γ pulled back to 0–1 within ~4 s. The −2 floor limits how far it can push:
  the first run sat at 2.3× the target (measured 4.7) with γ at the floor.
- Mirror (same day): when measured ≤ half the target, γ rises by at least 0.5 × error. Both far-off minimum
  steps are scaled by the controller's damping m, so a very sensitive model still settles (TdControllerTest).

## Tremolo first when TD is far under the target — trial (2026-11-26)

- Rule: when measured TD falls to at most half the target (above TD 4) after not being so, γ is held for 2 s
  (one update, logged why "tremolo first") so the OT's tremolo bonus acts first; then γ moves as usual.
  Once per episode. Where: `TdController::kTremoloFirstError / kTremoloFirstMs`. Branch: `sample_viewer`.
- The viewer marks each hold with a dot on the γ lane ("held: tremolo first").
- Effect (Stormy, TD 32, maestro_sample_07, 2 runs): at 6 s measured 4 → hold, tremolo bonus 1.0; by 8 s
  measured 35–45, almost all of it tremolo; then an overshoot to ~50 (1.6× target) that γ brings down only
  0.25 per 2 s (under 2× over, so the slow rule applies) while the tremolo bonus decays.
- Note: this branch still has the fast tremolo bonus (gain 0.5, max 4); `main` has the slower one (0.1, max 2).

## Negative γ stretches every gap — reverted (2026-11-26)

- Why: at γ −2 the onset distribution was unchanged (the model's top gap 45 cs: 41% before, 42% after). The old
  negative kernel only lifted gaps longer than g = 200 cs / TD (100 cs at TD 2), where the model has ~nothing.
- Rule: for γ < 0 the shift is |γ| · ln(c / g), capped at 4g, so the odds of gap c scale by c^|γ|
  (`TdOnsetBias::kernelFor`).
- Effect so far (Solemn, TD 2, 3 + 3 re-runs): still 2–4 onsets/s with γ at −2 in most runs (target 1/s), and
  in some runs a burst was followed by 5–8 s of silence (onsets pushed seconds ahead). Not clearly better than
  the old kernel; keep under review.

## Mood presets: a preset with tremolo strings needs TD 16+ — done (2026-11-26)

- One sounding tremolo note counts 8 onsets a second in the OT's TD meter, i.e. 16 TD units on its own, and the
  RT's TD controller reads that meter. Mysterious (tremolo, clarinet, bassoon) had TD 2: any tremolo put the
  measured TD far over target, so the controller pushed γ to its floor and thinned the reduction out.
- Mysterious is now TD 16 (`MoodPresets.h`, both `main` and `sample_viewer`). Stormy (32) and Grand (64), the
  other presets with tremolo, were already above 16.

## Duration bias only below TD 16 — done (2026-11-26)

- At TD 16, 32 and 64 the duration bias is off for both hybrid models; at TD 1–8 and AUTO it is on (if its
  switch is). Pitch and onset-gap biases are unaffected.
- Where: `RtLogitAdjustments::durationBiasAtTd` (`kDurationBiasBelowTd = 16`), used by the v1.1 and v1.2
  hybrids. v1.3 has no duration bias. Branch: `sample_viewer`.
- Check (maestro_sample_03, model 1): TD 16 has no duration τ in the log; TD 8 has it (0.1–0.37).

## Drop chord notes the model does not want — done (2026-11-26)

- Problem: chords are built bottom-up, each note above the last. Once the line is high, the notes above have
  only a few pitches left under the top limit (C8), and the mask gives those all the probability even when
  the model gave them ~1%. Result: clusters at 104–108 (sample viewer run 20261126_141517, maestro_sample_03).
- Rule: if the pitches still allowed for a stacked chord note hold less than 5% of the model's own pitch
  distribution (raw logits, before the biases), the chord ends there instead of forcing a note in.
- Where: `ForcedChordNote.h`; `Settings::minChordNoteShare` (0.05, 0 = off) in `ChordTokenEngine` (v1.2) and
  `InformativeWaitEngine` (v1.1). Precomputed chord branches don't check it yet. Branch: `sample_viewer`.
- Effect on maestro_sample_03 (same settings): notes at 104–108 went from 28 to 3 and 0 in two re-runs.

## Pitch τ′: a new interval halves it instead of resetting it — trial (2026-11-26)

- Before: any interval new to the 5 s window sent τ′ back to base, so a repeated pattern with small
  variations (−7 +6, with the odd −8) never ramped.
- Now: a new interval takes 10 notes off the ramp (τ′ halves).
- Effect so far is small (draws at base 196/304 → 162/296): τ′ rarely gets past 0.6 before the next new
  interval. A shorter hold (~3 notes) and a faster ramp (double every 5 notes) are the next thing to try.
- Where: `PitchTauScheduler::evaluate`, `app/source/reduction/DensePitchBias.cpp`. Branch: `sample_viewer`.

## Continuation in the sample viewer: model 1 listens during the prompt — done (2026-11-26)

- Model 1 takes the prompt's piano as context but draws nothing until the cut (`setSilentStartDelayMs`), so a
  continuation is a real continuation. Model 2 stays off. (v1.1: model 2 can still play during the prompt.)

## Reduction pitch range: C4 ± 2.5 octaves (F#1–F#6, 30–90) — done on `main`, not committed (2026-11-25)

- `RegisterRange::kRtLow / kRtHigh`; applies in every register, AUTO included.
- Not in the `sample_viewer` branch, which still allows up to C8.

## OT tremolo bonus: slower, no wind-up — done on `main`, not committed (2026-11-25)

- `TremoloBiasController`: gain 0.1 (was 0.5), max 2 (was 4), and it holds while below the target if nothing
  was orchestrated. Fixed TD overshooting to ~100 at TD 16.

## Register mix for the mood presets — done on `main` and `sample_viewer`, option off by default (2026-11-25)

- Shares below C3 / C3–C5 / above C5 steered by logit offsets from the last 16 played notes; a 0 share masks
  its band. Switch: "Presets: register mix" in the Logit adjustments widget.
- Copied into `sample_viewer` on 2026-11-26. The viewer can pick a mood preset ("Preset sets its register mix")
  and plots each band's target and current share, and the offsets.

## Simultaneous model (v1.1 model 2): coarse duration bins + τ′ 0.3 — done, committed (3931ecd)

- The duration bias penalises roughly log-spaced bins instead of 5 cs steps, so a "new" duration has to
  sound different. τ′ 0.3 in the version1.1 / version1.1_fast presets.

## Biases count committed notes only — done, committed (3931ecd)

- Pitch and duration statistics come from notes actually played; precomputed branches are biased by them but
  never counted until committed.

# Hybrid v1.2 (2026-11-24)

Hybrid v1.1 with model 2 replaced by the **chord-token model** ("m2c", run `ovn_m2c_r1`, vocab 14,071).
Model 1 (continuation, `rt_continuation_latest`) is unchanged.

- Spec: `orchestration/docs/model_specs/expressified_simultaneous.txt` B3f.
- Token layout: `orchestration/overnight/pkg/wait3.py`.
- Preset: `app/hybrid_v12.json`. Its ONNX name `rt_simultaneous_chord_latest` is a placeholder until the export exists.

## How it is wired

The preset keeps `"reductionType": 10`, so the same `HybridInformativeWaitReductionTransformer` runs it. Model 2's
ONNX vocab picks the variant: 11879 → v1.1, 14071 → v1.2. On 14071 the transformer turns on
`InformativeWaitEngine::Settings::chordTokens` for model 2's engine. It also hands that engine the voicing bank
named by the preset's `"voicingBank"`, given either as an absolute path or as a file name in `virtual_orch_models`.
If an absolute path is missing on this machine, the same file name is tried in `virtual_orch_models`.

## Model 2 in chord-token mode (InformativeWaitEngine)

| Step | What the engine does |
|---|---|
| Onset step | WAIT values under 6 cs are masked. After a WAIT, `[K k,+1]` is drawn. If d < 12 cs, the `+1` tokens are masked too, because that is the training floor with a leftover. |
| Fire | The target piano quad is written with `[DUR?]`. Then CHORD is drawn, masked to the types with k pitch classes, followed by SDUR, BASS and SVEL. If plus, the leftover quad follows with its onset forced to T. |
| Precompute | The top-5 branches decode these 4 tokens, or 7 with the leftover, instead of 1 + 3k. |
| OT notes | Only the OT gets the block as notes: one bank voicing at random, on BASS, all with duration q and velocity v. The context keeps the 4 tokens. Notes above the register drop an octave while they stay above the bass. The polyphony cap keeps the lowest notes. |
| Window close | WAIT < 300: `[LATE d]`, with d sampled and masked to the LATE values whose window has not passed. The new window is [close + d − 6, close + d + 6], starting no earlier than close. After `maxLates` (2) LATEs: `[EXPIRED]`, then the block (+ leftover) is decoded at once and plays now. WAIT 300: `[EXPIRED]`, then the next draw is a WAIT with the same K (the chain). |
| `[DUR?]` | Written at key-down. A release writes the real duration only while no orchestra quad, WAIT/K, chord block or LATE follows the quad. The median-of-16 estimate and the hold commit are not used. |
| After a piano onset | The orchestra may share the onset (B3f 1g): no 5 cs block. |

Decisions taken while Joel was away (each one easy to change):

1. **Known arrival sets LATE d.** Sometimes a piano onset that is already past the close is what shows the window
   closed (input lag, `expiryGraceCs` 20). Then LATE d = arrival − close (3 cs grid, ≤ 99) and the note fires.
   Training's LATE value is exactly this true delay, so sampling a d that cannot match would be worse.
2. **The chain repeats K without sampling** and forces the next token to be a WAIT. Training always has the same K
   there. A piano onset in between ends the chain.
3. **Temperatures.** CHORD, SDUR, BASS and SVEL use the duration/pitch/velocity settings (temperature 0.5, min-p
   0.25), since the block mirrors a quad. K and LATE use the onset temperature (1.0), as COUNT and WAIT do.
4. **BASS is masked to the Launchpad register.** The voicing then goes upward from the bass.
5. **The target piano quad is always `[DUR?]`.** In training it still sounds at its own chord block. This holds
   even when the key is up before the block is decoded.
6. **`noWaitAfterExpiry` is kept** (as v1.1) after a give-up. It does not apply to the chain.
7. **Monophonic cap.** Under the polyphony cap, the leftover is still written to the context, because K said +1,
   but it is not played.

## Model 1 and the shared music

v1.1 already never lets model 1 read model-2 tokens. Each model has its own engine and token stream, both get
every piano note, and every note one model outputs is mirrored into the other's context as a plain quad
(`mirrorNote`). In v1.2, model 2's output notes are the decoded chord notes and the leftover, so model 1 sees each
chord block as plain quads without any change. Model 1's piano durations come from its own engine (estimate, then
the release rewrite), so it never sees `[DUR?]`. Model 1's notes mirrored into model 2 are plain orchestra quads,
which m2c's training data has too. No new design decision was needed.

## Open points for Joel

- **Root-field offset: fixed (2026-11-24).** The first m2c tokenizer named CHORD roots from the token
  `11128 + p`, and 11128 % 12 = 4, so its roots were (true root + 4) mod 12. `wait3.analyse_stack` now uses MIDI
  pitches, the voicing bank was rebuilt on true roots (bass_iv = (bass pitch class − root) mod 12, so a
  root-position C major is `maj|3|0`), and the m2c run was restarted on the fixed tokenizer. The engine and
  `VoicingBank` now take the CHORD root as the true pitch class, with no offset. An ONNX from the first
  (pre-fix) run would decode every chord 4 semitones off with this engine, so don't use one.
- **The engine never writes a never-comes `[EXPIRED]` right after a WAIT.** Training's never-comes lines
  (q ≈ 3%) have that, while the engine always tries a LATE first. To match those lines, set `maxLates = 0`.
- **Order versus time in the `[DUR?]` rule.** The engine rule goes by order: a model unit written after the quad
  freezes `[DUR?]`. The training rule goes by time: is the note still sounding at the next model unit's onset?
  They differ when the orchestra note is scheduled in the future and the key goes up before it starts. This was
  kept as specified (B3f 3b).
- **Placeholder ONNX name.** `rt_simultaneous_chord_latest` has to match the export.

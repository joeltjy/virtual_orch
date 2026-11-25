# Hybrid v1.2 (2026-11-24)

Hybrid v1.1 with model 2 replaced by the **chord-token model** ("m2c", vocab 14,071). It needs an export of run
`ovn_m2c_r2` or later (the chord block after all piano notes, and LATE / EXPIRED choices).
Model 1 (continuation, `rt_continuation_latest`) is unchanged.

- Spec: `orchestration/docs/model_specs/expressified_simultaneous.txt` B3f.
- Token layout: `orchestration/overnight/pkg/wait3.py`.
- Preset: `app/hybrid_v12.json`. Model 2 = `simultaneous_chordtokens_m2c_r2b_kv.onnx` (run `ovn_m2c_r2b`, step 2500;
  verification report next to it), voicing bank `voicing_bank_m2c.json`, both in `virtual_orch_models`.

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
| Fire | All the piano notes of the onset are written first, each with `[DUR?]`. Then the chord block follows: a matching precomputed candidate's block as is, or an inferred CHORD and a live SDUR, BASS and SVEL. If plus, the leftover quad follows with its onset forced to T. See "Chord branches" below. |
| Precompute | The 5 branches are chord candidates, each with its full block (4 tokens, or 7 with the leftover). |
| OT notes | Only the OT gets the block as notes: one bank voicing at random, on BASS, all with duration q and velocity v. The context keeps the 4 tokens. Notes above the register drop an octave while they stay above the bass. The polyphony cap keeps the lowest notes. |
| Window close | At every close, of a WAIT (any d, 300 included) or of a LATE, the model draws `[LATE d]` or `[EXPIRED]`. The LATE values are masked to those whose window has not passed. LATE opens a new window [close + d − 6, close + d + 6], starting no earlier than close. After `maxLates` (2) LATEs, `[EXPIRED]` is forced. |
| After `[EXPIRED]` | The model draws a WAIT or a CHORD masked to the pending k. WAIT: a re-wait, and the next draw is a WAIT with the same K (training's > 300 cs chain). CHORD: give up. The rest of the block (+ leftover) is decoded at once and plays now with no piano. |
| `[DUR?]` | Written at key-down. A release writes the real duration only while no orchestra quad, WAIT/K, chord block or LATE follows the quad. The median-of-16 estimate and the hold commit are not used. |
| After a piano onset | The orchestra may share the onset (B3f 1g): no 5 cs block. |

Decisions taken while Joel was away (each one easy to change):

1. **Known arrival sets LATE d.** Sometimes a piano onset that is already past the close is what shows the window
   closed (input lag, `expiryGraceCs` 20). Then LATE d = arrival − close (3 cs grid, ≤ 99) and the note fires.
   Training's LATE value is exactly this true delay, so sampling a d that cannot match would be worse.
2. **A re-wait repeats K without sampling.** Training always has the same K there, since the same stack is still
   pending. A piano onset in between ends the re-wait. The choice after `[EXPIRED]` is drawn over WAIT ∪ CHORD at
   the close; on a WAIT, the WAIT value itself is then drawn by the next onset step from the same logits, within the
   WAIT range. Given a WAIT, that is the same distribution, and the WAITs whose window has passed by then are masked.
3. **Temperatures.** CHORD, SDUR, BASS and SVEL use the duration/pitch/velocity settings (temperature 0.5, min-p
   0.25), since the block mirrors a quad. K, LATE-or-EXPIRED and WAIT-or-CHORD use the onset temperature (1.0), as
   COUNT and WAIT do. A give-up's CHORD is the one drawn at 1.0 in that choice; SDUR, BASS and SVEL follow at 0.5.
4. **BASS is masked to the Launchpad register.** The voicing then goes upward from the bass.
5. **Every piano quad of the fired onset is `[DUR?]`.** In training they still sound at the chord block. This holds
   even when a key is up before the block is decoded.
6. **`noWaitAfterExpiry` is kept** (as v1.1) after a give-up. It does not apply to a re-wait.
7. **Monophonic cap.** Under the polyphony cap, the leftover is still written to the context, because K said +1,
   but it is not played.

## Model 1 and the shared music

v1.1 already never lets model 1 read model-2 tokens. Each model has its own engine and token stream, both get
every piano note, and every note one model outputs is mirrored into the other's context as a plain quad
(`mirrorNote`). In v1.2, model 2's output notes are the decoded chord notes and the leftover, so model 1 sees each
chord block as plain quads without any change. Model 1's piano durations come from its own engine (estimate, then
the release rewrite), so it never sees `[DUR?]`. Model 1's notes mirrored into model 2 are plain orchestra quads,
which m2c's training data has too. No new design decision was needed.

## LATE or EXPIRED is the model's choice (2026-11-24, Joel's "option 3")

Until then, the engine always wrote a LATE first and wrote `[EXPIRED]` only after the last LATE. Now the model chooses
at every close, matching the training data from `ovn_m2c_r2` on (`wait3.py`: `[EXPIRED]` carries loss, and 20 % of
synthetic late lines end in `LATE … [EXPIRED]` followed by the chord block alone):

- at a WAIT close: LATE (late pianist) or EXPIRED (never comes);
- at a LATE close: LATE (second late) or EXPIRED (gave up); after 2 LATEs, EXPIRED is forced;
- after EXPIRED: WAIT (re-wait) or CHORD (give up: the stack plays now).

A piano onset already known to be past the close still forces LATE d = its true delay, while a LATE is allowed.
`maxLates = 0` forces EXPIRED at every close. Stats: `expiredChosen` counts EXPIREDs the model drew where a LATE
was allowed; `giveUps` counts EXPIRED → CHORD; `chains` counts EXPIRED → WAIT.

**Use an export of `ovn_m2c_r2` or later.** `ovn_m2c_r1` was trained without loss on `[EXPIRED]`, so its
LATE-or-EXPIRED and WAIT-or-CHORD choices are untrained and will not behave well.

## Chord branches (2026-11-24, B3f "CHORD BRANCHES")

Training order from `ovn_m2c_r2` on: `[WAIT d][K] … [all piano quads of T][CHORD][SDUR][BASS][SVEL][leftover]`.
The older order (`ovn_m2c_r1`: the first piano quad, the block, then the other piano quads) does not match this
engine, so r1 exports should not be used with it.

**Precompute (A iii, B).** These steps run in the precompute, and each batched one is one call per engine step:
1. One batch-1 call on `[T, DUR?]` gives the top-5 piano pitches.
2. One batched call on `[pitch, velocity]` gives p(CHORD | pitch), masked to the types with k pitch classes.
3. Each chord type + root is scored by Σ p(pitch) · p(chord | pitch), and the top 5 are kept.
4. Each candidate is decoded after its largest contributor. The batch is tiled again from `[T, DUR?]` and fed
   `[pitch, velocity, CHORD]`, then SDUR, BASS, SVEL (+ the leftover). That is 4 batched calls, or 7 with plus, as before.

**Fire.**
- **Grouping.** The block goes after the piano notes that come in with the target, i.e. in the same input batch as
  it. That is the input's own grouping, and the engine adds no wait. A note of the same 5 cs onset arriving
  later is written after the block, as before.
- **Match.** A candidate matches if the onset's piano pitch classes plus its own form a triad, a seventh chord, or
  fewer than 3 pitch classes (`ChordTheory::recognized`). If several match, the best score wins. Its precomputed
  block is written as is and played: 0 calls.
- **No match** (an unfinished precompute counts as one). `ChordTheory::infer` gives the type and root from the
  piano pitch classes, by steps 1–4 of the INFERENCE RULE, preferring k pitch classes at each step. The model then
  picks n (CHORD logits masked to that type and root) and decodes SDUR, BASS and SVEL (+ leftover) live, in the
  fire's own step.
- **Stats.** `chordMatched` counts matches; `inferSteps[1..4]` counts inferences by step.

Decisions taken for this part (each easy to change):

8. **Candidates are chord type + root; n is summed.** "Notes means pitch classes", so candidates that differ only in
   n would match the same piano notes. With CHORD tokens as candidates, the top 5 would mostly be one chord with
   five values of n. Each candidate's n is drawn from its contributor's CHORD logits, restricted to that type and
   root.
9. **Re-tiling instead of regrouping rows.** The candidates' rows are re-tiled from `[T, DUR?]` and fed 3 tokens
   each (`[pitch, velocity, CHORD]`), instead of regrouping the pitch rows. This works with the KV cache kept on
   the GPU and costs no extra call (still 4 batched calls, 7 with plus).
10. **How the rules read where the spec leaves room:**
    - A unison or dyad with k > its count is extended to the k-pitch-class chords that contain it, ranked as in
      step 2.
    - "Canonical in the key" for a unison or dyad means all its notes are in the key's scale. For a minor key that
      scale is natural minor plus the raised 7th.
    - Step 3's "closest in the key" is read as the fewest differences: one extra or one missing counts 1, one swap
      counts 2. Then comes the most common degree, in the order I, V, IV, ii, vi, iii, vii.
    - Step 4 follows k: k = 1 gives the note itself, k = 2 the third, k = 4 min7 or dom7.
    - Any remaining tie goes to the lower type index, then the lower root.
11. **Key weights.** These are the piano and orchestra notes with onsets in the last 800 cs, each weighted by its
    duration (capped at 800). A `[DUR?]` note counts with the time it has been held so far. A chord block adds its
    chord's pitch classes with its SDUR. All-zero weights give C major.

## BASS is a chord tone (2026-11-25 fix)

BASS is masked to the CHORD's pitch classes (`bassRanges`) wherever a block is decoded: precomputed candidates,
the live decode after an inference, and a give-up. In training BASS is the lowest band note, so it is always a chord
tone. Without the mask the model often repeated the previous bass, e.g. B2 under E♭ dim. The voicing was then
stacked on a non-chord-tone bass, and the bank key's `bass_iv` was not a chord position. Joel heard this as an
"E♭ half-diminished over a dominant 7th".

Two edge cases:
- **No chord tone in the register.** If the register holds no chord tone at all, any pitch in the register is
  allowed rather than none.
- **Missing bank key.** If `type|n|bass_iv` is missing and no other n exists for that bass position, the chord
  tones are stacked upward from the bass, n notes. This was already the case, and it is now tested.

The match rule still accepts a candidate that is the upper structure of the piano's chord: E♭ dim with piano
B–E♭–A forms B7. With the bass masked to E♭ dim's tones, the block voices E♭ dim on one of its own tones, and together with the
piano's B it sounds as B7.

## Joel's v1.1 evening commits brought in (2026-11-24)

Only c9f419d and 3931ecd were taken from `origin/informative-wait`, as Joel asked. fae4bd7 (branches by pitch
class) was left out, because v1.2's branches are chord candidates. d59171a (a WAIT never expires) was left out,
because v1.2 keeps LATE / EXPIRED.

- **c9f419d** was cherry-picked as is. It sets the startup defaults (Performance view, the version1.1_fast preset)
  and turns on the logit adjustments for both Hybrid v1.1 models.
- **3931ecd** applied without conflicts. Its v1.1 behaviour is unchanged: biased branch notes,
  `countCommittedForBias` at a fire, and coarse duration bins with τ′ for model 2. The transformer sets
  `coarseDurationBins` for model 2's engine, so v1.2's chord-token model 2 gets the coarse bins too. In chord mode:
  - **Precomputed chord blocks.** SDUR and the leftover's duration take the duration bias; BASS and the leftover's
    pitch take the pitch bias. Both use the committed notes only, with copies of the τ schedulers.
  - **Live blocks.** A block decoded at the fire (no match) or at a give-up biases its SDUR and BASS the same way,
    with the engine's own schedulers.
  - **The leftover** of a live block goes through `drawBody`, which is already biased.
  - **A matched block** is counted with `countCommittedForBias`: 1 note, or 2 with the leftover.
  - **History.** The bias statistics now count each chord block as one orchestra quad: onset, SDUR, BASS, SVEL
    (`orchestraHistory` in chord mode). Without this, model 2's SDUR and BASS would only ever be compared against
    the leftovers and model 1's mirrored notes. The onset-gap bias reads the same history, so it now sees chord
    onsets too.

## Open points for Joel

- **Root-field offset: fixed (2026-11-24).** The first m2c tokenizer named CHORD roots from the token
  `11128 + p`, and 11128 % 12 = 4, so its roots were (true root + 4) mod 12. `wait3.analyse_stack` now uses MIDI
  pitches, the voicing bank was rebuilt on true roots (bass_iv = (bass pitch class − root) mod 12, so a
  root-position C major is `maj|3|0`), and the m2c run was restarted on the fixed tokenizer. The engine and
  `VoicingBank` now take the CHORD root as the true pitch class, with no offset. An ONNX from the first
  (pre-fix) run would decode every chord 4 semitones off with this engine, so don't use one.
- **Order versus time in the `[DUR?]` rule.** The engine rule goes by order: a model unit written after the quad
  freezes `[DUR?]`. The training rule goes by time: is the note still sounding at the next model unit's onset?
  They differ when the orchestra note is scheduled in the future and the key goes up before it starts. This was
  kept as specified (B3f 3b).
- **How a chord block should feed the pitch bias.** Counting a block as its bass note is the smallest reading. It
  could also count every voiced note, or the chord's pitch classes, which would hold back whole harmonies rather
  than bass pitches. The pitch bias on BASS only steers the bass, never the chord type or root, so CHORD itself is
  never biased. Is that what you want for v1.2?
- **Coarse bins on SDUR.** The bins (10–320 cs) were tuned on v1.1's Mono durations. Stack durations are longer on
  average, so the bins may be worth re-checking once r2 plays.

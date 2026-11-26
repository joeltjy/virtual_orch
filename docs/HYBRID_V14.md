# Hybrid v1.4 (2026-11-26)

Hybrid v1.2 with both models replaced by the **v1.4 group-token models**. An orchestra chord is one *group* token
sequence (a bass plus a shape) instead of one quad per note. Groups also travel from RT to the OT and playback as
**one** note that carries its shape. They are decoded into pitches only at the receiving end.

- Tokenization spec: `orchestration/docs/model_specs/v14_group_tokens.txt` (A: the group, B: m1_v4, C: m2_v4).
- Python token layout: `orchestration/overnight/pkg/groups4.py` (groups, shapes), `wait4.py` (m2_v4 WAIT blocks),
  `sample_v4.py` (the reference sampler this engine ports).
- Simultaneous-case sampling experiments: `orchestration/SIMULTANEOUS_SAMPLING_NOTES.md`.

## Using it

1. Put these in `~/Documents/virtual_orch_models/`:

   | file | what it is |
   |---|---|
   | `rt_continuation_chordal.onnx` | model 1, m1_v4 (piano silent): run `ovn_m1v4_r1`, step 6000, vocab 12,096 |
   | `rt_simultaneous_chordal.onnx` | model 2, m2_v4 (piano playing): run `ovn_m2v4_r1`, step 4500, vocab 12,239 |
   | `voicing_bank_v4.json` | how each (shape, N) is voiced, from corpus counts |

   Each `.onnx` has a verification report (`.onnx.json`) and a description (`.md`) next to it.
2. In the app, pick **Reduction → "Hybrid v1.4 (group tokens: m1_v4 silent / m2_v4 playing)"** (menu item 13).
   Or load the preset `app/presets/version1.4.json` (`"model": "hybrid_v14"`, `"reduction": "hybridGroup"`). The
   model config is `app/hybrid_v14.json`:
   - `onnx` / `silentOnnx`: the two models above;
   - `reductionType`: 13;
   - `voicingBank`: `voicing_bank_v4.json`;
   - `precompute`: true;
   - `iwMode`: unconstrained;
   - `startDelayMs`: 3000.
3. The preset's logit adjustments are the sampler's v3 settings:
   - on: pitch, duration, short gap, LogitTD;
   - off: onset gap;
   - pitch tau 0.2 / tau' 0.2, duration tau 0.1 / tau' 0.

   The silent side (`hybridV1*`) uses the same switches.

`init` checks the vocabularies: 12,096 for the silent model and 12,239 for the playing model. A wrong file fails with
a message in `lastInitError`.

## The group token

| event | tokens | notes |
|---|---|---|
| single note | `[onset][dur][pitch][vel]` | as before |
| group | `[onset][dur][BASS][SHAPE×N][vel]` | BASS is an orchestra pitch token |

- **SHAPE×N:** 43 shapes relative to the bass, each with a note-count range N, giving 328 tokens. They start at
  11,768 in m1_v4 and at 11,911 in m2_v4.
- **The shapes:** monad (N 2–5), the dyads +1..+11 (N 2–6), the triad inversions (N 3–10), and the seventh
  inversions (N 4–12). Aug and dim7 have one shape each.
- **Table:** `GroupChord` (`app/include/VirtualOrch/reduction/GroupChord.h`). It must stay in the order of
  `groups4.py`, which the `GroupChord.TableMatchesGroups4` test checks.
- **The draw:** after the pitch, the model draws either a velocity (a single note) or a SHAPE, then the velocity.
  There are no shapes in line modes or when `maxChordNotes` ≤ 1.

## RT → OT: a group is one note

`Token` has a new field, `chordShape` (−1 = a plain note, 0..327 = a groups4 offset).

| stage | what happens |
|---|---|
| RT (`GroupTokenEngine`) | A group or WAIT block leaves the engine as one `OutNote`: pitch = bass, shape = offset. `HybridGroupReductionTransformer` copies it to `Token::chordShape`. |
| OT input (`OrchestrationTransformer.cpp`) | After `drainIncoming`, `GroupChord::expandAll` turns each group token into its notes. |
| Reduction playback (`OutputPlayback.cpp`) | The reduction monitor expands each group with `GroupChord::expand` and schedules every note. |
| Mirroring between the models | The mirrored note keeps its shape, so each model sees the other's groups as groups. |

**How a group is decoded** (`GroupChord::expand`, `VoicingBankV4::decode`):

- **The voicing:** a draw from the bank's voicings for that offset, weighted by corpus count. With no bank entry,
  it falls back to close position upward from the bass.
- **The seed:** the token's `noteId`. The OT and playback therefore always get the same voicing for the same
  group.
- **The notes:** every note gets the group's time, duration, velocity and noteId.
- **The bank:** `loadVoicingBank` loads it once and shares it (`GroupChord::setSharedBank`).

## Model 2 (GroupTokenEngine, simultaneous)

The engine is a copy of v1.2's `ChordTokenEngine`, so v1.2 is unchanged. WAIT, K, LATE, EXPIRED, `[DUR?]`, the
window rules and the switching between models are as in `HYBRID_V12.md`. What differs:

| step | v1.4 |
|---|---|
| K | k = the main item's pitch classes (1–4); +1 = a leftover item follows. |
| Fire block | All piano quads of T (each with `[DUR?]`), then `[SDUR][BASS][SHAPE×N][SVEL]`, or `[SDUR][PITCH][SVEL]` when k = 1. Then the leftover item if +1. The block's SHAPE is masked to exactly k pitch classes. |
| Precompute | The top-5 anticipated piano pitches each decode a whole block, in stages: SDUR, BASS, SHAPE-or-velocity, SVEL, then the leftover. Each is scored by p(pitch) × p(BASS) × p(SHAPE). The single-vs-group choice is made once for all branches, so the batch stays aligned. A precomputed leftover is a single note. |
| Match rule | As v1.2, with the candidate's pitch classes taken from bass + shape. |
| Inference (no match) | BASS is masked to the inferred chord's pitch classes, and SHAPE to that chord's shape over the drawn bass (else to k pitch classes). SDUR and SVEL are decoded live. |
| After `[EXPIRED]` | The model draws a WAIT (re-wait) or an SDUR (give up: the block plays now). |
| Polyphony cap 1 | Only the block's bass plays. |

Model 1 (m1_v4) runs the same engine with no WAITs. It writes free notes and groups.

## Code

| file | what it is |
|---|---|
| `app/include/VirtualOrch/reduction/GroupChord.h`, `app/source/reduction/GroupChord.cpp` | shape table, pitch-class masks, labels (e.g. "C3 maj root x4"), `VoicingBankV4`, `expand` / `expandAll` |
| `.../GroupTokenEngine.h/.cpp` | the engine: constants `kShape0Group` 11911, `kShape0Continuation` 11768, `kGroupVocab` 12239, `kContinuationVocab` 12096 |
| `.../HybridGroupReductionTransformer.h/.cpp` | "Reduction Transformer Hybrid v1.4", label "Hybrid v1.4" |
| `ReductionTransformer.h` | `MusicModelArch::DenseHybridGroup` |
| `AppSession`, `MainComponent` | menu item 13, provider "hybridGroup", log tag `HYBRID14` |
| `LogitAdjustmentsWidget`, `ActiveModelWidget`, `WaitVisualizerWidget` | Hybrid v1.4 cases |
| `test/source/GroupChordTest.cpp` | shape table vs groups4, masks / labels, close-position fallback, seeded bank decode |
| `test/source/GroupTokenEngineTest.cpp` | scripted backend tests (see below), plus `HybridV14LoadsTheGroupModels` (loads the real exports; skipped if they are absent) |

The scripted backend tests check four cases:

- a fire served from the precomputed group block;
- a fire decoded on the spot without precompute;
- inference when no candidate matches;
- a free group leaving the engine as one note with its shape.

Run only these tests, so a run doesn't load the whole suite:

```bash
build/test/VirtualOrchTest_artefacts/Debug/VirtualOrchTest --gtest_filter='GroupTokenEngine.*:GroupChord.*'
```

The real-model test takes about 80 s in a Debug build.

## Not in the app yet

These are tried in `sample_v4.py` and tracked in `orchestration/SIMULTANEOUS_SAMPLING_NOTES.md`:

- the top-note seed;
- the forced first WAIT;
- onset top-p 0.8;
- the WAIT floor.

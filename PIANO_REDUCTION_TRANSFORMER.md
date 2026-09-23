# Piano Reduction Transformer

This document explains a new reduction model that was integrated into `virtual_orch`:
`ReductionTransformerPianoReduction`. It's written for someone with no prior context on
this change — if you already know the codebase well, skim the headings.

## What problem this solves

The app's pipeline is: MIDI keyboard → **reduction transformer** (turns a live
performance into a simplified "piano reduction") → **orchestration transformer** (turns
that reduction into a full orchestra) → playback.

Until now, every reduction model (`ReductionTransformerV1`, `ReductionTransformerV2`)
was trained on a single instrument: everything it ever saw or produced was "instrument
0." The new model is different — it was finetuned on **paired piano + piano-reduction**
MIDI, so it genuinely has two instruments:

- **Instrument 0 = the original piano.** This is what you play on the keyboard.
- **Instrument 1 = the reduction.** This is the *only* thing the model generates. It's
  what gets handed to the orchestration stage.

In other words: you play the piano part, the model watches it as context, and it writes
a second, separate line (the reduction) that the orchestration transformer will later
turn into a full orchestra.

## Where the model came from

Trained on the remote GPU host (`hai-res-mobile01.csail.mit.edu`), in a sibling
repository (`~/orchestration`, referred to elsewhere as `jam_bot`/`jordanai` — this is
**not** part of `virtual_orch`; the actual PyTorch training code, dataset construction,
and checkpoints all live there, not here).

- Training script: `orchestration/train/maestro_v8_piano_reduction_finetune.py`
- Base checkpoint it was finetuned from: `amt_dense_maestro_finetune_duration_v8`
  (the same "MAESTRO v8" lineage that `ReductionTransformerV2` targets — this is why
  the new class reuses V2's quantization grid, see below)
- Checkpoint used for this integration: **`checkpoint-4000`**, chosen over the later
  `checkpoint-14000` because eval loss was lowest there (1.4187) and had *risen* to
  1.672 by step 14000 — later checkpoints had started overfitting.
- Model shape: a plain HuggingFace `GPT2LMHeadModel`, 24 layers, 16 heads, 1024 hidden
  size, 1024-token context, vocab size 11768 — architecturally identical to the AMT
  dense models already used by V1/V2 (same dense event vocabulary, see below).

## How a piano note and a reduction note are told apart

Every note in this vocabulary is one of 4 integer tokens: `(onset, duration, note,
velocity)`. The `note` token encodes *both* pitch and instrument in one number:

```
note_token = NoteOffset + instrument * MaxPitch + pitch
```

With `MaxPitch = 128`, instrument 0 occupies note tokens `11000–11127` and instrument 1
occupies `11128–11255`. This scheme already existed in `DenseTypes.h` before this
change (`DenseConfig::MaxInstr = 5`, i.e. room for 5 instruments) — it just was never
exercised beyond instrument 0 until now. **No vocabulary change was needed**; this
checkpoint's vocab size (11768) matches the app's existing `DenseVocab::VocabSize`
exactly.

## What's actually new in the code

### `ReductionTransformerPianoReduction`

New class, in
[app/include/VirtualOrch/reduction/ReductionTransformerPianoReduction.h](app/include/VirtualOrch/reduction/ReductionTransformerPianoReduction.h)
/
[app/source/reduction/ReductionTransformerPianoReduction.cpp](app/source/reduction/ReductionTransformerPianoReduction.cpp),
inheriting directly from the shared `ReductionTransformer` base class (same base V1 and
V2 use). Structurally it's closest to V2 (same MAESTRO-v8 quantization/masking, same
"lighter" real-time clamps), plus three things unique to this model:

1. **Generation is hard-locked to instrument 1.** A new method,
   `generationInstrumentLogits`, masks out every note-vocab range except instrument 1's
   band before the model samples a note. This is a fixed constant, not driven by the
   existing `ModelConfig::outputInstruments` preset setting (the way V1's `instrLogits`
   works) — a wrong preset value should not be able to make this model resample piano
   notes as if it were generating them.
2. **A dedicated instrument-aware sort**, to fix a train/sample mismatch (see next
   section).
3. **An on/off toggle for the online pitch/duration anti-repetition bias** that V2
   always applies unconditionally. This model gets the same bias logic (ported
   verbatim from V2), but gated behind `biasEnabled` (default **on**), settable from
   the main window ("Reduction Bias" checkbox, persisted like the existing
   "Proportion Bias" toggle) or via `setBiasEnabled(bool)` in code.

### The train/sample mismatch that was found and fixed

This was the main risk called out before implementation, and it turned out to be real.

The offline training data (in the sibling `orchestration` repo,
`preprocessing/amt_dense.py`, function `_quad_sort_key`) sorts every note by
**`(onset, instrument)` ascending** — so if a piano note and a reduction note land on
the exact same onset, the piano note *always* comes first in the training sequence.

The app's existing pre-inference context sort,
[`NoteWindow::sortStridedByOnset`](app/include/VirtualOrch/NoteWindow.h:57), only
compares onset. Ties keep whatever order notes happened to arrive in. That's harmless
for V1/V2, because every note they ever see is instrument 0 anyway — a tie between two
instrument-0 notes has no "correct" order to violate. But for this model, if a live
piano note and a just-generated reduction note tie on onset, arrival order could easily
put the reduction note first — the opposite of what the model was trained on.

Fix: a new function,
[`DensePianoReductionOrder::sortStridedByOnsetThenInstrument`](app/include/VirtualOrch/reduction/DenseTypes.h)
(in `DenseTypes.h`), sorts by onset and then by ascending instrument on ties. It's used
only by `ReductionTransformerPianoReduction::generateNewToken`, in place of the plain
onset-only sort. It's additive — `NoteWindow::sortStridedByOnset` itself was not
changed, so V1/V2/AMT behavior is untouched.

### Everything else that was added

- `DenseConfig::PianoReductionInputInstrument` (= 0) and
  `PianoReductionOutputInstrument` (= 1) — named constants for the two fixed roles,
  in `DenseTypes.h`.
- `DenseQuantize::instrumentOfNoteToken` — given a packed note token, returns which
  instrument band it's in (used by the new sort).
- `MusicModelArch::DensePianoReduction` — new entry in the existing model-architecture
  enum ([ReductionTransformer.h](app/include/VirtualOrch/reduction/ReductionTransformer.h)),
  alongside `Amt`, `DenseV1`, `DenseV2`.
- `AppSession` gained a `reductionTransformerPianoReduction` member, wired into every
  place `reductionTransformerV1`/`V2` already were (thread start/stop, orchestration
  hookup, voice separation, ahead-throttle, etc.) — see `AppSession.h` / `AppSession.cpp`.
- The main window's "Reduction:" dropdown gained a 4th option, "Piano Reduction
  (2-instrument)", and a "Reduction Bias" checkbox next to the existing "Proportion
  Bias" one — see `MainComponent.cpp`.
- A checkpoint descriptor, `app/piano_reduction_checkpoint-4000.json`, so the app's
  checkpoint picker can find the exported model (paired with the `.onnx` file — see
  below for where that needs to live).

## Known limitation

`ReductionTauWidget` (the little live τ/τ′ bar-chart widget that visualizes V2's
anti-repetition bias in the workspace UI) only checks for `MusicModelArch::DenseV2`, so
it won't light up when this new model is active — even though the bias math is running
identically underneath. The bias itself works correctly; only that one debug
visualization doesn't know about the new architecture yet. Fixing it would mean
generalizing `ReductionTauWidget` to accept either V2 or the new class (they expose the
same `getPitchTau()`/`getDurationTau()`/etc. API) — left as a follow-up since it's
cosmetic.

## The exported model

The checkpoint was exported to ONNX on the remote host using the existing
`~/experiments/src/experiments/export_amt_dense_onnx.py` exporter (no changes needed —
same GPT-2 architecture as every other dense-AMT export it already handles):

```
python -m experiments.export_amt_dense_onnx \
  --checkpoint /data/scratch-oc40/joel/checkpoints/amt_dense_maestro_v8_piano_reduction/checkpoint-4000 \
  --out ~/amt_dense_piano_reduction_checkpoint-4000.onnx
```

Result: a 1.31 GB `.onnx` file, validated with `onnx.checker.check_model()`, with a
single dynamic `input_ids → logits` graph (no past-KV cache), matching the exact
contract `ReductionTransformerV1`/`V2`/this new class all expect. A copy was saved to
`~/Desktop/amt_dense_piano_reduction_checkpoint-4000.onnx` on this Mac.

To actually use it in the app, drop that `.onnx` file next to
`app/piano_reduction_checkpoint-4000.json` (either directly in `app/`, or in
`~/Documents/onnx_export_092126/` with a matching name — both locations are already
scanned by the app's checkpoint picker, see `ProjectPaths.h`), then pick "Piano
Reduction (2-instrument)" in the Reduction dropdown and select the checkpoint.

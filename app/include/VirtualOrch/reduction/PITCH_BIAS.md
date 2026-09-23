# V2 pitch bias (online)

`ReductionTransformerV2` can reshape the **note** logits before sampling so the model
is less likely to keep repeating the pitches (and pitch leaps) it has used recently.
The helpers live in `DensePitchBias.h` / `.cpp`. Bias is applied only on the note
field (onset / duration / velocity are unchanged).

This is independent of the orchestration Launchpad proportion bias (`τ_i` / `p_i` on
instrument-combo logits). Same Greek letter, different subsystem.

## Goal

When the recent pitch set has been stuck for a while, push probability mass away
from pitches (and intervals) that already dominate the last few seconds. When the
set of active pitches changes, ease off again so the model can settle into a new
area.

## Time base

Everything uses **centiseconds** (1 unit = 10 ms; `DenseConfig::TimeResolution = 100`
ticks per second).

**Now** matches offline `amt_causal` sampling: the last onset in the growing prefix,
so generated-ahead notes stay inside the 5 s window. Live RT also has a wall clock;
callers pass `clock->getTime()` and the helpers use `max(clock, lastOnset)` so that
(1) when generation is ahead of the clock, behaviour matches the offline prefix, and
(2) once the clock catches up, silence can still age the window and ramp τ. Pass
`nowCs < 0` (tests / no clock) to use last onset only.

## Sliding window

From dense `inputData` (4 ints per note: onset, duration, note id, velocity), keep
notes whose absolute onset is in `(now − 500, now]` — the last **5 seconds** of
musical time under the now rule above.

Pitch of a note token:

```
pitch = (noteId − NoteOffset) % 128   // MIDI pitch 0…127
```

Instrument band is ignored for statistics; the same pitch adjustment is later
written into every instrument band of the note vocab.

From that window we build, in stream order:

| Symbol | Meaning |
|--------|---------|
| ordered pitches | `[p₀, p₁, …]` as they appear in `inputData` |
| **π** | how often each MIDI pitch appears |
| **s_y** | how often each consecutive pitch *interval* appears |
| **S** | sorted unique pitches (drives the strength schedule only) |

Stats are **recomputed** each time we refresh (no forever-running accumulator).

### π — pitch unigram

Count occurrences of each pitch `p ∈ [0, 127]` in the window (`N` = note count).

Additive smoothing with `ε = 1/128`:

```
π[p] = (count[p] + ε) / (N + 128 · ε)
```

Unseen pitches still get a little mass so `log π` stays finite.

### s_y — pitch-interval distribution

For consecutive pitches in the window, `Δ = pᵢ₊₁ − pᵢ`. Only leaps in
**`[-12, 12]`** count (same as offline `amt_causal`); wider jumps are ignored for
both the histogram and the τ′ set **D**. Histogram with smoothing `ε = 0.01`
over 25 bins (index = `Δ + 12`):

```
s_y[Δ] = (count[Δ] + ε) / (numInRangePairs + 25 · ε)
```

When biasing a **candidate** pitch `p`, the interval used is vs the **last** pitch
still in the window: `Δ = p − lastPitch`. If `|Δ| > 12`, skip the τ′ term for
that candidate (unigram τ·log π still applies).

## Strength schedule (τ and τ′)

`τ` weights the unigram term; `τ′` weights the interval term. They share the same
hold/ramp shape but **independent clocks**.

Let **S** be the set of unique pitches in the 5 s window, and **D** the set of
unique consecutive pitch intervals with **Δ ∈ [-12, 12]** in that window.

### τ (pitch unigram)

1. Whenever **S gains a pitch** → set `τ = 0.5` and restart its hold clock.
   Pitches *leaving* the window do **not** reset.
2. Hold at 0.5 for **2 seconds**, then:

```
τ = 0.5 · 2^(seconds_since_last_new_pitch − 2)
```

### τ′ (pitch interval)

1. Whenever **D gains an interval** (a new Δ appears in the window) → set
   `τ′ = 0.5` and restart its hold clock. Intervals leaving do **not** reset.
2. Hold at 0.5 for **2 seconds**, then the same exponential ramp as τ.

Examples after the relevant set last gained an element: 2 s → 0.5, 3 s → 1,
4 s → 2, 5 s → 4.

So a long stretch without new pitch classes strengthens the pitch anti-repetition
push; a long stretch without new leap sizes strengthens the interval push. The two
can diverge (e.g. oscillating between the same two pitches keeps S stable while
only one Δ is present, so both ramp; introducing a new leap size resets τ′ only).

`ReductionTransformerV2` stores the latest `τ` / `τ′` in atomics via
`refreshPitchTauSchedule()` (called when generating, and from the OrchestrationDebug
**Reduction Tau** widget timer so values keep moving during silence).

## How logits are adjusted (note field)

Offline `amt_causal` (`sampling_notes.txt`) applies Menon τ / π **after** the new
onset (and duration) are already on the growing prefix. RT mirrors that: after
onset is sampled, π / s_y / τ are rebuilt with `now =` that onset, then the note
logits are adjusted before nucleus sample:

```
logits[note(p)]  -=  τ  · log(π[p])
logits[note(p)]  -=  τ′ · log(s_y[p − lastPitch])
```

Because `π` and `s_y` are probabilities ≤ 1, `log(·)` is ≤ 0, so subtracting
`τ · log(·)` **raises** logits for rare pitches / rare intervals and **lowers**
logits for common ones — more so as `τ` grows.

Then the usual nucleus sample runs (`top-p = 0.98`, temperature `0.7` by default).

If the window is empty, skip the bias (no last pitch / no meaningful π).

## What this is not

- Not the orchestration instrument-balance bias.
- Not applied on V1 (frozen older dense clamps / sampling).
- Context length for the ONNX forward remains the last **40 notes**; the 5 s window
  is only for π / s_y / S, not for transformer context.

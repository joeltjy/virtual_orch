# Hybrid v1.3 (2026-11-25): the event-tokenized pair

Spec: `orchestration/docs/model_specs/event_pair_proposal.txt` (sections 1-6 and 8). Tokenizers:
`orchestration/overnight/pkg/events.py` (m1e) and `events2.py` (m2e). Branch `claude/hybrid-v13`, cut from
`integration-2026-11-25` (v1.2 + the TD work).

## What a note is

Two events instead of one quad: ON = (time, note, vel), OFF = (time, OFF0 + note). Times are 0..9999 cs from the
context's base, events in time order, every OFF before every ON at one time, then by note id. No duration tokens.

| model | export | vocab | OFF0 | extra |
|---|---|---|---|---|
| model 1 event (silent side) | `v1.3_continuation` | 12024 | 11768 | |
| model 2 event (playing side) | `v1.3_simultaneous` | 14328 | 14071 | STACK_OFF 14327, the wait3 ids |

## Classes

- `EventEngine`: one model's event context and sampler. Piano key-down / key-up are written as ON / OFF at once.
  A draw is a time, then what happens at it (an orchestra ON with its velocity, or an OFF of a sounding orchestra
  note); it waits as the KV cache's scratch tail until the clock reaches it, and a piano event that comes first
  throws it away (it is drawn again after it), so the context stays in time order. Masks: times from the floor
  (last event / now) to +3 s; at one time OFFs before ONs and ascending note ids; no ON of a pitch already sounding;
  OFFs only of sounding notes; the register range and the polyphony cap.
- **Spec section 8:** a piano token after a drawn time is a no-op and the next time must be later. When the
  orchestra has been silent `forceAfterMs` (preset, default 2000), the piano tokens are masked (an orchestra event
  is forced). A note never released goes after the RT's 5 s cap. The context is rebuilt (newest ~384 tokens, new
  base) before its times pass 9999.
- `EventReductionTransformer`: the base (the RT's I/O for event models). Key-downs become ON events at once,
  key-ups (the filter's TokenUpdates, or a complete note's length) OFF events at their time. A model's ON goes out
  as a note with the provisional length (the RT's 5 s cap); its OFF sends the real length downstream as a
  TokenUpdate, like a key-up, and fixes the output history and the prompt view.
- **Model 2 event** (`Settings::chordTokens`, events2.py / section 4): where a time would be drawn the model may
  draw [WAIT d][K k,+1] (d >= 6 cs, +1 from 12 cs), measured from the event before it. While it is pending the
  orchestra may release notes (OFFs, STACK_OFF) before the target but not start any (Q9). A piano ON from 6 cs
  before the target to the window close (ref + d + 6, + 20 cs grace for live latency) fires: the onset's piano ONs
  (within 5 cs, written at its time), then [CHORD (k pitch classes)][BASS (a chord tone in the register)][SVEL],
  decoded into band notes with the voicing bank (folded into the register, the polyphony cap keeps the lowest),
  then the leftover ON at the onset when +1. The band notes are released together by one (time, STACK_OFF) event
  that the model draws later (the oldest sounding block); the leftover has its own OFF. At a close the model draws
  [LATE d] (at most 2; a piano ON already past the close sets d) or [EXPIRED]; after [EXPIRED] a WAIT (re-wait,
  the same K) or a CHORD (give up: the block plays alone at the close). No precompute / chord branches yet: the
  block is decoded at the fire (3 calls, 5 with the leftover).
- `HybridEventReductionTransformer` (reduction type 11, preset `app/hybrid_v13.json`): model 1 event silent,
  model 2 event playing, v1.1's switching rules (pedal release -> model 1; playing 1 s -> model 2; silent 0.5 s ->
  cut and model 1). Both engines get every piano event and each other's orchestra events. Model 2 draws nothing for
  the first `startDelayMs` (3000) and plays +16 velocity, as v1.2.

## Checks

- `EventEngineTest` (scripted model): ON / OFF timing and tokens, the piano no-op and the forced orchestra, piano
  input before a pending draw, the max hold, the rebuild; the transformer: provisional length then the TokenUpdate,
  the mirrors, the switch, the start delay; the real exports load (vocab check).
- `EventChordTest` (scripted m2e): the fire (piano ONs, then the block, band notes on C major tones with the
  bass), STACK_OFF releasing the band together, LATE then a fire, EXPIRED then a give-up, EXPIRED then a re-wait
  with the same K, the leftover ON at the onset.
- Real m2e (opt-in `IW_M2E=1`, sample_01's piano 15 s, voicing bank): seed 1: 86 ONs (24 from chord blocks,
  5.7/s), 18 WAITs, 17 fires, 8 LATE (all forced: the piano past the close), 8 STACK_OFF, 0 voicing fallbacks;
  seed 2: 80 ONs (14), 8 WAITs, 7 fires, 5 LATE (forced), 1 EXPIRED -> give-up, 5 STACK_OFF. 15 / 11 notes reached
  the 5 s cap (more blocks than STACK_OFFs: worth a look on real playing).
- Real m1e (opt-in `IW_M1E=1`, sample_01's piano for 6 s then 9 s silent): 1.7 / 4.2 ONs a second while silent
  (seeds 1 / 2), as the Python `--reject-piano` runs (1-5 notes/s).

## Not done yet / open

- **Model 2: no precompute / chord branches (B3f match / inference rules) yet.** The block is drawn at the fire
  (3-5 model calls: ~0.3-0.5 s at 110 ms a call on this Mac); v1.2's anticipation would hide that.
- **Downstream lengths.** The OT applies a TokenUpdate to the notes it made from a reduction note only when they
  match its onset / length / velocity / pitch; orchestrated notes that change pitch or velocity keep the provisional
  length (5 s) until the OT matches by the source note id. That is an OrchestrationTransformer change (the TD
  session's files): asked there.
- Not ported from v1.1: "2 s of predicted silence" (a note's length is only known at its OFF), Dynamic LogitTD and
  the logit adjustments (the TD session owns the TD paths), the WAIT visualizer and the RT_DRAW op log.
- `forceAfterMs` 2000 is a first guess (spec 8 leaves the threshold open).

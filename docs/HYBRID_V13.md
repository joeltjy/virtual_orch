# Hybrid v1.3 (2026-11-25): the event-tokenized pair

Spec: `orchestration/docs/model_specs/event_pair_proposal.txt` (sections 1-6 and 8). Tokenizers:
`orchestration/overnight/pkg/events.py` (m1e) and `events2.py` (m2e). Branch `claude/hybrid-v13`, cut from
`integration-2026-11-25` (v1.2 + the TD work).

## What a note is

Two events instead of one quad: ON = (time, note, vel), OFF = (time, OFF0 + note). Times are 0..9999 cs from the
context's base, events in time order, every OFF before every ON at one time, then by note id. No duration tokens.

| model | export | vocab | OFF0 | extra |
|---|---|---|---|---|
| model 1 event (silent side) | `continuation_events_m1e_r1_kv` | 12024 | 11768 | |
| model 2 event (playing side) | `simultaneous_events_chordtokens_m2e_r1_kv` | 14328 | 14071 | STACK_OFF 14327, the wait3 ids |

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
- `HybridEventReductionTransformer` (reduction type 11, preset `app/hybrid_v13.json`): model 1 event silent,
  model 2 event playing, v1.1's switching rules (pedal release -> model 1; playing 1 s -> model 2; silent 0.5 s ->
  cut and model 1). Both engines get every piano event and each other's orchestra events. Model 2 draws nothing for
  the first `startDelayMs` (3000) and plays +16 velocity, as v1.2.

## Checks

- `EventEngineTest` (scripted model): ON / OFF timing and tokens, the piano no-op and the forced orchestra, piano
  input before a pending draw, the max hold, the rebuild; the transformer: provisional length then the TokenUpdate,
  the mirrors, the switch, the start delay; the real exports load (vocab check).
- Real m1e (opt-in `IW_M1E=1`, sample_01's piano for 6 s then 9 s silent): 1.7 / 4.2 ONs a second while silent
  (seeds 1 / 2), as the Python `--reject-piano` runs (1-5 notes/s).

## Not done yet / open

- **Model 2's WAIT logic** (section 4: WAIT d / K at a strong chord, the fire with [piano ONs][CHORD][BASS][SVEL]
  [leftover ON], STACK_OFF releasing the band notes, LATE / EXPIRED): next step. Until then model 2 draws plain
  ON / OFF events (its WAIT / chord tokens masked).
- **Downstream lengths.** The OT applies a TokenUpdate to the notes it made from a reduction note only when they
  match its onset / length / velocity / pitch; orchestrated notes that change pitch or velocity keep the provisional
  length (5 s) until the OT matches by the source note id. That is an OrchestrationTransformer change (the TD
  session's files): asked there.
- Not ported from v1.1: "2 s of predicted silence" (a note's length is only known at its OFF), Dynamic LogitTD and
  the logit adjustments (the TD session owns the TD paths), the WAIT visualizer and the RT_DRAW op log.
- `forceAfterMs` 2000 is a first guess (spec 8 leaves the threshold open).

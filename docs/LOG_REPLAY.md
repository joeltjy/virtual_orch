# Log replay: checking the offline simulator against recorded sessions

`LogReplay` (`app/include/VirtualOrch/sim/LogReplay.h`, `app/source/sim/LogReplay.cpp`) replays a
recorded session log (`Documents/virtual-orch/Logs/sessions/session_*.log`) through the app's own
pipeline, offline, and compares what it logs with what the live run logged, line by line. It is
step 1 of the offline simulator: it proves the machinery around the models is right before the
real models are switched on.

The permanent test is `test/source/LogReplayTest.cpp`, on the four sessions of 22 Oct 2026
afternoon / evening (`test/resources/log_replay/*.log.gz`, 2.1 MB). The code that recorded them is
`main` @ 9563268 (committed 22 Oct 14:54).

## What is taken from the log, what is recomputed

Taken from the log:

- **Your inputs**: every `MIDI_IN` line (both pianos, the pedal, the Launchpad), delivered at its
  logged time through `MidiInputProcess::handleMidiFrom`, plus mode changes made in the UI and every
  Start / Stop (`stream reset` lines and `MIDI_OUT clear` lines).
- **The models' choices** ("logged samples"): each note the reduction (RT) drew, each W2/W3 WAIT
  decision (and its logged P(wait)), and each batch the orchestration model (OT) produced. The live
  run seeded its samplers from `std::random_device` and did not log the seed, so these are replayed
  instead of re-drawn. No model file is loaded: the RT runs on a zero-logit KV backend and the OT's
  model is a stand-in that returns the logged outputs for each input note (same onset, duration and
  pitch class).
- **Real-world timing the app does not control**, recovered from what the log shows:
  - when each model call finished (an RT note or WAIT is delivered at its logged time; the thread is
    busy until then; an OT batch is held until its logged time);
  - when the message-thread timer flushed the input window: the log shows when the RT had a new key
    (a Clear Queue cut on input, or a WAIT `resume`); the replay holds a flush until then, or adds
    one just before it;
  - when an idle RT pass ran: a hybrid switch (or the Clear Queue its cut logged just before), a
    WAIT `resume` / `timeout` / `reorder`, or an input cut marks the start of a live pass; the replay
    runs its pass at that moment instead of on its own 5 ms schedule;
  - the phase of the app clock's 10 ms ticks against the log's ms (estimated once per session), and
    the clock value at each Start.

Recomputed by the app's own code (and compared):

- the input window, note lengths and key releases;
- the RT's context token by token (`RT_IN`): given notes and their grouping, ties, TD and WAIT
  tokens, rebasing, rewinds and rewrites;
- cuts and Clear Queues (`CUT`, `RT`), the hybrid model switches (`HYBRID`), the ahead throttle,
  WAIT resume / timeout / reorder (`WAIT`);
- OT hold-back, Clear Queue pass-through, released note lengths, the order and timing of OT output
  (`OT`);
- playback scheduling and every MIDI out message (`MIDI_OUT`), the Launchpad actions (`LAUNCHPAD`)
  and mode changes (`MODE`).

The replay runs on simulated time (`AppTime`, sub-ms), event to event: a 26-minute session takes
about 10 s (about 0.6% of the session's length on average; idle sessions less).

## Running it

```bash
cd build && ninja
test/VirtualOrchTest_artefacts/Debug/VirtualOrchTest --gtest_filter='LogReplay*:Oct22/*'
```

Each session prints a table (recorded / replayed / matched lines per tag) and its first differences.
Debugging switches (environment variables):

| variable | what |
|---|---|
| `LOGREPLAY_DUMP="from,to"` | both logs side by side for that window (ms), `REC` = recorded, `SIM` = replayed, `IN` = your inputs |
| `LOGREPLAY_FULL=1` | do not cut long lines in the dump |
| `LOGREPLAY_KEEP=n` | list up to n differences per tag instead of 12 |

Example: `LOGREPLAY_DUMP="23000,23200" test/VirtualOrchTest_artefacts/Debug/VirtualOrchTest --gtest_filter='Oct22/*/1'`.

## Pass rules and results (21 Nov 2026)

A tag's match rate is matched lines / max(recorded, replayed); a line matches a recorded line with
the same text within 25 ms. The test requires every forced draw to have had a logged value, no
stall, `MIDI_IN` / `LAUNCHPAD` / `MODE` / `DROP` exact everywhere, every tag exact for 14:51, and for
the other sessions the minimums in the test (just under these numbers):

| | 14:51 (3 min) | 14:25 (26 min) | 18:30 (57 min) | 14:54 (3.6 h) |
|---|---|---|---|---|
| replay time | 2.3 s | 10 s | 8 s | 4.5 s |
| inputs, Launchpad, modes | exact | exact | exact | exact |
| RT notes and clears | exact | 99.9% | 99.9% | 99.7% |
| MIDI out | exact | 98.9% | 99.3% | 99.97% |
| OT | exact | 94.8% | 95.9% | 97.8% |
| WAIT | exact | 99.0% | exact | 94.6% |
| hybrid switches | exact | exact | 92.9% | 91.7% |
| cuts | – | exact | 98.9% | exact |
| RT context (`RT_IN`) | exact | 91.8% | 92.3% | 85.3% |

RT lines are compared without their voice id (see below).

## Accepted differences

- **Thread timing the log does not record.** A live thread's 5 ms wake-ups and the message-thread
  timer land at moments only some lines pin down; where nothing in the log marks the moment, the
  replay can be a clock tick off. Example: 14:25 at 535.9 s, the draw after a pedal release started
  ~2 ms later in the replay, so a held key was encoded 100 cs long instead of 95 (one token; the
  lines after it match again).
- **Given-note grouping at 811.68 s in 14:25 (open).** Live, F#3 and D3 formed a group (onset 729)
  separate from F#4 and D2 (728); by `AnticipatedStream::mergeIntoGroup` F#3 should join the 728
  group, which the replay does, so the stream has one WAIT token fewer and the `from N of M`
  positions of later `RT_IN` lines are off by one until the next rebase. The notes are unaffected.
  It needs F#3 to have reached the RT before F#4 / D2, which the log does not show; similar
  boundary cases probably account for part of the `RT_IN` gap in 18:30 and 14:54.
- **Voice ids.** RT notes' voices come from the vocsep model; the file the live run used
  (`Documents/onnx_export_*/vocsep.onnx` on the recording machine) is not on this Mac, and
  `virtual_orch_models/vocsep.onnx` (19 Nov) gives other ids. The replay switches vocsep off by
  default (`Options::voiceSeparation`); nothing it compares depends on voices.
- **OT notes never asked for**: 77 (14:25), 30 (18:30), 6 (14:54) logged OT notes the replay's OT
  never requested, following from the differences above.

## What changed in the app (behaviour unchanged when nothing is simulated)

- `AppTime.h`: one simulated clock for the pipeline; `Clock`, the Launchpad repeat filter, pedal
  and activity times, the hybrid rules, the ahead throttle, OT balance and playback read it.
- `SamplingRng` in `MusicToken.h`: `SamplingRng::seed(n)` makes every sampler draw from one seeded
  generator (step 2: seed 42).
- One-step loops: `TwoInstrumentAnticipatedReductionTransformer::beginLive` / `liveStep`
  (`threadRun` now calls them), `OrchestrationTransformer::beginSteps` / `stepOnce`,
  `OutputPlayback::beginSteps` / `stepOnce`.
- Replay hooks: `AnticipationSampler::forcedSample` / `forcedWait` / `drawHeld`,
  `ReductionTransformer::holdOutput` / `deliverOutputToken` / `afterOutputDelivered` (the ahead check
  now runs when a note is delivered, the same moment as before in the app), `OrchestrationTransformer::holdOutput`
  / `deliverOutputNote`, `ReductionTransformer::steppedBySimulator`, `SessionLog::setCapture`,
  `AppSession::startGenerationForReplay` / `stopGenerationForReplay`.

## Operation logs (oplogs) and the real models

The app now also writes an **operation log** of every performance
(`Documents/virtual-orch/Logs/oplogs/oplog_<start>.jsonl.gz`, `OpLog.h`): inputs, Launchpad presses,
every change to the RT's context, every queue hand-off, cut and drop with its reason, every MIDI
message out, and each model draw with the chosen token's probability and the model's ten most likely
alternatives (`OpLogDraw.h`); every 5 s each of the RT, OT and playback writes a checkpoint of its
state. The format is documented in `orchestration/performance_streams/oplogs/README.md`. When no
file is open, every hook returns at once.

`LogReplay` can run with the real models (`Options::rtSilentModel`, `rtWaitModel`, `otModel`): every
choice is still forced to the logged one, and the models supply the probabilities, so a recording
made before oplogs existed gets one (`Options::oplog`). `tools/oplog_replay` does that for a list of
session logs (resumable; writes `oplog_<stamp>.jsonl.gz`, the match summary `.replay.txt`, and the
replay's own session-log lines `.replay_session.log.gz` for checking):

```bash
oplog_replay --models ~/Documents/virtual_orch_models --out oplogs session_*.log
oplog_replay --zero --out oplogs session_*.log   # no models: zero logits, quick check
```

Checks (21 Nov 2026): with the oplog on, every replay table above is unchanged; the oplog alone
rebuilds the session log's RT notes and clears, OT notes, MIDI in / clears, modes and Launchpad
actions exactly (`orchestration/tools/performance_streams/oplog_check.py`, the four Oct 22
sessions); with the real models 14:51 is still exact on every tag, no forced choice has
probability 0, and 98–100% of the logged choices are in the model's top 10.

Fixed on the way: on Linux the first ONNX session of a process fell back to the CPU (the ORT
environment has to exist before the CUDA provider is appended: `StreamingKvCache.cpp`,
`IodPretrained.cpp`); `[kv]` lines on stderr now say which device each RT model runs on.

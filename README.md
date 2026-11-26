# Sympathetic Virtual Orchestra

Conduct a **generative** orchestra with your hands. This repository joins two projects:

- **virtual_orch** (Joel Tan; forked from jam_bot, git repo jordanai): a real-time app that takes piano input
  from a MIDI keyboard (or a MIDI file), continues it with a *reduction* model (piano music), orchestrates
  that note by note with an *orchestration* model (which instruments play, in which octaves), and plays the
  result through a virtual MIDI port into Reaper.
- **Sympathetic Orchestra** (Bob Tianqi Wei, Shm Garanganao Almeda, Ethan Tam, Dor Abrahamson and Bjoern
  Hartmann, UC Berkeley, 2024; [`bobtianqiwei/sympathetic-orchestra`](https://github.com/bobtianqiwei/sympathetic-orchestra)):
  a webcam-based virtual orchestra for conducting students. MediaPipe recognises hand gestures in the browser,
  and the gestures shape the playback of pre-recorded audio stems.

Here the gestures from Sympathetic Orchestra's Table 1 (point, palm spread, conducting, fist, relax, cueing)
drive the virtual_orch models instead of audio stems. The music is generated live, so the gestures change what
is generated and how it is played: a part can only play once it is cued, the conducted beat stretches or
compresses the predicted notes, and selecting a part brings it forward.

**Models:** download the ONNX models from
[this Google Drive folder](https://drive.google.com/drive/folders/1RsWv-WVZywiFP8WmT4IFHoXLgh49s1GJ?usp=drive_link)
and put them in `~/Documents/virtual_orch_models/` (see [Setup for Bob](#setup-for-bob)).

Branch `sympathetic_virtual_orchestra` of `joeltjy/virtual_orch`, based on `main` @ `6d90945`.

---

## Overview

```
                 browser (sympathetic/)                                virtual_orch (C++ / JUCE)
 webcam ─► MediaPipe hands + gestures ─► conduct_logic.js ─► POST ─► sym_server.js ─► OSC/UDP ─► OSCController
            (Bob's recognize_hands.js)   (gesture → event)            (Node relay)                    │
                                                                                                       ▼
 MIDI keyboard / MIDI file ─► reduction model ──warp──► orchestration model ──cue mask──► playback ──velocity──► Reaper
                              (piano continuation)      (which instruments)                (Select, Expression)
```

1. **Tracking (browser).** Bob's `recognize_hands.js` runs the MediaPipe gesture recognizer and hand
   landmarker on the webcam. `conduct_logic.js` turns each video frame into events: point at a part, palm
   spread, a beat (the lowest point of the conducting hand), stop and play.
2. **Relay.** A browser can't send UDP, so `sym_server.js` (Node, built-in modules only) serves the page and
   forwards each event to virtual_orch as an OSC message `/sym/<event>`.
3. **Engine.** `OSCController` updates the shared gesture state in `Sympathetic.h`, and the engine reads it
   at three points:
   - **after the reduction model draws a note:** the note is stretched by the warp factor (tempo);
   - **before the orchestration model runs:** it is only offered the instruments of cued parts (cueing);
   - **at note-on:** the velocity is raised or lowered (select, expression).

   Stop and Play go through the same code as the sustain pedal.

The gesture layer stays **off until the first `/sym/` message** arrives, so without the tracker the app behaves
exactly as before. Once it is on, nothing plays until a part is cued.

## Table 1 (from the paper) and what each function does here

| Function (action) | Paper: description | Here: what it does |
|---|---|---|
| **Expression** (palm spread) | Changes the modulation of the music, such as dynamics and expressiveness. | **Global velocity scale.** Wider spread means louder. Spread = thumb-tip to pinky-tip distance over wrist to middle-knuckle distance, mapped to 0–1, then to a scale of 0.5–1.5 (0.5 spread = unchanged). It multiplies every part's velocity before the Select boost/cut. Clamped to MIDI 1–127. Only read while a palm is open (MediaPipe `Open_Palm`). |
| **Select** (finger point) | Selects a specific part. | The point must rest on a part for 400 ms. Pointing at a part that is **already cued** selects it. The selected part's velocity goes **up** and every other part's goes **down** (default +20 / −20). It stays selected until another part is selected or the selection is released (point at empty space for 700 ms). |
| **Play** (relax naturally) | Continues playing. | What **pedal up** does in virtual_orch: generation resumes from now. It fires once no fist has been seen for 300 ms after a Stop. |
| **Stop** (fist) | Stops playing. | What **pedal down** does in virtual_orch: generation pauses, queued notes are dropped, sounding notes end within 0.3 s, and the models forget the cut future. The fist must be held 150 ms. Stop does not un-cue parts. |
| **Tempo** (conducting gestures) | Controls the tempo according to the score and corresponding beat. | Sets the **warp factor** (below). The tracker sends a beat at each ictus, i.e. the lowest point of the conducting hand's wrist after a fall of at least 4% of the frame height. The conducting hand is the right hand when it is free (not pointing or making a fist), otherwise the left. |
| **Cueing** (cursor appears one beat in advance in the corresponding part) | Cues the part that is about to start playing. | There is no score, so there is no advance cursor. **Only cued parts may play.** Every part starts uncued. Pointing at an **uncued** part cues it. Uncued instruments are **masked in the orchestration model's sampling**, so their material goes to cued parts instead of leaving holes. |

### Warp factor (tempo)

- `warp` starts at **1**. The beat period of the conductor's first four beats is the reference (`warp = 1`).
  After that, `warp = smoothed beat period / reference`, clamped to [0.5, 2]. Conducting faster gives
  `warp < 1`, so the music is compressed.
- Let `t` be the current clock time. Each predicted **reduction note**, with onset `o` and duration `d`, becomes

  ```
  onset'    = t + warp · (o − t)
  duration' = warp · d
  ```

- The warped values are the note from then on. They are applied right after the model draws the note,
  before it goes into the model's own context, the orchestration model, the session log / oplog and the
  performance streams. So the reduction model conditions on warped times.
- Onsets at or before `t` keep their time. A warped onset never goes before the previous onset, so a falling
  warp cannot reorder notes. Duration is clamped to [1, 999] cs, and the existing 5 s cap still applies.
- Beats closer than 200 ms are ignored (double detections). A gap over 2 s restarts the count without
  changing the reference. Smoothing gives the newest interval a weight of 0.4.

### Parts: Sympathetic Orchestra grid → virtual_orch instruments

| Grid part | virtual_orch local id | | Grid part | virtual_orch local id |
|---|---|---|---|---|
| Flute | 7 | | Timpani | 16 |
| Oboe | 8 | | Percussion | 17 |
| Clarinet | 9 | | Piano | 15 |
| Bassoon | 10 | | Violin 1, Violin 2 | 0 (one violin part) |
| French Horns | 12 | | Viola | 1 |
| Trumpets | 11 | | Cello | 2 |
| Trombones | 13 | | Bass | 3 |
| Tuba | 14 | | Harp, Conductor | — (no counterpart) |

- Tremolo strings (4), pizzicato strings (5) and string ensemble (6) are articulations, not parts. They are
  allowed whenever at least one solo string is cued. An ensemble note sounds only if the solo string its pitch
  routes to is cued. They are boosted together with a selected string part.
- Guitar (18) and choir (19) have no part in the grid, so they never play while the layer is on.
- A part plays only if it is cued **and** its instrument is switched on in virtual_orch's instrument pads.
- Violin 1 and Violin 2 both cue the one violin. Once it is cued, pointing at either selects it.

### OSC messages

| Address | Args | Function |
|---|---|---|
| `/sym/point` | part name (string) | Cue if uncued, else Select |
| `/sym/release` | — | Clear the selection |
| `/sym/expression` | spread 0–1 (float) | Expression |
| `/sym/beat` | time in ms on the tracker's clock (float) | Tempo (the engine computes the warp) |
| `/sym/stop`, `/sym/play` | — | Stop, Play |
| `/sym/reset` | — | Uncue all, clear the selection, warp = 1, new tempo reference |
| `/sym/warp` | float | Set the warp directly (testing) |
| `/sym/enable` | int 0/1 | Turn the layer off/on (it turns on by itself at the first gesture) |

Every gesture is logged as a `SYM` line in virtual_orch's session log.

## Where it currently falls short

- **Not yet tested end to end with a live camera.** The C++ and JS tests pass, and the conducting page and
  OSC relay were checked in a browser with mouse and keyboard. A full performance (webcam → gestures →
  models → Reaper) has not been run yet, and the gesture thresholds (dwell times, beat depth, palm-spread
  range) are first guesses that need tuning on real conducting.
- **Tempo is applied as notes are generated, not as they are played.** The reduction model runs ahead of the
  clock, so notes already generated keep the old warp, and a tempo change takes effect after that lead time.
  Tempo also only stretches time around "now". It doesn't follow a score or lock onto the beat phase.
- **Warp is only partly visible to some reduction models.** For the T3, Wait, Hybrid and Melody-first models,
  the output and everything downstream are warped, but each model's own streaming-KV context keeps the
  unwarped draw. MusicTransformer, V1, V2 and PianoReduction see the warp fully. Fixing this means rewriting
  the just-sampled event inside the KV stream.
- **There is no score, so cueing has no "one beat in advance" cursor.** The paper's cursor shows which part is
  about to enter. Here the conductor decides, and a part is silent until cued. A model-suggested entrance
  (highlighting the part the orchestration model "wants" next) would be the closest equivalent.
- **Cueing is one-way.** Parts can't be un-cued individually, only all at once with Reset.
- **Expression only changes velocity.** The paper's "modulation / expressiveness" could also map to
  articulation (e.g. tremolo), temporal density or instrument-family mix, which the models already condition
  on. None of that is wired up yet.
- **Play is "not a fist".** There is no explicit "relaxed hands" detector, so any non-fist pose after a Stop
  resumes.
- **Instrument coverage differs.** There is no Harp or Conductor part in virtual_orch, Violin 1 and 2 are one
  instrument, and guitar and choir can't be cued.
- **Manual connection.** The OSC port has to be entered and connected by hand in virtual_orch's Settings
  (the **Controller:** field).
- **Bob's original page** (`sympathetic/public/index.html`) is included unchanged but needs its MP3 stems,
  which are not in this repository.

---

## Setup for Bob

Your code is in **`sympathetic/`**, copied from `bobtianqiwei/sympathetic-orchestra` @ `261cbc4`. The audio
stems, p5 player and bundled `.task` files are left out; the page loads the MediaPipe models from Google's
CDN, as yours does.

| File | Status |
|---|---|
| `public/index.html`, `mySketch.js`, `hands_test.*`, `App.js`, `Original_Sympathetic_Orchestra.pde` | Yours, unchanged |
| `public/recognize_hands.js` | Yours, plus one additive change: it also stores all 21 landmarks per hand (`sharedData.leftLandmarks` / `rightLandmarks`) and when each hand was last seen (`leftSeenMs` / `rightSeenMs`) |
| `public/conduct.html`, `conduct.js` | New: the conducting page. Your grid and cursors; parts are grey until cued, white once cued, green when selected |
| `public/conduct_logic.js` | New: gesture → event rules and the beat (ictus) detector, camera-free so it can be tested |
| `sym_server.js` | New: serves `public/` and relays events to OSC (Node built-ins only, no `npm install`) |
| `test/conduct_logic.test.js` | New: `node --test sympathetic/test` |

Your code has no beat detection or cue logic, so both are new here. Your fist used to mean "everything except
piano drops to 0.01"; here it means Stop.

### 1. Try the tracker on its own (no virtual_orch needed)

Requires Node 18+ and Chrome.

```bash
git clone -b sympathetic_virtual_orchestra git@github.com:joeltjy/virtual_orch.git sympathetic_virtual_orchestra
```

```bash
node sympathetic_virtual_orchestra/sympathetic/sym_server.js
```

Open http://localhost:8000/conduct.html and press **Enable Camera** (it says "Loading Models..." until
MediaPipe has loaded). The status line shows playing/stopped, the beat BPM, expression and how many parts are
cued; the dot at the top right flashes on each detected beat. Without a camera: click a part to point at it,
B or space taps a beat, S stops, P plays, R resets. Events are sent to UDP 9001 whether or not anything is
listening, and the server's terminal prints each one.

### 2. Build virtual_orch (macOS, Apple Silicon)

Needs the Xcode command-line tools, CMake ≥ 3.22 and Ninja (`brew install cmake ninja`).

- **Libraries.** `libs` is a symlink to `../libs`, a folder next to the clone that CMake fills with JUCE,
  googletest and custommatplot on the first configure (that one needs internet). Create it first, in the
  folder that *contains* `sympathetic_virtual_orchestra`:

  ```bash
  mkdir -p libs
  ```

- **ONNX Runtime.** On Apple Silicon, install the arm64 Homebrew package and configure the project:

  ```bash
  brew install onnxruntime
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH="$(brew --prefix onnxruntime)"
  ```

  With ONNX Runtime 1.20 or later, the IodPretrained orchestration model uses the Core ML
  provider with CPU and GPU compute units. Core ML compiles the model during loading. If
  session creation fails, the app retries on CPU. To force CPU on Apple Silicon, launch the
  executable with `VIRTUAL_ORCH_ORT_BACKEND=cpu`. The streaming reduction models use CPU on
  macOS because the tested Core ML configuration increased single-token latency.

- **Build.** Build the JUCE header first (otherwise a fresh build can race on it), then everything.
  On a Mac with 8 GB of RAM keep `-j2`; a full build takes roughly 30–60 minutes.

  ```bash
  ninja -C build -j2 app/VirtualOrch_artefacts/JuceLibraryCode/JuceHeader.h && ninja -C build -j2
  ```

  The app is `build/app/VirtualOrch_artefacts/Debug/virtual-orch.app`. Tests:

  ```bash
  build/test/VirtualOrchTest_artefacts/Debug/VirtualOrchTest --gtest_filter='Sympathetic*'
  ```

- **Models.** Download the models from the
  [Google Drive folder](https://drive.google.com/drive/folders/1RsWv-WVZywiFP8WmT4IFHoXLgh49s1GJ?usp=drive_link)
  into `~/Documents/virtual_orch_models/` (the app and the model tests look there). Tests whose model file is
  missing fail with "File doesn't exist"; that is expected if you don't have that model.

### 3. Sound: Reaper

virtual_orch creates a virtual MIDI port called **"virtual-orch Virtual MIDI Output"**, with one MIDI channel
per instrument. Open `orchestra_output.RPP` in Reaper, enable that port as a MIDI input under
Preferences → MIDI Devices, and check that the tracks' instrument plugins load (replace any you don't have with
your own sampler, one track per channel).

### 4. Run everything together

1. Start Reaper with `orchestra_output.RPP`.
2. Start `virtual-orch.app`. In Settings, enter `9001` in the **Controller:** field (the OSC port) and press
   **Connect**. For MIDI Input, choose your keyboard, or "Scripted input (MIDI file)" if you don't have one.
3. Start the tracker (`node sympathetic/sym_server.js`) and open http://localhost:8000/conduct.html.
4. Start the orchestra in virtual_orch and play or start the MIDI file. Nothing sounds until you **point at a
   part** (hold 0.4 s) to cue it. Point at it again to select it. Conduct with your free hand to set the tempo,
   open your palm for dynamics, make a fist to stop and relax to resume. **Reset** on the page uncues everything.

Ports can be changed: `PORT=8000 OSC_PORT=9001 node sympathetic/sym_server.js`, with the same OSC port in
virtual_orch.

---

## Where it lives in the engine

| What | Where |
|---|---|
| State, part table, warp math, beat tracker, velocity rule | `app/include/VirtualOrch/Sympathetic.h` |
| Warp of each drawn reduction note | `ReductionTransformer::applySympatheticWarp`, called in MusicTransformer, V1, V2, PianoReduction, TwoInstrumentAnticipated and MelodyFirst |
| Cue mask (the model only sees cued instruments) | `OrchestrationTransformer::copyInstrumentLists` |
| Select / Expression velocity, uncued safety net | `OutputPlayback::handleNoteForOutput` (note-on) |
| Stop / Play | `MidiInputProcess::setPedal` (the sustain pedal uses it too) |
| OSC | `OSCController::handleSympathetic`, connected in `MainComponent` |
| Tests | `test/source/SympatheticTest.cpp`; `sympathetic/test/conduct_logic.test.js` |

The rest of virtual_orch is unchanged. Its threads (input, reduction, orchestration, playback) pass notes
through lock-free queues (`CircularFifo`). The gesture state is a set of atomics that the OSC thread writes
and the other threads read.

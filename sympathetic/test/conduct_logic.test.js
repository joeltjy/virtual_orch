// node --test sympathetic/test
const test = require("node:test");
const assert = require("node:assert/strict");
const { palmSpread, BeatDetector, GestureInterpreter } = require("../public/conduct_logic.js");
const { encodeOsc } = require("../sym_server.js");

function hand(gesture, x = 0, y = 0, landmarks = null) {
  return { gesture, x, y, landmarks };
}

function run(frames, partAt = () => null) {
  const events = [];
  const g = new GestureInterpreter(partAt, (...e) => events.push(e));
  for (const f of frames) g.update(f);
  return events;
}

test("fist held -> stop once, relax -> play once", () => {
  const frames = [];
  for (let t = 0; t <= 300; t += 30) frames.push({ nowMs: t, hands: [hand("Closed_Fist")] });
  for (let t = 330; t <= 800; t += 30) frames.push({ nowMs: t, hands: [hand("None")] });
  assert.deepEqual(run(frames), [["stop"], ["play"]]);
});

test("short fist flicker does not stop", () => {
  const frames = [
    { nowMs: 0, hands: [hand("Closed_Fist")] },
    { nowMs: 60, hands: [hand("Closed_Fist")] },
    { nowMs: 90, hands: [hand("None")] },
  ];
  assert.deepEqual(run(frames), []);
});

test("point dwell on a part sends it once; empty space releases", () => {
  const partAt = (x) => (x < 100 ? "Oboe" : null);
  const frames = [];
  for (let t = 0; t <= 600; t += 30) frames.push({ nowMs: t, hands: [hand("Pointing_Up", 50, 50)] });
  for (let t = 630; t <= 1500; t += 30) frames.push({ nowMs: t, hands: [hand("Pointing_Up", 500, 50)] });
  assert.deepEqual(run(frames, partAt), [["point", "Oboe"], ["release"]]);
});

test("passing over a part without dwelling sends nothing", () => {
  const partAt = (x) => (x < 100 ? "Oboe" : null);
  const frames = [
    { nowMs: 0, hands: [hand("Pointing_Up", 50, 50)] },
    { nowMs: 200, hands: [hand("Pointing_Up", 500, 50)] },
    { nowMs: 300, hands: [hand("None")] },
  ];
  assert.deepEqual(run(frames, partAt), []);
});

function landmarksWithSpread(ratio) {
  const lm = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 }));
  lm[0] = { x: 0.5, y: 0.6 };
  lm[9] = { x: 0.5, y: 0.5 }; // hand size 0.1
  lm[4] = { x: 0.5 - ratio * 0.05, y: 0.5 };
  lm[20] = { x: 0.5 + ratio * 0.05, y: 0.5 };
  return lm;
}

test("palm spread maps to 0..1", () => {
  assert.equal(palmSpread(landmarksWithSpread(0.3)), 0);
  assert.equal(palmSpread(landmarksWithSpread(2.5)), 1);
  assert.ok(Math.abs(palmSpread(landmarksWithSpread(1.2)) - 0.5) < 1e-9);
  assert.equal(palmSpread(null), null);
});

test("expression is rate-limited and only sent on change", () => {
  const frames = [];
  for (let t = 0; t <= 1000; t += 20)
    frames.push({ nowMs: t, hands: [hand("Open_Palm", 0, 0, landmarksWithSpread(1.2))] });
  const events = run(frames);
  assert.equal(events.length, 1);
  assert.equal(events[0][0], "expression");
  assert.ok(Math.abs(events[0][1] - 0.5) < 1e-9);
});

function conductingFrames(periodMs, beats, start = 0) {
  // y = lowest (largest) at each ictus: a cosine bounce, sampled at 30 fps
  const frames = [];
  for (let t = start; t <= start + periodMs * beats; t += 33) {
    const phase = ((t - start) % periodMs) / periodMs;
    const y = 0.5 - 0.1 * Math.cos(2 * Math.PI * phase); // top at phase 0, bottom at 0.5
    frames.push({ nowMs: t, hands: [hand("None")], conductY: y });
  }
  return frames;
}

test("beats are detected once per conducting period", () => {
  const events = run(conductingFrames(600, 8)).filter((e) => e[0] === "beat");
  assert.ok(events.length >= 7 && events.length <= 8, `got ${events.length} beats`);
  const times = events.map((e) => e[1]);
  for (let i = 1; i < times.length; i++) {
    const gap = times[i] - times[i - 1];
    assert.ok(Math.abs(gap - 600) <= 70, `gap ${gap}`);
  }
});

test("small jitter is not a beat", () => {
  const d = new BeatDetector();
  let beats = 0;
  for (let t = 0; t < 3000; t += 33) if (d.update(t, 0.5 + 0.005 * Math.sin(t / 50)) != null) beats++;
  assert.equal(beats, 0);
});

test("no beats while stopped", () => {
  const frames = conductingFrames(600, 4).map((f) => ({ ...f, hands: [hand("Closed_Fist")] }));
  const events = run(frames).filter((e) => e[0] === "beat");
  assert.equal(events.length, 0);
});

test("OSC encoding", () => {
  assert.equal(encodeOsc("/sym/point", ["Oboe"]).toString("hex"),
    "2f73796d2f706f696e7400002c7300004f626f6500000000");
  assert.equal(encodeOsc("/sym/stop", []).toString("hex"), "2f73796d2f73746f700000002c000000");
  assert.equal(encodeOsc("/sym/expression", [0.5]).toString("hex"),
    "2f73796d2f65787072657373696f6e002c6600003f000000");
});

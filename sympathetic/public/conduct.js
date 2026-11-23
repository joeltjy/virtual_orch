// Sympathetic Virtual Orchestra: conducting page (p5 global mode).
//
// Bob Tianqi Wei et al.'s Sympathetic Orchestra grid and hand tracking (recognize_hands.js), with
// the music coming from virtual_orch instead of audio stems. Gestures are turned into events by
// conduct_logic.js and POSTed to sym_server.js, which forwards them to virtual_orch as OSC.
//
// Without a camera: click a part = point at it, B / space = beat (tap tempo), S = stop,
// P = play, R = reset.

/* global ConductLogic */

// ---- grid (from Bob's mySketch.js) ---------------------------------------------------------
const n_grid_X = 16, n_grid_Y = 7;
const unitX = 70, unitY = 70;
const smallSquareSize = 50, smallRectRad = 25, bigSquareMargin = 5, bigRectRad = 30;
const smallSquareOffset = (unitX - smallSquareSize) / 2;
// {row, column, rows, columns, grey}
const units = [[2, 5, 1, 2, 255], [2, 7, 1, 2, 255], [2, 9, 1, 2, 255], [2, 11, 1, 2, 255],
               [0, 8, 1, 4, 255], [1, 8, 1, 2, 255], [1, 10, 1, 2, 255], [1, 12, 1, 1, 255],
               [0, 6, 2, 2, 255], [0, 4, 2, 2, 255], [1, 3, 2, 1, 255], [5, 0, 2, 7, 255],
               [3, 1, 2, 6, 255], [3, 7, 2, 4, 255], [5, 9, 2, 7, 255], [3, 11, 2, 4, 255],
               [2, 4, 1, 1, 255], [5, 7, 2, 2, 100]];
const texts = ["Flute", "Oboe", "Clarinet", "Bassoon", "French Horns", "Trumpets", "Trombones",
               "Tuba", "Timpani", "Percussion", "Piano", "Violin 1", "Violin 2", "Viola", "Cello",
               "Bass", "Harp", "Conductor"];

// Parts with no virtual_orch counterpart (see README.md).
const NO_COUNTERPART = new Set(["Harp", "Conductor"]);
// Parts that share one virtual_orch instrument cue together.
const instrumentKey = (part) => (part === "Violin 1" || part === "Violin 2" ? "violin" : part);

let globalX = 50, globalY = 50, gridScale = 1;
let boxes = []; // [x, y, w, h] in screen pixels

function layout() {
  const gridW = n_grid_X * unitX, gridH = n_grid_Y * unitY;
  gridScale = Math.max(0.1, Math.min(1, (windowWidth - 40) / gridW, (windowHeight - 140) / gridH));
  globalX = (windowWidth - gridW * gridScale) / 2;
  globalY = (windowHeight - gridH * gridScale) / 2;
  boxes = units.map((u) => [
    globalX + (u[1] * unitX + smallSquareOffset - bigSquareMargin) * gridScale,
    globalY + (u[0] * unitY + smallSquareOffset - bigSquareMargin) * gridScale,
    (u[3] * unitX - 2 * bigSquareMargin) * gridScale,
    (u[2] * unitY - 2 * bigSquareMargin) * gridScale,
  ]);
}

function partAt(x, y) {
  for (let i = 0; i < boxes.length; i++) {
    const [bx, by, bw, bh] = boxes[i];
    if (x > bx && x < bx + bw && y > by && y < by + bh)
      return NO_COUNTERPART.has(texts[i]) ? null : texts[i];
  }
  return null;
}

// ---- engine link -----------------------------------------------------------------------------
const mirror = { cued: new Set(), selected: null, stopped: false, expression: null };
let lastEvent = "", lastEventMs = 0, linkOk = true;
const beatTimes = [];

function send(name, ...args) {
  fetch("/sym", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ address: "/sym/" + name, args }),
  })
    .then((r) => { linkOk = r.ok; })
    .catch(() => { linkOk = false; });
}

// Same rule as the engine (Sympathetic::State::point): uncued -> cue, cued -> select.
function onEvent(name, ...args) {
  if (name === "point") {
    const part = args[0];
    const key = instrumentKey(part);
    if (!mirror.cued.has(key)) {
      mirror.cued.add(key);
      lastEvent = "cue " + part;
    } else {
      mirror.selected = part;
      lastEvent = "select " + part;
    }
  } else if (name === "release") {
    mirror.selected = null;
    lastEvent = "release";
  } else if (name === "stop") {
    mirror.stopped = true;
    lastEvent = "stop";
  } else if (name === "play") {
    mirror.stopped = false;
    lastEvent = "play";
  } else if (name === "expression") {
    mirror.expression = args[0];
  } else if (name === "beat") {
    beatTimes.push(args[0]);
    while (beatTimes.length > 5) beatTimes.shift();
    lastEvent = "beat";
  } else if (name === "reset") {
    mirror.cued.clear();
    mirror.selected = null;
    beatTimes.length = 0;
    lastEvent = "reset";
  }
  if (name !== "expression") lastEventMs = performance.now();
  send(name, ...args);
}

let interpreter;

// ---- p5 --------------------------------------------------------------------------------------
function setup() {
  createCanvas(windowWidth, windowHeight);
  layout();
  interpreter = new ConductLogic.GestureInterpreter(partAt, onEvent);
  const reset = document.getElementById("resetButton");
  if (reset) reset.addEventListener("click", () => onEvent("reset"));
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  layout();
}

function handFrame(side, nowMs) {
  const d = window.sharedData;
  if (!d) return null;
  const seen = d[side + "SeenMs"];
  if (seen < 0 || nowMs - seen > ConductLogic.DEFAULTS.staleMs) return null;
  return {
    gesture: d[side + "GestureData"].gestureName,
    x: (1 - d[side + "HandCursorX"]) * windowWidth, // mirrored, as in Bob's sketch
    y: d[side + "HandCursorY"] * windowHeight,
    landmarks: d[side + "Landmarks"],
  };
}

function conductingY(hands) {
  // The conducting hand: right if it is free (not pointing / fist), else left.
  for (const h of [hands[1], hands[0]]) {
    if (h && h.landmarks && h.gesture !== "Pointing_Up" && h.gesture !== "Closed_Fist")
      return h.landmarks[0].y; // wrist
  }
  return null;
}

function draw() {
  const nowMs = performance.now();
  background(200);

  const hands = [handFrame("left", nowMs), handFrame("right", nowMs)];
  interpreter.update({ nowMs, hands, conductY: conductingY(hands) });

  drawParts();
  drawHands(hands);
  drawStatus(nowMs);
}

function drawParts() {
  noStroke();
  textAlign(CENTER, CENTER);
  for (let i = 0; i < units.length; i++) {
    const [x, y, w, h] = boxes[i];
    const part = texts[i];
    const none = NO_COUNTERPART.has(part);
    const cued = mirror.cued.has(instrumentKey(part));
    if (none) fill(170);
    else if (mirror.selected === part) fill(169, 213, 175); // selected: green (Bob's colour)
    else if (cued) fill(255);
    else fill(140); // not cued yet: silent
    rect(x, y, w, h, Math.min(bigRectRad * gridScale, w / 2, h / 2));

    fill(mirror.selected === part || cued ? 30 : 235);
    textSize(14 * Math.max(0.7, gridScale));
    text(part + (none ? " (n/a)" : cued ? "" : " (cue)"), x + w / 2, y + h / 2);
  }
}

function drawHands(hands) {
  noStroke();
  for (const h of hands) {
    if (!h) continue;
    if (h.gesture === "Pointing_Up") {
      fill("green");
      triangle(h.x, h.y, h.x + 25, h.y + 26, h.x + 1, h.y + 35);
    } else if (h.gesture === "Open_Palm") {
      fill(255, 165, 0);
      rect(h.x - 25, h.y - 7, 45, 15);
    } else if (h.gesture === "Closed_Fist") {
      fill("red");
      circle(h.x, h.y, 12);
    } else {
      fill(60, 60, 200);
      circle(h.x, h.y, 22);
    }
  }
}

function drawStatus(nowMs) {
  let bpm = "–";
  if (beatTimes.length >= 2) {
    // Same window as the engine's beat tracker (200 ms .. 2 s between beats).
    const gaps = beatTimes.slice(1).map((t, i) => t - beatTimes[i]).filter((g) => g >= 200 && g <= 2000);
    if (gaps.length) bpm = Math.round(60000 / (gaps.reduce((a, b) => a + b, 0) / gaps.length)).toString();
  }
  const parts = [
    mirror.stopped ? "STOPPED (fist)" : "playing",
    "beat " + bpm + " BPM",
    "expression " + (mirror.expression == null ? "–" : mirror.expression.toFixed(2)),
    "cued " + mirror.cued.size,
    linkOk ? "" : "engine relay unreachable",
  ].filter(Boolean);
  fill(40);
  textAlign(LEFT, TOP);
  textSize(13);
  text(parts.join("   ·   "), 12, 34);
  if (lastEvent && nowMs - lastEventMs < 1200) {
    textSize(18);
    text(lastEvent, 12, 54);
  }
  // beat flash
  if (beatTimes.length && nowMs - beatTimes[beatTimes.length - 1] < 120) {
    fill(255, 165, 0);
    circle(width - 30, 40, 24);
  }
}

// ---- no-camera controls ------------------------------------------------------------------
function mousePressed() {
  const part = partAt(mouseX, mouseY);
  if (part) onEvent("point", part);
}

function keyPressed() {
  const k = key.toLowerCase();
  if (k === "b" || key === " ") onEvent("beat", performance.now());
  else if (k === "s") onEvent("stop");
  else if (k === "p") onEvent("play");
  else if (k === "r") onEvent("reset");
}

// Sympathetic Virtual Orchestra: gesture -> event logic (Table 1 of the paper).
// Pure and camera-free so it can be tested in Node (test/conduct_logic.test.js).
//
// Events (all sent on to virtual_orch as OSC /sym/<name>):
//   point <part>      finger point held on a part: the engine cues it if uncued, else selects it
//   release           finger point held on empty space: clear the selection
//   expression <0..1> palm spread while a palm is open
//   beat <ms>         ictus (lowest point) of the conducting hand
//   stop / play       fist / relax

(function (root) {
  "use strict";

  const DEFAULTS = {
    staleMs: 250,          // a hand not seen for this long counts as absent
    fistHoldMs: 150,       // fist must be held this long to Stop
    relaxHoldMs: 300,      // no fist for this long after a Stop -> Play
    pointDwellMs: 400,     // finger point must rest on a part this long
    releaseDwellMs: 700,   // ... or on empty space this long to release the selection
    expressionMinIntervalMs: 100,
    expressionMinChange: 0.02,
    // palm spread = |thumb tip - pinky tip| / |wrist - middle MCP|, mapped linearly to 0..1
    spreadLow: 0.6,
    spreadHigh: 1.8,
    // beats
    beatMinIntervalMs: 200,
    beatMinDrop: 0.04,     // the hand must fall at least this far (normalised height) per beat
    beatSmoothing: 0.5,
  };

  /** Palm spread 0..1 from 21 MediaPipe hand landmarks, or null. */
  function palmSpread(landmarks, opts = DEFAULTS) {
    if (!landmarks || landmarks.length < 21) return null;
    const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const size = d(landmarks[0], landmarks[9]);
    if (size <= 1e-6) return null;
    const ratio = d(landmarks[4], landmarks[20]) / size;
    const t = (ratio - opts.spreadLow) / (opts.spreadHigh - opts.spreadLow);
    return Math.min(1, Math.max(0, t));
  }

  /**
   * Ictus detector on the conducting hand's height (y grows downwards, normalised 0..1).
   * A beat is the turn from falling to rising after a fall of at least beatMinDrop.
   */
  class BeatDetector {
    constructor(opts = DEFAULTS) {
      this.opts = opts;
      this.reset();
    }
    reset() {
      this.y = null;
      this.falling = false;
      this.top = null;      // highest point (smallest y) since the last beat
      this.bottom = null;   // lowest point so far in this fall
      this.bottomMs = 0;
      this.lastBeatMs = -Infinity;
    }
    /** Feed one sample; returns the beat time (ms) when a beat is detected, else null. */
    update(nowMs, rawY) {
      const o = this.opts;
      if (rawY == null) {
        this.reset();
        return null;
      }
      const prev = this.y;
      this.y = prev == null ? rawY : o.beatSmoothing * rawY + (1 - o.beatSmoothing) * prev;
      if (prev == null) {
        this.top = this.y;
        return null;
      }
      const dy = this.y - prev;
      let beat = null;
      if (dy > 0) {
        // falling
        if (!this.falling) {
          this.falling = true;
          this.bottom = this.y;
        }
        if (this.bottom == null || this.y >= this.bottom) {
          this.bottom = this.y;
          this.bottomMs = nowMs;
        }
      } else if (dy < 0) {
        if (this.falling) {
          // turned upwards: was the fall deep enough?
          const drop = this.bottom - (this.top ?? this.bottom);
          if (drop >= o.beatMinDrop && this.bottomMs - this.lastBeatMs >= o.beatMinIntervalMs) {
            beat = this.bottomMs;
            this.lastBeatMs = this.bottomMs;
          }
          this.falling = false;
          this.top = this.y;
        } else if (this.top == null || this.y < this.top) {
          this.top = this.y;
        }
      }
      return beat;
    }
  }

  /**
   * Frame-by-frame interpreter. Each frame:
   *   { nowMs, hands: [{ gesture, x, y, landmarks } | null, ...] (screen coords for x/y),
   *     conductY: normalised y of the conducting hand or null }
   * `partAt(x, y)` returns a part name or null. Calls `emit(name, ...args)`.
   */
  class GestureInterpreter {
    constructor(partAt, emit, opts = {}) {
      this.opts = Object.assign({}, DEFAULTS, opts);
      this.partAt = partAt;
      this.emit = emit;
      this.beats = new BeatDetector(this.opts);
      this.stopped = false;
      this.fistSince = null;
      this.noFistSince = null;
      this.pointTarget = undefined; // undefined = not pointing; null = pointing at nothing
      this.pointSince = 0;
      this.pointFired = false;
      this.lastExpression = null;
      this.lastExpressionMs = -Infinity;
    }

    update(frame) {
      const { nowMs } = frame;
      const hands = (frame.hands || []).filter(Boolean);
      const has = (g) => hands.some((h) => h.gesture === g);

      // Stop (fist) / Play (relax)
      if (has("Closed_Fist")) {
        this.noFistSince = null;
        if (this.fistSince == null) this.fistSince = nowMs;
        if (!this.stopped && nowMs - this.fistSince >= this.opts.fistHoldMs) {
          this.stopped = true;
          this.emit("stop");
        }
      } else {
        this.fistSince = null;
        if (this.noFistSince == null) this.noFistSince = nowMs;
        if (this.stopped && nowMs - this.noFistSince >= this.opts.relaxHoldMs) {
          this.stopped = false;
          this.emit("play");
        }
      }

      // Select / Cue (finger point with dwell)
      const pointer = hands.find((h) => h.gesture === "Pointing_Up");
      if (pointer) {
        const target = this.partAt(pointer.x, pointer.y);
        if (target !== this.pointTarget) {
          this.pointTarget = target;
          this.pointSince = nowMs;
          this.pointFired = false;
        }
        const dwell = target == null ? this.opts.releaseDwellMs : this.opts.pointDwellMs;
        if (!this.pointFired && nowMs - this.pointSince >= dwell) {
          this.pointFired = true;
          if (target == null) this.emit("release");
          else this.emit("point", target);
        }
      } else {
        this.pointTarget = undefined;
      }

      // Expression (palm spread while a palm is open)
      const palm = hands.find((h) => h.gesture === "Open_Palm");
      if (palm) {
        const spread = palmSpread(palm.landmarks, this.opts);
        if (
          spread != null &&
          nowMs - this.lastExpressionMs >= this.opts.expressionMinIntervalMs &&
          (this.lastExpression == null ||
            Math.abs(spread - this.lastExpression) >= this.opts.expressionMinChange)
        ) {
          this.lastExpression = spread;
          this.lastExpressionMs = nowMs;
          this.emit("expression", spread);
        }
      }

      // Tempo (beats of the conducting hand)
      const beat = this.beats.update(nowMs, this.stopped ? null : frame.conductY);
      if (beat != null) this.emit("beat", beat);
    }
  }

  const api = { DEFAULTS, palmSpread, BeatDetector, GestureInterpreter };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ConductLogic = api;
})(typeof window !== "undefined" ? window : globalThis);

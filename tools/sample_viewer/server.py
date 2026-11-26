#!/usr/bin/env python3
"""Sample viewer backend (2026-11-26): on every Generate, play a sample into the real app pipeline (RT + OT) with
the chosen version and settings (tools/sample_render, headless, real time) and return its tracks and draws.

    python3 tools/sample_viewer/server.py [--port 8810] [--render build/tools/SampleRender_artefacts/Release/sample_render]
                                          [--models ~/Documents/virtual_orch_models] [--out ~/sample_viewer_runs]

Stdlib only. Open http://localhost:8810/ (or the Desktop copy of sample_viewer.html, pointed at this server; through
ssh: ssh -L 8810:localhost:8810 HOST). One generation runs at a time: it takes as long as the music (the app runs in
real time), plus loading the models.

Samples: samples/maestro_sample_00..09 (10 MAESTRO test clips, 15 s: orchestration/samples/maestro_fixtures_10),
samples/maestro_val_00..19 (20 MAESTRO validation clips, 15 s: make_samples.py) and samples/test_simultaneous
(Joel's 22.5 s waltz pattern). Each run is kept in --out/<stamp>_<system>_<sample>/
(oplog, settings, stdout, result.json).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import http.server
import json
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
APP = os.path.join(REPO, "app")
SAMPLES = os.path.join(HERE, "samples")

SYSTEMS = {  # name -> preset file (app/presets/version<name>.json); kind of model 2
    "v1.1": "informative WAIT (model 1 rt_continuation_latest / model 2 rt_simultaneous_latest)",
    "v1.2": "chord tokens, round 3b (model 1 rt_continuation_latest / model 2 v1.2_simultaneous)",
    "v1.2_r4": "chord tokens, round 4 (model 1 rt_continuation_latest / model 2 v1.2_r4_simultaneous)",
    "v1.3": "event models (model 1 v1.3_continuation / model 2 v1.3_simultaneous)",
}
INSTRUMENTS = {0: "Violin", 1: "Viola", 2: "Cello", 3: "Contrabass", 4: "Tremolo Strings", 5: "Pizzicato Strings",
               6: "String Ensemble", 7: "Flute", 8: "Oboe", 9: "Clarinet", 10: "Bassoon", 11: "Trumpet",
               12: "French Horn", 13: "Trombone", 14: "Tuba", 15: "Piano (orch)", 16: "Timpani", 17: "Snare",
               18: "Guitar", 19: "Choir"}
GM = {0: 40, 1: 41, 2: 42, 3: 43, 4: 44, 5: 45, 6: 48, 7: 73, 8: 68, 9: 71, 10: 70, 11: 56, 12: 60, 13: 57,
      14: 58, 15: 0, 16: 47, 17: 116, 18: 25, 19: 52}
NOTE_NAMES = "C C# D Eb E F F# G Ab A Bb B".split()


def pname(p):
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


# ---------------------------------------------------------------- MIDI input (a small reader: notes and CC64)

def read_midi(path):
    """[(start_s, end_s, pitch, vel)] and [(t_s, value)] for CC64, every track merged, tempo map applied."""
    data = open(path, "rb").read()
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    hlen = struct.unpack(">I", data[4:8])[0]
    _, ntrk, div = struct.unpack(">HHH", data[8:14])
    pos = 8 + hlen
    events, tempos = [], [(0, 500000)]
    for _ in range(ntrk):
        if data[pos:pos + 4] != b"MTrk":
            break
        ln = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        p, end, tick, status = pos + 8, pos + 8 + ln, 0, 0
        while p < end:
            v = 0
            while True:
                b = data[p]; p += 1
                v = (v << 7) | (b & 0x7F)
                if b < 0x80:
                    break
            tick += v
            b = data[p]
            if b == 0xFF:
                typ, p = data[p + 1], p + 2
                ml = 0
                while True:
                    c = data[p]; p += 1
                    ml = (ml << 7) | (c & 0x7F)
                    if c < 0x80:
                        break
                if typ == 0x51:
                    tempos.append((tick, int.from_bytes(data[p:p + 3], "big")))
                p += ml
                continue
            if b in (0xF0, 0xF7):
                p += 1
                ml = 0
                while True:
                    c = data[p]; p += 1
                    ml = (ml << 7) | (c & 0x7F)
                    if c < 0x80:
                        break
                p += ml
                continue
            if b & 0x80:
                status, p = b, p + 1
            kind = status & 0xF0
            n = 1 if kind in (0xC0, 0xD0) else 2
            a = data[p:p + n]; p += n
            if kind in (0x80, 0x90):
                events.append((tick, "on" if kind == 0x90 and a[1] > 0 else "off", a[0], a[1]))
            elif kind == 0xB0 and a[0] == 64:
                events.append((tick, "cc", 64, a[1]))
        pos = end
    tempos.sort()

    def sec(t):
        s, last_t, last_tempo = 0.0, 0, 500000
        for tt, tempo in tempos:
            if tt > t:
                break
            s += (tt - last_t) * last_tempo / 1e6 / div
            last_t, last_tempo = tt, tempo
        return s + (t - last_t) * last_tempo / 1e6 / div

    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
    notes, pedal, open_ = [], [], {}
    for tick, kind, a, b in events:
        t = sec(tick)
        if kind == "on":
            open_.setdefault(a, []).append((t, b))
        elif kind == "off" and open_.get(a):
            s, v = open_[a].pop(0)
            notes.append((s, t, a, v))
        elif kind == "cc":
            pedal.append((t, b))
    for a, lst in open_.items():
        for s, v in lst:
            notes.append((s, s + 0.5, a, v))
    notes.sort()
    return notes, pedal


def samples():
    out = []
    for d in sorted(os.listdir(SAMPLES)):
        mid = os.path.join(SAMPLES, d, "input.mid")
        if not os.path.exists(mid):
            continue
        info = {}
        pj = os.path.join(SAMPLES, d, "piece.json")
        if os.path.exists(pj):
            info = json.load(open(pj))
        notes, _ = read_midi(mid)
        label = d
        if info.get("composer"):
            label += f"  ({info['composer'].split()[-1]}: {info['title'][:40]} @{info.get('clip_start_s', 0):.0f}s)"
        elif info.get("midi_filename"):
            label += "  (MAESTRO test " + os.path.basename(info["midi_filename"])[:30] + f" @{info.get('clip_start_s', 0):.0f}s)"
        out.append(dict(id=d, label=label, notes=len(notes), seconds=round(max((n[1] for n in notes), default=0), 2)))
    return out


def systems():
    out = []
    for name, what in SYSTEMS.items():
        f = os.path.join(APP, "presets", f"version{name.lstrip('v')}.json")
        if not os.path.exists(f):
            continue
        preset = json.load(open(f))
        cfg_path = os.path.join(APP, preset.get("model", "") + ".json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        out.append(dict(name=name, what=what, logitAdjustments=preset.get("logitAdjustments", {}),
                        outputTemperatures=preset.get("outputTemperatures", [0.5, 0.5, 0.5]),
                        orchestrationMode=preset.get("orchestrationMode", "jam"),
                        startDelayMs=cfg.get("startDelayMs", 0 if name == "v1.1" else 3000),
                        onnx=cfg.get("onnx"), silentOnnx=cfg.get("silentOnnx")))
    return out


# ---------------------------------------------------------------- oplog -> tracks and draws

def read_oplog(path):
    rows = []
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except (EOFError, OSError):
        pass
    return rows


def token_label(tok, field, base, clock0, model):
    """A readable label for an RT token (absolute ids; the dense layout + each model 2's extra tokens)."""
    if field == "onset" and tok < 10000:
        cs = base + tok
        return f"t {(cs - clock0) / 100:.2f}s" if clock0 is not None else f"{cs} cs"
    if 10000 <= tok < 11000:
        return f"dur {(tok - 10000) / 100:.2f}s"
    if 11000 <= tok < 11640:
        ins, p = divmod(tok - 11000, 128)
        return ("piano " if ins == 0 else "ON ") + pname(p)
    if 11640 <= tok < 11768:
        return f"vel {tok - 11640}"
    if model == "m1e":
        if 11768 <= tok < 12024:
            ins, p = divmod(tok - 11768, 128)
            return ("piano OFF " if ins == 0 else "OFF ") + pname(p)
        return str(tok)
    if field in ("onset", "count", "k", "late", "chord") or tok >= 11768:
        if model == "m2":  # v1.1 informative WAIT vocab
            if 11768 <= tok < 11868:
                return f"WAIT {(tok - 11768 + 1) * 3} cs"
            if 11868 <= tok < 11878:
                return f"COUNT {tok - 11868 + 1}"
            if tok == 11878:
                return "EXPIRED"
        if 11768 <= tok < 11868:
            return f"WAIT {(tok - 11768 + 1) * 3} cs"
        if 11868 <= tok < 11876:
            return f"K {(tok - 11868) // 2 + 1}{',+1' if (tok - 11868) % 2 else ''}"
        if tok == 11876:
            return "EXPIRED"
        if 11877 <= tok < 11910:
            return f"LATE {(tok - 11877 + 1) * 3} cs"
        if tok == 11910:
            return "DUR?"
        if 11911 <= tok < 14071:
            ci = tok - 11911
            return f"CHORD type {ci // 120} root {NOTE_NAMES[(ci // 10) % 12]} n {ci % 10 + 1}"
        if model == "m2e" and 14071 <= tok < 14328:
            ins, p = divmod(tok - 14071, 128)
            return ("piano OFF " if ins == 0 else "OFF ") + pname(p)
    return str(tok)


def ot_label(field, tok):
    if field == "groups":
        return f"groups {tok + 1:04b}"
    return f"{field} {tok:b}"


def parse(rows, piano, prompt_s=None):
    """The viewer's result: tracks (piano, RT, one per OT instrument), notes with ids, every draw."""
    t0 = next((r["t"] for r in rows if r.get("op") == "SAMPLE_T0"), rows[0]["t"] if rows else 0.0)
    # clock (cs) at t0: from the records around it (clock = floor(ms / 10) - k while running)
    clock_pairs = [(r["t"], r["clock"]) for r in rows if r.get("clock", -1) >= 0]
    clock0 = None
    if clock_pairs:
        t, c = min(clock_pairs, key=lambda x: abs(x[0] - t0))
        clock0 = c - (t - t0) / 10.0

    def cs_to_s(cs):
        return (cs - clock0) / 100.0 if clock0 is not None else cs / 100.0

    rt_draws, ot_draws, ot_from, rt_notes, waits = {}, {}, {}, [], []
    for r in rows:
        op = r.get("op")
        if op == "RT_DRAW":
            model = r.get("model", "")
            fields = []
            for f in r.get("fields", []):
                fields.append(dict(field=f["field"], chosen=f["chosen"], p=f.get("p"), p_model=f.get("p_model"),
                                   label=token_label(f["chosen"], f["field"], r.get("base", 0), clock0, model),
                                   forced=f.get("forced", False),
                                   top=[[tk, p, pm, token_label(tk, f["field"], r.get("base", 0), clock0, model)]
                                        for tk, p, pm in f.get("top", [])]))
            rt_draws[r["r"]] = dict(model=model, t=round((r["t"] - t0) / 1000, 3), source=r.get("source", ""),
                                    context_len=r.get("context_len"), fields=fields)
        elif op == "OT_DRAW":
            steps = [dict(field=f["field"], chosen=f["chosen"], p=f.get("p"), p_model=f.get("p_model"),
                          label=ot_label(f["field"], f["chosen"]),
                          top=[[tk, p, pm, ot_label(f["field"], tk)] for tk, p, pm in f.get("top", [])])
                     for f in r.get("steps", [])]
            ot_draws[r["for"]] = dict(t=round((r["t"] - t0) / 1000, 3), instruments=r.get("instruments"),
                                      combo=r.get("combo"), steps=steps)
        elif op == "OT_NOTE":
            ot_from[r["o"]] = r.get("from")
        elif op == "RT_NOTE":
            rt_notes.append(dict(id=r["r"], start=round(cs_to_s(r["onset"]), 3),
                                 end=round(cs_to_s(r["onset"] + max(1, r.get("dur", 10))), 3),
                                 pitch=r["pitch"], vel=r.get("vel", 64), model=r.get("model", "")))
        elif op == "RT_WAIT":
            waits.append(dict(t=round((r["t"] - t0) / 1000, 3), decision=r.get("decision"), text=r.get("text", "")))
    # RT_NOTE "model" is empty for some reductions: take the draw's
    for n in rt_notes:
        if not n["model"] and n["id"] in rt_draws:
            n["model"] = rt_draws[n["id"]]["model"]
        n["drawn"] = n["id"] in rt_draws
    # OT: what was played (MIDI_OUT on / off), per instrument
    ot, sounding = {}, {}
    for r in rows:
        if r.get("op") != "MIDI_OUT":
            continue
        t = (r["t"] - t0) / 1000.0
        k = r.get("kind")
        if k == "on" and "instrument" in r:
            sounding.setdefault((r["instrument"], r["pitch"]), []).append((t, r.get("vel", 64), r.get("o")))
        elif k == "off" and "instrument" in r and sounding.get((r["instrument"], r["pitch"])):
            s, v, o = sounding[(r["instrument"], r["pitch"])].pop(0)
            ot.setdefault(r["instrument"], []).append((s, t, r["pitch"], v, o))
        elif k == "clear":
            for (ins, p), lst in sounding.items():
                for s, v, o in lst:
                    ot.setdefault(ins, []).append((s, t, p, v, o))
            sounding = {}
    end = max([r["t"] for r in rows] or [t0]) / 1000.0 - t0 / 1000.0
    for (ins, p), lst in sounding.items():
        for s, v, o in lst:
            ot.setdefault(ins, []).append((s, end, p, v, o))
    tracks = [dict(key="piano", name="Piano (input)", kind="piano", program=0,
                   notes=[dict(start=round(s, 3), end=round(e, 3), pitch=p, vel=v) for s, e, p, v in piano])]
    tracks.append(dict(key="rt", name="RT reduction", kind="rt", program=0, notes=rt_notes))
    for ins in sorted(ot):
        notes = [dict(id=o, start=round(s, 3), end=round(e, 3), pitch=p, vel=v, src=ot_from.get(o))
                 for s, e, p, v, o in sorted(ot[ins])]
        tracks.append(dict(key=f"ot{ins}", name=INSTRUMENTS.get(ins, f"instrument {ins}"), kind="ot", instrument=ins,
                           program=GM.get(ins, 0), notes=notes))
    return dict(tracks=tracks, rt_draws=rt_draws, ot_draws=ot_draws, waits=waits,
                duration=round(max([end] + [n["end"] for t in tracks for n in t["notes"]]), 2), prompt_s=prompt_s)


# ---------------------------------------------------------------- jobs

class Jobs:
    def __init__(self, a):
        self.a, self.lock, self.jobs, self.busy = a, threading.Lock(), {}, threading.Lock()

    def start(self, req):
        sample = req.get("sample", "")
        if not os.path.exists(os.path.join(SAMPLES, sample, "input.mid")):
            raise ValueError("unknown sample " + sample)
        if req.get("system") not in SYSTEMS:
            raise ValueError("unknown system")
        jid = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{req['system']}_{req.get('mode', 'simultaneous')[:4]}_{sample}"
        job = dict(id=jid, state="queued", log=[], req=req)
        with self.lock:
            self.jobs[jid] = job
        threading.Thread(target=self.run, args=(job,), daemon=True).start()
        return jid

    def run(self, job):
        with self.busy:  # one at a time: the app runs in real time
            job["state"] = "running"
            req, a = job["req"], self.a
            out = os.path.join(a.out, job["id"])
            os.makedirs(out, exist_ok=True)
            settings = req.get("settings", {})
            json.dump(req, open(os.path.join(out, "request.json"), "w"), indent=1)
            json.dump(settings, open(os.path.join(out, "settings.json"), "w"), indent=1)
            mid = os.path.join(SAMPLES, req["sample"], "input.mid")
            mode = req.get("mode", "simultaneous")
            prompt_s = float(req.get("prompt_s", 5.0))
            cmd = [a.render, "--app", APP, "--models", a.models, "--system", req["system"], "--mode", mode,
                   "--input", mid, "--oplog", os.path.join(out, "oplog.jsonl.gz"),
                   "--settings", os.path.join(out, "settings.json"), "--prompt-s", str(prompt_s),
                   "--seconds", str(float(req.get("seconds", 15.0))), "--tail-s", str(float(req.get("tail_s", 3.0)))]
            job["log"].append(" ".join(cmd))
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                with open(os.path.join(out, "stdout.txt"), "w") as log:
                    for line in p.stdout:
                        log.write(line)
                        job["log"].append(line.rstrip())
                p.wait()
                rows = read_oplog(os.path.join(out, "oplog.jsonl.gz"))
                if p.returncode != 0 or not rows:
                    raise RuntimeError(f"sample_render exited {p.returncode}")
                notes, _ = read_midi(mid)
                if mode == "continuation":
                    notes = [(s, min(e, prompt_s), pi, v) for s, e, pi, v in notes if s < prompt_s]
                res = parse(rows, notes, prompt_s if mode == "continuation" else None)
                res.update(id=job["id"], request=req)
                json.dump(res, open(os.path.join(out, "result.json"), "w"))
                job["result"], job["state"] = res, "done"
            except Exception as e:  # noqa: BLE001 - report anything to the page
                job["log"].append(f"error: {e}")
                job["state"] = "failed"

    def status(self, jid, with_result):
        with self.lock:
            j = self.jobs.get(jid)
        if j is None:
            return None
        out = dict(id=j["id"], state=j["state"], log=j["log"][-12:])
        if with_result and j["state"] == "done":
            out["result"] = j["result"]
        return out

    def history(self):
        runs = []
        if os.path.isdir(self.a.out):
            for d in sorted(os.listdir(self.a.out), reverse=True):
                if os.path.exists(os.path.join(self.a.out, d, "result.json")):
                    runs.append(d)
        return runs[:200]


def handler(jobs):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")  # the Desktop copy is opened from file://
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(b)

        def do_OPTIONS(self):
            self.send(204, b"")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path in ("/", "/index.html", "/sample_viewer.html"):
                return self.send(200, open(os.path.join(HERE, "sample_viewer.html"), "rb").read(), "text/html")
            if u.path == "/api/config":
                return self.send(200, dict(samples=samples(), systems=systems(), instruments=INSTRUMENTS,
                                           gm=GM, render=jobs.a.render, models=jobs.a.models))
            if u.path == "/api/job":
                s = jobs.status(q.get("id", [""])[0], q.get("result", ["0"])[0] == "1")
                return self.send(200 if s else 404, s or dict(error="no such job"))
            if u.path == "/api/history":
                return self.send(200, dict(runs=jobs.history()))
            if u.path == "/api/run":
                d = os.path.basename(q.get("id", [""])[0])
                f = os.path.join(jobs.a.out, d, "result.json")
                return self.send(200, open(f, "rb").read()) if os.path.exists(f) else self.send(404, dict(error="no run"))
            return self.send(404, dict(error="not found"))

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/generate":
                    return self.send(200, dict(id=jobs.start(req)))
            except Exception as e:  # noqa: BLE001
                return self.send(400, dict(error=str(e)))
            return self.send(404, dict(error="not found"))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--render", default=os.path.join(REPO, "build/tools/SampleRender_artefacts/Release/sample_render"))
    ap.add_argument("--models", default=os.path.expanduser("~/Documents/virtual_orch_models"))
    ap.add_argument("--out", default=os.path.expanduser("~/sample_viewer_runs"))
    ap.add_argument("--parse", help="parse an existing run dir (oplog.jsonl.gz + request.json) and print a summary")
    a = ap.parse_args()
    if a.parse:
        req = json.load(open(os.path.join(a.parse, "request.json")))
        notes, _ = read_midi(os.path.join(SAMPLES, req["sample"], "input.mid"))
        res = parse(read_oplog(os.path.join(a.parse, "oplog.jsonl.gz")), notes)
        for t in res["tracks"]:
            print(f"{t['name']:24s} {len(t['notes'])} notes")
        print(len(res["rt_draws"]), "RT draws,", len(res["ot_draws"]), "OT draws")
        return
    if not os.path.exists(a.render):
        print(f"warning: no sample_render at {a.render} (build it: cmake --build build --target SampleRender)")
    os.makedirs(a.out, exist_ok=True)
    srv = http.server.ThreadingHTTPServer((a.host, a.port), handler(Jobs(a)))
    print(f"sample viewer: http://localhost:{a.port}/  (runs in {a.out})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stagnant viewer backend (2026-11-26): look at samples that are already on disk. No model, no GPU, no app run: it
only reads files. The page (stagnant_viewer.html) shows the same per-track scrolling piano rolls as the sample viewer,
plays them, and clicking a note shows the top 10 of every draw that made it (p / p_model), when the sample stored them.

    python3 tools/stagnant_viewer/server.py [--port 8811] [--samples DIR ...] [--runs DIR ...] [--oplogs DIR ...]
                                            [--midi NAME=DIR ...]

Sources (each a dropdown entry on the page):
  samples  folders holding generated.mid (the overnight samplers: orchestration/overnight/samples/<set>/<fixture>/).
           With draws.json.gz (sample_v4 / sample_m2c write it; overnight/trace_to_draws.py makes it from an m2c
           trace.json.gz) every orchestra note has its draws; without it, only the notes (from generated.mid).
  runs     the sample viewer's Generate runs (<out>/<run>/result.json).
  oplogs   the app's oplogs (logs/<day>/oplogs/*.jsonl.gz), one entry per run (RUN_START .. RUN_STOP): piano = MIDI in,
           RT notes with their RT_DRAW top 10s, OT notes (what was played) with their OT_DRAW top 10s; mode changes.
  midi     any folder of .mid files (one lane per MIDI track, no draws).
Stdlib only; the oplog parsing is the sample viewer's own (tools/sample_viewer/server.py).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import http.server
import json
import os
import re
import struct
import sys
import threading
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("sample_viewer_server", os.path.join(os.path.dirname(HERE), "sample_viewer", "server.py"))
SV = importlib.util.module_from_spec(_spec)  # tools/sample_viewer/server.py: parse, INSTRUMENTS, GM
_spec.loader.exec_module(SV)

HOME = os.path.expanduser("~")


# ---------------------------------------------------------------- MIDI, per track

def read_midi_tracks(path):
    """[(track name, program, [(start_s, end_s, pitch, vel)])], tempo map applied (from the first track)."""
    data = open(path, "rb").read()
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    hlen = struct.unpack(">I", data[4:8])[0]
    _, ntrk, div = struct.unpack(">HHH", data[8:14])
    pos, tracks, tempos = 8 + hlen, [], [(0, 500000)]
    for _ in range(ntrk):
        if data[pos:pos + 4] != b"MTrk":
            break
        ln = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        p, end, tick, status = pos + 8, pos + 8 + ln, 0, 0
        name, program, ev = "", None, []

        def vlq():
            nonlocal p
            v = 0
            while True:
                b = data[p]
                p += 1
                v = (v << 7) | (b & 0x7F)
                if b < 0x80:
                    return v
        while p < end:
            tick += vlq()
            b = data[p]
            if b == 0xFF:
                typ = data[p + 1]
                p += 2
                ml = vlq()
                if typ == 0x51:
                    tempos.append((tick, int.from_bytes(data[p:p + 3], "big")))
                elif typ == 0x03 and not name:
                    name = data[p:p + ml].decode("latin-1", "replace")
                p += ml
                continue
            if b in (0xF0, 0xF7):
                p += 1
                p += vlq()
                continue
            if b & 0x80:
                status, p = b, p + 1
            kind = status & 0xF0
            n = 1 if kind in (0xC0, 0xD0) else 2
            a = data[p:p + n]
            p += n
            if kind in (0x80, 0x90):
                ev.append((tick, 0 if kind == 0x80 or a[1] == 0 else 1, a[0], a[1], status & 0x0F))
            elif kind == 0xC0 and program is None:
                program = a[0]
        tracks.append((name, program, ev))
        pos = end
    tempos.sort()

    def sec(t):
        s, lt, lv = 0.0, 0, 500000
        for tt, tempo in tempos:
            if tt > t:
                break
            s += (tt - lt) * lv / 1e6 / div
            lt, lv = tt, tempo
        return s + (t - lt) * lv / 1e6 / div

    out = []
    for name, program, ev in tracks:
        ev.sort(key=lambda e: (e[0], e[1]))
        open_, notes = {}, []
        for tick, on, pitch, vel, ch in ev:
            if on:
                open_.setdefault((ch, pitch), []).append((sec(tick), vel))
            elif open_.get((ch, pitch)):
                s, v = open_[(ch, pitch)].pop(0)
                notes.append((s, sec(tick), pitch, v))
        for (ch, pitch), lst in open_.items():
            for s, v in lst:
                notes.append((s, s + 0.5, pitch, v))
        if notes:
            out.append((name, program or 0, sorted(notes)))
    return out


def midi_result(path):
    tracks = []
    for i, (name, program, notes) in enumerate(read_midi_tracks(path)):
        piano = "piano" in name.lower() and "orch" not in name.lower()
        tracks.append(dict(key=f"t{i}", name=name or f"track {i}", kind="piano" if piano else "rt", program=program,
                           notes=[dict(id=f"t{i}n{j}", start=round(s, 3), end=round(e, 3), pitch=p, vel=v)
                                  for j, (s, e, p, v) in enumerate(notes)]))
    dur = max([n["end"] for t in tracks for n in t["notes"]] or [0])
    return dict(tracks=tracks, rt_draws={}, ot_draws={}, waits=[], duration=round(dur, 2))


# ---------------------------------------------------------------- oplogs, per run

_index_lock, _index, _parsed = threading.Lock(), {}, {}
RUN_RE = re.compile(r'"op": ?"(RUN_START|RUN_STOP)"')


def oplog_runs(path):
    """[(run name, first line, last line, rt)] by a cheap line scan (cached by mtime)."""
    st = os.stat(path)
    key = (path, st.st_mtime, st.st_size)
    with _index_lock:
        if key in _index:
            return _index[key]
    runs, cur = [], None
    try:
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                m = RUN_RE.search(line[:200])
                if not m:
                    continue
                r = json.loads(line)
                if m.group(1) == "RUN_START":
                    cur = [r.get("run", f"run{len(runs)}"), i, None, r.get("rt", "")]
                elif cur is not None:
                    cur[2] = i
                    runs.append(tuple(cur))
                    cur = None
    except (EOFError, OSError, json.JSONDecodeError):
        pass  # a log still being written / cut short: keep what was read
    if cur is not None:
        runs.append((cur[0], cur[1], None, cur[3]))
    with _index_lock:
        _index[key] = runs
    return runs


def oplog_result(path, run):
    key = (path, run, os.path.getmtime(path))
    if key in _parsed:
        return _parsed[key]
    runs = {r[0]: r for r in oplog_runs(path)}
    name, a, b, rt = runs[run]
    rows = []
    try:
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                if i < a:
                    continue
                if b is not None and i > b:
                    break
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except (EOFError, OSError):
        pass
    t0 = rows[0]["t"] if rows else 0.0
    # the piano: MIDI in from the piano device (note on / off bytes)
    piano, open_ = [], {}
    for r in rows:
        if r.get("op") != "MIDI_IN" or "piano" not in r.get("roles", []):
            continue
        bs = [int(x, 16) for x in r.get("bytes", "").split()]
        if len(bs) < 3 or bs[0] & 0xF0 not in (0x80, 0x90):
            continue
        t = (r["t"] - t0) / 1000.0
        if bs[0] & 0xF0 == 0x90 and bs[2] > 0:
            open_.setdefault(bs[1], []).append((t, bs[2]))
        elif open_.get(bs[1]):
            s, v = open_[bs[1]].pop(0)
            piano.append((s, t, bs[1], v))
    end = (rows[-1]["t"] - t0) / 1000.0 if rows else 0.0
    for p, lst in open_.items():
        for s, v in lst:
            piano.append((s, end, p, v))
    rows_t0 = [dict(op="SAMPLE_T0", t=t0)] + rows
    res = SV.parse(rows_t0, sorted(piano))
    events = [dict(t=w["t"], text=f"WAIT {w.get('decision') or ''} {w.get('text') or ''}".strip()) for w in res["waits"]]
    for r in rows:
        if r.get("op") == "MODE" and r.get("from") != r.get("to"):
            events.append(dict(t=round((r["t"] - t0) / 1000, 3), text=f"mode {r.get('from')} → {r.get('to')}", kind="mode"))
        elif r.get("op") == "PAD_ACTION":
            events.append(dict(t=round((r["t"] - t0) / 1000, 3), text="pad: " + str(r.get("action", r.get("desc", ""))),
                               kind="pad"))
    res["waits"] = sorted(events, key=lambda e: e["t"])
    res["meta"] = dict(oplog=path, run=name, rt=rt, rows=len(rows))
    if len(_parsed) > 6:
        _parsed.clear()
    _parsed[key] = res
    return res


# ---------------------------------------------------------------- sources

class Sources:
    def __init__(self, a):
        self.src = {}
        for d in a.samples:
            self.src["samples: " + d.replace(HOME, "~")] = ("samples", d)
        for d in a.runs:
            self.src["sample viewer runs: " + d.replace(HOME, "~")] = ("runs", d)
        for d in a.oplogs:
            self.src["app oplogs: " + d.replace(HOME, "~")] = ("oplogs", d)
        for spec in a.midi:
            name, _, d = spec.partition("=")
            self.src["MIDI: " + name] = ("midi", d)
        self.src = {k: v for k, v in self.src.items() if os.path.isdir(v[1])}

    def listing(self):
        return [dict(name=k, kind=v[0], root=v[1]) for k, v in self.src.items()]

    def items(self, name):
        kind, root = self.src[name]
        out = []
        if kind == "samples":
            for dp, dirs, files in os.walk(root):
                dirs.sort()
                if "generated.mid" in files:
                    rel = os.path.relpath(dp, root)
                    out.append(dict(id=rel, group=rel.split(os.sep)[0], label=rel,
                                    draws="draws.json.gz" in files))
        elif kind == "runs":
            for d in sorted(os.listdir(root), reverse=True):
                if os.path.exists(os.path.join(root, d, "result.json")):
                    out.append(dict(id=d, group=d[:8], label=d, draws=True))
        elif kind == "oplogs":
            for f in sorted(glob.glob(os.path.join(root, "*", "oplogs", "*.jsonl.gz")), reverse=True):
                rel = os.path.relpath(f, root)
                for run, a, b, rt in oplog_runs(f):
                    out.append(dict(id=rel + "#" + run, group=rel.split(os.sep)[0],
                                    label=f"{os.path.basename(f)[6:-9]} {run} ({rt.replace('Reduction Transformer ', '')})"
                                          + ("" if b is not None else " (no RUN_STOP)"), draws=True))
        elif kind == "midi":
            for dp, dirs, files in os.walk(root):
                dirs.sort()
                for f in sorted(files):
                    if f.lower().endswith((".mid", ".midi")):
                        rel = os.path.relpath(os.path.join(dp, f), root)
                        out.append(dict(id=rel, group=os.path.dirname(rel) or ".", label=rel, draws=False))
        return out

    def path(self, root, rel):
        p = os.path.realpath(os.path.join(root, rel))
        if not p.startswith(os.path.realpath(root) + os.sep) and p != os.path.realpath(root):
            raise ValueError("bad path")
        return p

    def load(self, name, item):
        kind, root = self.src[name]
        if kind == "samples":
            d = self.path(root, item)
            info = {}
            sj = os.path.join(os.path.dirname(d), "summary.json")
            if os.path.exists(sj):
                s = json.load(open(sj))
                info = dict({k: v for k, v in s.items() if k != "samples"},
                            sample=next((x for x in s.get("samples", []) if x.get("fixture") == os.path.basename(d)), None))
            if os.path.exists(os.path.join(d, "draws.json.gz")):
                res = json.load(gzip.open(os.path.join(d, "draws.json.gz"), "rt"))
            else:
                res = midi_result(os.path.join(d, "generated.mid"))
                res["note"] = "no draws.json.gz here: notes only (sampled before the samplers logged draws; " \
                              "an m2c trace.json.gz can be converted with overnight/trace_to_draws.py)"
            res["info"] = info
            return res
        if kind == "runs":
            return json.load(open(os.path.join(self.path(root, item), "result.json")))
        if kind == "oplogs":
            rel, _, run = item.partition("#")
            return oplog_result(self.path(root, rel), run)
        if kind == "midi":
            return midi_result(self.path(root, item))
        raise ValueError(kind)


def handler(S):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html", "/stagnant_viewer.html"):
                    return self.send(200, open(os.path.join(HERE, "stagnant_viewer.html"), "rb").read(), "text/html")
                if u.path == "/api/sources":
                    return self.send(200, dict(sources=S.listing(), instruments=SV.INSTRUMENTS, gm=SV.GM))
                if u.path == "/api/items":
                    return self.send(200, dict(items=S.items(q["source"])))
                if u.path == "/api/load":
                    return self.send(200, S.load(q["source"], q["id"]))
            except Exception as e:  # noqa: BLE001
                return self.send(500, dict(error=f"{type(e).__name__}: {e}"))
            return self.send(404, dict(error="not found"))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--samples", nargs="*", default=[f"{HOME}/orchestration/overnight/samples",
                                                     f"{HOME}/orchestration/samples/overnight_112226/v2"])
    ap.add_argument("--runs", nargs="*", default=[f"{HOME}/sample_viewer_runs"])
    ap.add_argument("--oplogs", nargs="*", default=sorted(glob.glob(f"{HOME}/virtual_orch*/logs")))
    ap.add_argument("--midi", nargs="*", default=[f"expressifier_listen={HOME}/orchestration/tools/expressifier_listen/midi",
                                                  f"reductions_v4={HOME}/Desktop/2026-2027/orchestration_v2/reductions_v4/midi"])
    a = ap.parse_args()
    S = Sources(a)
    srv = http.server.ThreadingHTTPServer((a.host, a.port), handler(S))
    print(f"stagnant viewer: http://localhost:{a.port}/  ({len(S.src)} sources)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

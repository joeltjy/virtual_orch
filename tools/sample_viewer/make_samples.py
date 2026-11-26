#!/usr/bin/env python3
"""Cut MAESTRO v3 clips for the sample viewer (2026-11-26): samples/maestro_val_NN/input.mid + piece.json.

    python3 tools/sample_viewer/make_samples.py ~/Downloads/maestro-v3.0.0 [--split validation] [--n 20] [--seed 20261126]

Each clip: 15 s from a random point of a random piece of the split (seeded), notes that start in the window, cut at
its end, re-based to 0, with the sustain pedal (CC64). Clips with fewer than 40 notes are skipped. Needs mido.
The 10 test clips (samples/maestro_sample_00..09) are orchestration/samples/maestro_fixtures_10, cut the same way.
"""
import argparse
import csv
import json
import os
import random

import mido

HERE = os.path.dirname(os.path.abspath(__file__))


def notes_of(path):
    mid = mido.MidiFile(path)
    t, out, pedal, open_ = 0.0, [], [], {}
    for m in mid:  # merged tracks, time in seconds
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            open_.setdefault(m.note, []).append((t, m.velocity))
        elif m.type in ("note_on", "note_off") and open_.get(m.note):
            s, v = open_[m.note].pop(0)
            out.append((s, t, m.note, v))
        elif m.type == "control_change" and m.control == 64:
            pedal.append((t, m.value))
    return out, pedal, t


def write(path, notes, pedal, length):
    ev = [(s, "on", p, v) for s, e, p, v in notes] + [(min(e, length), "off", p, 0) for s, e, p, v in notes]
    ev += [(t, "cc", 64, v) for t, v in pedal]
    ev.sort(key=lambda x: (x[0], {"off": 0, "cc": 1, "on": 2}[x[1]]))
    mf = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    last = 0
    for t, k, a, b in ev:
        tick = int(round(t * 960))  # 120 bpm: 960 ticks per second
        dt, last = tick - last, tick
        if k == "on":
            tr.append(mido.Message("note_on", note=a, velocity=b, time=dt))
        elif k == "off":
            tr.append(mido.Message("note_off", note=a, velocity=0, time=dt))
        else:
            tr.append(mido.Message("control_change", control=64, value=b, time=dt))
    mf.tracks.append(tr)
    mf.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maestro")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--prefix", default="maestro_val")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(a.maestro, "maestro-v3.0.0.csv"))) if r["split"] == a.split]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    made = 0
    for r in rows:
        if made >= a.n:
            break
        notes, pedal, total = notes_of(os.path.join(a.maestro, r["midi_filename"]))
        if total < a.seconds + 10:
            continue
        start = rng.uniform(5.0, total - a.seconds - 5.0)
        end = start + a.seconds
        clip = [(s - start, min(e, end) - start, p, v) for s, e, p, v in notes if start <= s < end]
        if len(clip) < 40:
            continue
        down = [v for t, v in pedal if t < start]
        ped = ([(0.0, down[-1])] if down and down[-1] >= 64 else []) + [(t - start, v) for t, v in pedal if start <= t < end]
        if ped and ped[-1][1] >= 64:
            ped.append((a.seconds, 0))
        d = os.path.join(HERE, "samples", f"{a.prefix}_{made:02d}")
        os.makedirs(d, exist_ok=True)
        write(os.path.join(d, "input.mid"), clip, ped, a.seconds)
        json.dump(dict(sample=os.path.basename(d), midi_filename=r["midi_filename"], composer=r["canonical_composer"],
                       title=r["canonical_title"], clip_start_s=round(start, 3), total_s=a.seconds, source="maestro_" + a.split,
                       seed=a.seed, n_notes=len(clip)), open(os.path.join(d, "piece.json"), "w"), indent=1)
        print(f"{os.path.basename(d)}: {r['canonical_composer']} - {r['canonical_title'][:50]} @{start:.0f}s, {len(clip)} notes")
        made += 1


if __name__ == "__main__":
    main()

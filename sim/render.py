"""Render many scenes in parallel.

    python -m sim.render --out data/sim_v1 --start 0 --count 1000 --workers 8 [--heldout] [--prefix pc]

Each scene writes <split>/<name>.flac (24-bit, full scale = 120 dB SPL (acoustics.FULL_SCALE_DB)), .npz (frame labels), .json (meta).
Scenes already on disk are skipped, so a run can be restarted.
"""
import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from . import acoustics as ac
from .scene import FS, sample_and_render

LABEL_VERSION = "snr_lines2"   # bump when audio or labels change; --relabel re-renders older scenes


def split_of(seed, heldout):
    if heldout:
        return "test_heldout"
    return {0: "val", 1: "test_sim"}.get(seed % 20, "train")


def render_one(job):
    seed, heldout, out, prefix, relabel = job
    split = split_of(seed, heldout)
    name = f"{prefix}{'h' if heldout else ''}{seed:07d}"
    d = Path(out) / split
    js = d / f"{name}.json"
    if js.exists() and (not relabel or f'"labels": "{LABEL_VERSION}"' in js.read_text()):
        return name, "skip", 0.0
    t0 = time.time()
    try:
        audio, labels, meta = sample_and_render(seed + (10_000_000 if heldout else 0), heldout)
        x = audio[0] / ac.FULL_SCALE_PA
        meta["clipped_frac"] = float(np.mean(np.abs(x) >= 1.0))
        meta["split"] = split
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{name}.flac", np.clip(x, -1, 1 - 2 ** -23), FS, subtype="PCM_24")
        np.savez_compressed(d / f"{name}.npz", **labels)
        meta["render_s"] = time.time() - t0
        meta["labels"] = LABEL_VERSION
        (d / f"{name}.json").write_text(json.dumps(meta, default=float))
        return name, "ok", meta["render_s"]
    except Exception:
        return name, "error: " + traceback.format_exc(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--relabel", action="store_true", help="re-render scenes rendered with an older LABEL_VERSION")
    ap.add_argument("--only-split", choices=["train", "val", "test_sim", "test_heldout"],
                    help="render only the seeds in this range that fall in one split (e.g. test_sim for testing)")
    a = ap.parse_args()
    seeds = [s for s in range(a.start, a.start + a.count) if a.only_split in (None, split_of(s, a.heldout))]
    jobs = [(s, a.heldout, a.out, a.prefix, a.relabel) for s in seeds]
    t0, done = time.time(), 0
    log = Path(a.out) / f"render_{a.prefix or 'x'}{'h' if a.heldout else ''}_{a.start}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, maxtasksperchild=20) as pool, open(log, "a") as f:
        for name, status, secs in pool.imap_unordered(render_one, jobs):
            done += 1
            f.write(f"{name}\t{status}\t{secs:.1f}\n")
            f.flush()
            if done % 10 == 0 or status.startswith("error"):
                print(f"{done}/{len(jobs)} {name} {status[:200]} {secs:.1f}s elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"DONE {done}/{len(jobs)} elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

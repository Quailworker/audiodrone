"""Render multi-cluster localisation scenes (tasks/phase1_localisation.md step 1).

    python -m loc.render --config tetra_pole_100 --out data/loc_v1 --count 300 --workers 8

Scenes are the ordinary sim scenes (sim/scene.py) heard by every mic of a field layout (sim/layouts.py),
seeds from SEED0 up, keeping only seeds whose scene has a drone. The same seeds are used for every
configuration, so configurations are compared on identical flights.

Per scene <out>/<config>/<name>:
  .c<k>.flac   cluster k's 4 mics, 8 kHz, 24-bit, full scale = 120 dB SPL. Direction finding uses < 2 kHz,
               so the 32 kHz render is decimated by 4 to keep storage small (~1/4 of 32 kHz).
  .truth.npz   per drone: control-rate track, velocity, per-rotor shaft rate; per cluster and analysis
               frame (FRAME_HZ): true direction at emission time, range, height, horizontal distance;
               per cluster: tone-line SNRs of the drone categories at its reference mic (audibility) and
               the drone's SNR inside the direction-finding bands (DF_BANDS).
  .json        scene metadata (as sim/render.py) plus config, cluster centres and mic positions.
"""
import argparse
import json
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from sim import acoustics as ac
from sim.layouts import CONFIGS, field
from sim.scene import FS, kinematics, sample_and_render

SEED0 = 20_000_000                 # cannot collide with sim_v1 (0-4,000, 1,000,000-1,002,000, +10M held out)
FS_OUT = 8000
FRAME_HZ = 10                      # analysis frames for direction finding and tracking (EKF rate)
DRONE_CATS = ("drone_electric", "drone_engine")
DF_BANDS = {"df860": (100, 860), "df2k": (100, 2000)}          # direction-finding bands (loc/doa.py)


def has_drone(seed):
    """Mirrors the first draw of sample_and_render."""
    return np.random.default_rng(seed).uniform() < 0.75


def drone_seeds(count):
    seeds, s = [], SEED0
    while len(seeds) < count:
        if has_drone(s):
            seeds.append(s)
        s += 1
    return seeds


def frame_times(duration):
    return np.arange(0, duration, 1 / FRAME_HZ) + 0.5 / FRAME_HZ


def truth(drone_tracks, centres, labels, duration):
    """Ground truth per drone and per cluster at the analysis frame times."""
    tf = frame_times(duration)
    out = {"frame_t": tf, "centres": centres}
    for i, (q, t, p, shaft) in enumerate(drone_tracks):
        v, _ = kinematics(t, p)
        out.update({f"d{i}/t": t, f"d{i}/p": p.astype(np.float32), f"d{i}/v": v.astype(np.float32),
                    f"d{i}/shaft_hz": shaft.astype(np.float32), f"d{i}/blades": q["blades"]})
        for k, c in enumerate(centres):
            tau, r = ac.emission_time(tf, t, p, c)
            pe = ac.interp_path(t, p, tau)
            ve = np.stack([np.interp(tau, t, v[:, j]) for j in range(3)], 1)
            u = (pe - c) / r[:, None]
            out[f"d{i}/c{k}/u"] = u.astype(np.float32)                       # true direction (emission time)
            out[f"d{i}/c{k}/r"] = r.astype(np.float32)
            out[f"d{i}/c{k}/rho"] = np.hypot(*(pe[:, :2] - c[:2]).T).astype(np.float32)
            out[f"d{i}/c{k}/rdot"] = np.einsum("ij,ij->i", u, ve).astype(np.float32)
            out[f"d{i}/c{k}/tau"] = tau.astype(np.float32)
        tau0, _ = ac.emission_time(tf, t, p, centres[0])
        out[f"d{i}/pe"] = ac.interp_path(t, p, tau0).astype(np.float32)          # position at c0's emission time
    # audibility per cluster at the reference mic: tone-line SNRs (the CNN label rule, 50 Hz-8 kHz) and the
    # drone's SNR inside each direction-finding band (from the 48 coarse label bands at 25 frames/s)
    edges = np.geomspace(40, 14000, 49)
    centre_hz = np.sqrt(edges[:-1] * edges[1:])
    for k, lab in enumerate(labels):
        lt = lab["frame_t"]
        idx = np.clip(np.searchsorted(lt, tf), 0, len(lt) - 1)
        power = {key[6:]: 10 ** (lab[key].astype(np.float64) / 10) for key in lab if key.startswith("bands/")}
        total = sum(power.values())
        j = np.clip((tf * 25).astype(int), 0, total.shape[1] - 1)
        for cat in DRONE_CATS:
            key = f"snr_lines/{cat}"
            if key in lab:
                lines = lab[key].astype(np.float32)
                out[f"c{k}/{cat}/lines01"] = lines[idx, :2]
                for name, (lo, hi) in DF_BANDS.items():
                    b = (centre_hz >= lo) & (centre_hz <= hi)
                    pd = power[cat][b].sum(0)
                    out[f"c{k}/{cat}/snr_{name}"] = (10 * np.log10(pd / np.maximum(total[b].sum(0) - pd, 1e-30) + 1e-30))[j].astype(np.float32)
    return out


def render_one(job):
    seed, config, out = job
    layout, kind, mount = CONFIGS[config]
    name = f"L{seed:08d}"
    d = Path(out) / config
    js = d / f"{name}.json"
    if js.exists():
        return name, "skip", 0.0
    t0 = time.time()
    try:
        centres, mics = field(layout, kind, mount)
        ref = list(range(0, len(mics), 4))
        audio, labels, meta, tracks = sample_and_render(seed, False, [tuple(m) for m in mics], label_mics=ref)
        x = signal.resample_poly(audio / ac.FULL_SCALE_PA, 1, FS // FS_OUT, axis=1).astype(np.float32)
        meta["clipped_frac"] = float(np.mean(np.abs(audio[ref] / ac.FULL_SCALE_PA) >= 1.0))
        d.mkdir(parents=True, exist_ok=True)
        for k in range(len(centres)):
            sf.write(d / f"{name}.c{k}.flac", np.clip(x[4 * k: 4 * k + 4].T, -1, 1 - 2 ** -23), FS_OUT, subtype="PCM_24")
        np.savez_compressed(d / f"{name}.truth.npz", **truth(tracks, centres, labels, meta["duration"]))
        meta.update(config=config, layout=layout, cluster=kind, mount=mount, centres=centres.tolist(),
                    fs=FS_OUT, render_s=time.time() - t0)
        js.write_text(json.dumps(meta, default=float))
        return name, "ok", meta["render_s"]
    except Exception:
        return name, "error: " + traceback.format_exc(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--out", default="data/loc_v1")
    ap.add_argument("--count", type=int, default=300, help="number of drone scenes")
    ap.add_argument("--skip", type=int, default=0, help="start at this drone-scene index (pilot / chunks)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = drone_seeds(a.skip + a.count)[a.skip:]
    jobs = [(s, a.config, a.out) for s in seeds]
    t0, done = time.time(), 0
    log = Path(a.out) / a.config / f"render_{a.skip}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, maxtasksperchild=10) as pool, open(log, "a") as f:
        for name, status, secs in pool.imap_unordered(render_one, jobs):
            done += 1
            f.write(f"{name}\t{status}\t{secs:.1f}\n")
            f.flush()
            if done % 5 == 0 or status.startswith("error"):
                print(f"{done}/{len(jobs)} {name} {status[:300]} {secs:.1f}s elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"DONE {done}/{len(jobs)} elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

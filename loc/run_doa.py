"""Direction finding on every rendered localisation scene (task step 2).

    python -m loc.run_doa --config tetra_pole_100 [--variants]

Writes <data>/<config>/<scene>.doa.npz with, per method and cluster k, "<method>/c<k>/u" (frames, 3) and
"<method>/c<k>/coh" at the truth file's frame times. Main methods: srp, ls (pairwise PHAT-beta GCC) and
bf (reflection-aware beamformer). --variants adds the experiments: srp with beta 1 and 0, srp and bf over
100-2000 Hz (aliasing), and srp and bf on audio decimated to 4 kHz. --comb adds "bf_comb": the
beamformer restricted to bins near the drone's own shaft-rate harmonics as received at each cluster
(true per-rotor shaft rates, Doppler-shifted by the true radial speed). It is an oracle: the upper bound of
steering direction finding with the comb tracker's lines (design 5.2, 6) instead of the whole band.
"bf_comb2k" is the same over 100-2000 Hz (the drone's lines above the tetrahedron's 860 Hz alias limit).
--trkcomb adds "bf_trk" / "bf_trk2k": the same beamformer guided by the frequency tracker's own lines
(loc/freqtrack.py, both tracked tones at the reference mic, no truth): the real, non-oracle version.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from sim.layouts import MOUNT_Z, cluster_offsets

from .doa import DOA, BeamformerDOA


def estimators(kind, mount, fs, variants):
    off = cluster_offsets(kind)
    h = MOUNT_Z[mount]
    est = {"main": (fs, DOA(off, fs), BeamformerDOA(off, fs, h))}
    if variants:
        est["beta1"] = (fs, DOA(off, fs, phat_beta=1.0), None)
        est["beta0"] = (fs, DOA(off, fs, phat_beta=0.0), None)
        est["band2k"] = (fs, DOA(off, fs, band=(100, 2000)), BeamformerDOA(off, fs, h, band=(100, 2000)))
        est["fs4k"] = (4000, DOA(off, 4000), BeamformerDOA(off, 4000, h))
    return est


def run_scene(stem, est, n_clusters):
    tf = np.load(f"{stem}.truth.npz")["frame_t"]
    out = {}
    for k in range(n_clusters):
        x, fs = sf.read(f"{stem}.c{k}.flac", dtype="float32")
        x = x.T
        for name, (fs_e, pair, bf) in est.items():
            xe = signal.resample_poly(x, 1, fs // fs_e, axis=1).astype(np.float32) if fs_e != fs else x
            methods = ("srp", "ls") if name == "main" else ("srp",)
            res = pair.run(xe, tf, methods=methods)
            for m in methods:
                tag = m if name == "main" else f"{m}_{name}"
                out[f"{tag}/c{k}/u"] = res[m]["u"].astype(np.float32)
                out[f"{tag}/c{k}/coh"] = res[m]["coh"].astype(np.float16)
            if bf is not None:
                r = bf.run(xe, tf)
                tag = "bf" if name == "main" else f"bf_{name}"
                out[f"{tag}/c{k}/u"] = r["u"].astype(np.float32)
                out[f"{tag}/c{k}/coh"] = r["coh"].astype(np.float16)
    return out


def comb_weights(truth, k, f_bins, tol_hz=4.0):
    """(frames, bins) 0/1 mask: bins within tol of a multiple of any drone rotor's received shaft rate."""
    tf = truth["frame_t"]
    w = np.zeros((len(tf), len(f_bins)), np.float32)
    i = 0
    while f"d{i}/t" in truth:
        tau = truth[f"d{i}/c{k}/tau"]
        dop = 1 / (1 + truth[f"d{i}/c{k}/rdot"] / 343.0)
        for shaft in truth[f"d{i}/shaft_hz"]:
            f0 = np.interp(tau, truth[f"d{i}/t"], shaft) * dop                     # (frames,)
            n = np.round(f_bins[None] / f0[:, None])
            w = np.maximum(w, ((n >= 1) & (np.abs(f_bins[None] - n * f0[:, None]) <= tol_hz)).astype(np.float32))
        i += 1
    return w


COMB_VARIANTS = {"bf_comb": (100.0, 860.0), "bf_comb2k": (100.0, 2000.0)}
TRK_VARIANTS = {"bf_trk": (100.0, 860.0), "bf_trk2k": (100.0, 2000.0)}


def tracker_weights(fr, k, f_bins, tol_hz=4.0):
    """(frames, bins) 0/1 mask: bins within tol of a harmonic of either tone the tracker found at cluster k."""
    f0 = np.where(np.isfinite(fr[f"c{k}/f0_fine"]), fr[f"c{k}/f0_fine"], fr[f"c{k}/f0"]).astype(float)   # (F, tones)
    w = np.zeros((len(f0), len(f_bins)), np.float32)
    for j in range(f0.shape[1]):
        f = f0[:, j]
        on = np.isfinite(f)
        n = np.round(f_bins[None] / np.where(on, f, 1.0)[:, None])
        hit = on[:, None] & (n >= 1) & (np.abs(f_bins[None] - n * np.where(on, f, 1.0)[:, None]) <= tol_hz)
        w = np.maximum(w, hit.astype(np.float32))
    return w


def add_trk(stem, kind, mount, fs, n_clusters):
    """Add the tracker-guided beamformer to an existing .doa.npz (runs the tracker if needed)."""
    from loc.eval_freq import track_scene
    fr = track_scene(stem)
    tf = np.load(f"{stem}.truth.npz")["frame_t"]
    out = dict(np.load(f"{stem}.doa.npz"))
    for tag, band in TRK_VARIANTS.items():
        if f"{tag}/c0/u" in out:
            continue
        bf = BeamformerDOA(cluster_offsets(kind), fs, MOUNT_Z[mount], band=band)
        for k in range(n_clusters):
            x, _ = sf.read(f"{stem}.c{k}.flac", dtype="float32")
            r = bf.run(x.T, tf, chunk=4, weights=tracker_weights(fr, k, bf.base.f))
            out[f"{tag}/c{k}/u"] = r["u"].astype(np.float32)
            out[f"{tag}/c{k}/coh"] = r["coh"].astype(np.float16)
        del bf
        torch.cuda.empty_cache()
    np.savez_compressed(f"{stem}.doa.npz", **out)


def add_comb(stem, kind, mount, fs, n_clusters):
    """Add the oracle comb-guided beamformer (860 Hz and 2 kHz bands) to an existing .doa.npz."""
    truth = dict(np.load(f"{stem}.truth.npz"))
    out = dict(np.load(f"{stem}.doa.npz"))
    for tag, band in COMB_VARIANTS.items():
        if f"{tag}/c0/u" in out:
            continue
        bf = BeamformerDOA(cluster_offsets(kind), fs, MOUNT_Z[mount], band=band)
        for k in range(n_clusters):
            x, _ = sf.read(f"{stem}.c{k}.flac", dtype="float32")
            r = bf.run(x.T, truth["frame_t"], chunk=4, weights=comb_weights(truth, k, bf.base.f))
            out[f"{tag}/c{k}/u"] = r["u"].astype(np.float32)
            out[f"{tag}/c{k}/coh"] = r["coh"].astype(np.float16)
        del bf
        torch.cuda.empty_cache()
    np.savez_compressed(f"{stem}.doa.npz", **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/loc_v1")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes (sorted by name)")
    ap.add_argument("--wait-for", type=int, default=0, help="keep polling until this many scenes are processed")
    ap.add_argument("--comb", action="store_true", help="add the oracle comb-guided beamformer to existing results")
    ap.add_argument("--trkcomb", action="store_true", help="add the frequency-tracker-guided beamformer")
    a = ap.parse_args()
    d = Path(a.data) / a.config
    if a.comb or a.trkcomb:
        variants, add = (COMB_VARIANTS, add_comb) if a.comb else (TRK_VARIANTS, add_trk)
        stems = sorted(str(p)[:-8] for p in d.glob("L*.doa.npz"))[: a.limit or None]
        for i, stem in enumerate(stems):
            if all(f"{t}/c0/u" in np.load(stem + ".doa.npz") for t in variants):
                continue
            meta = json.loads(Path(stem + ".json").read_text())
            add(stem, meta["cluster"], meta["mount"], meta["fs"], len(meta["centres"]))
            print(f"{'comb' if a.comb else 'trk'} {i + 1}/{len(stems)}", flush=True)
        return
    est = None
    t0, n = time.time(), 0
    while True:
        scenes = sorted(js for js in d.glob("L*.json") if Path(str(js)[:-5] + ".truth.npz").exists())
        if a.limit:
            scenes = scenes[: a.limit]
        todo = [js for js in scenes if not Path(str(js)[:-5] + ".doa.npz").exists()]
        for js in todo:
            meta = json.loads(js.read_text())
            if est is None:
                est = estimators(meta["cluster"], meta["mount"], meta["fs"], a.variants)
            stem = str(js)[:-5]
            np.savez_compressed(stem + ".doa.npz", **run_scene(stem, est, len(meta["centres"])))
            n += 1
            print(f"{n} {js.stem} {time.time() - t0:.0f}s", flush=True)
        n_done = len(list(d.glob("L*.doa.npz")))
        if n_done >= a.wait_for:
            break
        time.sleep(60)
    print(f"DONE {n} scenes {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

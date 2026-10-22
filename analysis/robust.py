"""Robustness to recording-chain effects a real mic, recorder or phone adds, which the simulator does not.

    python -m analysis.robust --data data/sim_v1 --split test_sim --model models/B_logf_best.pt --frontend logf \
        --out analysis/out/B_logf.robust.json

Each perturbation is applied to every scene's waveform before the model; frame F1, recall and the false
positive rate on negative frames are compared with the clean audio (labels unchanged).
"""
import argparse
import io
import json
import time

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from analysis.dump import load_model
from cnn.data import frame_targets, scene_list
from cnn.evaluate import predict
from cnn.model import CLASSES, SR
from sim import sources as src
from sim.acoustics import FULL_SCALE_PA, set_level_dba


def _butter(x, kind, hz, order=4):
    sos = signal.butter(order, hz / (SR / 2), btype=kind, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _codec(x, fmt, subtype, level, sr=SR):
    """Encode and decode with a lossy codec (at `sr`; Opus needs 48 kHz)."""
    y = signal.resample_poly(x, sr // np.gcd(sr, SR), SR // np.gcd(sr, SR)) if sr != SR else x
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format=fmt, subtype=subtype, compression_level=level)
    buf.seek(0)
    y = sf.read(buf, dtype="float32")[0]
    if sr != SR:
        y = signal.resample_poly(y, SR // np.gcd(sr, SR), sr // np.gcd(sr, SR))
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)].astype(np.float32)


def _resample_via(x, sr):
    g = np.gcd(sr, SR)
    y = signal.resample_poly(x, sr // g, SR // g)
    return signal.resample_poly(y, SR // g, sr // g).astype(np.float32)[: len(x)]


def _clip_frac(x, frac):
    """Scale so that `frac` of samples exceed full scale, hard clip, scale back (a recorder gain set too high)."""
    a = np.quantile(np.abs(x), 1 - frac)
    g = 1.0 / max(a, 1e-9)
    return (np.clip(x * g, -1, 1) / g).astype(np.float32)


def _noise_spl(rng, n, dba):
    """White noise at a given A-weighted level, in the file's full-scale units."""
    return set_level_dba(rng.standard_normal(n).astype(np.float32), SR, dba) / FULL_SCALE_PA


def _tilt(x, db_per_decade):
    """Spectral tilt around 1 kHz (mic/enclosure response), via an FIR designed on a log-frequency ramp."""
    f = np.linspace(0, SR / 2, 513)
    gain_db = db_per_decade * np.log10(np.maximum(f, 20) / 1000)
    taps = signal.firwin2(1023, f / (SR / 2), 10 ** (gain_db / 20))
    return signal.fftconvolve(x, taps, mode="same").astype(np.float32)


def perturbations(rng):
    return {
        "clean": lambda x: x,
        "gain -30 dB": lambda x: x * 10 ** (-30 / 20),
        "gain +20 dB (clipped)": lambda x: np.clip(x * 10, -1, 1),
        "clip 0.1% samples": lambda x: _clip_frac(x, 0.001),
        "clip 2% samples": lambda x: _clip_frac(x, 0.02),
        "high-pass 100 Hz": lambda x: _butter(x, "high", 100),
        "high-pass 200 Hz": lambda x: _butter(x, "high", 200),
        "high-pass 400 Hz": lambda x: _butter(x, "high", 400),
        "low-pass 8 kHz (16 kHz recorder)": lambda x: _resample_via(x, 16000),
        "low-pass 4 kHz (8 kHz phone)": lambda x: _resample_via(x, 8000),
        "48 kHz round trip": lambda x: _resample_via(x, 48000),
        "MP3 low bitrate": lambda x: _codec(x, "MP3", "MPEG_LAYER_III", 0.9),
        "MP3 high bitrate": lambda x: _codec(x, "MP3", "MPEG_LAYER_III", 0.1),
        "Opus low bitrate": lambda x: _codec(x, "OGG", "OPUS", 0.9, 48000),
        "tilt -6 dB/decade (dull mic)": lambda x: _tilt(x, -6),
        "tilt +6 dB/decade (bright mic)": lambda x: _tilt(x, 6),
        "white noise 45 dBA (noisy preamp)": lambda x: x + _noise_spl(rng, len(x), 45),
        "white noise 55 dBA": lambda x: x + _noise_spl(rng, len(x), 55),
        "wind on mic 75 dBA": lambda x: x + set_level_dba(src.mic_wind(rng, len(x), SR), SR, 75) / FULL_SCALE_PA,
        "50 Hz hum + DC": lambda x: x + 0.02 + 0.01 * np.sin(2 * np.pi * 50 * np.arange(len(x)) / SR).astype(np.float32),
        "time shift 5 ms": lambda x: np.r_[np.zeros(160, np.float32), x[:-160]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--split", default="test_sim")
    ap.add_argument("--model", required=True)
    ap.add_argument("--frontend", required=True)
    ap.add_argument("--arch", default="cnn14")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="run only these perturbations")
    ap.add_argument("--append", action="store_true", help="add to an existing results file")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.model, a.frontend, a.arch, device)
    items = scene_list(a.data, a.split)
    scenes = []
    for stem in items:
        wave = sf.read(str(stem) + ".flac", dtype="float32")[0]
        y, m = frame_targets(np.load(str(stem) + ".npz"), len(wave) // 320 + 1)
        scenes.append((wave, y, m))
    results = json.load(open(a.out)) if a.append else {}
    for name, fn in perturbations(np.random.default_rng(0)).items():
        if a.only and name not in a.only and name != "clean":
            continue
        t0 = time.time()
        tp, fp, fn_, neg = (np.zeros(len(CLASSES)) for _ in range(4))
        for wave, y, m in scenes:
            x = fn(wave.copy()).astype(np.float32)
            pred = predict(model, x, device)[: len(y)] >= 0.5
            yy, mm = y[: len(pred)] > 0, m[: len(pred)] > 0
            tp += (pred & yy & mm).sum(0)
            fp += (pred & ~yy & mm).sum(0)
            fn_ += (~pred & yy & mm).sum(0)
            neg += (~yy & mm).sum(0)
        res = {c: {"f1": float(2 * tp[i] / max(2 * tp[i] + fp[i] + fn_[i], 1)), "recall": float(tp[i] / max(tp[i] + fn_[i], 1)),
                   "fp_rate": float(fp[i] / max(neg[i], 1))} for i, c in enumerate(CLASSES)}
        results[name] = res
        print(f"{name:36s} " + "  ".join(f"{c[:10]} F1 {r['f1']:.3f} R {r['recall']:.3f} FP {100 * r['fp_rate']:.2f}%" for c, r in res.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

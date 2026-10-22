"""Real recordings (test only, never used for training, design D-25).

    python -m analysis.real esc50 --root data/real/ESC-50-master --out analysis/out/real_esc50.pkl
    python -m analysis.real idmt  --root data/real/idmt --out analysis/out/real_idmt.pkl

esc50: 2,000 real 5 s environmental clips in 50 classes (no drones; CC BY-NC). Per clip and model, the
        peak probability per class and whether a >= 0.5 s event fires. Measures real false alarms.
idmt:  Fraunhofer IDMT outdoor drone flights (DJI Phantom/Mavic, Align; CC BY 4.0) in 30 s segments with
        GPS positions. Per 40 ms frame, the model probabilities and the drone's distance. Measures real
        detection against distance.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import signal

from analysis.dump import load_model
from cnn.evaluate import predict
from cnn.model import CLASSES, SR

MODELS = {"B": ("models/B_logf_best.pt", "logf"), "A": ("models/A_mel_best.pt", "mel")}


def load_mono(path, channel=0):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, min(channel, x.shape[1] - 1)]
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def event(p, threshold=0.5, min_frames=50):
    on = np.r_[0, (p >= threshold).astype(int), 0]
    d = np.diff(on)
    return bool(np.any(np.flatnonzero(d == -1) - np.flatnonzero(d == 1) >= min_frames))


def esc50(root, models, device, gain_db=0.0):
    rows = []
    meta = list(csv.DictReader(open(Path(root) / "meta" / "esc50.csv")))
    for i, r in enumerate(meta):
        x = load_mono(Path(root) / "audio" / r["filename"]) * 10 ** (gain_db / 20)
        row = {"file": r["filename"], "category": r["category"], "rms_dbfs": 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)}
        for name, model in models.items():
            p = predict(model, x, device)
            for c, cat in enumerate(CLASSES):
                row[f"{name}_max_{cat}"] = float(p[:, c].max())
                row[f"{name}_event_{cat}"] = event(p[:, c])
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"esc50 {i + 1}/{len(meta)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["esc50", "idmt"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gain_db", type=float, default=0.0, help="esc50: scale clips (they are normalised loud)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {k: load_model(p, f, "cnn14", device) for k, (p, f) in MODELS.items()}
    df = esc50(a.root, models, device, a.gain_db) if a.dataset == "esc50" else idmt(a.root, models, device)
    df.to_pickle(a.out)
    print(f"{len(df)} rows -> {a.out}")


def idmt(root, models, device):
    """Per 40 ms frame: model probabilities and the drone's horizontal distance (the GPS log has no altitude)."""
    rows = []
    segments = sorted(d for d in (Path(root) / "MicArrayData30s").iterdir() if d.is_dir())
    for i, seg in enumerate(segments):
        wav = next(seg.glob("*.wav"))
        pos = pd.read_csv(seg / "DOATruePos.csv")
        x = load_mono(wav, channel=0)
        n = len(x) // 320 + 1
        idx = np.arange(2, n, 4)
        t = idx / 100
        r = np.interp(t, pos.time, np.hypot(pos.X, pos.Y))
        flight = seg.name.rsplit("_", 2)[0]
        drone = next(k for k in ("Phantom4", "Mavic", "Align") if k in flight)
        kind = next(k for k in ("static", "linear", "random") if k in flight)
        row = {"segment": seg.name, "flight": flight, "drone": drone, "kind": kind, "t": t, "r": r,
               "rms_dbfs": 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)}
        for name, model in models.items():
            p = predict(model, x, device)
            for c, cat in enumerate(CLASSES):
                row[f"{name}_{cat}"] = p[idx[idx < len(p)], c] if len(p) >= n else np.interp(idx, np.arange(len(p)), p[:, c])
        rows.append(pd.DataFrame(row))
        if (i + 1) % 40 == 0:
            print(f"idmt {i + 1}/{len(segments)}", flush=True)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    main()

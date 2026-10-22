"""Run a trained model on a real recording (field test, design.md 14.7).

    python -m cnn.infer recording.wav --model runs/logf_v1/best.pt --frontend logf [--channel 0]

Writes <recording>.detections.png (spectrogram + per-class probability) and .csv (detected events).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
import torch

from .evaluate import predict
from .model import CLASSES, SR, DroneCNN14, SmallCNN


def events(prob, threshold, fps=100, min_len_s=0.5):
    on = prob >= threshold
    edges = np.diff(np.r_[0, on.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(s / fps, e / fps, float(prob[s:e].max())) for s, e in zip(starts, ends, strict=True) if e - s >= min_len_s * fps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", required=True)
    ap.add_argument("--frontend", default="logf")
    ap.add_argument("--arch", default="cnn14", choices=["cnn14", "small"])
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    x, sr = sf.read(a.wav, dtype="float32", always_2d=True)
    x = x[:, a.channel]
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    model = (DroneCNN14(a.frontend) if a.arch == "cnn14" else SmallCNN(a.frontend)).cuda().eval()
    model.load_state_dict(torch.load(a.model, map_location="cuda"))
    probs = predict(model, x, "cuda")

    out = Path(a.wav).with_suffix("")
    rows = ["class,start_s,end_s,max_prob"]
    for c, name in enumerate(CLASSES):
        for s, e, p in events(probs[:, c], a.threshold):
            rows.append(f"{name},{s:.2f},{e:.2f},{p:.3f}")
    Path(f"{out}.detections.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    f, t, X = signal.stft(x, SR, nperseg=4096, noverlap=4096 - 1280)
    P = 20 * np.log10(np.abs(X) + 1e-9)
    ax0.pcolormesh(t, f, P, vmin=np.percentile(P, 40), vmax=P.max(), shading="auto", cmap="magma")
    ax0.set_yscale("log")
    ax0.set_ylim(40, 8000)
    ax0.set_ylabel("Hz")
    tt = np.arange(len(probs)) / 100
    for c, name in enumerate(CLASSES):
        ax1.plot(tt, probs[:, c], label=name)
    ax1.axhline(a.threshold, color="k", ls=":")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("s")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"{out}.detections.png", dpi=80)


if __name__ == "__main__":
    main()

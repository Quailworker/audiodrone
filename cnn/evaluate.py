"""Evaluate a trained model on whole simulated scenes (design.md 5.8.7).

    python -m cnn.evaluate --data data/sim_v1 --split test_sim test_heldout --model runs/logf_v1/best.pt --frontend logf

Reports per class: frame precision/recall/F1 (ignore band excluded), detection probability vs distance,
and false alarms per hour on scenes with no drone. Writes <model dir>/eval_<split>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .data import frame_targets, load_meta, scene_list
from .model import CLASSES, DroneCNN14, SmallCNN

R_BINS = [0, 25, 50, 100, 200, 400, 800, 1600, 5000]


@torch.no_grad()
def predict(model, wave, device, chunk_s=60):
    """Frame probabilities for a whole clip, in overlapping chunks to bound memory."""
    sr, hop = 32000, 320
    n = len(wave)
    n_frames = n // hop + 1
    probs = np.zeros((n_frames, len(CLASSES)), np.float32)
    weight = np.zeros(n_frames, np.float32)
    step, size = chunk_s * sr // 2, chunk_s * sr
    for start in range(0, max(1, n - size // 2), step):
        x = torch.from_numpy(wave[start:start + size].astype(np.float32))[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            p = torch.sigmoid(model(x).float())[0].cpu().numpy()
        f0 = start // hop
        f1 = min(n_frames, f0 + len(p))
        probs[f0:f1] += p[: f1 - f0]
        weight[f0:f1] += 1
    return probs / np.maximum(weight, 1)[:, None]


def events_per_hour(pred, fps=100, min_len_s=0.5):
    on = np.flatnonzero(np.diff(np.r_[0, pred.astype(int), 0]) == 1)
    off = np.flatnonzero(np.diff(np.r_[0, pred.astype(int), 0]) == -1)
    return int(np.sum((off - on) >= min_len_s * fps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--split", nargs="+", default=["test_sim"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--frontend", default="logf")
    ap.add_argument("--arch", default="cnn14", choices=["cnn14", "small"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = a.device
    model = (DroneCNN14(a.frontend) if a.arch == "cnn14" else SmallCNN(a.frontend)).to(device).eval()
    model.load_state_dict(torch.load(a.model, map_location=device))

    for split in a.split:
        items = scene_list(a.data, split)
        tp, fp, fn = (np.zeros(len(CLASSES)) for _ in range(3))
        hits = {c: np.zeros(len(R_BINS) - 1) for c in CLASSES[:2]}
        tots = {c: np.zeros(len(R_BINS) - 1) for c in CLASSES[:2]}
        neg_hours, neg_events = 0.0, np.zeros(len(CLASSES))
        for stem in items:
            wave, _ = sf.read(str(stem) + ".flac", dtype="float32")
            npz = np.load(str(stem) + ".npz")
            meta = load_meta(stem)
            probs = predict(model, wave, device)
            y, m = frame_targets(npz, len(probs))
            pred = probs >= a.threshold
            tp += (pred & (y > 0) & (m > 0)).sum(0)
            fp += (pred & (y == 0) & (m > 0)).sum(0)
            fn += (~pred & (y > 0) & (m > 0)).sum(0)
            drones = [s for s in meta["sources"] if "cls" in s]
            ranges = [npz[f"drone{i}/r"][: len(probs)] for i in range(len(drones))]
            for i, s in enumerate(drones):
                c = CLASSES.index(s["category"])
                r = ranges[i]
                # with two drones of one class, a frame's detection is credited to the nearer one only
                same = [ranges[j] for j, o in enumerate(drones) if o["category"] == s["category"]]
                own = r <= np.min(same, axis=0)
                b = np.clip(np.digitize(r, R_BINS) - 1, 0, len(R_BINS) - 2)[own]
                np.add.at(tots[s["category"]], b, 1)
                np.add.at(hits[s["category"]], b, pred[: len(r), c][own])
            if not drones:
                neg_hours += len(wave) / 32000 / 3600
                false_pos = pred & (y == 0) & (m > 0)    # a correctly detected chainsaw is not a false alarm
                neg_events += [events_per_hour(false_pos[:, c]) for c in range(len(CLASSES))]
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        res = {"split": split, "scenes": len(items),
               "frame": {c: {"precision": float(tp[i] / max(tp[i] + fp[i], 1)),
                             "recall": float(tp[i] / max(tp[i] + fn[i], 1)), "f1": float(f1[i])}
                         for i, c in enumerate(CLASSES)},
               "p_detect_vs_range_m": {c: {f"{R_BINS[j]}-{R_BINS[j + 1]}": (float(hits[c][j] / tots[c][j]) if tots[c][j] else None)
                                           for j in range(len(R_BINS) - 1)} for c in hits},
               "false_alarms_per_hour_no_drone_scenes": {c: float(neg_events[i] / max(neg_hours, 1e-9)) for i, c in enumerate(CLASSES)},
               "no_drone_hours": neg_hours}
        print(json.dumps(res, indent=1))
        (Path(a.model).parent / f"eval_{split}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

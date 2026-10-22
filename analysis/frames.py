"""Join model predictions with everything known about each frame, for slicing errors.

    python -m analysis.frames --data data/sim_v1 data/sim_eval --pred analysis/out/B_logf --out analysis/out/B_logf.frames.pkl

One row per 40 ms (every 4th label frame, aligned with the 25 fps coarse spectra stored per source):
probabilities, targets and masks per class, the tone-region SNRs behind the labels, the nearest drone's
geometry and parameters, scene conditions, and the loudest interfering source in 50 Hz - 8 kHz at that moment.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cnn.data import frame_targets
from cnn.model import CLASSES

EDGES = np.geomspace(40, 14000, 49)
IN_BAND = (EDGES[:-1] >= 50) & (EDGES[1:] <= 8000)
DRONE_CATS = CLASSES[:2]


def band_db(bands):
    """Coarse per-source spectrum (48, n) in dB -> total power in 50 Hz - 8 kHz, dB."""
    p = 10 ** (bands.astype(np.float32)[IN_BAND] / 10)
    return 10 * np.log10(p.sum(0) + 1e-20)


def scene_rows(args):
    stem, pred_path = args
    probs = np.load(pred_path).astype(np.float32)
    npz = np.load(str(stem) + ".npz")
    meta = json.loads(Path(str(stem) + ".json").read_text())
    n = min(len(probs), len(npz["frame_t"]))
    y, m = frame_targets(npz, n)
    idx = np.arange(2, n, 4)
    k = idx // 4
    row = {"scene": stem.name, "split": stem.parent.name, "t": idx / 100, "t_end": (n - idx) / 100}
    for c, cat in enumerate(CLASSES):
        row[f"p_{cat}"] = probs[idx, c]
        row[f"y_{cat}"] = y[idx, c]
        row[f"m_{cat}"] = m[idx, c]
        lines = npz[f"snr_lines/{cat}"].astype(np.float32)[idx] if f"snr_lines/{cat}" in npz else np.full((len(idx), 4), -100.0)
        row[f"l0_{cat}"], row[f"l1_{cat}"] = lines[:, 0], lines[:, 1]

    # source levels in band (25 fps); the loudest non-drone source is the "interferer"
    levels = {}
    for key in npz.files:
        if key.startswith("bands/"):
            lv = band_db(npz[key])
            levels[key[6:]] = np.pad(lv, (0, max(0, k.max() + 1 - len(lv))), mode="edge")[k]
    others = [c for c in levels if c not in DRONE_CATS]
    stack = np.stack([levels[c] for c in others])
    row["interferer"] = np.array(others)[stack.argmax(0)]
    row["interferer_db"] = stack.max(0)
    row["bed_db"] = levels["bed"]
    for cat in DRONE_CATS:
        row[f"lvl_{cat}"] = levels.get(cat, np.full(len(idx), -200.0))

    # nearest drone at each frame
    drones = [s for s in meta["sources"] if "cls" in s]
    row["n_drones"] = len(drones)
    if drones:
        r = np.stack([npz[f"drone{i}/r"][idx] for i in range(len(drones))])
        near = r.argmin(0)
        pick = lambda key: np.stack([npz[f"drone{i}/{key}"][idx] for i in range(len(drones))])[near, np.arange(len(idx))]  # noqa: E731
        row.update(r=r.min(0), elev=pick("elev_deg"), rdot=pick("rdot"), bpf_rx=pick("bpf_rx"), bpf_emit=pick("bpf_emit"))
        for key in ["category", "cls", "family", "blades", "rotors", "prop_in", "rpm_hover", "level", "speed", "alt", "cpa_m", "pole_pairs"]:
            row[f"d_{key}"] = np.array([d.get(key, np.nan) if d.get(key) is not None else np.nan for d in drones], dtype=object)[near]
    kinds = sorted({s.get("kind", s["category"]) for s in meta["sources"] if "cls" not in s})
    row.update(confusers="+".join(kinds), beds="+".join(meta["beds"]), bed_level=meta["bed_level"],
               wind_level=meta["wind_level"] or 0.0, mic_h=meta["mic_pos"][0][2], ground_r=meta["ground_r"],
               heldout=meta["heldout"], duration=meta["duration"])
    return pd.DataFrame(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for pred in sorted(Path(a.pred).glob("*/*.npy")):
        for root in a.data:
            stem = Path(root) / pred.parent.name / pred.stem
            if stem.with_suffix(".npz").exists():
                jobs.append((stem, pred))
                break
    with ProcessPoolExecutor(a.workers) as pool:
        df = pd.concat(pool.map(scene_rows, jobs, chunksize=8), ignore_index=True)
    for col in ["scene", "split", "interferer", "confusers", "beds", "d_category", "d_cls", "d_family"]:
        if col in df:
            df[col] = df[col].astype("category")
    df.to_pickle(a.out)
    print(f"{len(jobs)} scenes, {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()

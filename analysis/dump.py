"""Run a model over scene folders and save its per-frame probabilities, for the failure analysis.

    python -m analysis.dump --data data/sim_v1 data/sim_eval --split test_sim test_heldout train val \
        --model models/B_logf_best.pt --frontend logf --arch cnn14 --out analysis/out/B_logf

Writes <out>/<split>/<scene>.npy (float16, frames x classes). Scenes already dumped are skipped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from cnn.data import scene_list
from cnn.evaluate import predict
from cnn.model import DroneCNN14, SmallCNN


def load_model(path, frontend, arch, device):
    model = (DroneCNN14(frontend) if arch == "cnn14" else SmallCNN(frontend)).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frontend", required=True)
    ap.add_argument("--arch", default="cnn14")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.model, a.frontend, a.arch, device)
    jobs = []
    for split in a.split:
        for stem in scene_list(a.data, split):
            dst = Path(a.out) / split / f"{stem.name}.npy"
            if not dst.exists():
                jobs.append((stem, dst))
    print(f"{len(jobs)} scenes to predict", flush=True)

    def read(job):
        return job, sf.read(str(job[0]) + ".flac", dtype="float32")[0]

    with ThreadPoolExecutor(4) as pool:
        for i, ((_stem, dst), wave) in enumerate(pool.map(read, jobs)):
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, predict(model, wave, device).astype(np.float16))
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

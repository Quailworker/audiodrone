"""Scene manifest: check that a re-rendered dataset matches the one the published models were trained on.

    python scripts/manifest.py make   data/sim_v1 results/sim_v1_manifest.csv
    python scripts/manifest.py verify data/sim_v1 results/sim_v1_manifest.csv [--split test_sim test_heldout]

Rendering is deterministic per seed. On the same platform the FLAC files are bit-identical; across platforms
(the reference set was rendered on Linux and Windows) samples may differ by 1 LSB of 24-bit audio (-150 dB),
so scenes are compared by length, level and label counts with small tolerances, not by file hash.
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn.data import frame_targets  # noqa: E402
from cnn.model import CLASSES  # noqa: E402

FIELDS = ["name", "split", "n_samples", "rms_db", "sources", *[f"pos_{c}" for c in CLASSES], *[f"neg_{c}" for c in CLASSES]]


def describe(stem):
    x, _ = sf.read(str(stem) + ".flac", dtype="float64")
    npz = np.load(str(stem) + ".npz")
    meta = json.loads(Path(str(stem) + ".json").read_text())
    y, m = frame_targets(npz, len(npz["frame_t"]))
    row = {"name": stem.name, "split": stem.parent.name, "n_samples": len(x),
           "rms_db": round(10 * np.log10(np.mean(x ** 2) + 1e-30), 3),
           "sources": "+".join(sorted(s["category"] for s in meta["sources"]))}
    for i, c in enumerate(CLASSES):
        row[f"pos_{c}"] = int(((y[:, i] > 0) & (m[:, i] > 0)).sum())
        row[f"neg_{c}"] = int(((y[:, i] == 0) & (m[:, i] > 0)).sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "verify"])
    ap.add_argument("data")
    ap.add_argument("manifest")
    ap.add_argument("--split", nargs="*", help="verify only these splits")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    stems = sorted(js.with_suffix("") for js in Path(a.data).glob("*/*.json"))
    if a.split:
        stems = [s for s in stems if s.parent.name in a.split]
    with Pool(a.workers) as pool:
        rows = pool.map(describe, stems, chunksize=16)

    if a.mode == "make":
        with open(a.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} scenes to {a.manifest}")
        return

    ref = {r["name"]: r for r in csv.DictReader(open(a.manifest))}
    bad = 0
    for r in rows:
        q = ref.get(r["name"])
        if q is None:
            print("not in manifest:", r["name"])
            bad += 1
            continue
        problems = []
        if int(q["n_samples"]) != r["n_samples"] or q["sources"] != r["sources"] or q["split"] != r["split"]:
            problems.append("different scene")
        if abs(float(q["rms_db"]) - r["rms_db"]) > 0.01:
            problems.append(f"level {r['rms_db']} vs {q['rms_db']} dB")
        for k in FIELDS[5:]:
            if abs(int(q[k]) - r[k]) > max(2, 0.005 * int(q[k])):
                problems.append(f"{k} {r[k]} vs {q[k]}")
        if problems:
            print(r["name"], "; ".join(problems))
            bad += 1
    print(f"checked {len(rows)} scenes, {bad} mismatched")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

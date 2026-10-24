"""Cluster spacing study on flat open ground (UPGRADES / NEXT_STEPS item: tracks fragment and give false
"over it now" alerts with clusters 100-300 m apart).

    python scripts/spacing_study.py --out results/spacing --workers 4 [--chain v1|v2]

Layout: five flat clusters, four on the corners of a square of side S and one in the middle, a protected
site (radius 60 m) at the middle. S = 100, 300, 600, 1000 m. Flights, each in its own run:
  head-on    straight at the site from 3 km
  offset     a straight pass 400 m to the side of the site (should never give take-cover)
for three classes (mid 7-12 in electric, fpv, ts_mr two-stroke multirotor). Flat ground, desert ambient.

Per run (results/spacing/runs.csv):
  first_cnn_s, first_track_s, first_take_cover_s, warning_s (site entry - take-cover)
  track_ids        distinct confirmed track ids that followed the drone (fragmentation: 1 is ideal)
  false_cover      take-cover alerts in an offset run
  over_now_far     "over it now" alerts sent while the drone was > 2 site radii from the site
  track_err_m      median horizontal track error
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SIDES = [100, 300, 600, 1000]
CLASSES = {"mid": {"agl": 60, "speed": 15}, "fpv": {"agl": 30, "speed": 25}, "ts_mr": {"agl": 100, "speed": 15}}
SITE_R = 60.0


def scenario(side, cls, flight, seed):
    h = side / 2
    clusters = [[-h, -h], [h, -h], [h, h], [-h, h], [0, 0]]
    y = 0.0 if flight == "head_on" else 400.0
    end = [0, 0] if flight == "head_on" else [1500, y]
    return {"origin": [23.13, 53.77], "half_m": 4000, "terrain": False, "mount": "ground",
            "clusters": clusters, "sites": [{"name": "Camp", "x": 0, "y": 0, "radius_m": SITE_R}],
            "drones": [{"cls": cls, "waypoints": [[-3000, y], end], "alt_mode": "agl", **CLASSES[cls]}],
            "ambient": "desert", "temp_c": 35, "rh": 30, "seed": seed,
            "title": f"spacing {side} m, {cls}, {flight}"}


def measure(res, side, cls, flight):
    s = res["summary"]
    d = res["drones"][0]
    tt, pp = np.array(d["t"]), np.array(d["p"])
    tf = np.array(res["t"])
    px, py = np.interp(tf, tt, pp[:, 0]), np.interp(tf, tt, pp[:, 1])
    ids, errs = set(), []
    for i, f in enumerate(res["frames"]):
        for tr in f["tracks"]:
            e = np.hypot(tr["x"] - px[i], tr["y"] - py[i])
            if e < 150:
                ids.add(tr["id"])
        if f["tracks"]:
            errs.append(min(np.hypot(tr["x"] - px[i], tr["y"] - py[i]) for tr in f["tracks"]))
    covers = [a for a in res["alerts"] if a["tier"] == "take_cover" and a["to"] == "operators"]
    over_far = 0
    for a in covers:
        if re.search(r"over it now", a["text"]):
            i = min(np.searchsorted(tf, a["t"]), len(tf) - 1)
            over_far += int(np.hypot(px[i], py[i]) > 2 * SITE_R)
    return {"side_m": side, "cls": cls, "flight": flight,
            "first_cnn_s": s["first_cnn_s"], "first_track_s": s["first_track_s"],
            "first_take_cover_s": s["first_take_cover_s"], "enters_s": s["drone_enters_site_s"],
            "warning_s": s["warning_s"], "track_ids": len(ids), "false_cover": len(covers) if flight == "offset" else 0,
            "over_now_far": over_far, "track_err_m": round(float(np.median(errs)), 1) if errs else None,
            "first_detect_range_m": None if s["first_cnn_s"] is None else
            round(float(np.hypot(np.interp(s["first_cnn_s"], tt, pp[:, 0]), np.interp(s["first_cnn_s"], tt, pp[:, 1])))),
            "run_s": res["timing"]["total_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/spacing")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chain", default="v2", choices=["v1", "v2"], help="v1: CNN + plain beamformer, directions only;"
                    " v2: + frequency tracker, tracker-guided beamformer and Doppler")
    ap.add_argument("--sides", type=int, nargs="+", default=SIDES)
    ap.add_argument("--classes", nargs="+", default=list(CLASSES))
    a = ap.parse_args()
    from terra.pipeline import run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"runs_{a.chain}.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(int(r["side_m"]), r["cls"], r["flight"]) for r in csv.DictReader(f)}
    for side in a.sides:
        for cls in a.classes:
            for flight in ("head_on", "offset"):
                if (side, cls, flight) in done:
                    continue
                scn = scenario(side, cls, flight, seed=7)
                scn["chain"] = a.chain
                t0 = time.time()
                res = run(scn, out / "runs" / f"{a.chain}_{side}_{cls}_{flight}", progress=lambda m: None, workers=a.workers)
                row = measure(res, side, cls, flight)
                new = not rows_path.exists()
                with open(rows_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(json.dumps(row), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

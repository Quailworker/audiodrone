"""Simulated border incidents for the Border watch screen: each one rendered on the real terrain at a point
of the UAE land border, heard by the deployment clusters there (terra/border.py), and run through the whole
chain (CNN, frequency tracker, tracker-guided beamformer, EKF, alert core, border monitor).

    python scripts/border_incidents.py [--only HATTA-1 ...] [--workers 3]

Full runs go to runs_border/<id>/ (audio, alerts, result.json). A compact replay of each, small enough to
ship with the repo, goes to terra/data/incidents/<id>.json; the screen replays them on one watch timeline
(t0 = when the incident starts in the watch).

Everything an operator sees in the replay is chain output. The simulated truth (where the drone really
was) is stored separately and only shown when "truth" is switched on.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from terra import replay
from terra.border import load, local_border, local_clusters, nearest_border_point

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "terra" / "data" / "incidents"
RUNS = ROOT / "runs_border"

INCIDENTS = [
    {"id": "HATTA-1", "name": "Hatta, Hajar mountains", "near": (24.80, 56.13), "cls": "fpv", "speed": 25, "agl": 40,
     "start_m": 2600, "in_m": 1500, "angle_deg": 20, "ambient": "desert_wind", "t0": 45, "seed": 101},
    {"id": "AIN-1", "name": "Al Ain / Buraimi", "near": (24.24, 55.80), "cls": "mid", "speed": 14, "agl": 60,
     "start_m": 2000, "in_m": 1400, "angle_deg": -15, "ambient": "town", "t0": 330, "seed": 102},
    {"id": "RAK-1", "name": "Musandam border, Jebel Jais", "near": (25.90, 56.15), "cls": "mid", "speed": 12, "agl": 80,
     "start_m": 1800, "in_m": 1200, "angle_deg": 10, "ambient": "desert", "t0": 660, "seed": 103},
    {"id": "GHW-1", "name": "Ghuwaifat, Saudi border", "near": (24.10, 51.75), "cls": "ts_fw", "speed": 30, "agl": 150,
     "start_m": 4500, "in_m": 2400, "angle_deg": 25, "ambient": "desert_wind", "t0": 960, "seed": 104},
    {"id": "LIWA-1", "name": "South of Liwa, Saudi border", "near": (22.85, 53.60), "cls": "ts_mr", "speed": 15,
     "agl": 100, "start_m": 2800, "in_m": 1500, "angle_deg": -10, "ambient": "desert", "t0": 1260, "seed": 105},
    {"id": "SAU-P", "name": "Saudi side, flight along the border", "near": (22.75, 54.40), "cls": "mid", "speed": 14,
     "agl": 60, "parallel": True, "offset_m": 1000, "length_m": 3500, "ambient": "desert", "t0": 1560, "seed": 106},
]


def scenario(inc, doc):
    sector, lon_b, lat_b, tan = nearest_border_point(doc, *inc["near"])
    n = np.array([-tan[1], tan[0]])                                    # into the UAE
    if inc.get("parallel"):
        a = -inc["offset_m"] * n - inc["length_m"] / 2 * tan
        b = -inc["offset_m"] * n + inc["length_m"] / 2 * tan
        wps, sites = [a, b], []
        half = max(inc["length_m"] / 2, inc["offset_m"]) + 1500
    else:
        ang = math.radians(inc["angle_deg"])
        d = math.cos(ang) * n + math.sin(ang) * tan
        wps = [-inc["start_m"] * d, inc["in_m"] * d]
        post = (inc["in_m"] - 300) * d
        sites = [{"name": f"Post {inc['id']}", "x": round(float(post[0]), 1), "y": round(float(post[1]), 1), "radius_m": 100}]
        half = max(inc["start_m"], inc["in_m"]) + 500
    half = float(min(half, 5000))
    xy, idx = local_clusters(doc, lat_b, lon_b, radius_m=3500, limit=24)
    scn = {"origin": [round(lat_b, 6), round(lon_b, 6)], "half_m": half, "terrain": True, "mount": "ground",
           "clusters": np.round(xy, 1).tolist(), "sites": sites,
           "drones": [{"cls": inc["cls"], "waypoints": [np.round(w, 1).tolist() for w in wps], "agl": inc["agl"],
                       "speed": inc["speed"], "alt_mode": "agl"}],
           "border": local_border(doc, lat_b, lon_b, half),
           "ambient": inc["ambient"], "temp_c": 38, "rh": 25, "seed": inc["seed"], "chain": "v2",
           "area_name": inc["name"], "title": f"{inc['id']} {inc['name']}"}
    return scn, sector, idx


def compact(inc, scn, sector, idx, res):
    """Replay file for the screen (terra/replay.py)."""
    return replay.compact(res, scn, {"id": inc["id"], "name": inc["name"], "t0": inc["t0"], "sector": sector["id"],
                                     "neighbour": sector["neighbour"], "cluster_index": idx})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from terra.pipeline import run
    doc = load()
    OUT.mkdir(parents=True, exist_ok=True)
    for inc in INCIDENTS:
        if a.only and inc["id"] not in a.only:
            continue
        scn, sector, idx = scenario(inc, doc)
        t0 = time.time()
        out = RUNS / inc["id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "scenario.json").write_text(json.dumps(scn, indent=1))
        res = run(scn, out, progress=lambda m: None, workers=a.workers)
        (out / "result.json").write_text(json.dumps(res))
        rep = compact(inc, scn, sector, idx, res)
        (OUT / f"{inc['id']}.json").write_text(json.dumps(rep, separators=(",", ":")))
        tiers = [x["tier"] for x in res["alerts"]]
        print(json.dumps({"id": inc["id"], "sector": sector["id"], "clusters": len(scn["clusters"]),
                          "first_cnn_s": res["summary"]["first_cnn_s"], "first_track_s": res["summary"]["first_track_s"],
                          "track_err_m": res["summary"]["median_track_error_m"],
                          "border_watch": tiers.count("border_watch"), "border_cross": tiers.count("border_cross"),
                          "take_cover": tiers.count("take_cover"), "s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()

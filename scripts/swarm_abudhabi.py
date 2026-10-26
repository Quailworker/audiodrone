"""Simulated swarm on a protected site in central Abu Dhabi: how do the tracker and the alerts cope with several
drones at once? (A defensive stress test of our chain; one drone or two were all we had tested.)

    python scripts/swarm_abudhabi.py [--workers 3] [--layout ring|dense]

Scenario (terra/scenario.py format; seed fixed):
  site      a generic protected site, radius 300 m, at the origin (no real facility is modelled)
  clusters  "ring": 16 flat ground clusters at 1.5 km plus 4 at 600 m (what works in open desert);
            "dense": 12 at 400 m, 8 at 800 m and 1 at the site (what a city needs); any in the sea are dropped
  swarm     6 drones from six directions: 3 electric 7-12 in quads, 2 FPVs and 1 two-stroke multirotor,
            each starting far enough out to arrive with the others (~160 s), then orbiting the site at
            200 m for about a minute
  ambient   town noise (urban bed + traffic); the elevation model includes the buildings, which screen sound

Writes the run to runs_terra/swarm-abudhabi[-dense]/ (open it in the terrain tool under "Previous runs") and
the tracker analysis to results/swarm_abudhabi/<layout>/report.json and report.png.

Result of the first ("ring") run: in town noise and among buildings the drones were barely heard (147 of
~33,000 cluster-frames) and no track ever formed. A small drone 500 m or more from the nearest cluster is
below the city's background noise.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs_terra" / "swarm-abudhabi"
OUT = ROOT / "results" / "swarm_abudhabi"
ORIGIN = [24.470, 54.370]
ARRIVE_S = 160.0
MATCH_M = 150.0
# (class, bearing it comes from in degrees from north, agl m, speed m/s, loiter offset east/north m)
SWARM = [("mid", 315, 70, 14, (-120, 80)), ("mid", 0, 80, 13, (0, 150)), ("fpv", 270, 35, 24, (-160, -40)),
         ("mid", 135, 60, 14, (120, -90)), ("fpv", 90, 30, 25, (150, 40)), ("ts_mr", 200, 100, 17, (-30, -160))]


LAYOUTS = {"ring": ((1500.0, 16), (600.0, 4)), "dense": ((400.0, 12), (800.0, 8), (0.0, 1))}
ORBIT_M = 200.0


def scenario(layout="ring"):
    from terra.dem import Terrain
    T = Terrain(*ORIGIN, half_m=4500.0, res_m=30.0)
    clusters = []
    for i, (r, n) in enumerate(LAYOUTS[layout]):
        for k in range(n):
            a = 2 * math.pi * k / n + (math.pi / n if i % 2 else 0)        # alternate rings staggered
            x, y = r * math.sin(a), r * math.cos(a)
            if T.height(x, y) > 0.5:                                       # on land
                clusters.append([round(x, 1), round(y, 1)])
    drones = []
    for cls, brg, agl, speed, (lx, ly) in SWARM:
        d0 = speed * ARRIVE_S
        a = math.radians(brg)
        start = [round(d0 * math.sin(a), 1), round(d0 * math.cos(a), 1)]
        wps = [start, [lx, ly]]
        if layout != "ring":                                               # then orbit the site for about a minute
            a0 = math.atan2(lx, ly)
            wps += [[round(ORBIT_M * math.sin(a0 + j * math.pi / 3), 1), round(ORBIT_M * math.cos(a0 + j * math.pi / 3), 1)]
                    for j in range(1, 7)]
        drones.append({"cls": cls, "waypoints": wps, "agl": agl, "speed": speed, "alt_mode": "agl"})
    return {"origin": ORIGIN, "half_m": 4500.0, "terrain": True, "mount": "ground", "clusters": clusters,
            "sites": [{"name": "Protected site", "x": 0.0, "y": 0.0, "radius_m": 300.0}],
            "drones": drones, "ambient": "town", "temp_c": 34, "rh": 45, "seed": 2026, "chain": "v2",
            "area_name": f"Abu Dhabi: simulated swarm on a protected site ({layout} layout)",
            "title": f"Swarm of 6 on a protected site, central Abu Dhabi, {layout} layout"}


def analyse(res):
    """Per drone: when it was first heard, first tracked (a confirmed track within MATCH_M), how much of its
    flight was tracked, how many track ids followed it, the median error. Overall: true drones against
    confirmed tracks over time, and ghost tracks (no drone within MATCH_M)."""
    tf = np.asarray(res["t"])
    truth = []
    for d in res["drones"]:
        tt, pp = np.asarray(d["t"]), np.asarray(d["p"])
        truth.append(np.column_stack([np.interp(tf, tt, pp[:, i]) for i in range(3)]))
    audible = np.asarray(res["audible"], bool)                              # (K, F): any drone audible
    n_drone = len(truth)
    per = [{"drone": i, "cls": res["drones"][i]["cls"], "first_track_s": None, "tracked_frac": 0.0,
            "track_ids": set(), "errs": []} for i in range(n_drone)]
    n_tracks, n_ghost, n_true_near = [], [], []
    for f, fr in enumerate(res["frames"]):
        trs = [tr for tr in fr["tracks"] if tr.get("n", 0) >= 5]
        n_tracks.append(len(trs))
        used = set()
        for tr in trs:
            d = [math.hypot(tr["x"] - P[f, 0], tr["y"] - P[f, 1]) for P in truth]
            j = int(np.argmin(d))
            if d[j] <= MATCH_M:
                used.add(tr["id"])
                per[j]["track_ids"].add(tr["id"])
        n_ghost.append(sum(1 for tr in trs if tr["id"] not in used))
        near = 0
        for i, P in enumerate(truth):
            d = [math.hypot(tr["x"] - P[f, 0], tr["y"] - P[f, 1]) for tr in trs]
            if d and min(d) <= MATCH_M:
                near += 1
                per[i]["errs"].append(min(d))
                if per[i]["first_track_s"] is None:
                    per[i]["first_track_s"] = round(float(tf[f]), 1)
        n_true_near.append(near)
    for i, p in enumerate(per):
        p["tracked_frac"] = round(len(p["errs"]) / len(tf), 3)
        p["track_ids"] = len(p["track_ids"])
        p["median_err_m"] = round(float(np.median(p["errs"])), 1) if p["errs"] else None
        p["dist_at_first_track_m"] = None
        if p["first_track_s"] is not None:
            f = int(np.searchsorted(tf, p["first_track_s"]))
            p["dist_at_first_track_m"] = round(float(np.hypot(*truth[i][f, :2])))
        del p["errs"]
    alerts = [a for a in res["alerts"] if a["to"] == "operators"]
    tiers = {}
    for a in alerts:
        tiers[a["tier"]] = tiers.get(a["tier"], 0) + 1
    first_cover = next((a["t"] for a in alerts if a["tier"] == "take_cover"), None)
    return {"drones": n_drone, "clusters": len(res["clusters"]), "duration_s": res["duration"],
            "first_heard_s": res["summary"]["first_cnn_s"], "per_drone": per,
            "max_drones_tracked_at_once": int(max(n_true_near)) if n_true_near else 0,
            "max_confirmed_tracks_at_once": int(max(n_tracks)) if n_tracks else 0,
            "ghost_track_frames_frac": round(float(np.mean([g > 0 for g in n_ghost])), 3),
            "first_take_cover_s": first_cover, "operator_alerts": tiers,
            "cluster_frames_hearing_a_drone": int(audible.sum()),
            "series": {"t": tf.round(1).tolist(), "tracks": n_tracks, "drones_tracked": n_true_near, "ghosts": n_ghost}}


def plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = rep["series"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={"height_ratios": [2, 1.3]})
    ax[0].plot(s["t"], s["tracks"], label="confirmed tracks", color="#d9534f")
    ax[0].plot(s["t"], s["drones_tracked"], label="drones with a track within 150 m", color="#2b8a3e")
    ax[0].plot(s["t"], s["ghosts"], label="ghost tracks (no drone near)", color="#868e96", lw=1)
    ax[0].axhline(rep["drones"], color="#1c7ed6", ls="--", lw=1, label=f"drones in the swarm ({rep['drones']})")
    if rep["first_take_cover_s"] is not None:
        ax[0].axvline(rep["first_take_cover_s"], color="#e8590c", ls=":", lw=1.5, label="first take-cover")
    ax[0].set_ylabel("count")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("Simulated swarm on a protected site, central Abu Dhabi: what the tracker held")
    names = [f"{p['drone']}: {p['cls']}" for p in rep["per_drone"]]
    frac = [p["tracked_frac"] for p in rep["per_drone"]]
    ids = [p["track_ids"] for p in rep["per_drone"]]
    ax[1].barh(names, frac, color="#2b8a3e")
    for k, (f, n) in enumerate(zip(frac, ids, strict=True)):
        ax[1].text(f + 0.01, k, f"{f:.0%} tracked, {n} track id(s)", va="center", fontsize=8)
    ax[1].set_xlim(0, 1.15)
    ax[1].set_xlabel("fraction of the flight with a track within 150 m")
    ax[1].invert_yaxis()
    ax[0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--layout", choices=list(LAYOUTS), default="dense")
    a = ap.parse_args()
    from terra.pipeline import run
    run_dir = RUN if a.layout == "ring" else RUN.with_name(f"{RUN.name}-{a.layout}")
    out = OUT / a.layout
    scn = scenario(a.layout)
    print(f"{len(scn['clusters'])} clusters on land, {len(scn['drones'])} drones", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scenario.json").write_text(json.dumps(scn, indent=1))
    t0 = time.time()
    res = run(scn, run_dir, progress=lambda m: print(f"  {time.time() - t0:6.0f}s {m}", flush=True), workers=a.workers)
    res["id"] = run_dir.name
    (run_dir / "result.json").write_text(json.dumps(res))
    out.mkdir(parents=True, exist_ok=True)
    rep = analyse(res)
    rep["layout"] = a.layout
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    plot(rep, out / "report.png")
    print(json.dumps({k: v for k, v in rep.items() if k != "series"}, indent=1))


if __name__ == "__main__":
    main()

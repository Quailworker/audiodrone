"""Re-run tracking and alerts on saved runs with the current tracker settings, without rendering again.

    python -m terra.retrack runs_border/LIWA-1 runs_terra/B1026-162047 ...
    python -m terra.retrack --all            # every border incident and every test built on the screen

A run directory holds the scenario and the full result, which keeps every cluster's direction, CNN probability
and pitch per frame. Only terra.pipeline.track_and_alert runs again; the result's frames, alerts and the
summary's track and alert times are replaced (the old result is kept as result.before-retrack.json), and the
replay the Border watch screen plays is rewritten from it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import P_DET, track_and_alert
from .replay import compact
from .scenario import load_terrain

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"


def retrack_result(run_dir):
    run_dir = Path(run_dir)
    scn = json.loads((run_dir / "scenario.json").read_text())
    res = json.loads((run_dir / "result.json").read_text())
    terrain = load_terrain(scn)
    centres = np.array([[c["x"], c["y"], c["z"]] for c in res["clusters"]])
    tf = np.asarray(res["t"])
    p_cnn = np.asarray(res["p_cnn"], float)
    U = np.asarray(res["u"], float)
    use = np.asarray(res["used"], bool)
    det = p_cnn >= P_DET
    pitch = np.asarray(res.get("pitch_hz", np.zeros_like(p_cnn)), float)[..., None].repeat(2, -1)
    pitch[pitch <= 0] = np.nan
    frames, events = track_and_alert(scn, terrain, centres, tf, U, use, det, p_cnn, pitch,
                                      scn.get("chain", "v2") == "v2" and scn.get("doppler", False), run_dir / "alerts.jsonl")
    alerts = [{"t": e["t"], "tier": e["tier"], "to": e["to"], "text": e["text"]} for e in events if e["tier"] != "log"]
    s = res["summary"]
    s["first_track_s"] = next((f["t"] for f in frames if f["tracks"]), None)
    s["first_alert_s"] = alerts[0]["t"] if alerts else None
    s["first_take_cover_s"] = next((a["t"] for a in alerts if a["tier"] == "take_cover"), None)
    entered = s.get("drone_enters_site_s")
    s["warning_s"] = round(entered - s["first_take_cover_s"], 1) if entered is not None and s["first_take_cover_s"] is not None else None
    if res["drones"]:
        tt, pp = np.asarray(res["drones"][0]["t"]), np.asarray(res["drones"][0]["p"])
        errs = [min(np.hypot(tr["x"] - np.interp(f["t"], tt, pp[:, 0]), tr["y"] - np.interp(f["t"], tt, pp[:, 1]))
                    for tr in f["tracks"]) for f in frames if f["tracks"]]
        s["median_track_error_m"] = round(float(np.median(errs)), 1) if errs else None
    backup = run_dir / "result.before-retrack.json"
    if not backup.exists():
        (run_dir / "result.json").replace(backup)
    res["frames"], res["alerts"], res["retracked"] = frames, alerts, True
    (run_dir / "result.json").write_text(json.dumps(res))
    return scn, res


def rewrite_replay(run_dir, scn, res):
    """Rewrite the screen's replay for this run, keeping its id, name, watch time and sector."""
    for f in (DATA / "incidents" / f"{Path(run_dir).name}.json", DATA / "user" / f"{Path(run_dir).name}.json"):
        if f.exists():
            old = json.loads(f.read_text())
            meta = {k: old[k] for k in ("id", "name", "t0", "sector", "neighbour", "cluster_index", "built") if k in old}
            f.write_text(json.dumps(compact(res, scn, meta), separators=(",", ":")))
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every border incident and every built test")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    if a.all:
        runs += [ROOT / "runs_border" / f.stem for f in sorted((DATA / "incidents").glob("*.json"))]
        runs += [ROOT / "runs_terra" / f.stem for f in sorted((DATA / "user").glob("*.json"))]
    for r in runs:
        if not (r / "result.json").exists():
            print(f"{r}: no saved result, skipped")
            continue
        scn, res = retrack_result(r)
        ids = {tr["id"] for f in res["frames"] for tr in f["tracks"]}
        most = max((len(f["tracks"]) for f in res["frames"]), default=0)
        print(f"{r.name}: {len(ids)} track ids, at most {most} at once, {len(res['alerts'])} alerts; replay {rewrite_replay(r, scn, res)}")


if __name__ == "__main__":
    main()
